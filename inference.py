import os
import sys
sys.path.append(".")
sys.path.append("..")
# sys.path.append("...")
import torch
import accelerate 
import logging
import hydra
from collections import Counter
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np 
from torch import optim
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import json
import time
from utils.logging import setup_logging
import random
from accelerate.utils import gather_object


from torch.nn.parallel import DistributedDataParallel
# import tensorflow_hub as hub

from evaluation.calvin.utils.multistep_sequences import get_sequences
from evaluation.calvin.utils.infer import get_env_state_for_initial_condition, join_vis_lang
from evaluation.calvin.utils.rollout_video import RolloutVideo

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (idx, self.data[idx])
        
def get_video_tag(i):
    # if dist.is_available() and dist.is_initialized():
    #     i = i * dist.get_world_size() + dist.get_rank()
    return f"sequence_{i}"

def count_success(results, seq_len=5):
    count = Counter(results)
    step_success = []
    for i in range(1, seq_len + 1):
        n_success = sum(count[j] for j in reversed(range(i, seq_len + 1)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, cfg, log_dir):
    
    # sequences = get_sequences(cfg.inference.num_sequences, cfg.inference.seq_len)

    current_data = {}
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results, seq_len=cfg.inference.seq_len))}
    logger.info(f"Average successful sequence length: {avg_seq_len}")
    logger.info("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        logger.info(f"{i}: {sr:.3f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, sequence in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        logger.info(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    # wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
    current_data = data

    json_data = current_data
    with open(os.path.join(log_dir, "results.json"), "w") as file:
        json.dump(json_data, file, indent=4)

def evaluate_policy(cfg, model, env, accelerator, log_dir):
    progress = Progress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not accelerator.is_main_process,
    ) 
    progress.start()

    task_oracle = hydra.utils.instantiate(cfg.task)
    seq_len = cfg.inference.seq_len
    eval_sequences = get_sequences(
        num_sequences=cfg.inference.num_sequences, 
        seq_len=seq_len,
    )

    dataset = ListDataset(eval_sequences)
    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    dataloader = accelerator.prepare(eval_loader)

    results = []
    sequences = []
    plans = []
    
    eval_task = progress.add_task("Evaluating sequences", total=len(dataloader))
    

    #####
    val_annotations = cfg.annotation
    # lang_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    lang_embeddings = {}
    # for task, annotation in val_annotations.items():
    #     lang_embeddings[task] = lang_model([annotation[0]]).numpy()
        # logger.info(f"Loaded language embedding for task {task}: {annotation} with shape {lang_embeddings[task].shape}")
    
    rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=os.path.join(log_dir, "rollout_videos"),
            resolution_scale=1,
        )

    rollout_video2 = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=os.path.join(log_dir, "flow_videos"),
            resolution_scale=1,
        )

    record = cfg.inference.record_rollout_videos
    device = next(model.parameters()).device
    ###
    for i, data in enumerate(dataloader):
        idx, (initial_state, eval_sequence) = data[0]
        start_time = time.time()
        task_sequence = " -> ".join(eval_sequence)
        description = f"Evaluating sequence {i + 1}/{len(dataloader)}: {task_sequence}"
        logger.info(description)
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, progress, cfg, record, rollout_video, rollout_video2, idx
        )
        # result = idx
        end_time = time.time()
        results.append(result)
        sequences.append(eval_sequence)
        success_rates = count_success(results, seq_len)
        average_rate = sum(success_rates) / len(success_rates) * seq_len
        description = f"Device {device}: " + " ".join([f"{i + 1}/{seq_len} : {v:.3f}% |" for i, v in enumerate(success_rates)])
        description += f" Average: {average_rate:.3f} | Time: {end_time - start_time:.2f}s | "    
        logger.info(description)

        if record:
            logger.info(f"Writing rollout video for sequence {idx}...")
            rollout_video.write_to_tmp()
            rollout_video._log_currentvideos_to_file(idx, result, save_as_video=True)
            if cfg.inference.record_flow:
                rollout_video2.write_to_tmp()
                rollout_video2._log_currentvideos_to_file(idx, result, save_as_video=True)

        progress.update(eval_task, advance=1)

    progress.stop()
    logger.info("Evaluation completed. Waiting for all processes to finish...")
    results = gather_object(results)
    sequences = gather_object(sequences)
    # logger.info(results)
    # logger.info(sequences)
    return results, sequences

def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, progress, cfg, record, rollout_video, rollout_video2, i
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
        if cfg.inference.record_flow:
            rollout_video2.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    # accelerate.utils.set_seed(cfg.seed)
    for idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()
            if cfg.inference.record_flow:
                rollout_video2.new_subtask()
        # success = random.randint(0, 1)
        success = rollout(env, model, task_checker, cfg, idx, subtask, lang_embeddings, val_annotations, progress, record, rollout_video, rollout_video2)
        if record:
            rollout_video.draw_outcome(success)
            if cfg.inference.record_flow:
                rollout_video2.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def get_transform(image_size=128):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])

def reset(model):
    if isinstance(model, DistributedDataParallel):
        model.module.reset()
    else:
        model.reset()
    return model

def rollout(env, model, task_oracle, cfg, idx, subtask, lang_embeddings, val_annotations, progress, record=False, rollout_video=None, rollout_video2=None):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    
    model = reset(model)
    start_info = env.get_info()
    transform = get_transform(cfg.inference.image_size)
    device = next(model.parameters()).device

    history = {}
    history["rgb_static"] = [transform(image=obs["rgb_obs"]["rgb_static"])["image"][None, None, ].to(device) / 255. for _ in range(2)]
    history["rgb_gripper"] = [transform(image=obs["rgb_obs"]["rgb_gripper"])["image"][None, None].to(device) / 255. for _ in range(2)]

    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding

    # goal = lang_embeddings[subtask]
    # goal['lang_text'] = val_annotations[subtask][0]

    robot_state_min = torch.tensor([-0.4322, -0.4838,  0.2963, -3.1416, -0.7520, -3.1415, -0.0256, -2.4121,                                    
                                          -0.8907,  1.1649, -3.0606, -2.1438,  1.0935, -1.8253, -1.0000])
    robot_state_max = torch.tensor([ 0.4215,  0.1230,  0.7387,  3.1416,  0.6386,  3.1416,  0.0907,  0.3987,                         
                                      1.6911,  2.8208, -0.4788,  0.5587,  2.7564,  2.7591,  1.0000])
    robot_state_range = torch.tensor([0.8537, 0.6068, 0.4424, 6.2832, 1.3906, 6.2831, 0.1163, 2.8108, 2.5818,                       
                                     1.6559, 2.5818, 2.7024, 1.6629, 4.5844, 2.0000]) 

    bar = progress.add_task(f"Rollout {idx}. {subtask}", total=cfg.inference.ep_len)
    total_time = 0
    for step in range(cfg.inference.ep_len):
        robot_obs = torch.tensor(obs["robot_obs"])
        robot_obs = (robot_obs - robot_state_min) / robot_state_range

        inputs = {
            "image": {
                "rgb_static": torch.cat(history["rgb_static"][-2:], dim=1),
                "rgb_gripper": torch.cat(history["rgb_gripper"][-2:], dim=1),
            },
            "language": lang_annotation,
        }

        stime = time.time()
        if isinstance(model, DistributedDataParallel):
            action_output = model.module.step(inputs, visualize=cfg.inference.record_flow)
        else:   
            action_output = model.step(inputs, visualize=cfg.inference.record_flow)
        etime= time.time()
        # logger.info(f"Inference time step {step}: {etime - stime:.2f}s")
        total_time += etime - stime
        action = action_output["action"]
        viz_flow = action_output["viz_flow"]
        # logger.info(f"Step {step + 1}: {action}")
        action[:-1] = action[:-1].clamp(-1, 1)  # Clamp action to [-1, 1]
        action[-1] = (action[-1] > 0).long() * 2 - 1
        #print('obs_max:',obs["rgb_obs"]['cond_static'].max())
        #print('obs_shape:', obs["rgb_obs"]['cond_static'].shape)
        # print(action, type(action))
        # print(action.min(), action.max(), action)
        # print(env.relative_actions)
        # exit(0)
        obs, _, _, current_info = env.step(action.numpy())
        # update history
        if step % cfg.inference.multistep == 0:
            model = reset(model)
        else:
            history["rgb_static"].pop(-1)
            history["rgb_static"].pop(-1)
            history["rgb_gripper"].pop(-1)
            history["rgb_gripper"].pop(-1)
            
        history["rgb_static"].append(transform(image=obs["rgb_obs"]["rgb_static"])["image"][None, None, ].to(device) / 255.)
        history["rgb_static"].append(transform(image=obs["rgb_obs"]["rgb_static"])["image"][None, None, ].to(device) / 255.)
        history["rgb_gripper"].append(transform(image=obs["rgb_obs"]["rgb_gripper"])["image"][None, None].to(device) / 255.)
        history["rgb_gripper"].append(transform(image=obs["rgb_obs"]["rgb_gripper"])["image"][None, None].to(device) / 255.)
        # remove oldest history
        while len(history["rgb_static"]) > 3:
            history["rgb_static"].pop(0)
        while len(history["rgb_gripper"]) > 3:
            history["rgb_gripper"].pop(0)
            
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
            if cfg.inference.record_flow and viz_flow is not None:
                rollout_video2.update(viz_flow)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            model = reset(model)
            logger.info(f"Average step time: {total_time / (step + 1):.2f}s")
            progress.update(bar, advance=cfg.inference.ep_len - step)
            progress.remove_task(bar)

            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
                if cfg.inference.record_flow:
                    rollout_video2.add_language_instruction(lang_annotation)
            return True

        else:
            progress.update(bar, advance=1)
    

    logger.info(f"Average step time: {total_time / cfg.inference.ep_len:.2f}s")
    progress.remove_task(bar)
    logger.info(f"Failed to solve task {subtask}:{lang_annotation} in order {idx}.")
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
        if cfg.inference.record_flow:
            rollout_video2.add_language_instruction(lang_annotation)
    return False


def main(cfg):

    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=864000))
    accelerator = accelerate.Accelerator(**cfg.accelerator, kwargs_handlers=[kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(cfg.seed)
    
    log_dir = cfg.inference.save_dir
    setup_logging(accelerator.is_main_process, log_dir=log_dir)

    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model)
    
    if cfg.weights:
        logger.info(f"Model weights specified: {cfg.weights}")
        model.from_pretrained(cfg.weights)

    model = accelerator.prepare(model)
    model.eval()

    env = get_env(cfg.inference.dataset, show_gui=False)
    
    results, sequences = evaluate_policy(cfg, model, env, accelerator, log_dir=log_dir)
    if accelerator.is_main_process:
        print_and_save(results, sequences, cfg, log_dir=log_dir)
        
if __name__ == "__main__":
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="infer", overrides=sys.argv[1:])
        OmegaConf.resolve(cfg)
        main(cfg)