import accelerate
import logging 
import os
import torch
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from collections import defaultdict
import time
import humanfriendly

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, 
        cfg,
        accelerator, 
        model, 
        optimizer, 
        train_loader, 
        val_loader,
        scheduler,
        checkpoint_path: str = None
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = os.path.join(cfg.save_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)


        self.cur_step = 0        
        self.load_checkpoint(checkpoint_path)
        # self.model.load_weights()
    
        # Prepare the model and optimizer with the accelerator
        logger.info(f"Preparing model and optimizer with {self.accelerator.__class__.__name__}.")
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.scheduler,
            self.train_loader, 
            self.val_loader
        )
        logger.info(f"Model and optimizer prepared. Model: {self.model.__class__.__name__}, Optimizer: {self.optimizer.__class__.__name__}")

        self.progress = Progress(
            TextColumn("{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not self.accelerator.is_main_process,
        ) 
        self.progress.start()

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().item())
        return float(value)

    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if checkpoint_path:
            logger.info(f"Model weights specified: {checkpoint_path}")
            self.model.from_pretrained(checkpoint_path)
        # if checkpoint_path is not None:
        #     logger.info(f"Loading model weights from {checkpoint_path}.")
        #     if os.path.exists(checkpoint_path):
        #         ckpt = torch.load(checkpoint_path, map_location="cpu")
        #         state_dict = self.model.state_dict()
        #         new_state_dict = {}
        #         for k, v in ckpt.items():
        #             if k in state_dict:
        #                 module = state_dict.pop(k)
        #                 if v.shape == module.shape:
        #                     new_state_dict[k] = v
        #                 else:
        #                     logger.warning(f"[pink]Skipping loading {k} from checkpoint: Shape mismatch: checkpoint: {v.shape}, model: {module.shape}")
        #             else:
        #                 logger.warning(f"[pink]Skipping loading {k} from checkpoint: Not found in model.")

        #         logger.info(self.model.load_state_dict(new_state_dict, strict=False))
        #         if self.cfg.resume:
        #             try:
        #                 self.cur_step = int(checkpoint_path.split('_')[-1].split('.')[0])
        #                 logger.info(f"Resuming training from iter {self.cur_step}")
        #             except:
        #                 pass
        #     else:
        #         logger.warning(f"Weights file {checkpoint_path} does not exist. Skipping loading weights.")

    def save_checkpoint(self, k=5, last=False):
        """
        Save the model and optimizer state to a checkpoint.
        Args:
            checkpoint_path (str): Path to save the checkpoint file.
            k: Keep last k checkpoints
        """
        ckpt_lst = sorted([x for x in os.listdir(self.save_dir) if x.startswith("model_") and x.endswith(".pth")])
        while len(ckpt_lst) >= k:
            logger.info(f"Removing old checkpoint: {ckpt_lst[0]}")
            os.remove(os.path.join(self.save_dir, ckpt_lst[0]))
            ckpt_lst = ckpt_lst[1:]
        
        if not last:
            ckpt = os.path.join(self.save_dir, f"model_{self.cur_step:07d}.pth")
        else:
            ckpt = os.path.join(self.save_dir, f"model_final.pth")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), ckpt)
        logger.info(f"Saved checkpoint to {ckpt}")
        
    def train(self):
        self.model.train()
        logger.info(f"Starting training at step {self.cur_step} for {self.cfg.total_steps} steps.")
        train_task = self.progress.add_task(
            "[bold blue]Training...", 
            total=self.cfg.total_steps, 
            completed=self.cur_step,
        )

        losses = defaultdict(list)
        data_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        while self.cur_step < self.cfg.total_steps:
            for batch in self.train_loader:
                losses["data_time"].append(time.time() - data_time)
                
                # Forward pass
                outputs = {}
                with self.accelerator.accumulate(self.model):
                    start_time = time.time()
                    outputs = self.model(batch)
                    training_time = time.time() - start_time
                    losses["time"].append(training_time)

                    # Compute loss
                    loss = outputs["total_loss"]
                        
                    # Backward pass
                    self.accelerator.backward(loss)

                    # Update parameters
                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        self.scheduler.step()  # Update learning rate
                        self.optimizer.zero_grad(set_to_none=True)
                        self.cur_step += 1
                        self.progress.update(train_task, advance=1)
                
                # Log 
                if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                    # History of losses
                    for k, v in outputs.items():
                        if "loss" in k:
                            losses[k].append(self._to_float(v))
                
                    # Log every log_interval steps
                    if self.cur_step > 0 and self.cur_step % self.cfg.log_interval == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        dct_loss = {k: sum(v) / len(v) for k, v in losses.items()}
                        losses = defaultdict(list)  # Reset losses for next logging
                        loss_string = ", ".join([f"{k}: {v:.7f}" for k, v in dct_loss.items()])
                        self.accelerator.log({
                            **dct_loss,
                            "step": self.cur_step,
                            "lr": current_lr,
                        }, step=self.cur_step
                        )
                        logger.info(f"Step {self.cur_step} {loss_string}, LR: {current_lr:.7f}, GPU: {self._get_gpu_memory()}")
                
                    # Save checkpoint
                    if self.cur_step > 0 and self.cur_step % self.cfg.save_interval == 0:
                        self.save_checkpoint()
                        
                # Validation step
                if self.cur_step > 0 and self.cur_step % self.cfg.val_interval == 0:
                    self.validate(self.train_loader, split="train", num_steps=1)
                    self.validate(self.val_loader, split="val")

                if self.cur_step >= self.cfg.total_steps:
                    break

                data_time = time.time()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.save_checkpoint(last=True)
        self.validate(self.train_loader, split="train", num_steps=1)
        self.validate(self.val_loader, split="val")
        self.progress.stop()
                        
    def validate(self, val_loader, split="val", num_steps=-1):
        self.model.eval()
        if num_steps <= 0:
            num_steps = len(val_loader)

        logger.info(f"Starting validation on {split} set for {num_steps} steps.")
        val_task = self.progress.add_task(
            "[bold green]Validating...",
            total=num_steps,
        )
        
        cnt = 0
        losses = defaultdict(float)

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                outputs = self.model(batch, split=split)
                for k, v in outputs.items():
                    if "loss" in k:
                        losses[k] += self._to_float(v)
                cnt += 1
                self.progress.update(val_task, advance=1)

                # Log images
                if self.accelerator.is_main_process and i == 0:
                    images=None
                    try:
                        images = self.accelerator.unwrap_model(self.model).visualize(batch, outputs)
                    except Exception as e:
                        logger.warning(f"Visualization failed at step {self.cur_step} during {split} validation: {e}")
                        images = None
                    
                    if images is not None:
                        logger.info(f"Logging validation images for {split} at step {self.cur_step}.")
                        self.accelerator.log({
                                f"{split}/visualize": images,
                            }, step = self.cur_step
                        )
                    else:
                        logger.warning(f"No images to log for {split} at step {self.cur_step}.")
                
                if cnt == num_steps:
                    break 

        # Reduce metrics across all processes.
        cnt_t = torch.tensor(float(cnt), device=self.accelerator.device)
        cnt_global = self.accelerator.reduce(cnt_t, reduction="sum").item()
        reduced_losses = {}
        for k, v in losses.items():
            v_t = torch.tensor(float(v), device=self.accelerator.device)
            v_global = self.accelerator.reduce(v_t, reduction="sum").item()
            reduced_losses[f"Validation/{split}_{k}"] = v_global / max(cnt_global, 1.0)

        losses = reduced_losses
        loss_string = ", ".join([f"{k}: {v:.7f}" for k, v in losses.items()])
        logger.info(f"Validation at step {self.cur_step}: {loss_string}")
        if self.accelerator.is_main_process:
            self.accelerator.log({
                **losses,
                "step": self.cur_step,
            }, step = self.cur_step)

        self.model.train()
        self.progress.remove_task(val_task)
    

    def _get_gpu_memory(self):
        if torch.cuda.is_available():
            return humanfriendly.format_size(torch.cuda.max_memory_allocated())