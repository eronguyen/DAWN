"""Diagnostic: run VPP's own, unmodified `VPP_Policy` through DAWN's own
`inference.py` rollout harness (env creation, per-step observation history,
action post-processing, task-success checking) instead of DAWNArch.

Purpose: isolate whether the large gap we measured between native VPP eval
(avg_seq_len ~4.9/5, using vpp-hiva's own env/harness) and our DAWNArch
integration (avg_seq_len ~0.5-0.8/5, using DAWN's inference.py harness) with
the *same checkpoint* comes from (a) a bug in the DAWNArch-hosted re-
implementation (SVDMotionExtractor / VPPCompatiblePolicy), or (b) something in
DAWN's own rollout harness (env, observation construction, action
post-processing) that the real VPP model is also sensitive to. If this hybrid
run also scores far below ~4.9/5, the bug is in DAWN's harness/env, not the
port; if it scores close to ~4.9/5, the bug is in the port.

The only "glue" code below is a tiny adapter translating between DAWN's
`model.step(inputs, visualize=...)` interface (dict with `inputs["image"]`,
`inputs["language"]`) and VPP's own `model.step(obs, goal)` interface (dict
with `obs["rgb_obs"]`, `goal["lang_text"]`) -- both models' own internals
(SVD feature extraction, Video-Former, diffusion sampling, multistep chunking)
are untouched, real code from each repo.

Usage (single GPU, matching both smoke tests' process count)::

    accelerate launch --num_processes=1 scripts/native_vpp_in_dawn_harness.py \
        inference.num_sequences=10 inference.record_rollout_videos=False inference.record_flow=False
"""

from __future__ import annotations

import os
import sys

_DAWN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VPP_ROOT = "/home/colligo/Codes/HiVA/vpp-hiva"
for p in (_DAWN_ROOT, _VPP_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(_DAWN_ROOT)

import hydra
import torch
from torch import nn
from omegaconf import OmegaConf

import inference as dawn_inference  # DAWN's own inference.py, imported as a module

VPP_CHECKPOINT_DIR = (
    "/mnt/localssd/.cache/huggingface/hub/models--yjguo--dp-calvin/"
    "snapshots/199fc56a4e61fd72df55d1fe5dacd293504b42e4"
)
VPP_SVD_PATH = (
    "/mnt/localssd/.cache/huggingface/hub/models--yjguo--svd-robot-calvin-ft/"
    "snapshots/d7178b195d5f934f5d5b268e25fdf011f16b7bfb"
)
VPP_CLIP_PATH = (
    "/mnt/localssd/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/"
    "snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)


class NativeVPPAdapter(nn.Module):
    """Wraps a real, unmodified `VPP_Policy` so it plugs into DAWN's own
    `evaluate_policy`/`rollout()` (`inference.py`) with zero changes there."""

    def __init__(self, vpp_model) -> None:
        super().__init__()
        self.vpp = vpp_model

    def reset(self) -> None:
        self.vpp.reset()

    @torch.no_grad()
    def step(self, inputs, visualize: bool = False):
        device = next(self.vpp.parameters()).device

        # DAWN's `inputs["image"]["rgb_static"]` is (B, 2, C, H, W) in [0, 1]
        # (a 2-frame history stack; both frames are the current frame after
        # step 0 -- see inference.py's history bookkeeping). VPP's own env
        # (HulcWrapper) produces (B, 1, C, H, W) normalized to [-1, 1] via
        # torchvision Normalize(mean=0.5, std=0.5); reproduce both here.
        rgb_static = inputs["image"]["rgb_static"][:, -1:].to(device)
        rgb_gripper = inputs["image"]["rgb_gripper"][:, -1:].to(device)
        rgb_static = rgb_static * 2.0 - 1.0
        rgb_gripper = rgb_gripper * 2.0 - 1.0

        obs = {"rgb_obs": {"rgb_static": rgb_static, "rgb_gripper": rgb_gripper}}
        goal = {"lang_text": inputs["language"]}

        action = self.vpp.step(obs, goal)
        action = action.reshape(-1).detach().cpu()
        return {"action": action, "viz_flow": None}


def build_native_vpp_model(device: torch.device):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(_VPP_ROOT, "policy_conf"), job_name="calvin_evaluate_all.yaml"):
        vpp_cfg = compose(config_name="calvin_evaluate_all.yaml")

    vpp_cfg.model.pretrained_model_path = VPP_SVD_PATH
    vpp_cfg.model.text_encoder_path = VPP_CLIP_PATH

    ckpt_path = os.path.join(VPP_CHECKPOINT_DIR, "last.pt")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    vpp_model = hydra.utils.instantiate(vpp_cfg.model)
    missing_unexpected = vpp_model.load_state_dict(state_dict["model"], strict=False)
    print("VPP_Policy.load_state_dict:", missing_unexpected)
    vpp_model.freeze()
    vpp_model = vpp_model.to(device)

    # Match native eval's own inference-time overrides (policy_conf/calvin_evaluate_all.yaml).
    vpp_model.num_sampling_steps = vpp_cfg.num_sampling_steps
    vpp_model.sampler_type = vpp_cfg.sampler_type
    vpp_model.multistep = vpp_cfg.multistep
    if vpp_cfg.sigma_min is not None:
        vpp_model.sigma_min = vpp_cfg.sigma_min
    if vpp_cfg.sigma_max is not None:
        vpp_model.sigma_max = vpp_cfg.sigma_max
    if vpp_cfg.noise_scheduler is not None:
        vpp_model.noise_scheduler = vpp_cfg.noise_scheduler
    vpp_model.process_device()
    vpp_model.eval()
    return vpp_model


def main() -> None:
    import shutil
    import accelerate
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    from calvin_env.envs.play_table_env import get_env
    from utils.logging import setup_logging, save_config_yaml

    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="infer", overrides=sys.argv[1:])
        OmegaConf.resolve(cfg)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=864000))
    accelerator = accelerate.Accelerator(**cfg.accelerator, kwargs_handlers=[kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(cfg.seed)

    log_dir = cfg.inference.save_dir
    setup_logging(accelerator.is_main_process, log_dir=log_dir)
    accelerator.init_trackers(project_name=cfg.project, config=OmegaConf.to_container(cfg, resolve=True))

    if accelerator.is_main_process:
        cfg_path = save_config_yaml(cfg, log_dir, filename="config.yaml")
        print(f"Saved resolved config to: {cfg_path}")

    print("Building native VPP_Policy (real vpp-hiva code, unmodified)...")
    model = build_native_vpp_model(device)
    model = NativeVPPAdapter(model)
    model.eval()

    try:
        from calvin_env.utils.utils import set_egl_device

        set_egl_device(accelerator.device)
    except Exception as e:
        os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")
        print(f"set_egl_device failed ({e}); fell back to EGL_VISIBLE_DEVICES=0")

    env = get_env(cfg.inference.dataset, show_gui=False)

    results, sequences = dawn_inference.evaluate_policy(cfg, model, env, accelerator, log_dir=log_dir)
    if accelerator.is_main_process:
        dawn_inference.print_and_save(results, sequences, cfg, log_dir=log_dir, accelerator=accelerator)

    accelerator.end_training()


if __name__ == "__main__":
    main()
