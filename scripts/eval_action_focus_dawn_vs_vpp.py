"""Action-focused, apples-to-apples DAWN vs VPP-integration comparison.

Both models are run under the exact same, genuinely non-privileged inference
condition (motion_source=predicted -- no ground-truth flow, matching real
rollout conditions) on the same real validation samples, and action-chunk MSE
is aggregated over many samples, broken down by horizon (step-1, first-5,
all-10) exactly like evaluation/eval_action_expert.py's original design --
this is what makes the comparison fair, unlike DAWN's own eval tool's default
of motion_source=estimator (ground-truth-derived flow it never has at real
inference time).

Usage
-----
    python scripts/eval_action_focus_dawn_vs_vpp.py --num-samples 200
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

from dawn.data.calvin.calvin import CalvinDataset

DAWN_CHECKPOINT = "outputs/DAWN-action-expert-edp2/edp2-hsv-bs128-skip10-lr1e-4-total100k/2026-09-12_18-24/checkpoints/model_final.pth"
VPP_CHECKPOINT = "/mnt/localssd/.cache/huggingface/dawn_vpp/dp_calvin_dawn_arch.pth"
DATASET_PATH = "/home/colligo/Codes/localssd/calvin-dawn/dataset_opt_local/task_ABC_D"


def build_model(config_name: str, checkpoint_path: str, device: str):
    with initialize(version_base=None, config_path="../configs/model"):
        cfg = compose(config_name=config_name)
    cfg.motion_source = "predicted"  # genuine inference for both: no ground-truth flow
    model = instantiate(cfg).to(device)
    model.from_pretrained({"model": checkpoint_path})
    model.eval()
    return model


def horizon_mse(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    return {
        "step1": (pred[0] - gt[0]).pow(2).mean().item(),
        "first5": (pred[:5] - gt[:5]).pow(2).mean().item(),
        "all10": (pred - gt).pow(2).mean().item(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"

    print("Loading validation dataset...")
    ds = CalvinDataset(
        data_path=DATASET_PATH, split="validation", image_size=256,
        num_frames=2, num_actions=10, min_skip=10, max_skip=30,
    )

    print("Loading DAWN (LDM + edp2)...")
    dawn_model = build_model("DAWN_stage2", DAWN_CHECKPOINT, device)

    print("Loading VPP-integrated (SVDMotionExtractor + VPPCompatiblePolicy)...")
    vpp_model = build_model("DAWN_stage2_vpp", VPP_CHECKPOINT, device)

    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(ds), size=args.num_samples, replace=False)

    dawn_agg = {"step1": [], "first5": [], "all10": []}
    vpp_agg = {"step1": [], "first5": [], "all10": []}

    for idx in tqdm(idxs, desc="Evaluating"):
        item = ds[int(idx)]
        batch = {
            "image": {k: v.unsqueeze(0).to(device) for k, v in item["image"].items()},
            "language": [item["language"]],
            "action": item["action"].unsqueeze(0).float().to(device),
        }
        gt_chunk = batch["action"][0]

        with torch.no_grad():
            dawn_out = dawn_model(batch)
            vpp_out = vpp_model(batch)

        dawn_chunk = dawn_out["action"]["logits"][0]
        vpp_chunk = vpp_out["action"]["logits"][0]

        for k, v in horizon_mse(dawn_chunk, gt_chunk).items():
            dawn_agg[k].append(v)
        for k, v in horizon_mse(vpp_chunk, gt_chunk).items():
            vpp_agg[k].append(v)

    print()
    print(f"Action-focused MSE, motion_source=predicted (non-privileged, both models), N={args.num_samples}")
    print(f"{'horizon':<10} {'DAWN (edp2)':>15} {'VPP-integration':>18}")
    for k in ["step1", "first5", "all10"]:
        dawn_mean = float(np.mean(dawn_agg[k]))
        vpp_mean = float(np.mean(vpp_agg[k]))
        print(f"{k:<10} {dawn_mean:>15.4f} {vpp_mean:>18.4f}")


if __name__ == "__main__":
    main()
