"""Load DAWN's native stack (LDM + edp2) and the VPP-integrated stack
(SVDMotionExtractor + VPPCompatiblePolicy) side by side, run both on the same
real validation frames in genuine inference mode (motion_source=predicted --
no ground-truth flow, matching real rollout conditions), and print/save a
side-by-side comparison of their predicted action chunks plus DAWN's
predicted motion visualization (VPP has no analogous visual output -- its
conditioning is SVD features, not a pixel-space flow image).

Usage
-----
    python scripts/compare_dawn_vpp_inference.py --num-samples 5
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
from PIL import Image

from dawn.data.calvin.calvin import CalvinDataset

DAWN_CHECKPOINT = "/home/colligo/Codes/localssd/DAWN/2026-09-12_18-24/checkpoints/model_0047000.pth"
VPP_CHECKPOINT = "/mnt/localssd/.cache/huggingface/dawn_vpp/dp_calvin_dawn_arch.pth"
DATASET_PATH = "/home/colligo/Codes/localssd/calvin-dawn/dataset_opt_local/task_ABC_D"


def build_model(config_name: str, checkpoint_path: str, device: str):
    with initialize(version_base=None, config_path="../configs/model"):
        cfg = compose(config_name=config_name)
    cfg.motion_source = "predicted"  # genuine inference: no ground-truth flow
    model = instantiate(cfg).to(device)
    model.from_pretrained({"model": checkpoint_path})
    model.eval()
    return model


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().float().clamp(0, 1).cpu().numpy()
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--out-dir", default="outputs/dawn_vs_vpp_compare")
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading validation dataset...")
    ds = CalvinDataset(
        data_path=DATASET_PATH, split="validation", image_size=256,
        num_frames=2, num_actions=10, min_skip=10, max_skip=30,
    )

    print("Loading DAWN (LDM + edp2)...")
    dawn_model = build_model("DAWN_stage2", DAWN_CHECKPOINT, device)

    print("Loading VPP-integrated (SVDMotionExtractor + VPPCompatiblePolicy)...")
    vpp_model = build_model("DAWN_stage2_vpp", VPP_CHECKPOINT, device)

    idxs = np.linspace(0, len(ds) - 1, args.num_samples * 37, dtype=int)[:: 37][: args.num_samples]

    print()
    print(f"{'#':>2}  {'language':<45} {'GT action[0]':<45}")
    for i, idx in enumerate(idxs):
        item = ds[int(idx)]
        batch = {
            "image": {k: v.unsqueeze(0).to(device) for k, v in item["image"].items()},
            "language": [item["language"]],
            "action": item["action"].unsqueeze(0).float().to(device),
        }
        gt = batch["action"][0, 0]

        with torch.no_grad():
            dawn_out = dawn_model(batch)
            vpp_out = vpp_model(batch)

        dawn_pred = dawn_out["action"]["logits"][0, 0]
        vpp_pred = vpp_out["action"]["logits"][0, 0]

        def fmt(t):
            return "[" + ", ".join(f"{x:+.3f}" for x in t.tolist()) + "]"

        dawn_l1 = (dawn_pred - gt).abs().mean().item()
        dawn_mse = (dawn_pred - gt).pow(2).mean().item()
        vpp_l1 = (vpp_pred - gt).abs().mean().item()
        vpp_mse = (vpp_pred - gt).pow(2).mean().item()

        print(f"\n=== sample {i} (idx={idx}) ===")
        print(f"language : {item['language']}")
        print(f"GT       : {fmt(gt)}")
        print(f"DAWN     : {fmt(dawn_pred)}  |  L1: {dawn_l1:.4f}  MSE: {dawn_mse:.4f}")
        print(f"VPP      : {fmt(vpp_pred)}  |  L1: {vpp_l1:.4f}  MSE: {vpp_mse:.4f}")

        if i == 0:
            # Save a visual: input frame + DAWN's predicted motion (VPP has no
            # analogous pixel-space output -- its conditioning is SVD features).
            input_frame = to_uint8(item["image"]["rgb_static"][0])
            dawn_flow = dawn_out["motion"].get("predicted_flow_rgb")
            panels = [Image.fromarray(input_frame)]
            if dawn_flow is not None:
                panels.append(Image.fromarray(to_uint8(dawn_flow[0])))
            w = sum(im.width for im in panels)
            h = max(im.height for im in panels)
            canvas = Image.new("RGB", (w, h))
            x = 0
            for im in panels:
                canvas.paste(im, (x, 0))
                x += im.width
            out_path = os.path.join(args.out_dir, f"sample_{idx}_input_and_dawn_flow.png")
            canvas.save(out_path)
            print(f"Saved visual: {out_path}")


if __name__ == "__main__":
    main()
