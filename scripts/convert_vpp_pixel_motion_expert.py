"""Transfer VPP's (yjguo/dp-calvin) transformer-body and language-goal weights
into `VPPCompatiblePolicy(use_video_former=False, obs_dim=<encoder_dim>)`, for
pairing with DAWN's own pixel-motion `motion_director` (e.g. `LDM`) instead of
VPP's SVD+Video_Former perception stack.

Only `model.inner_model.*` and `language_goal.*` transfer -- `Video_Former.*`
and `TVP_encoder.*` are dropped entirely since this configuration doesn't use
either (the `motion_director` supplies `pixel_motion_feat` directly, see
`dawn/models/action_expert/vpp_edp.py`'s module docstring). Within
`model.inner_model`, only `tok_emb.*` is skipped (its input dim depends on the
upstream feature dim -- 384 for VPP's Video_Former output vs the encoder's own
dim, e.g. 768 for DINOv3-ConvNeXt-Small -- so it can't transfer regardless of
which encoder is used); every other transformer-body weight (attention/MLP
blocks, sigma/action embeddings, action projections, and -- unlike
`scripts/convert_vpp_action_expert.py`'s edp2 target -- `lang_emb.0` too, since
`goal_dim=512` is unchanged here) transfers with matching shapes.

Usage
-----
    python scripts/convert_vpp_pixel_motion_expert.py \
        --vpp-checkpoint <path to dp-calvin last.pt> \
        --obs-dim 768 \
        --out outputs/pretrained_transfer/dp_calvin_pixel_motion_expert.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dawn.models.action_expert.vpp_edp import VPPCompatiblePolicy

ATOMIC_SKIP_ON_MISMATCH = ["tok_emb"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpp-checkpoint", required=True, help="Path to VPP's (dp-calvin) last.pt.")
    p.add_argument("--obs-dim", type=int, default=768, help="DAWN encoder's feature dim (DINOv3-ConvNeXt-Small = 768).")
    p.add_argument("--goal-dim", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--proprio-dim", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.vpp_checkpoint, map_location="cpu", weights_only=False)
    vpp_sd = ckpt["model"] if "model" in ckpt else ckpt

    inner_sd = {k[len("model.inner_model."):]: v for k, v in vpp_sd.items() if k.startswith("model.inner_model.")}
    lang_sd = {k: v for k, v in vpp_sd.items() if k.startswith("language_goal.")}

    ref = VPPCompatiblePolicy(
        obs_dim=args.obs_dim,
        goal_dim=args.goal_dim,
        latent_dim=args.latent_dim,
        proprio_dim=args.proprio_dim,
        use_video_former=False,
    )
    ours_inner_sd = ref.model.inner_model.state_dict()

    transferred, skipped_atomic, skipped_mismatch = {}, [], []
    for k, v in ours_inner_sd.items():
        if k not in inner_sd:
            continue
        if any(k == a or k.startswith(a + ".") for a in ATOMIC_SKIP_ON_MISMATCH):
            skipped_atomic.append(k)
            continue
        if tuple(inner_sd[k].shape) != tuple(v.shape):
            skipped_mismatch.append(k)
            continue
        transferred[k] = inner_sd[k].clone()

    out_sd = {f"action_expert.model.inner_model.{k}": v for k, v in transferred.items()}
    out_sd.update({f"action_expert.{k}": v for k, v in lang_sd.items()})

    total_inner_params = sum(v.numel() for v in ours_inner_sd.values())
    transferred_params = sum(v.numel() for v in transferred.values())

    print(f"inner_model total params: {total_inner_params:,}")
    print(f"inner_model transferred: {len(transferred)}/{len(ours_inner_sd)} keys, "
          f"{transferred_params:,} params ({transferred_params / total_inner_params:.1%})")
    print(f"inner_model skipped (atomic, obs-dim mismatch): {sorted(set(k.split('.')[0] for k in skipped_atomic))}")
    if skipped_mismatch:
        print(f"inner_model skipped (other shape mismatch): {skipped_mismatch}")
    print(f"language_goal transferred: {len(lang_sd)} keys")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out_sd, args.out)
    print(f"Wrote {args.out} ({len(out_sd)} keys)")


if __name__ == "__main__":
    main()
