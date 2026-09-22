"""Transfer as much of yjguo/dp-calvin's (Video Prediction Policy) action-expert
transformer body into DAWN's edp2 TransformerDiffusionPolicy as shape-compatible.

VPP's `model.inner_model` and our `action_expert.inner_model` (dawn.models.modules.
diffusion.diffusion_transformer.DiffusionTransformer) are the same architecture
(GPT-style encoder + FiLM cross-attention decoder, 4/4 layers, 8 heads, 224 obs
tokens, 10-step action chunks, 7-dim actions, EDM/k-diffusion sigma conditioning)
at embed_dim=384, so nearly all of the core transformer body transfers exactly:
encoder/decoder attention+MLP blocks, final layernorms, the sigma/timestep
embedding, and the action in/out projections (action_dim=7 in both).

What does NOT transfer: `tok_emb` and the first `lang_emb` layer, because their
*input* side depends on upstream feature dims that differ from ours (VPP resamples
video to 384-dim tokens and encodes language with CLIP ViT-B/32 -> 512-dim; DAWN
feeds raw CLIP ViT-L/14 features at 768-dim directly). VPP's `goal_emb` and
`proprio_emb` also have no counterpart at all (DAWN's proprio_emb is currently
disabled / `pos_emb` is unused), so they're dropped entirely.

Usage
-----
    python scripts/convert_vpp_action_expert.py \
        --vpp-checkpoint <path to last.pt> \
        --embed-dim 384 \
        --out outputs/pretrained_transfer/dp_calvin_action_expert_embed384.pth

The output is a flat state dict with `action_expert.inner_model.*` keys, loadable
via `weights.action_expert=<out>` (DAWNArch.from_pretrained filters for that
prefix and calls `self.action_expert.load_state_dict(..., strict=False)`, so the
rest of TransformerDiffusionPolicy -- image_proj, motion_encoder, motion_proj --
stays randomly initialized).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dawn.models.modules.diffusion.diffusion_transformer import DiffusionTransformer

# Modules whose *input* side depends on an upstream feature dim VPP and DAWN don't
# share (see module docstring) -- skip weight AND bias together so we never mix a
# transferred bias with a randomly-initialized weight for the same layer.
ATOMIC_SKIP_ON_MISMATCH = ["tok_emb", "lang_emb.0"]


def build_reference_module(action_dim: int, obs_dim: int, goal_dim: int, proprio_dim: int, embed_dim: int) -> DiffusionTransformer:
    return DiffusionTransformer(
        action_dim=action_dim,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        proprio_dim=proprio_dim,
        goal_conditioned=True,
        embed_dim=embed_dim,
        n_dec_layers=4,
        n_enc_layers=4,
        n_obs_token=224,
        goal_seq_len=1,
        obs_seq_len=1,
        action_seq_len=10,
        embed_pdrob=0,
        goal_drop=0,
        attn_pdrop=0.3,
        resid_pdrop=0.1,
        mlp_pdrop=0.05,
        n_heads=8,
        use_mlp_goal=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpp-checkpoint", required=True)
    p.add_argument("--embed-dim", type=int, default=384, help="Must match VPP's embed_dim (latent_dim=384 in its config).")
    p.add_argument("--obs-dim", type=int, default=768, help="DAWN's shared-encoder feature dim (CLIP ViT-L/14).")
    p.add_argument("--goal-dim", type=int, default=768)
    p.add_argument("--proprio-dim", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    vpp_ckpt = torch.load(args.vpp_checkpoint, map_location="cpu", weights_only=False)["model"]
    vpp_sd = {k[len("model.inner_model."):]: v for k, v in vpp_ckpt.items() if k.startswith("model.inner_model.")}

    ref = build_reference_module(7, args.obs_dim, args.goal_dim, args.proprio_dim, args.embed_dim)
    ours_sd = ref.state_dict()

    transferred, skipped_mismatch, skipped_atomic = {}, [], []
    for k, v in ours_sd.items():
        if k not in vpp_sd:
            continue
        if any(k == a or k.startswith(a + ".") for a in ATOMIC_SKIP_ON_MISMATCH) and tuple(vpp_sd[k].shape) != tuple(v.shape):
            skipped_atomic.append(k)
            continue
        if tuple(vpp_sd[k].shape) != tuple(v.shape):
            skipped_mismatch.append(k)
            continue
        transferred[k] = vpp_sd[k].clone()

    # Also drop any sibling of an atomically-skipped module (e.g. tok_emb.bias)
    # even if its own shape happened to match trivially (bias shape = out_dim only).
    for k in list(transferred.keys()):
        if any(k.startswith(a + ".") for a in ATOMIC_SKIP_ON_MISMATCH):
            del transferred[k]

    out_sd = {f"action_expert.inner_model.{k}": v for k, v in transferred.items()}

    total_params = sum(v.numel() for v in ours_sd.values())
    transferred_params = sum(v.numel() for v in transferred.values())

    print(f"inner_model total params: {total_params:,}")
    print(f"transferred: {len(transferred)} keys, {transferred_params:,} params ({transferred_params / total_params:.1%})")
    print(f"skipped (atomic, upstream-dim mismatch): {sorted(set(k.split('.')[0] for k in skipped_atomic))}")
    if skipped_mismatch:
        print(f"skipped (other shape mismatch): {skipped_mismatch}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_sd, args.out)
    print(f"Wrote {args.out} ({len(out_sd)} keys)")


if __name__ == "__main__":
    main()
