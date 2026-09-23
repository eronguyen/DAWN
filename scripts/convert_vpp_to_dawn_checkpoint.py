"""Remap yjguo/dp-calvin's (Video Prediction Policy) raw checkpoint keys onto
DAWNArch's `motion_director`/`action_expert` submodule prefixes, so the
resulting file can be loaded directly via the existing, unmodified
`weights.model=<out>` path in `DAWNArch.from_pretrained`.

VPP's checkpoint (`last.pt`, `{"model": state_dict, "args": ...}`) was saved
from `VPP_Policy` directly, so its keys have no `action_expert.`/
`motion_director.` prefix at all -- they start with one of four top-level
names: `TVP_encoder.*`, `Video_Former.*`, `language_goal.*`,
`model.inner_model.*`. This is a pure rename, not a filter or shape-matched
transfer (contrast `scripts/convert_vpp_action_expert.py`, which
shape-matches a subset of `model.inner_model.*` onto edp2's differently-sized
transformer): here, `dawn.models.motion_director.svd_extractor
.SVDMotionExtractor` (attribute `self.TVP_encoder`) and
`dawn.models.action_expert.vpp_edp.VPPCompatiblePolicy` (attributes
`self.Video_Former`, `self.language_goal`, `self.model.inner_model`) were
built with exactly VPP's own attribute names and dims, so every key should
transfer with an unchanged shape.

Usage
-----
    python scripts/convert_vpp_to_dawn_checkpoint.py \
        --vpp-checkpoint /mnt/localssd/.cache/huggingface/hub/models--yjguo--dp-calvin/snapshots/*/last.pt \
        --out outputs/pretrained_transfer/dp_calvin_dawn_arch.pth
"""

from __future__ import annotations

import argparse
import os

import torch

# Top-level VPP checkpoint prefixes -> DAWNArch submodule prefix they get
# renamed under. Order matters: more specific prefixes are not needed here
# since none of these four names overlap.
PREFIX_MAP = {
    "TVP_encoder.": "motion_director.TVP_encoder.",
    "Video_Former.": "action_expert.Video_Former.",
    "language_goal.": "action_expert.language_goal.",
    "model.inner_model.": "action_expert.model.inner_model.",
}


def remap_key(key: str) -> str:
    for old_prefix, new_prefix in PREFIX_MAP.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    raise ValueError(f"Unrecognized VPP checkpoint key (no known prefix): {key!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpp-checkpoint", required=True, help="Path to VPP's last.pt.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.vpp_checkpoint, map_location="cpu", weights_only=False)
    vpp_sd = ckpt["model"] if "model" in ckpt else ckpt

    out_sd = {}
    counts = {new: 0 for new in PREFIX_MAP.values()}
    for k, v in vpp_sd.items():
        new_k = remap_key(k)
        out_sd[new_k] = v
        for new_prefix in PREFIX_MAP.values():
            if new_k.startswith(new_prefix):
                counts[new_prefix] += 1
                break

    print(f"Read {len(vpp_sd)} keys from {args.vpp_checkpoint}")
    for prefix, n in counts.items():
        print(f"  {prefix:<40} {n} keys")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out_sd, args.out)
    print(f"Wrote {args.out} ({len(out_sd)} keys)")


if __name__ == "__main__":
    main()
