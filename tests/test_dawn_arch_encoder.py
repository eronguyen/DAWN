from __future__ import annotations
import sys
sys.path.append(".")  # Ensure project root is in path for imports.

import torch

from dawn.models.arch.dawn import DAWNArch
from dawn.models.encoder import Encoder


def _shape(x):
    return None if x is None else tuple(x.shape)


def run_encoder_only_arch_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = DAWNArch(
        encoder=Encoder(pretrained="stable-diffusion-v1-5/stable-diffusion-v1-5").to(device),
        motion_director=None,
        action_expert=None,
    ).to(device)

    batch_data = {
        "image": {
            "rgb_static": torch.rand(2, 4, 3, 224, 224, device=device),
            "rgb_gripper": torch.rand(2, 4, 3, 224, 224, device=device),
        },
        "language": ["pick up the block", "move to the target"],
    }

    with torch.inference_mode():
        enc = arch.encode_batch(batch_data)

    print("encoder outputs:")
    print("- text_feat:", _shape(enc.get("text_feat")))
    
    view_feats = enc.get("view_image_feat")
    if isinstance(view_feats, dict):
        for view_name, feat in view_feats.items():
            print(f"- view_image_feat[{view_name}]:", _shape(feat))
    else:
        print("- view_image_feat:", view_feats)


if __name__ == "__main__":
    run_encoder_only_arch_test()
