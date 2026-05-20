from __future__ import annotations
import sys
sys.path.append(".")  # Ensure project root is in path for imports.
import torch

from dawn.models.motion_director.ldm import LDM


def build_dummy_batch(batch_size=2, frames=2, h=256, w=256):
    return {
        "image": {
            "rgb_static": torch.rand(batch_size, frames, 3, h, w),
            "rgb_gripper": torch.rand(batch_size, frames, 3, h, w),
        },
        "target_flow_rgb": torch.rand(batch_size, 3, h, w),
        "language": ["pick up the block"] * batch_size,
    }


def build_dummy_encoder_outputs(batch_size=2, dim=768):
    return {
        "text_feat": torch.randn(batch_size, 77, dim),
        "image_feat": torch.randn(batch_size, dim),
        "view_image_feat": {
            "rgb_static": torch.randn(batch_size, dim),
            "rgb_gripper": torch.randn(batch_size, dim),
        },
    }


def run_dummy_io_once(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    # This uses the real LDM class and real model components.
    # Make sure SD/vae weights are available (it may download on first run).
    model = LDM(
        pretrained="stable-diffusion-v1-5/stable-diffusion-v1-5",
        image_size=256,
        condition_dim=768,
        conditioning_mode="text+visual",
        num_inference_steps=8,
        guidance_scale=2.0,
        use_condition_latents=True,
        input_type="rgb_static",
    ).to(device)

    batch_data = build_dummy_batch(batch_size=1, frames=2, h=256, w=256)
    encoder_outputs = build_dummy_encoder_outputs(batch_size=1, dim=768)

    # Move tensor inputs to model device.
    for k, v in batch_data["image"].items():
        batch_data["image"][k] = v.to(device)
    batch_data["target_flow_rgb"] = batch_data["target_flow_rgb"].to(device)

    for k, v in encoder_outputs.items():
        if isinstance(v, dict):
            for vk, vv in v.items():
                v[vk] = vv.to(device)
        elif torch.is_tensor(v):
            encoder_outputs[k] = v.to(device)

    model.train()
    train_out = model(batch_data, encoder_outputs=encoder_outputs)
    print("[train] keys:", sorted(train_out.keys()))
    print("[train] loss:", float(train_out["loss"].detach().cpu()))

    model.eval()
    with torch.inference_mode():
        eval_out = model(
            batch_data,
            encoder_outputs=encoder_outputs,
            num_inference_steps=6,
            guidance_scale=1.5,
            return_estimator=False,
        )
    print("[eval] keys:", sorted(eval_out.keys()))
    print("[eval] predicted_flow_rgb shape:", tuple(eval_out["predicted_flow_rgb"].shape))


if __name__ == "__main__":
    run_dummy_io_once()
