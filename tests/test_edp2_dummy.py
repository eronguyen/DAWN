from __future__ import annotations
import sys
sys.path.append(".")  # Ensure project root is in path for imports.
import torch

from dawn.models.action_expert.edp2 import TransformerDiffusionPolicy


def build_dummy_batch(batch_size=2, act_seq_len=10, action_dim=7, h=224, w=224):
    return {
        "image": {
            "rgb_static": torch.rand(batch_size, 1, 3, h, w),
            "rgb_gripper": torch.rand(batch_size, 1, 3, h, w),
        },
        "action": torch.randn(batch_size, act_seq_len, action_dim),
    }


def build_dummy_encoder_outputs(batch_size=2, obs_dim=768, n_view_tokens=197, dim=768, text_len=20):
    return {
        "view_image_feat": {
            "rgb_static": torch.randn(batch_size, n_view_tokens, obs_dim),
            "rgb_gripper": torch.randn(batch_size, n_view_tokens, obs_dim),
        },
        "text_feat": torch.randn(batch_size, text_len, dim),
    }


def build_dummy_motion_outputs(batch_size=2, h=224, w=224):
    return {"predicted_flow_rgb": torch.rand(batch_size, 3, h, w)}


def run_dummy_io_once(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model = TransformerDiffusionPolicy(
        action_dim=7,
        obs_dim=768,
        goal_dim=768,
        embed_dim=768,
        num_latents=224,
        goal_window_size=1,
        obs_seq_len=1,
        act_seq_len=10,
        proprio_dim=8,
        num_sampling_steps=10,
        sigma_data=0.5,
        sigma_min=0.001,
        sigma_max=80,
        noise_scheduler="exponential",
        sigma_sample_density_type="loglogistic",
        sampler_type="ddim",
        motion_pretrained="google/vit-base-patch16-224-in21k",
        motion_in_channels=6,
        motion_rgb_view="rgb_static",
        motion_input_size=224,
        freeze_motion_encoder=False,
    ).to(device)

    batch_size = 2
    batch_data = build_dummy_batch(batch_size=batch_size)
    encoder_outputs = build_dummy_encoder_outputs(batch_size=batch_size)
    motion_outputs = build_dummy_motion_outputs(batch_size=batch_size)

    for k, v in batch_data["image"].items():
        batch_data["image"][k] = v.to(device)
    batch_data["action"] = batch_data["action"].to(device)

    for k, v in encoder_outputs.items():
        if isinstance(v, dict):
            for vk, vv in v.items():
                v[vk] = vv.to(device)
        elif torch.is_tensor(v):
            encoder_outputs[k] = v.to(device)

    for k, v in motion_outputs.items():
        motion_outputs[k] = v.to(device)

    # --- train forward + backward -----------------------------------------
    model.train()
    train_out = model(batch_data, encoder_outputs=encoder_outputs, motion_outputs=motion_outputs)
    print("[train] keys:", sorted(train_out.keys()))
    loss = train_out["total_loss"]
    print("[train] total_loss:", float(loss.detach().cpu()))
    loss.backward()

    patch_embed_grad = None
    for name, p in model.motion_encoder.named_parameters():
        if "patch_embeddings" in name and "weight" in name and p.grad is not None:
            patch_embed_grad = (name, p.grad.abs().sum().item())
            break
    print("[train] motion_encoder patch-embed grad (name, |grad|.sum()):", patch_embed_grad)
    assert patch_embed_grad is not None and patch_embed_grad[1] > 0, \
        "expected non-zero gradient on the widened patch-embed conv (motion encoder should be trainable)"

    # --- eval forward --------------------------------------------------
    model.eval()
    eval_out = model(batch_data, encoder_outputs=encoder_outputs, motion_outputs=motion_outputs)
    print("[eval] keys:", sorted(eval_out.keys()))
    print("[eval] logits shape:", tuple(eval_out["logits"].shape))
    assert eval_out["logits"].shape == (batch_size, model.act_window_size, model.action_dim)

    print("Smoke test passed.")


if __name__ == "__main__":
    run_dummy_io_once()
