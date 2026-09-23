"""SVD-based diffusion feature extractor, ported from VPP (yjguo/dp-calvin /
yjguo/svd-robot-calvin-ft) so its checkpoint's `TVP_encoder.*` weights load
directly and its frozen SVD backbone can be loaded via
`StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)` exactly
as VPP itself does.

Source: vpp-hiva/policy_models/module/diffusion_extract.py, trimmed of an
unused, fully-commented-out `step_unet` variant.
"""

from __future__ import annotations

import contextlib
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from diffusers import StableVideoDiffusionPipeline


@contextlib.contextmanager
def _sdpa_math_fallback():
    """The temporal-mixing attention in the SVD UNet flattens spatial positions
    into the batch dim, which can exceed a grid-size limit in PyTorch's
    flash/mem-efficient SDPA CUDA kernels and fail with
    ``CUDA error: invalid argument``. That path has a tiny sequence length
    (num_frames), so falling back to the math backend just for the failing
    call is cheap; forcing math everywhere else would OOM on the
    large-sequence spatial attention blocks.
    """
    orig = torch.nn.functional.scaled_dot_product_attention

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        try:
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        except RuntimeError:
            with sdpa_kernel(SDPBackend.MATH):
                return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = wrapped
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig


def load_primary_models(pretrained_model_path: str, eval: bool = False):
    """Mirrors VPP's `VPP_policy.load_primary_models`."""
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline


class Diffusion_feature_extractor(nn.Module):
    def __init__(
        self,
        pipeline: StableVideoDiffusionPipeline,
        tokenizer: AutoTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        position_encoding: bool = True,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.num_frames = 16
        self.position_encoding = position_encoding

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        texts,
        timestep: Union[torch.Tensor, float, int],
        extract_layer_idx: Union[torch.Tensor, float, int],
        all_layer: bool = False,
        step_time: int = 1,
        max_length: int = 20,
    ):
        height = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor // 3
        width = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor // 3
        self.pipeline.vae.eval()
        self.pipeline.image_encoder.eval()
        device = self.pipeline.unet.device
        dtype = self.pipeline.vae.dtype
        vae = self.pipeline.vae

        num_videos_per_prompt = 1
        batch_size = pixel_values.shape[0]
        pixel_values = rearrange(pixel_values, "b f c h w-> (b f) c h w").to(dtype)

        with torch.no_grad():
            encoder_hidden_states = self.encode_text(
                texts, self.tokenizer, self.text_encoder, position_encode=self.position_encoding,
                use_clip=True, max_length=max_length,
            )
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        image_embeddings = encoder_hidden_states

        if pixel_values.shape[-3] == 4:
            image_latents = pixel_values / vae.config.scaling_factor
        else:
            image_latents = self.pipeline._encode_vae_image(pixel_values, device, num_videos_per_prompt, False)
        image_latents = image_latents.to(image_embeddings.dtype)

        num_frames = self.num_frames
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        fps = 4
        motion_bucket_id = 127
        added_time_ids = self.pipeline._get_add_time_ids(
            fps, motion_bucket_id, 0, image_embeddings.dtype, batch_size, num_videos_per_prompt, False,
        )
        added_time_ids = added_time_ids.to(device)

        self.pipeline.scheduler.set_timesteps(timestep, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_videos_per_prompt, num_frames, num_channels_latents, height, width,
            image_embeddings.dtype, device, None, None,
        )

        feature_pred = None
        for i, t in enumerate(timesteps):
            complete = i != step_time - 1

            latent_model_input = self.pipeline.scheduler.scale_model_input(latents, t)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            with _sdpa_math_fallback():
                feature_pred = self.step_unet(
                    latent_model_input, t, encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids, use_layer_idx=extract_layer_idx,
                    all_layer=all_layer, complete=complete,
                )[0]

            if not complete:
                break
            latents = self.pipeline.scheduler.step(feature_pred, t, latents).prev_sample

        return feature_pred

    def step_unet(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        use_layer_idx: int = 5,
        all_layer: bool = False,
        complete: bool = False,
    ):
        """Runs the SVD UNet's down/mid blocks fully, then stops partway through the
        up blocks (at `use_layer_idx`) to return intermediate features instead of a
        final denoised sample -- unless `complete`, in which case it runs to
        completion (used for the last partial-denoising step)."""
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            dtype = (torch.float32 if is_mps else torch.float64) if isinstance(timestep, float) else (torch.int32 if is_mps else torch.int64)
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.pipeline.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.pipeline.unet.time_embedding(t_emb)

        time_embeds = self.pipeline.unet.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.pipeline.unet.add_embedding(time_embeds)
        emb = emb + aug_emb

        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        sample = self.pipeline.unet.conv_in(sample)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.pipeline.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        sample = self.pipeline.unet.mid_block(
            hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        feature_list = []
        for i, upsample_block in enumerate(self.pipeline.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states, image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
            if i < use_layer_idx:
                factor = 2 ** (use_layer_idx - i)
                feature_list.append(torch.nn.functional.interpolate(sample, scale_factor=factor))
            if i == use_layer_idx and not complete:
                feature_list.append(sample)
                break

        if not complete:
            if all_layer:
                sample = torch.cat(feature_list, dim=1)
            sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
            return (sample,)

        sample = self.pipeline.unet.conv_norm_out(sample)
        sample = self.pipeline.unet.conv_act(sample)
        sample = self.pipeline.unet.conv_out(sample)
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        return (sample,)

    @torch.no_grad()
    def encode_text(
        self, texts, tokenizer, text_encoder,
        position_encode: bool = True, use_clip: bool = False, max_length: int = 20,
    ) -> torch.Tensor:
        def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
            assert embed_dim % 2 == 0
            omega = np.arange(embed_dim // 2, dtype=np.float64)
            omega /= embed_dim / 2.0
            omega = 1.0 / 10000**omega
            pos = pos.reshape(-1)
            out = np.einsum("m,d->md", pos, omega)
            emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
            return emb

        if use_clip:
            inputs = tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True, max_length=max_length).to(text_encoder.device)
            outputs = text_encoder(**inputs)
            encoder_hidden_states = outputs.last_hidden_state
            if position_encode:
                embed_dim, pos_num = encoder_hidden_states.shape[-1], encoder_hidden_states.shape[1]
                pos = np.arange(pos_num, dtype=np.float64)
                pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
                pos_embed = torch.tensor(pos_embed, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype, requires_grad=False)
                encoder_hidden_states = encoder_hidden_states + pos_embed
            assert encoder_hidden_states.shape[-1] == 512
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=-1)
        else:
            inputs = tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True, max_length=32).to(text_encoder.device)
            outputs = text_encoder(**inputs)
            encoder_hidden_states = outputs.last_hidden_state
            assert encoder_hidden_states.shape[1:] == (32, 1024)

        return encoder_hidden_states
