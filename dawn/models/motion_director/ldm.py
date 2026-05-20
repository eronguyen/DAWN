from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from dawn.models.motion_director.base import BaseMotionDirector
from dawn.models.pixel_motion.estimator import PixelMotionEstimator
from dawn.models.pixel_motion.utils import PixelMotionVisualizer

logger = logging.getLogger(__name__)

class LDM(BaseMotionDirector):
    """
    Motion director trained as a DDPM latent diffusion model.

    Training:
    - encode target optical-flow RGB to VAE latents
    - encode condition image to VAE latents
    - sample timestep t and noise eps
    - noisy_target = scheduler.add_noise(target_latents, eps, t)
    - noisy_condition = scheduler.add_noise(condition_latents, eps_cond, t)
    - model_input = concat([noisy_target, noisy_condition], channel_dim)
    - UNet predicts scheduler target (epsilon or v)
    """

    def __init__(
        self,
        pretrained: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        image_size: int = 256,
        condition_dim: int = 768,
        conditioning_mode: str = "text",
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        condition_channel_init: str = "zeros",
        img2img_strength: float = 1.0,
        use_cfg: bool = True,
        cfg_dropout_prob: float = 0.1,
        flow_to_rgb: bool = True,
        flow_rgb_mode: str = "torchvision",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        input_type: str = "rgb_static",
        support_types: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            input_type=input_type,
            support_types=support_types or ["rgb_static", "rgb_gripper"],
        )
        self.image_size = image_size
        self.condition_dim = condition_dim
        self.conditioning_mode = conditioning_mode
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.condition_channel_init = condition_channel_init
        self.img2img_strength = float(img2img_strength)
        self.use_cfg = use_cfg
        self.cfg_dropout_prob = float(cfg_dropout_prob)

        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
        self._expand_unet_in_channels_for_condition(condition_channel_init)
        # self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(num_train_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        self.flow_estimator = PixelMotionEstimator(
            flow_to_rgb=flow_to_rgb,
            flow_rgb_mode=flow_rgb_mode,
        )

        self.vae.requires_grad_(False)
        self.flow_estimator.requires_grad_(False)
        # Project visual embeddings into UNet cross-attention dimension.
        # self.visual_proj = nn.LazyLinear(condition_dim)

        logger.info(
            (
                "Initialized LDM(image_size=%s, input_type=%s, flow_rgb_mode=%s, "
                "conditioning_mode=%s, num_inference_steps=%s, guidance_scale=%s, "
                "img2img_strength=%s, "
                "condition_channel_init=%s, prediction_type=%s)."
            ),
            image_size,
            input_type,
            flow_rgb_mode,
            conditioning_mode,
            self.num_inference_steps,
            self.guidance_scale,
            self.img2img_strength,
            self.condition_channel_init,
            prediction_type,
        )

    @staticmethod
    def _select_frame(images: torch.Tensor, which: str = "first") -> torch.Tensor:
        # Accepts BCHW or BTCHW. Returns BCHW.
        if images.ndim == 4:
            return images
        if images.ndim != 5:
            raise ValueError(f"Expected image shape (B,C,H,W) or (B,T,C,H,W), got {tuple(images.shape)}.")
        if which == "first":
            return images[:, 0]
        return images[:, -1]

    def _get_images_tensor(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        images = batch_data["image"]
        if isinstance(images, dict):
            if self.input_type in images:
                images = images[self.input_type]
            elif len(images) == 1:
                images = next(iter(images.values()))
            else:
                raise ValueError(
                    f"`image` is a multi-view dict but input_type='{self.input_type}' was not found. "
                    f"Available views: {list(images.keys())}."
                )
        if not torch.is_tensor(images):
            raise ValueError(f"Expected image tensor or dict[str, tensor], got {type(images)}.")
        return images

    def _to_vae_latents(self, image_bchw: torch.Tensor, scale: bool = True) -> torch.Tensor:
        image_bchw = image_bchw.to(next(self.vae.parameters()).device)
        image_bchw = (image_bchw * 2.0 - 1.0).clamp(-1.0, 1.0)
        latents = self.vae.encode(image_bchw).latent_dist.mode()
        if scale:
            latents = latents * self.vae.config.scaling_factor
        return latents

    def _decode_vae_latents(self, latents: torch.Tensor, scale: bool = True) -> torch.Tensor:
        if scale:
            latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        return (image / 2.0 + 0.5).clamp(0.0, 1.0)

    def _expand_unet_in_channels_for_condition(self, init_mode: str = "zeros") -> None:
        """
        Expand UNet input channels from C to 2C so we can concatenate:
        [noisy_target_latents, condition_latents_t].

        First C channels keep pretrained weights; extra C channels use init_mode.
        """
        conv_in = self.unet.conv_in
        in_c = conv_in.in_channels
        out_c = conv_in.out_channels
        k = conv_in.kernel_size
        s = conv_in.stride
        p = conv_in.padding
        has_bias = conv_in.bias is not None

        if in_c == 8:
            return 
        
        new_conv = nn.Conv2d(
            in_channels=in_c * 2,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=has_bias,
            device=conv_in.weight.device,
            dtype=conv_in.weight.dtype,
        )
        with torch.no_grad():
            new_conv.weight[:, :in_c].copy_(conv_in.weight)
            if init_mode == "zeros":
                new_conv.weight[:, in_c:].zero_()
            elif init_mode == "normal":
                nn.init.normal_(new_conv.weight[:, in_c:], mean=0.0, std=0.02)
            else:
                raise ValueError("condition_channel_init must be 'zeros' or 'normal'.")
            if has_bias:
                new_conv.bias.copy_(conv_in.bias)

        self.unet.conv_in = new_conv
        if hasattr(self.unet.config, "in_channels"):
            self.unet.config.in_channels = in_c * 2

    def get_target_flow_rgb(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        target = batch_data.get("target_flow_rgb")
        if target is None:
            target = batch_data.get("target_flow")
        if target is None:
            # If there is no temporal pair (single frame), flow is undefined.
            # Return zeros to keep inference/eval paths stable.
            images = self._get_images_tensor(batch_data)
            if images.ndim == 4:
                b, _, h, w = images.shape
                return torch.zeros((b, 3, h, w), dtype=images.dtype, device=images.device)
            if images.ndim == 5 and images.shape[1] < 2:
                b, _, _, h, w = images.shape
                return torch.zeros((b, 3, h, w), dtype=images.dtype, device=images.device)

            # Fallback target from optical-flow estimator output.
            est = self.flow_estimator.estimate_flow(images)
            if est.ndim != 5:
                raise ValueError(f"Expected estimator output with 5 dims, got {tuple(est.shape)}.")
            target = est[:, 0]
        elif target.ndim == 5:
            target = target[:, 0]

        if target.ndim != 4:
            raise ValueError(f"Expected target flow RGB shape (B,C,H,W), got {tuple(target.shape)}.")
        return target

    @staticmethod
    def _default_prompt_embeds(batch_size: int, device: torch.device, condition_dim: int) -> torch.Tensor:
        # 77 is CLIP context length used by SD1.x text encoder.
        return torch.zeros((batch_size, 77, condition_dim), device=device)

    @staticmethod
    def _as_batch_timesteps(t: torch.Tensor | int | float, batch_size: int, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(t):
            if t.ndim == 0:
                t = t.long().view(1).repeat(batch_size)
            elif t.ndim == 1 and t.shape[0] == batch_size:
                t = t.long()
            else:
                t = t.reshape(-1)[0].long().view(1).repeat(batch_size)
            return t.to(device)
        return torch.full((batch_size,), int(t), device=device, dtype=torch.long)

    def _to_condition_tokens(
        self,
        encoder_outputs: Optional[Dict[str, Any]],
        batch_size: int,
        device: torch.device,
        conditioning_mode: Optional[str] = None,
    ) -> torch.Tensor:
        mode = conditioning_mode or self.conditioning_mode
        if mode not in ("text", "text+visual"):
            raise ValueError(f"Unsupported conditioning_mode: {mode}. Use 'text' or 'text+visual'.")

        text_tokens = None
        if encoder_outputs is not None:
            text_tokens = encoder_outputs.get("text_feat")
        if text_tokens is not None:
            text_tokens = text_tokens.to(device)

        if mode == "text":
            if text_tokens is None:
                return self._default_prompt_embeds(batch_size, device, self.condition_dim)
            return text_tokens

        # mode == "text+visual"
        visual_tokens = []
        if encoder_outputs is not None:
            view_image_feat = encoder_outputs.get("view_image_feat")
            if isinstance(view_image_feat, dict):
                for feat in view_image_feat.values():
                    if feat is None:
                        continue
                    if feat.ndim == 2:
                        visual_tokens.append(feat.unsqueeze(1))
                    elif feat.ndim == 3:
                        visual_tokens.append(feat)

        if visual_tokens:
            visual_tokens = torch.cat([v.to(device) for v in visual_tokens], dim=1)
            # visual_tokens = self.visual_proj(visual_tokens)
        else:
            visual_tokens = None

        if text_tokens is None and visual_tokens is None:
            return self._default_prompt_embeds(batch_size, device, self.condition_dim)
        if text_tokens is None:
            return visual_tokens
        if visual_tokens is None:
            return text_tokens
        return torch.cat([text_tokens, visual_tokens], dim=1)

    @staticmethod
    def _apply_cfg_dropout(condition_tokens: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if drop_prob <= 0.0:
            return condition_tokens
        if drop_prob >= 1.0:
            return torch.zeros_like(condition_tokens)
        # Per-sample conditional dropout.
        keep_mask = (torch.rand(condition_tokens.shape[0], device=condition_tokens.device) > drop_prob).float()
        keep_mask = keep_mask.view(-1, 1, 1)
        return condition_tokens * keep_mask

    def _scheduler_target(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v_prediction":
            return self.scheduler.get_velocity(clean_latents, noise, timesteps)
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")

    def _compose_model_input(self, noisy_latents: torch.Tensor, cond_latents_t: torch.Tensor) -> torch.Tensor:
        # Concatenate along channel axis: [B, 2C, H, W]
        return torch.cat([noisy_latents, cond_latents_t], dim=1)

    def _prepare_img2img_latents(
        self,
        condition_latents: torch.Tensor,
        num_inference_steps: int,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = condition_latents.device
        bsz = condition_latents.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        strength = float(max(0.0, min(1.0, strength)))
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        if init_timestep <= 0:
            return condition_latents.clone(), timesteps[-1:]

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        noise = torch.randn_like(condition_latents)
        latent_timestep = timesteps[:1].repeat(bsz)
        latents = self.scheduler.add_noise(condition_latents, noise, latent_timestep)
        return latents, timesteps

    def forward_train(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        images = self._get_images_tensor(batch_data)
        condition_frame = self._select_frame(images, which="first")
        target_flow_rgb = self.get_target_flow_rgb(batch_data)

        condition_latents = self._to_vae_latents(condition_frame, scale=False)
        target_latents = self._to_vae_latents(target_flow_rgb)

        bsz = target_latents.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=target_latents.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(target_latents)
        noisy_target_latents = self.scheduler.add_noise(target_latents, noise, timesteps)

        model_input = self._compose_model_input(noisy_target_latents, condition_latents)

        prompt_embeds = self._to_condition_tokens(
            encoder_outputs=encoder_outputs,
            batch_size=bsz,
            device=target_latents.device,
            conditioning_mode=kwargs.get("conditioning_mode"),
        )
        if self.use_cfg:
            prompt_embeds = self._apply_cfg_dropout(prompt_embeds, self.cfg_dropout_prob)

        model_pred = self.unet(
            model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds.to(target_latents.device),
        ).sample

        loss_mse = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        return {
            "pred_noise": model_pred,
            "target_noise": noise,
            "timesteps": timesteps,
            "model_input": model_input,
            "condition_latents": condition_latents,
            "target_latents": target_latents,
            "target_flow_rgb": target_flow_rgb,
            "loss_mse": loss_mse,
            "total_loss": loss_mse,
        }

    def forward_eval(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        images = self._get_images_tensor(batch_data)
        condition_frame = self._select_frame(images, which="first")
        condition_latents = self._to_vae_latents(condition_frame, scale=False)

        bsz = condition_latents.shape[0]
        prompt_embeds = self._to_condition_tokens(
            encoder_outputs=encoder_outputs,
            batch_size=bsz,
            device=condition_latents.device,
            conditioning_mode=kwargs.get("conditioning_mode"),
        )

        num_inference_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        guidance_scale = float(kwargs.get("guidance_scale", self.guidance_scale))
        strength = float(kwargs.get("strength", self.img2img_strength))

        latents_history = []
        
        if num_inference_steps == 0:
            latents = condition_latents
        else:
            # latents, timesteps = self._prepare_img2img_latents(
            #     condition_latents=condition_latents,
            #     num_inference_steps=num_inference_steps,
            #     strength=strength,
            # )
            latents = torch.randn_like(condition_latents)
            self.scheduler.set_timesteps(num_inference_steps, device=condition_latents.device)
            timesteps = self.scheduler.timesteps
            
            uncond_tokens = torch.zeros_like(prompt_embeds)
            
            for t in timesteps:
                
                model_input = self._compose_model_input(latents, condition_latents)

                if guidance_scale > 1.0:
                    noise_uncond = self.unet(model_input, t, encoder_hidden_states=uncond_tokens).sample
                    noise_cond = self.unet(model_input, t, encoder_hidden_states=prompt_embeds).sample
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = self.unet(model_input, t, encoder_hidden_states=prompt_embeds).sample

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                latents_history.append((t, latents.clone()))

        predicted_flow_rgb = self._decode_vae_latents(latents)
        
        output = {
            "prompt_embeds": prompt_embeds,
            "condition_latents": condition_latents,
            "predicted_flow_latents": latents,
            "predicted_flow_rgb": predicted_flow_rgb,
            "predicted_pixel_motion": predicted_flow_rgb.unsqueeze(1),
            "predicted_flow_history": latents_history,
        }

        target_flow_rgb = self.get_target_flow_rgb(batch_data)
        target_latents = self._to_vae_latents(target_flow_rgb)
        output["target_flow_rgb"] = target_flow_rgb
        output["target_flow_latents"] = target_latents

        mse_flow_latents = torch.nn.functional.mse_loss(
            latents.float(),
            target_latents.float(),
            reduction="mean",
        )

        mse_flow_rgb = torch.nn.functional.mse_loss(
            predicted_flow_rgb.float(),
            target_flow_rgb.float(),
            reduction="mean",
        )

        target_flow_raw = self.flow_estimator.flow_converter.rgb_to_flow(target_flow_rgb)
        predicted_flow_raw = self.flow_estimator.flow_converter.rgb_to_flow(predicted_flow_rgb)

        mse_flow_raw = torch.nn.functional.mse_loss(
            predicted_flow_raw.float(),
            target_flow_raw.float(),
            reduction="mean",
        )
        output["loss_mse_flow"] = mse_flow_raw
        output["loss_mse_rgb"] = mse_flow_rgb
        output["loss_mse_latents"] = mse_flow_latents
        output["total_loss"] = mse_flow_rgb + mse_flow_latents

        return output

    def visualize(
        self,
        batch_data: Dict[str, Any],
        outputs: Dict[str, Any],
        max_items: int = 8,
        step: int = 16,
        overlay_alpha: float = 0.6,
        min_magnitude: float = 1.0,
        title_prefix: str = "Stage-1 Predicted vs Target Motion",
        predict_only: bool = False,
    ) -> list[Any]:
        images = self._get_images_tensor(batch_data)
        pred_frame = self._select_frame(images, which="first")
        target_frame = self._select_frame(images, which="last") if images.ndim == 5 and images.shape[1] > 1 else pred_frame

        pred_flow_rgb = outputs.get("predicted_flow_rgb")
        if pred_flow_rgb is None:
            return []
        target_flow_rgb = outputs.get("target_flow_rgb", batch_data.get("target_flow_rgb"))

        bsz = pred_flow_rgb.shape[0]
        n_items = max(1, min(int(max_items), int(bsz)))

        visualizer = PixelMotionVisualizer(
            flow_rgb_mode=self.flow_estimator.flow_converter.mode,
            mag_norm=self.flow_estimator.flow_converter.mag_norm,
        )

        result = []
        for i in range(n_items):
            pred_flow_i = pred_flow_rgb[i]
            target_flow_i = target_flow_rgb[i] if target_flow_rgb is not None else None
            
            if predict_only:
                img = visualizer.visualize_flow_vectors_as_pil(
                    image=pred_frame[i],
                    flow=pred_flow_i,
                    flow_is_rgb=True,
                    step=step,
                    title="",
                )
            else:
                img = visualizer.visualize_pred_target_grid_as_pil(
                    pred_image=pred_frame[i],
                    pred_flow=pred_flow_i,
                    pred_flow_is_rgb=True,
                    target_image=target_frame[i],
                    target_flow=target_flow_i,
                    target_flow_is_rgb=target_flow_i is not None,
                    step=step,
                    overlay_alpha=overlay_alpha,
                    min_magnitude=min_magnitude,
                    title=f"{title_prefix} [sample={i}]",
                )
            result.append(img)
        return result
