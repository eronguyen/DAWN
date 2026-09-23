"""Action expert that hosts VPP's own Video-Former + diffusion-transformer
policy inside `DAWNArch`, so its checkpoint (`yjguo/dp-calvin`) can be loaded
and run without any DAWN-side training.

Consumes `SVDMotionExtractor`'s raw `motion_feat` (see
`dawn/models/motion_director/svd_extractor.py`), resamples it with a ported
`Video_Former_3D`, encodes the language goal with a ported `LangClip`, and
feeds both into DAWN's existing `DiffusionTransformer` (architecturally
identical to VPP's own `model.inner_model`) at VPP's dims
(embed_dim=384, obs_dim=384, goal_dim=512).

The diffusion-loss / DDIM-sampling machinery is shared with `edp2.py` rather
than reimplemented (`sample_ddim`, `append_dims`, `dawn.models.action_expert
.utils`), since the two are the same EDM/k-diffusion formulation.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dawn.models.action_expert.base import BaseActionExpert
from dawn.models.action_expert.edp2 import append_dims, sample_ddim
from dawn.models.action_expert.utils import rand_log_logistic, rand_log_uniform, rand_uniform
from dawn.models.modules.clip_lang_encoder import LangClip
from dawn.models.modules.diffusion.diffusion_transformer import DiffusionTransformer
from dawn.models.modules.gc_sampling import (
    get_sigmas_exponential,
    get_sigmas_karras,
    get_sigmas_linear,
    get_sigmas_vp,
)
from dawn.models.modules.video_former import Video_Former_3D

logger = logging.getLogger(__name__)


class VPPCompatiblePolicy(BaseActionExpert):
    def __init__(
        self,
        action_dim: int = 7,
        latent_dim: int = 384,
        goal_dim: int = 512,
        num_latents: int = 224,
        num_frame: int = 16,
        Former_depth: int = 6,
        Former_heads: int = 8,
        Former_dim_head: int = 64,
        Former_num_time_embeds: int = 16,
        condition_dim: int = 2560,
        use_Former_temporal: bool = True,
        goal_window_size: int = 1,
        obs_seq_len: int = 1,
        act_seq_len: int = 10,
        proprio_dim: int = 8,
        noise_scheduler: str = "exponential",
        sigma_sample_density_type: str = "loglogistic",
        sampler_type: str = "ddim",
        num_sampling_steps: int = 10,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80,
        lang_model_name: str = "ViT-B/32",
        freeze_lang_encoder: bool = True,
    ) -> None:
        super().__init__()
        logger.info("Initializing %s.", __class__.__name__)

        self.Video_Former = Video_Former_3D(
            dim=latent_dim,
            depth=Former_depth,
            condition_dim=condition_dim,
            dim_head=Former_dim_head,
            heads=Former_heads,
            num_latents=num_latents,
            num_frame=num_frame,
            num_time_embeds=Former_num_time_embeds,
            use_temporal=use_Former_temporal,
        )
        self.language_goal = LangClip(freeze_backbone=freeze_lang_encoder, model_name=lang_model_name)

        self.model = nn.Module()
        self.model.inner_model = DiffusionTransformer(
            action_dim=action_dim,
            obs_dim=latent_dim,
            goal_dim=goal_dim,
            proprio_dim=proprio_dim,
            goal_conditioned=True,
            embed_dim=latent_dim,
            n_dec_layers=4,
            n_enc_layers=4,
            n_obs_token=num_latents,
            goal_seq_len=goal_window_size,
            obs_seq_len=obs_seq_len,
            action_seq_len=act_seq_len,
            embed_pdrob=0,
            goal_drop=0,
            attn_pdrop=0.3,
            resid_pdrop=0.1,
            mlp_pdrop=0.05,
            n_heads=8,
            use_mlp_goal=True,
        )

        self.sigma_data = sigma_data
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_sampling_steps = num_sampling_steps
        self.sampler_type = sampler_type
        self.noise_scheduler = noise_scheduler
        self.act_window_size = act_seq_len
        self.action_dim = action_dim
        self.criterion = F.mse_loss

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _language_list(batch_data: Dict[str, Any], batch_size: int) -> list:
        language = batch_data.get("language", batch_data.get("text"))
        if language is None:
            return [""] * batch_size
        if isinstance(language, str):
            return [language] * batch_size
        return [str(x) for x in language]

    def _get_motion_feat(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]],
        motion_outputs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        motion_feat = None
        if motion_outputs is not None:
            motion_feat = motion_outputs.get("motion_feat")
        if motion_feat is None and encoder_outputs is not None:
            motion_feat = encoder_outputs.get("motion_feat")
        if motion_feat is None:
            raise ValueError(
                "VPPCompatiblePolicy requires a `motion_feat` tensor from "
                "`SVDMotionExtractor` in motion_outputs/encoder_outputs."
            )
        return motion_feat

    def encode_inputs(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        motion_outputs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        motion_feat = self._get_motion_feat(batch_data, encoder_outputs, motion_outputs)
        state_images = self.Video_Former(motion_feat.to(self.device))

        batch_size = state_images.shape[0]
        language = self._language_list(batch_data, batch_size)
        latent_goal = self.language_goal(language).to(state_images.dtype)

        perceptual_emb = {"state_images": state_images, "modality": "language"}
        return perceptual_emb, latent_goal

    def forward_train(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        motion_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        perceptual_emb, latent_goal = self.encode_inputs(batch_data, encoder_outputs, motion_outputs)
        actions = batch_data["action"].to(self.device)

        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return {"total_loss": loss}

    @torch.inference_mode()
    def forward_eval(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        motion_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        perceptual_emb, latent_goal = self.encode_inputs(batch_data, encoder_outputs, motion_outputs)
        labels = batch_data.get("action")
        if labels is not None:
            labels = labels.to(self.device)
        return self.eval_forward(perceptual_emb, latent_goal, labels)

    def loss(self, state, action, goal, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.model.inner_model(state, noised_input * c_in, goal, sigma, **kwargs)
        target = (action - c_skip * noised_input) / c_out
        loss = (model_output - target).pow(2).flatten(1).mean()
        return loss, model_output

    def step(self, state, action, goal, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.model.inner_model(state, action * c_in, goal, sigma, **kwargs) * c_out + action * c_skip

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def make_sample_density(self):
        if self.sigma_sample_density_type == "loglogistic":
            loc = math.log(self.sigma_data)
            scale = 0.5
            return partial(
                rand_log_logistic, loc=loc, scale=scale, min_value=self.sigma_min, max_value=self.sigma_max
            )
        if self.sigma_sample_density_type == "loguniform":
            return partial(rand_log_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        if self.sigma_sample_density_type == "uniform":
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        raise ValueError(f"Unknown sample density type: {self.sigma_sample_density_type}")

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        if noise_schedule_type == "karras":
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7, self.device)
        if noise_schedule_type == "exponential":
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        if noise_schedule_type == "vp":
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        if noise_schedule_type == "linear":
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError(f"Unknown noise schedule type: {noise_schedule_type}")

    def sample_loop(self, sigmas, x_t, state, goal, sampler_type, extra_args=None):
        extra_args = extra_args or {}
        if sampler_type == "ddim":
            return sample_ddim(self, state, x_t, goal, sigmas, disable=True, extra_args=extra_args)
        raise ValueError(f"VPPCompatiblePolicy only supports sampler_type='ddim', got {sampler_type!r}.")

    def denoise_actions(self, perceptual_emb, latent_goal, inference: bool = False) -> torch.Tensor:
        sampling_steps = self.num_sampling_steps if inference else 10
        if latent_goal.ndim < perceptual_emb["state_images"].ndim:
            latent_goal = latent_goal.unsqueeze(1)
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        x = torch.randn(
            (latent_goal.shape[0], self.act_window_size, self.action_dim), device=self.device
        ) * self.sigma_max
        return self.sample_loop(sigmas, x, perceptual_emb, latent_goal, self.sampler_type)

    @torch.inference_mode()
    def eval_forward(
        self,
        perceptual_emb: Dict[str, Any],
        latent_goal: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        act_seq = self.denoise_actions(perceptual_emb, latent_goal, inference=True)
        return_dict = {"logits": act_seq}
        if labels is not None:
            return_dict["total_loss"] = self.criterion(
                act_seq.flatten(start_dim=1), labels.flatten(start_dim=1)
            )
        return return_dict
