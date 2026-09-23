"""SVD-based motion "director" that hosts VPP's own frozen feature-extraction
pipeline inside `DAWNArch`, so VPP's checkpoint (`yjguo/dp-calvin`) can be
loaded and run without any DAWN-side training.

Unlike `LDM` (which predicts a pixel-space optical-flow RGB image as its
stage-1 objective), this module has no training objective of its own -- it is
a frozen perception backbone, exactly as it is used inside VPP. It is only
meant for `motion_source`-agnostic stage-2 use: `forward_train`/`forward_eval`
are identical (both just run the frozen extractor).

Returns raw multi-frame, multi-view features already reshaped/concatenated the
way `VPP_Policy.extract_predictive_feature` does, i.e. shaped
`(B, num_frames, 2 * H' * W', C)` (static-view and gripper-view tokens
concatenated along the token axis) -- ready to feed directly into a
`Video_Former_3D` in the action expert.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from dawn.models.motion_director.base import BaseMotionDirector
from dawn.models.modules.vpp_diffusion_extractor import Diffusion_feature_extractor, load_primary_models

logger = logging.getLogger(__name__)


class SVDMotionExtractor(BaseMotionDirector):
    """Wraps VPP's `TVP_encoder` (an SVD-backbone diffusion feature extractor)."""

    def __init__(
        self,
        pretrained_model_path: str,
        text_encoder_path: str,
        timestep: int = 20,
        extract_layer_idx: int = 1,
        use_all_layer: bool = True,
        max_length: int = 20,
        position_encoding: bool = False,
        num_time_embeds: int = 16,
        eval_dtype: bool = True,
        input_type: str = "rgb_static",
        support_types: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            input_type=input_type,
            support_types=support_types or ["rgb_static", "rgb_gripper"],
        )
        self.timestep = timestep
        self.extract_layer_idx = extract_layer_idx
        self.use_all_layer = use_all_layer
        self.max_length = max_length
        self.num_time_embeds = num_time_embeds

        pipeline = load_primary_models(pretrained_model_path, eval=eval_dtype)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)
        text_encoder = text_encoder.eval()

        pipeline.image_encoder.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        pipeline.unet.eval()

        self.TVP_encoder = Diffusion_feature_extractor(
            pipeline=pipeline,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            position_encoding=position_encoding,
        )

        self.requires_grad_(False)
        self.eval()

        logger.info(
            "Initialized SVDMotionExtractor(pretrained_model_path=%s, text_encoder_path=%s, "
            "timestep=%s, extract_layer_idx=%s, use_all_layer=%s).",
            pretrained_model_path,
            text_encoder_path,
            timestep,
            extract_layer_idx,
            use_all_layer,
        )

    def process_device(self) -> None:
        """Move the wrapped diffusers pipeline to match this module's device.

        `self.TVP_encoder.pipeline` is a `DiffusionPipeline`, not an
        `nn.Module`, so it is not a registered submodule and `.to(device)` on
        this module (or its parents) silently leaves it in place. Mirrors
        VPP's own `VPP_Policy.process_device()`.
        """
        device = next(self.TVP_encoder.text_encoder.parameters()).device
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(device)

    def to(self, *args: Any, **kwargs: Any):  # noqa: D102
        ret = super().to(*args, **kwargs)
        self.process_device()
        return ret

    def cuda(self, device: Any = None):  # noqa: D102
        ret = super().cuda(device)
        self.process_device()
        return ret

    def _apply(self, fn, recurse: bool = True):
        """Called by a *parent* module's `.to()/.cuda()` (e.g. `DAWNArch.to(device)`)
        to recursively convert submodules -- unlike `.to()`, this is not something a
        parent calls on us by name, so overriding `.to()`/`.cuda()` alone (above)
        only helps when this module is moved directly, not as part of a larger
        model. `nn.Module.to()` internally recurses via `child._apply(fn)`, not
        `child.to(device)`, so this is the hook that actually fires in that case.
        """
        ret = super()._apply(fn, recurse=recurse)
        self.process_device()
        return ret

    def _get_view(self, batch_data: Dict[str, Any], view: str) -> torch.Tensor:
        images = batch_data["image"]
        rgb = images[view]
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
        return rgb[:, :1]  # current frame only, kept as (B, 1, C, H, W)

    @staticmethod
    def _to_svd_range(rgb01: torch.Tensor) -> torch.Tensor:
        # Our dataset convention is [0, 1]; VPP/SVD expects [-1, 1].
        return (rgb01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    @staticmethod
    def _resize_if_needed(rgb: torch.Tensor, size: int = 256) -> torch.Tensor:
        if rgb.shape[-2:] == (size, size):
            return rgb
        b, f, c, h, w = rgb.shape
        rgb = rearrange(rgb, "b f c h w -> (b f) c h w")
        rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False)
        return rearrange(rgb, "(b f) c h w -> b f c h w", b=b)

    @staticmethod
    def _language_list(batch_data: Dict[str, Any], batch_size: int) -> List[str]:
        language = batch_data.get("language", batch_data.get("text"))
        if language is None:
            return [""] * batch_size
        if isinstance(language, str):
            return [language] * batch_size
        return [str(x) for x in language]

    def _extract(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        rgb_static = self._resize_if_needed(self._to_svd_range(self._get_view(batch_data, "rgb_static")))
        rgb_gripper = self._resize_if_needed(self._to_svd_range(self._get_view(batch_data, "rgb_gripper")))
        batch_size = rgb_static.shape[0]

        language = self._language_list(batch_data, batch_size)
        input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
        doubled_language = language + language

        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                input_rgb,
                doubled_language,
                self.timestep,
                self.extract_layer_idx,
                all_layer=self.use_all_layer,
                step_time=1,
                max_length=self.max_length,
            )

        # (2B, F, C, H, W) -> (2B, F, L, C)
        perceptual_features = rearrange(perceptual_features, "b f c h w -> b f c (h w)")
        perceptual_features = rearrange(perceptual_features, "b f c l -> b f l c")
        perceptual_features = perceptual_features[:, : self.num_time_embeds]

        static_feat, gripper_feat = torch.split(perceptual_features, [batch_size, batch_size], dim=0)
        # Concatenate static-view and gripper-view tokens along the token axis.
        motion_feat = torch.cat([static_feat, gripper_feat], dim=2).float()
        return motion_feat

    def forward_train(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"motion_feat": self._extract(batch_data)}

    @torch.no_grad()
    def forward_eval(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"motion_feat": self._extract(batch_data)}
