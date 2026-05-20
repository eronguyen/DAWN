from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import numpy as np
from torch import nn
from logging import getLogger
from humanfriendly import format_size

from dawn.models.action_expert.base import BaseActionExpert
from dawn.models.encoder import Encoder
from dawn.models.motion_director.base import BaseMotionDirector

logger = getLogger(__name__)

class DAWNArch(nn.Module):
    """
    Top-level DAWN architecture.

    It composes:
    - a shared `Encoder` for image/text features
    - a `MotionDirector`
    - an `ActionExpert`
    """

    SUPPORTED_MODES = ("motion", "action", "joint")
    SUPPORTED_STAGES = (1, 2)
    SUPPORTED_MOTION_SOURCES = ("predicted", "estimator")

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        motion_director: Optional[BaseMotionDirector] = None,
        action_expert: Optional[BaseActionExpert] = None,
        stage: int = 1,
        motion_source: str = "predicted",
        use_predicted_motion_eval: bool = False,

    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.motion_director = motion_director
        self.action_expert = action_expert

        self.stage = stage
        self.motion_source = motion_source
        self.use_predicted_motion_eval = use_predicted_motion_eval

        if self.motion_source == "estimator":
            self.motion_director.requires_grad_(False)

        if self.stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Invalid stage: {self.stage}. Supported stages: {self.SUPPORTED_STAGES}.")
        if self.motion_source not in self.SUPPORTED_MOTION_SOURCES:
            raise ValueError(
                f"Invalid motion_source: {self.motion_source}. "
                f"Supported: {self.SUPPORTED_MOTION_SOURCES}."
            )

        # if self.stage == 2:
        #     self.encoder.image_proj.requires_grad_(False) # Freeze the projector
        # self.encoder.requires_grad_(True)
        
        logger.info("Model parameter summary:\n%s", self.module_parameter_table())

    def from_pretrained(self, weights) -> None:
        if "model" in weights and weights["model"] is not None:
            logger.info("Loading model weights from %s", weights["model"])
            ckpt = torch.load(weights["model"], map_location="cpu")
            logger.info(self.load_state_dict(ckpt, strict=False))
        
        if "motion_director" in weights and self.motion_director is not None and weights["motion_director"] is not None:
            logger.info("Loading motion_director weights from %s", weights["motion_director"])
            ckpt = torch.load(weights["motion_director"], map_location="cpu")
            ckpt = {k.replace("motion_director.","") : v for k, v in ckpt.items() if k.startswith("motion_director.")}
            logger.info(self.motion_director.load_state_dict(ckpt, strict=False))
        
        if "action_expert" in weights and self.action_expert is not None and weights["action_expert"] is not None:
            logger.info("Loading action_expert weights from %s", weights["action_expert"])
            ckpt = torch.load(weights["action_expert"], map_location="cpu")
            ckpt = {k.replace("action_expert.","") : v for k, v in ckpt.items() if k.startswith("action_expert.")}
            logger.info(self.action_expert.load_state_dict(ckpt, strict=False))
        
    def encode_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.encoder is None:
            return {}

        encoder_outputs: Dict[str, Any] = {}

        image = batch_data.get("image")
        text = batch_data.get("language", batch_data.get("text"))
        text_feat = self.encoder.encode_text(text) if text is not None else None

        view_image_feat: Dict[str, Any] = {}

        for view_name, view_tensor in image.items():
            view_input = self._prepare_encoder_image(view_tensor)
            feat = self.encoder.encode_image(view_input)
            view_image_feat[view_name] = feat
            
        encoder_outputs["view_image_feat"] = view_image_feat if view_image_feat else None
        encoder_outputs["text_feat"] = text_feat
        return encoder_outputs

    @staticmethod
    def _prepare_encoder_image(image: torch.Tensor) -> torch.Tensor:
        # Accept BCHW or BTCHW; use first frame for shared CLIP visual embedding.
        if torch.is_tensor(image) and image.ndim == 5:
            return image[:, 0]
        return image

    def forward(
        self,
        batch_data: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        outputs["encoder"] = self.encode_batch(batch_data)

        if self.stage == 1:
            if self.motion_director is None:
                raise ValueError("Stage 1 requires `motion_director`.")
            
            outputs["motion"] = self.motion_director(
                batch_data,
                encoder_outputs=outputs["encoder"],
                **kwargs,
            )

            return outputs["motion"]

        # stage == 2
        if self.action_expert is None:
            raise ValueError("Stage 2 requires `action_expert`.")
        
        if self.motion_source == "predicted" or (self.use_predicted_motion_eval and not self.training):
            if self.motion_director is None:
                raise ValueError("motion_source='predicted' requires `motion_director`.")
            motion_outputs = self.motion_director(
                batch_data,
                encoder_outputs=outputs["encoder"],
                **kwargs,
            )
            pixel_motion = self._extract_pixel_motion(motion_outputs)

        elif self.motion_source == "estimator" or not self.use_predicted_motion_eval:
            if self.motion_director is None or not hasattr(self.motion_director, "get_target_flow_rgb"):
                raise ValueError(
                    "motion_source='estimator' requires motion_director with get_estimated_pixel_motion()."
                )
            pixel_motion = self.motion_director.get_target_flow_rgb(batch_data)
            motion_outputs = {
                "estimated_flow_rgb": pixel_motion,
                "predicted_flow_rgb": pixel_motion,  # For visualization consistency
            }
        else:
            raise ValueError(f"Unsupported motion_source: {self.motion_source}")

        pixel_motion_feat = self.encoder.encode_image(pixel_motion, motion=True)
        

        action_encoder_outputs = dict(outputs["encoder"])
        # action_encoder_outputs["pixel_motion"] = pixel_motion
        # action_encoder_outputs["pixel_motion_feat"] = pixel_motion_feat

        outputs["motion"] = motion_outputs
        outputs["action"] = self.action_expert(
            batch_data,
            encoder_outputs=action_encoder_outputs,
            pixel_motion=pixel_motion,
            motion_outputs=motion_outputs,
            **kwargs,
        )

        outputs["total_loss"] = outputs["action"].get("total_loss", 0.0)
        return outputs

    @staticmethod
    def _extract_pixel_motion(motion_outputs: Optional[Dict[str, Any]]) -> Any:
        if not isinstance(motion_outputs, dict):
            return None
        for key in ("predicted_flow_rgb", "estimated_flow_rgb", "target_flow_rgb", "flow", "pixel_motion"):
            if key in motion_outputs:
                return motion_outputs[key]
        return None

    @staticmethod
    def _count_parameters(module: Optional[nn.Module]) -> Dict[str, int]:
        if module is None:
            return {"total": 0, "trainable": 0}
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def module_parameter_summary(self) -> Dict[str, Dict[str, int]]:
        rows = {
            "encoder": self._count_parameters(self.encoder),
            "motion_director": self._count_parameters(self.motion_director),
            "action_expert": self._count_parameters(self.action_expert),
        }
        rows["total"] = {
            "total": rows["encoder"]["total"] + rows["motion_director"]["total"] + rows["action_expert"]["total"],
            "trainable": rows["encoder"]["trainable"]
            + rows["motion_director"]["trainable"]
            + rows["action_expert"]["trainable"],
        }
        return rows

    def module_parameter_table(self) -> str:
        rows = self.module_parameter_summary()
        header = f"{'module':<18} {'total_params':>15} {'trainable_params':>18}"
        sep = "-" * len(header)

        lines = [header, sep]
        for name in ("encoder", "motion_director", "action_expert", "total"):
            lines.append(
                f"{name:<18} {format_size(rows[name]['total']):>15} {format_size(rows[name]['trainable']):>18}"
            )
        return "\n".join(lines)

    @staticmethod
    def _language_to_list(language: Any, batch_size: int) -> list[str]:
        if language is None:
            return [""] * batch_size
        if isinstance(language, str):
            return [language] * batch_size
        if isinstance(language, (list, tuple)):
            vals = [str(x) for x in language]
            if len(vals) >= batch_size:
                return vals[:batch_size]
            return vals + [""] * (batch_size - len(vals))
        return [str(language)] * batch_size

    def visualize(
        self,
        batch_data: Dict[str, Any],
        outputs: Dict[str, Any],
        max_items: int = 8,
        **kwargs: Any,
    ) -> list[Any]:
        if self.motion_director is None or not hasattr(self.motion_director, "visualize"):
            return []

        motion_outputs = outputs if "predicted_flow_rgb" in outputs else outputs.get("motion", outputs)
        pil_images = self.motion_director.visualize(
            batch_data=batch_data,
            outputs=motion_outputs,
            max_items=max_items,
            **kwargs,
        )
        if not pil_images:
            return []

        try:
            import wandb
        except Exception as e:
            logger.warning("wandb is unavailable in DAWNArch.visualize: %s", e)
            return []

        languages = self._language_to_list(batch_data.get("language", batch_data.get("text")), len(pil_images))
        wandb_images = []
        for i, img in enumerate(pil_images):
            caption = languages[i] if i < len(languages) else ""
            wandb_images.append(wandb.Image(img, caption=caption))
        return wandb_images
    

    def reset(self):
        self.precalc_outputs = None 
    
    @torch.no_grad()
    def step(self, data, visualize=True):
        if self.precalc_outputs is None or self.precalc_outputs["action"]["logits"].shape[1] == 0:
            # logger.info("No precalculated actions available, generating new actions.")
            # Generate actions
            outputs = self.forward(data)
            self.precalc_outputs = outputs
        else:
            outputs = self.precalc_outputs

        if visualize:
            vis_image = self.motion_director.visualize(
                batch_data=data,
                outputs=outputs["motion"],
                predict_only=True,
            )[0]
            vis_image = np.array(vis_image)
        else:
            vis_image = None
        # Use precalculated actions if available

        this_action = self.precalc_outputs["action"]["logits"][0, 0].detach().cpu()
        self.precalc_outputs["action"]['logits'] = self.precalc_outputs["action"]['logits'][:, 1:]
        # logger.info(f"Using precalculated action: {this_action}")
        return {
            "action": this_action,
            "viz_flow": vis_image,
        }
    