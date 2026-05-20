from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseActionExpert


class DiffusionPolicyActionExpert(BaseActionExpert):
    """
    Minimal scaffold for an action expert implementation.

    Replace the `NotImplementedError` blocks with the concrete diffusion-policy logic.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward_train(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pixel_motion = kwargs.get("pixel_motion")
        if pixel_motion is None and encoder_outputs is not None:
            pixel_motion = encoder_outputs.get("pixel_motion")

        return {
            "pixel_motion": pixel_motion,
            "image_feat": None if encoder_outputs is None else encoder_outputs.get("image_feat"),
            "text_feat": None if encoder_outputs is None else encoder_outputs.get("text_feat"),
        }

    def forward_eval(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return self.forward_train(batch_data, encoder_outputs=encoder_outputs, **kwargs)
