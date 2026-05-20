from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch import nn


class BaseMotionDirector(nn.Module, ABC):
    """Base class for motion director modules."""

    def __init__(
        self,
        input_type: str = "rgb_static",
        support_types: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.input_type = input_type
        self.support_types = support_types or ["rgb_static", "rgb_gripper"]
        self._validate_input_type()

    def _validate_input_type(self) -> None:
        if self.input_type not in self.support_types:
            raise ValueError(
                f"Invalid input_type: {self.input_type}. Supported types: {self.support_types}."
            )

    def forward(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if self.training:
            return self.forward_train(batch_data, encoder_outputs=encoder_outputs, **kwargs)
        return self.forward_eval(batch_data, encoder_outputs=encoder_outputs, **kwargs)

    @abstractmethod
    def forward_train(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Training-time forward pass."""

    @abstractmethod
    def forward_eval(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Evaluation-time forward pass."""
