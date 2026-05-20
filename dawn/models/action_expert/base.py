from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch import nn


class BaseActionExpert(nn.Module, ABC):
    """Base class for action expert modules."""

    def forward(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if self.training:
            return self.forward_train(batch_data, encoder_outputs=encoder_outputs, **kwargs)
        return self.forward_eval(batch_data, encoder_outputs=encoder_outputs, **kwargs)

    # @abstractmethod
    def forward_train(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Training-time forward pass."""

    # @abstractmethod
    def forward_eval(
        self,
        batch_data: Dict[str, Any],
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Evaluation-time forward pass."""
