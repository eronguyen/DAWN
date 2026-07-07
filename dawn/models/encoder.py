#TODO:  Create a torch module class that encode input image/text from CLIP model
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoImageProcessor, AutoModel, AutoTokenizer

import logging

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(
        self,
        text_pretrained: str = "google-t5/t5-base", 
        image_pretrained: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m", #"openai/clip-vit-large-patch14", #"facebook/dinov2-base",#"facebook/dinov3-convnext-small-pretrain-lvd1689m",
        max_length: int = 77,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        if "clip" in text_pretrained:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_pretrained)
            self.text_encoder = CLIPTextModel.from_pretrained(text_pretrained)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(text_pretrained)
            self.text_encoder = AutoModel.from_pretrained(text_pretrained)

        self.processor = AutoImageProcessor.from_pretrained(image_pretrained)
        
        if "clip" in image_pretrained:
            self.image_encoder = CLIPVisionModel.from_pretrained(image_pretrained)
        else:
            self.image_encoder = AutoModel.from_pretrained(image_pretrained)

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.image_encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def encode_text(
        self,
        text: Union[Sequence[str], Tensor],
        max_length: Optional[int] = None,
    ) -> Tensor:

        # logger.debug(f"Encoding text: {text}")
        # print(text)
        max_length = max_length or self.max_length
        text_condition = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        text_condition = self.text_encoder(**text_condition, return_dict=False)
        feat = text_condition[0]
        return feat

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        with torch.inference_mode():
            pixel_values = (pixel_values * 255.0).to(torch.uint8)
            inp = self.processor(images=pixel_values, return_tensors="pt").to(self.device)
            outputs = self.image_encoder(**inp)
            return outputs.last_hidden_state

    @torch.inference_mode()
    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Union[Sequence[str], Tensor]] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if image is None and text is None:
            raise ValueError("At least one of `image` or `text` must be provided.")

        image_feat = self.encode_image(image) if image is not None else None
        text_feat = self.encode_text(text) if text is not None else None
        return image_feat, text_feat
