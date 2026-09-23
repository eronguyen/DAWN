"""Language goal encoder, ported from VPP (yjguo/dp-calvin) so its checkpoint's
`language_goal.clip_rn50.*` weights load directly.

Source: vpp-hiva/policy_models/module/clip_lang_encoder.py. Despite the
attribute name `clip_rn50`, VPP always instantiates this with
`model_name="ViT-B/32"` for the checkpoints we target; the name is just
inherited from an earlier RN50 variant and kept for checkpoint compatibility.
"""

from __future__ import annotations

import torch
from torch import nn

from dawn.models.modules.clip_openai import build_model, load_clip, tokenize


class LangClip(nn.Module):
    def __init__(self, freeze_backbone: bool = True, model_name: str = "ViT-B/32"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        model, _ = load_clip(model_name, device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)

    def forward(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)
