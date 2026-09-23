"""Vendored OpenAI CLIP (MIT License, https://github.com/openai/CLIP), ported
from VPP (yjguo/dp-calvin: `vpp-hiva/policy_models/module/clip.py` +
`utils/clip_tokenizer.py`) so its checkpoint's `language_goal.clip_rn50.*`
weights load directly. Kept as a near-verbatim copy for exact compatibility.
"""

from .clip import available_models, build_model, load_clip, tokenize  # noqa: F401
