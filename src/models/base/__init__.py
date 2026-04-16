"""Bagel-style custom model components based on Qwen3 architecture.

All components are standalone — they do NOT import from `transformers` for
forward-pass logic. Only `PreTrainedModel` is used as a base class for
checkpoint compatibility (from_pretrained / save_pretrained).
"""

from .config import BaseModelConfig
from .normalization import RMSNorm, FunctionalRMSNorm
from .embeddings import RotaryEmbedding, apply_rotary_pos_emb
from .attention import Attention
from .mlp import MLP
from .output_head import LMHead

__all__ = [
    "BaseModelConfig",
    "RMSNorm",
    "FunctionalRMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "Attention",
    "MLP",
    "LMHead",
]
