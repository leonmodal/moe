"""Base model configuration for all model types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseModelConfig:
    """Configuration shared by all model types (dense, MoE variants)."""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    # QK normalization (Qwen3-style)
    qk_norm: bool = True
    # Normalization type: "learned" (standard RMSNorm) or "functional" (F.rms_norm, parameterless)
    norm_type: str = "learned"
    # Output head options
    use_fp8_lm_head: bool = False
    logit_softcapping: float = 0.0  # 0 = disabled, >0 = sigmoid softcapping scale
    # Attention mode: "sdpa" or "flex" (FlexAttention with doc masking + sliding window)
    attn_implementation: str = "sdpa"
    sliding_window: int = 0  # 0 = disabled
