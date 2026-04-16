"""Rotary Position Embeddings (RoPE) — standalone implementation based on Qwen3."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings with configurable theta."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1_000_000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Input tensor (for device/dtype reference).
            position_ids: (batch, seq_len) position indices.

        Returns:
            (cos, sin) each of shape (batch, seq_len, dim).
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_kv_heads, seq_len, head_dim)
        cos: (batch, seq_len, head_dim) or broadcastable
        sin: (batch, seq_len, head_dim) or broadcastable

    Returns:
        (q_embed, k_embed) with rotary embeddings applied.
    """
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
