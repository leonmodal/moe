"""Normalization layers: learned RMSNorm and parameterless FunctionalRMSNorm."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learned scale (Qwen3-style)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FunctionalRMSNorm(nn.Module):
    """Parameterless RMSNorm using F.rms_norm.

    Extracted from speedrun_mixture_of_everything.py. This variant has no learnable
    parameters — it just normalizes by root mean square.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, (self.hidden_size,), eps=self.eps)


def build_norm(hidden_size: int, eps: float = 1e-6, norm_type: str = "learned") -> nn.Module:
    """Factory to build normalization layer from config."""
    if norm_type == "functional":
        return FunctionalRMSNorm(hidden_size, eps)
    return RMSNorm(hidden_size, eps)
