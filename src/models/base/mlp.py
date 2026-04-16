"""Standalone SwiGLU MLP based on Qwen3 architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """SwiGLU MLP (gate + up projection, then down projection)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
