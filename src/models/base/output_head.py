"""LM output head with optional FP8 matmul and sigmoid logit softcapping.

FP8 lm_head and sigmoid softcapping extracted from speedrun_gpt.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMHead(nn.Module):
    """Language model output head.

    Options:
    - use_fp8: Use FP8 matrix multiplication for the final projection (requires torch >= 2.4)
    - softcapping: Apply sigmoid softcapping to logits. 0.0 = disabled.
      When enabled, logits = scale * tanh(logits / scale).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        use_fp8: bool = False,
        softcapping: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        self.use_fp8 = use_fp8
        self.softcapping = softcapping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            logits = self._fp8_forward(hidden_states)
        else:
            logits = self.linear(hidden_states)

        if self.softcapping > 0.0:
            logits = self.softcapping * torch.tanh(logits / self.softcapping)

        return logits

    def _fp8_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP8 matrix multiply for the output projection."""
        try:
            return torch._scaled_mm(
                hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float8_e4m3fn),
                self.linear.weight.T.contiguous().to(torch.float8_e4m3fn),
                out_dtype=hidden_states.dtype,
                scale_a=torch.ones(1, device=hidden_states.device),
                scale_b=torch.ones(1, device=hidden_states.device),
            ).reshape(*hidden_states.shape[:-1], self.linear.weight.shape[0])
        except (RuntimeError, AttributeError):
            # Fall back to standard matmul if FP8 is not supported
            return self.linear(hidden_states)
