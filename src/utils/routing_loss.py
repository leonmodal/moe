"""Switch-style auxiliary load balancing loss for per-head routers."""

from __future__ import annotations

import torch
from torch import Tensor


def switch_load_balancing_loss(expert_ids: Tensor, probs: Tensor, num_experts: int) -> Tensor:
    """Switch Transformer aux loss: E * sum_i(f_i * P_i).

    expert_ids: (N,) selected expert per token
    probs: (N, E) full softmax probabilities
    num_experts: E

    Encourages uniform expert utilization.
    """
    N = expert_ids.shape[0]
    if N == 0:
        return expert_ids.new_tensor(0.0, dtype=torch.float32)
    # f_i = fraction of tokens routed to expert i
    f = torch.zeros(num_experts, device=expert_ids.device, dtype=torch.float32)
    f.scatter_add_(0, expert_ids.long(), torch.ones(N, device=expert_ids.device, dtype=torch.float32))
    f = f / N
    # P_i = mean routing probability for expert i
    P = probs.float().mean(dim=0)
    return num_experts * (f * P).sum()
