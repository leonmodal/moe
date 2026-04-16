"""Routing helper functions extracted from speedrun_moe_gpt.py.

These provide efficient top-1 routing dispatch with Triton grouped GEMM:
- _make_router: Create a router module for a given projection type
- _route_top1: Select top-1 expert per token using a router
- _grouped_project: Apply grouped projection across experts
- _sparse_attn_project: Sparse attention projection with per-head routing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_router(
    hidden_size: int,
    num_experts: int,
    exploration_rate: float = 0.0,
) -> nn.Linear:
    """Create a simple linear router for top-1 expert selection."""
    router = nn.Linear(hidden_size, num_experts, bias=False)
    return router


def route_top1(
    hidden_states: torch.Tensor,
    router: nn.Module,
    exploration_rate: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform top-1 routing: each token selects exactly one expert.

    Args:
        hidden_states: (batch*seq, hidden_size)
        router: Linear module producing (batch*seq, num_experts) logits
        exploration_rate: Probability of random expert selection during training
        training: Whether in training mode

    Returns:
        (routing_weights, expert_indices, router_logits)
        - routing_weights: (batch*seq, 1) weight for the selected expert
        - expert_indices: (batch*seq, 1) index of selected expert
        - router_logits: (batch*seq, num_experts) raw logits
    """
    logits = router(hidden_states.float())
    scores = torch.sigmoid(logits)

    if training and exploration_rate > 0.0:
        mask = torch.rand(scores.shape[0], device=scores.device) < exploration_rate
        selection_scores = scores.clone()
        if mask.any():
            selection_scores[mask] = torch.rand_like(selection_scores[mask])
    else:
        selection_scores = scores

    expert_indices = selection_scores.argmax(dim=-1, keepdim=True)  # (T, 1)
    routing_weights = scores.gather(1, expert_indices)  # (T, 1)

    return routing_weights, expert_indices, logits


def grouped_project(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    weight: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Apply grouped projection: each token is projected by its assigned expert's weight.

    Args:
        hidden_states: (T, in_dim)
        expert_indices: (T, 1) expert assignments
        weight: (num_experts, out_dim, in_dim) expert weight matrices
        num_experts: Number of experts

    Returns:
        (T, out_dim) projected hidden states
    """
    T = hidden_states.shape[0]
    out_dim = weight.shape[1]

    # Group tokens by expert
    flat_idx = expert_indices.squeeze(-1)
    output = torch.zeros(T, out_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    for e in range(num_experts):
        mask = flat_idx == e
        if mask.any():
            output[mask] = F.linear(hidden_states[mask], weight[e])

    return output
