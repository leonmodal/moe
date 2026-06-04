"""Routing helper functions extracted from speedrun_moe_gpt.py.

These provide efficient top-1 routing dispatch with Triton grouped GEMM:
- make_router: Create a router module for a given projection type
- route_top1: Select top-1 expert per token using a router
- grouped_project: Apply grouped projection across experts
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


def make_top1_router(input_dim: int, num_experts: int, config=None):
    """Create a top-1 router for per-head-slot expert selection.

    Uses DeepSeek or Exploration router depending on config.use_deepseek_routing.
    Kaiming-initialized weights for diverse expert preferences from the start.
    """
    import math
    from types import SimpleNamespace
    from src.models.router import DeepSeekRouter, ExplorationTopKRouter

    norm_topk_prob = getattr(config, "norm_topk_prob", True) if config else True
    exploration_rate = getattr(config, "router_exploration_rate", 0.0) if config else 0.0
    use_deepseek = getattr(config, "use_deepseek_routing", False) if config else False

    base_cfg = SimpleNamespace(
        hidden_size=input_dim,
        num_local_experts=num_experts,
        num_experts=num_experts,
        num_experts_per_tok=1,
        norm_topk_prob=norm_topk_prob,
        router_exploration_rate=exploration_rate,
    )
    if use_deepseek:
        base_cfg.topk_scaling_factor = getattr(config, "topk_scaling_factor", None) if config else None
        base_cfg.num_groups = None
        base_cfg.group_topk = None
        router = DeepSeekRouter(base_cfg)
    else:
        router = ExplorationTopKRouter(base_cfg)
    nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))
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
