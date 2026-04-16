"""Unified router implementations.

This module is the source of truth for all router classes:
- DeepSeekRouter: Sigmoid + expert-bias router (DeepSeek V3 style)
- ExplorationTopKRouter: Softmax top-k with random exploration
- BranchRouter: Binary attention/MLP branch selection per token
- BranchRouterRecorder: Parameterless recorder for deterministic routing

Re-exports from src.models.router for DeepSeek/ExplorationTopK routers
(which are tightly coupled to the Qwen3MoeTopKRouter base class).
BranchRouter and BranchRouterRecorder are defined here as the source of truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.router import (
    DeepSeekRouter,
    ExplorationTopKRouter,
    group_limited_topk,
    sample_router_exploration_mask,
    apply_router_exploration,
    collect_router_topk_indices,
    checkpoint_recompute_context,
    is_checkpoint_recompute,
)


def _straight_through_ones(probs: torch.Tensor) -> torch.Tensor:
    """Forward: returns ones. Backward: gradient flows through probs."""
    return probs + (1.0 - probs).detach()


class BranchRouter(nn.Module):
    """Binary router: ATTN (0) or MLP (1) per token.

    Hard routing: each token picks one branch via argmax.
    The selected branch output is scaled by its softmax probability
    for gradient flow (same pattern as MoE expert routing).

    NOTE: BranchRouter has NO load-balancing loss by design.
    """

    def __init__(self, hidden_size: int, exploration_rate: float = 0.0,
                 scale_by_routing_weight: bool = True):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)
        self.exploration_rate = exploration_rate
        self.scale_by_routing_weight = scale_by_routing_weight
        self.last_probs = None
        self.last_selected_experts = None

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate(hidden_states.float())
        probs = F.softmax(logits, dim=-1).to(hidden_states.dtype)
        self.last_probs = probs
        choice_scores = probs.float()
        if self.training and self.exploration_rate > 0.0:
            explore_mask = torch.rand(choice_scores.shape[:-1], device=choice_scores.device) < self.exploration_rate
            if explore_mask.any():
                choice_scores = choice_scores.clone()
                choice_scores[explore_mask] = torch.rand_like(choice_scores[explore_mask])
        choice = choice_scores.argmax(dim=-1)
        self.last_selected_experts = choice.unsqueeze(-1).detach()
        attn_mask = (choice == 0).unsqueeze(-1)
        mlp_mask = (choice == 1).unsqueeze(-1)
        if self.scale_by_routing_weight:
            w_attn = probs[..., 0:1] * attn_mask
            w_mlp = probs[..., 1:2] * mlp_mask
        else:
            w_attn = _straight_through_ones(probs[..., 0:1]) * attn_mask
            w_mlp = _straight_through_ones(probs[..., 1:2]) * mlp_mask
        return w_attn, w_mlp, attn_mask, mlp_mask


class BranchRouterRecorder(nn.Module):
    """Parameterless recorder used when branch routing is deterministic."""

    def __init__(self):
        super().__init__()
        self.last_probs = None
        self.last_selected_experts = None


__all__ = [
    "DeepSeekRouter",
    "ExplorationTopKRouter",
    "BranchRouter",
    "BranchRouterRecorder",
    "group_limited_topk",
    "sample_router_exploration_mask",
    "apply_router_exploration",
    "collect_router_topk_indices",
    "checkpoint_recompute_context",
    "is_checkpoint_recompute",
]
