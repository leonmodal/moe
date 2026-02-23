"""
Routing statistics computed from output.router_logits.

output.router_logits is a tuple of [T, E] softmax-probability tensors,
one per MoE layer, already produced by the forward pass — zero extra
overhead to compute these stats.

Metrics
-------
Per-layer (both standard and global):
  routing/layer_NN_entropy      — normalised entropy of mean routing probs [0,1]
                                  1.0 = perfectly uniform, 0.0 = single expert
  routing/layer_NN_utilization  — fraction of experts that received ≥1 token
  routing/layer_NN_top_expert   — index of most-used expert this batch

Global MoE only:
  routing/cross_layer_sim_mean  — mean cosine-similarity between per-layer
                                  utilization vectors; high → layers select
                                  similar experts (no depth specialisation)
  routing/cross_layer_sim_min   — minimum pairwise similarity
  routing/expert_NN_layers      — how many layers activated expert NN
                                  (logged as a histogram for all 2048 experts)
"""
import math
from typing import Sequence

import torch


def compute_routing_stats(
    router_logits: Sequence[torch.Tensor],   # tuple[Tensor[T, E]] per layer
    num_experts_per_tok: int,
    is_global: bool = False,
) -> dict:
    """
    Args:
        router_logits:       tuple of [T, E] softmax-prob tensors, one per layer.
                             Already on whatever device the model ran on.
        num_experts_per_tok: top-k value.
        is_global:           whether to compute cross-layer metrics.

    Returns:
        Flat dict of scalar metrics ready to pass to accelerator.log().
        Histogram data (for wandb) is included as lists under
        '_hist/<key>' keys and should be logged with wandb.Histogram.
    """
    stats: dict = {}
    mean_probs_per_layer: list[torch.Tensor] = []

    for layer_idx, probs in enumerate(router_logits):
        probs = probs.detach().float()          # [T, E]
        E = probs.shape[1]
        tag = f"routing/layer_{layer_idx:02d}"

        # ── mean routing probability vector ─────────────────────────
        mean_p = probs.mean(dim=0)              # [E]  sums to 1

        # Normalised entropy  (1 = uniform, 0 = collapsed)
        entropy = -(mean_p * (mean_p + 1e-10).log()).sum().item()
        stats[f"{tag}_entropy"] = entropy / math.log(E)

        # Fraction of experts that received ≥1 token this batch
        _, top_k_idx = torch.topk(probs, num_experts_per_tok, dim=-1)  # [T, K]
        counts = torch.bincount(top_k_idx.reshape(-1), minlength=E)    # [E]
        stats[f"{tag}_utilization"] = (counts > 0).float().mean().item()
        stats[f"{tag}_top_expert"]  = int(counts.argmax().item())

        mean_probs_per_layer.append(mean_p)

    # ── Global MoE: cross-layer similarity & per-expert depth coverage ──
    if is_global and len(mean_probs_per_layer) > 1:
        mat   = torch.stack(mean_probs_per_layer)      # [L, E]
        norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        normed = mat / norms                           # [L, E]
        sim   = normed @ normed.T                      # [L, L]

        L = sim.shape[0]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        off_diag = sim[mask]

        stats["routing/cross_layer_sim_mean"] = off_diag.mean().item()
        stats["routing/cross_layer_sim_min"]  = off_diag.min().item()
        stats["routing/cross_layer_sim_max"]  = off_diag.max().item()

        # Per-expert: how many layers activated it (has top-k probability)
        # Threshold: expert is "active" in a layer if it has top-k prob
        # We use mean_p to get a continuous "depth coverage" score
        depth_coverage = (mat > (1.0 / mat.shape[1])).float().sum(dim=0)  # [E]
        stats["routing/expert_depth_coverage_mean"] = depth_coverage.mean().item()
        stats["routing/expert_depth_coverage_max"]  = depth_coverage.max().item()
        # Store raw coverage as hist data for wandb (logged separately)
        stats["_hist/expert_depth_coverage"] = depth_coverage.cpu().tolist()

    return stats
