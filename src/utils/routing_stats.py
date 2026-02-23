"""
Routing statistics computed from output.router_logits.

output.router_logits is a tuple of [T, E] softmax-probability tensors,
one per MoE layer, already produced by the forward pass — zero extra overhead.

Metrics
-------
Per-layer (both standard and global):
  routing/layer_NN_load_imbalance   — max_expert_tokens / ideal_expert_tokens
                                      1.0 = perfect, >1 = overloaded experts
  routing/layer_NN_load_cv          — std(token_counts) / mean(token_counts)
                                      0.0 = perfectly uniform
  routing/layer_NN_utilization      — fraction of experts that received ≥1 token
  routing/layer_NN_entropy          — normalised entropy of mean routing probs [0,1]
                                      1.0 = uniform, 0.0 = collapsed
  routing/layer_NN_top_expert       — index of most-loaded expert this batch

  _hist/layer_NN_expert_load_frac   — per-expert fraction of total routing slots
                                      (list of E floats, log as wandb.Histogram)
                                      ideal value = top_k / E for every expert

Global MoE only:
  routing/cross_layer_sim_mean      — mean cosine-sim between per-layer load vectors
                                      high → layers select same experts (bad specialisation)
  routing/cross_layer_sim_min       — minimum pairwise similarity
  routing/expert_depth_coverage_mean — avg # of layers each expert is active in
  routing/expert_depth_coverage_max
  _hist/expert_depth_coverage       — histogram across all experts
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
        num_experts_per_tok: top-k value (K).
        is_global:           whether to compute cross-layer metrics.

    Returns:
        Flat dict of scalar metrics + '_hist/*' lists for wandb.Histogram.
    """
    stats: dict = {}
    load_vecs: list[torch.Tensor] = []   # [E] integer counts per layer, for cross-layer sim

    for layer_idx, probs in enumerate(router_logits):
        probs = probs.detach().float()          # [T, E]
        T, E  = probs.shape
        tag   = f"routing/layer_{layer_idx:02d}"

        # ── Actual token counts per expert ───────────────────────────────
        # top-k indices for every token → tally how many tokens hit each expert
        _, top_k_idx = torch.topk(probs, num_experts_per_tok, dim=-1)  # [T, K]
        counts = torch.bincount(top_k_idx.reshape(-1), minlength=E).float()  # [E]

        total_slots = float(T * num_experts_per_tok)    # = T*K routing assignments
        ideal_load  = total_slots / E                    # tokens per expert if uniform

        # Load imbalance: max expert load vs ideal  (1.0 = perfect)
        stats[f"{tag}_load_imbalance"] = (counts.max() / ideal_load).item()

        # Coefficient of variation of loads (0.0 = perfect)
        stats[f"{tag}_load_cv"] = (counts.std() / counts.mean()).item() if counts.mean() > 0 else 0.0

        # Fraction of experts that received ≥1 token
        stats[f"{tag}_utilization"] = (counts > 0).float().mean().item()

        # Index of most-loaded expert
        stats[f"{tag}_top_expert"] = int(counts.argmax().item())

        # Per-expert load fraction (each value = fraction of total routing slots)
        # ideal = 1/E for every expert; logged as histogram
        load_frac = (counts / total_slots).cpu().tolist()   # list of E floats
        stats[f"_hist/{tag}_expert_load_frac"] = load_frac

        # ── Entropy of mean routing probabilities ────────────────────────
        mean_p  = probs.mean(dim=0)                     # [E]
        entropy = -(mean_p * (mean_p + 1e-10).log()).sum().item()
        stats[f"{tag}_entropy"] = entropy / math.log(E)

        load_vecs.append(counts)

    # ── Global MoE: cross-layer metrics ─────────────────────────────────────
    if is_global and len(load_vecs) > 1:
        mat   = torch.stack(load_vecs).float()          # [L, E]
        norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        normed = mat / norms
        sim   = normed @ normed.T                       # [L, L]

        L    = sim.shape[0]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        off  = sim[mask]

        stats["routing/cross_layer_sim_mean"] = off.mean().item()
        stats["routing/cross_layer_sim_min"]  = off.min().item()
        stats["routing/cross_layer_sim_max"]  = off.max().item()

        # Per-expert: how many layers activated it above uniform threshold
        depth_coverage = (mat > (mat.sum(dim=1, keepdim=True) / E)).float().sum(dim=0)  # [E]
        stats["routing/expert_depth_coverage_mean"] = depth_coverage.mean().item()
        stats["routing/expert_depth_coverage_max"]  = depth_coverage.max().item()
        stats["_hist/expert_depth_coverage"]        = depth_coverage.cpu().tolist()

    return stats
