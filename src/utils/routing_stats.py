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
  routing/layer_NN_entropy          — normalised entropy of mean routing probs [0,1]
                                      1.0 = uniform, 0.0 = collapsed
  routing/layer_NN_num_active_experts — count of experts that received ≥1 token
  routing/layer_NN_top_expert       — index of most-loaded expert this batch
  routing/layer_NN_router_margin_mean — mean gap between k-th and (k+1)-th prob
                                        large = confident routing, ~0 = random
  routing/layer_NN_router_margin_min  — min margin (worst-case token)

  _hist/layer_NN_expert_load_frac   — per-expert fraction of total routing slots
                                      (list of E floats, log as wandb.Histogram)
                                      ideal value = top_k / E for every expert

Global MoE only:
  routing/cross_layer_sim_mean      — mean cosine-sim between per-layer load vectors
                                      high → layers select same experts (bad specialisation)
  routing/cross_layer_sim_min       — minimum pairwise similarity
  routing/cross_layer_sim_max       — maximum pairwise similarity

  _hist/global_expert_layer_count      — per-expert: how many layers use it (0=dead, L=everywhere)

  routing/global_pool_load_imbalance   — max expert total tokens / ideal across all layers
  routing/global_pool_load_cv          — CV of total token counts across the shared expert pool
  routing/global_pool_num_active       — count of experts used by at least one layer
  routing/global_pool_top_expert       — index of most-loaded expert across all layers
  routing/global_pool_entropy          — normalised entropy of aggregate load distribution

  _hist/global_pool_expert_load_frac   — per-expert fraction of total routing slots across all layers

Cumulative tables (via accumulate_expert_counts + expert_counts_to_tables):
  _table/layer_NN_expert_tokens     — list of (expert_index, token_count) per layer,
                                      accumulated over routing_log_every steps, all-reduced across ranks
  _table/global_pool_expert_tokens  — list of (expert_index, token_count) summed across all layers
"""
import math
from typing import Sequence

import torch


def accumulate_expert_counts(
    router_logits: Sequence[torch.Tensor],
    num_experts_per_tok: int,
    accumulator: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Add per-layer expert token counts from one batch into an accumulator.

    Args:
        router_logits: tuple of [T, E] prob tensors, one per layer.
        num_experts_per_tok: top-k value.
        accumulator: existing {layer_idx: [E] counts} dict, or None to create new.

    Returns:
        Updated accumulator dict.
    """
    if accumulator is None:
        accumulator = {}
    for layer_idx, probs in enumerate(router_logits):
        probs = probs.detach().float()
        T, E = probs.shape
        _, top_k_idx = torch.topk(probs, num_experts_per_tok, dim=-1)
        counts = torch.bincount(top_k_idx.reshape(-1), minlength=E).float()
        if layer_idx in accumulator:
            accumulator[layer_idx] += counts
        else:
            accumulator[layer_idx] = counts
    return accumulator


def expert_counts_to_tables(
    accumulator: dict[int, torch.Tensor],
    is_global: bool = False,
) -> dict:
    """
    Convert accumulated expert counts into _table/ entries for logging.

    Args:
        accumulator: {layer_idx: [E] token counts} accumulated over multiple steps/ranks.
        is_global: whether to also emit a global pool table.

    Returns:
        Dict with _table/ keys.
    """
    stats = {}
    for layer_idx in sorted(accumulator):
        counts = accumulator[layer_idx]
        E = counts.shape[0]
        counts_list = counts.cpu().tolist()
        tag = f"routing/layer_{layer_idx:02d}"
        stats[f"_table/{tag}_expert_tokens"] = list(zip(range(E), counts_list))

    if is_global and len(accumulator) > 1:
        all_counts = torch.stack([accumulator[i] for i in sorted(accumulator)])
        pool_counts = all_counts.sum(dim=0)
        E = pool_counts.shape[0]
        pool_list = pool_counts.cpu().tolist()
        stats["_table/global_pool_expert_tokens"] = list(zip(range(E), pool_list))

    return stats


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

        active = (counts > 0).sum().item()
        stats[f"{tag}_num_active_experts"] = int(active)

        # Index of most-loaded expert
        stats[f"{tag}_top_expert"] = int(counts.argmax().item())

        # Per-expert load fraction (each value = fraction of total routing slots)
        # ideal = 1/E for every expert; logged as histogram
        load_frac = (counts / total_slots).cpu().tolist()   # list of E floats
        stats[f"_hist/{tag}_expert_load_frac"] = load_frac

        # ── Entropy of expert assignment distribution ─────────────────────
        # Use actual routing counts (proper distribution summing to 1)
        load_dist = counts / (counts.sum() + 1e-10)     # [E], sums to 1
        entropy = -(load_dist * (load_dist + 1e-10).log()).sum().item()
        stats[f"{tag}_entropy"] = entropy / math.log(E) if E > 1 else 0.0

        # ── Router confidence: gap between k-th and (k+1)-th prob ─────
        # Large margin → router is confident in its choices
        # Near zero → routing is essentially random
        if E > num_experts_per_tok:
            sorted_probs, _ = probs.sort(dim=-1, descending=True)  # [T, E]
            margin = sorted_probs[:, num_experts_per_tok - 1] - sorted_probs[:, num_experts_per_tok]  # [T]
            stats[f"{tag}_router_margin_mean"] = margin.mean().item()
            stats[f"{tag}_router_margin_min"]  = margin.min().item()

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

        # ── Per-expert layer usage (raw: got ≥1 token from that layer) ──
        E = mat.shape[1]
        used_mask = mat > 0                                    # [L, E] bool
        layer_count = used_mask.float().sum(dim=0)             # [E] how many layers use each expert

        # Histogram: for each expert, how many layers use it (0 = dead, L = everywhere)
        stats["_hist/global_expert_layer_count"] = layer_count.cpu().tolist()

        # ── Pool-level aggregate stats (sum across all layers) ────────────
        pool_counts = mat.sum(dim=0)                           # [E] total tokens per expert
        pool_total  = pool_counts.sum().item()                 # total routing slots across all layers
        pool_ideal  = pool_total / E

        stats["routing/global_pool_load_imbalance"] = (pool_counts.max() / pool_ideal).item()
        stats["routing/global_pool_load_cv"] = (pool_counts.std() / pool_counts.mean()).item() if pool_counts.mean() > 0 else 0.0

        pool_active = (pool_counts > 0).sum().item()
        stats["routing/global_pool_num_active"] = int(pool_active)
        stats["routing/global_pool_top_expert"] = int(pool_counts.argmax().item())

        pool_frac = (pool_counts / pool_total).cpu()
        stats["_hist/global_pool_expert_load_frac"] = pool_frac.tolist()

        pool_entropy = -(pool_frac * (pool_frac + 1e-10).log()).sum().item()
        stats["routing/global_pool_entropy"] = pool_entropy / math.log(E) if E > 1 else 0.0

    return stats
