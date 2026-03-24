"""
Routing statistics computed from router probability tensors.

When actual selected experts are available (for example DeepSeek biased routing),
pass them via ``selected_experts`` so load metrics reflect the real assignments
instead of recomputing top-k from the unbiased router scores.
"""
import math
from typing import Sequence

import torch


def _normalize_selected_experts(selected: torch.Tensor | None) -> torch.Tensor | None:
    if selected is None:
        return None
    selected = selected.detach()
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    return selected


def _counts_from_router(
    probs: torch.Tensor,
    num_experts_per_tok: int,
    selected_experts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = probs.detach().float()
    selected_experts = _normalize_selected_experts(selected_experts)
    if selected_experts is None:
        _, selected_experts = torch.topk(probs, num_experts_per_tok, dim=-1)
    counts = torch.bincount(selected_experts.reshape(-1), minlength=probs.shape[1]).float()
    return probs, counts


def accumulate_expert_counts(
    router_logits: Sequence[torch.Tensor],
    num_experts_per_tok: int,
    accumulator: dict[int, torch.Tensor] | None = None,
    selected_experts: Sequence[torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """Add per-layer expert token counts from one batch into an accumulator."""
    if accumulator is None:
        accumulator = {}

    for layer_idx, probs in enumerate(router_logits):
        selected = None
        if selected_experts is not None and layer_idx < len(selected_experts):
            selected = selected_experts[layer_idx]
        probs, counts = _counts_from_router(probs, num_experts_per_tok, selected)
        if layer_idx in accumulator:
            accumulator[layer_idx] += counts.to(accumulator[layer_idx].device)
        else:
            accumulator[layer_idx] = counts
    return accumulator


def accumulate_router_margins(
    router_logits: Sequence[torch.Tensor],
    num_experts_per_tok: int,
    accumulator: dict[int, dict[str, torch.Tensor]] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Accumulate router top-k margin stats across batches."""
    if accumulator is None:
        accumulator = {}

    for layer_idx, probs in enumerate(router_logits):
        probs = probs.detach().float()
        if probs.shape[1] <= num_experts_per_tok:
            continue

        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        margin = sorted_probs[:, num_experts_per_tok - 1] - sorted_probs[:, num_experts_per_tok]

        entry = accumulator.get(layer_idx)
        if entry is None:
            entry = {
                "sum": torch.zeros((), device=probs.device, dtype=torch.float64),
                "count": torch.zeros((), device=probs.device, dtype=torch.float64),
                "min": torch.full((), float("inf"), device=probs.device, dtype=torch.float64),
            }
            accumulator[layer_idx] = entry

        entry["sum"] += margin.sum(dtype=torch.float64)
        entry["count"] += torch.tensor(margin.numel(), device=probs.device, dtype=torch.float64)
        entry["min"] = torch.minimum(entry["min"], margin.min().to(torch.float64))

    return accumulator


def router_margin_accumulator_to_stats(
    accumulator: dict[int, dict[str, torch.Tensor]] | None,
    prefix: str = "routing",
) -> dict:
    """Convert accumulated router margin stats into scalars."""
    if not accumulator:
        return {}

    stats = {}
    for layer_idx in sorted(accumulator):
        entry = accumulator[layer_idx]
        count = entry["count"].item()
        if count <= 0:
            continue
        tag = f"{prefix}/layer_{layer_idx:02d}"
        stats[f"{tag}_router_margin_mean"] = (entry["sum"] / entry["count"]).item()
        stats[f"{tag}_router_margin_min"] = entry["min"].item()
    return stats


def _per_layer_stats_from_counts(counts: torch.Tensor, layer_idx: int, prefix: str) -> dict:
    counts = counts.detach().float()
    total_slots = counts.sum().item()
    num_experts = counts.shape[0]
    ideal_load = total_slots / max(1, num_experts)
    tag = f"{prefix}/layer_{layer_idx:02d}"

    stats: dict[str, float | list[float] | int] = {}
    if ideal_load > 0:
        stats[f"{tag}_load_imbalance"] = (counts.max() / ideal_load).item()
    else:
        stats[f"{tag}_load_imbalance"] = 0.0

    stats[f"{tag}_load_cv"] = (counts.std() / counts.mean()).item() if counts.mean() > 0 else 0.0

    active = int((counts > 0).sum().item())
    stats[f"{tag}_num_active_experts"] = active
    stats[f"{tag}_utilization"] = active / max(1, num_experts)
    stats[f"{tag}_top_expert"] = int(counts.argmax().item())

    load_frac = (counts / max(total_slots, 1.0)).cpu().tolist()
    stats[f"_hist/{tag}_expert_load_frac"] = load_frac

    load_dist = counts / (counts.sum() + 1e-10)
    entropy = -(load_dist * (load_dist + 1e-10).log()).sum().item()
    stats[f"{tag}_entropy"] = entropy / math.log(num_experts) if num_experts > 1 else 0.0
    return stats


def compute_routing_stats_from_counts(
    accumulator: dict[int, torch.Tensor] | None,
    is_global: bool = False,
    prefix: str = "routing",
) -> dict:
    """Compute routing stats from accumulated selected-expert counts."""
    if not accumulator:
        return {}

    stats: dict[str, float | list[float] | int] = {}
    load_vecs: list[torch.Tensor] = []

    for layer_idx in sorted(accumulator):
        counts = accumulator[layer_idx]
        stats.update(_per_layer_stats_from_counts(counts, layer_idx, prefix))
        load_vecs.append(counts.detach().float())

    if is_global and len(load_vecs) > 1:
        mat = torch.stack(load_vecs).float()
        norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        normed = mat / norms
        sim = normed @ normed.T

        num_layers = sim.shape[0]
        mask = torch.triu(torch.ones(num_layers, num_layers, dtype=torch.bool), diagonal=1)
        off_diag = sim[mask]
        if off_diag.numel() > 0:
            stats[f"{prefix}/cross_layer_sim_mean"] = off_diag.mean().item()
            stats[f"{prefix}/cross_layer_sim_min"] = off_diag.min().item()
            stats[f"{prefix}/cross_layer_sim_max"] = off_diag.max().item()

        used_mask = mat > 0
        layer_count = used_mask.float().sum(dim=0)
        coverage = layer_count / max(1, num_layers)
        coverage_list = coverage.cpu().tolist()
        stats[f"{prefix}/expert_depth_coverage_mean"] = coverage.mean().item()
        stats[f"{prefix}/expert_depth_coverage_max"] = coverage.max().item()
        stats[f"_hist/{prefix}/expert_depth_coverage"] = coverage_list
        if prefix == "routing":
            stats["_hist/expert_depth_coverage"] = coverage_list

        pool_counts = mat.sum(dim=0)
        pool_total = pool_counts.sum().item()
        num_experts = pool_counts.shape[0]
        pool_ideal = pool_total / max(1, num_experts)

        if pool_ideal > 0:
            stats[f"{prefix}/global_pool_load_imbalance"] = (pool_counts.max() / pool_ideal).item()
        else:
            stats[f"{prefix}/global_pool_load_imbalance"] = 0.0
        stats[f"{prefix}/global_pool_load_cv"] = (
            (pool_counts.std() / pool_counts.mean()).item() if pool_counts.mean() > 0 else 0.0
        )
        stats[f"{prefix}/global_pool_num_active"] = int((pool_counts > 0).sum().item())
        stats[f"{prefix}/global_pool_top_expert"] = int(pool_counts.argmax().item())

        pool_frac = (pool_counts / max(pool_total, 1.0)).cpu()
        stats[f"_hist/{prefix}/global_pool_expert_load_frac"] = pool_frac.tolist()
        if prefix == "routing":
            stats["_hist/global_pool_expert_load_frac"] = pool_frac.tolist()
            stats["_hist/global_expert_layer_count"] = layer_count.cpu().tolist()

        pool_entropy = -(pool_frac * (pool_frac + 1e-10).log()).sum().item()
        stats[f"{prefix}/global_pool_entropy"] = pool_entropy / math.log(num_experts) if num_experts > 1 else 0.0

    return stats


def expert_counts_to_tables(
    accumulator: dict[int, torch.Tensor],
    is_global: bool = False,
    prefix: str = "routing",
) -> dict:
    """Convert accumulated expert counts into ``_table/`` entries for logging."""
    stats = {}
    for layer_idx in sorted(accumulator):
        counts = accumulator[layer_idx]
        num_experts = counts.shape[0]
        counts_list = counts.cpu().tolist()
        tag = f"{prefix}/layer_{layer_idx:02d}"
        stats[f"_table/{tag}_expert_tokens"] = list(zip(range(num_experts), counts_list))

    if is_global and len(accumulator) > 1:
        all_counts = torch.stack([accumulator[i] for i in sorted(accumulator)])
        pool_counts = all_counts.sum(dim=0)
        num_experts = pool_counts.shape[0]
        stats[f"_table/{prefix}/global_pool_expert_tokens"] = list(
            zip(range(num_experts), pool_counts.cpu().tolist())
        )

    return stats


def compute_routing_stats(
    router_logits: Sequence[torch.Tensor],
    num_experts_per_tok: int,
    is_global: bool = False,
    selected_experts: Sequence[torch.Tensor] | None = None,
    prefix: str = "routing",
) -> dict:
    """Compute scalar and histogram routing metrics.

    Args:
        router_logits: tuple of ``[T, E]`` probability tensors, one per layer.
        num_experts_per_tok: top-k value.
        is_global: whether to compute shared-pool cross-layer metrics.
        selected_experts: optional actual assignments per layer. If provided,
            load metrics use these assignments instead of recomputed top-k.
        prefix: metric namespace, e.g. ``routing`` or ``routing/attention/q``.
    """
    stats: dict[str, float | list[float] | int] = {}
    load_vecs: list[torch.Tensor] = []

    for layer_idx, probs in enumerate(router_logits):
        selected = None
        if selected_experts is not None and layer_idx < len(selected_experts):
            selected = selected_experts[layer_idx]
        probs, counts = _counts_from_router(probs, num_experts_per_tok, selected)

        total_slots = counts.sum().item()
        num_experts = probs.shape[1]
        ideal_load = total_slots / max(1, num_experts)
        tag = f"{prefix}/layer_{layer_idx:02d}"

        if ideal_load > 0:
            stats[f"{tag}_load_imbalance"] = (counts.max() / ideal_load).item()
        else:
            stats[f"{tag}_load_imbalance"] = 0.0

        stats[f"{tag}_load_cv"] = (counts.std() / counts.mean()).item() if counts.mean() > 0 else 0.0

        active = int((counts > 0).sum().item())
        stats[f"{tag}_num_active_experts"] = active
        stats[f"{tag}_utilization"] = active / max(1, num_experts)
        stats[f"{tag}_top_expert"] = int(counts.argmax().item())

        load_frac = (counts / max(total_slots, 1.0)).cpu().tolist()
        stats[f"_hist/{tag}_expert_load_frac"] = load_frac

        load_dist = counts / (counts.sum() + 1e-10)
        entropy = -(load_dist * (load_dist + 1e-10).log()).sum().item()
        stats[f"{tag}_entropy"] = entropy / math.log(num_experts) if num_experts > 1 else 0.0

        if num_experts > num_experts_per_tok:
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            margin = sorted_probs[:, num_experts_per_tok - 1] - sorted_probs[:, num_experts_per_tok]
            stats[f"{tag}_router_margin_mean"] = margin.mean().item()
            stats[f"{tag}_router_margin_min"] = margin.min().item()

        load_vecs.append(counts)

    if is_global and len(load_vecs) > 1:
        mat = torch.stack(load_vecs).float()
        norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        normed = mat / norms
        sim = normed @ normed.T

        num_layers = sim.shape[0]
        mask = torch.triu(torch.ones(num_layers, num_layers, dtype=torch.bool), diagonal=1)
        off_diag = sim[mask]
        if off_diag.numel() > 0:
            stats[f"{prefix}/cross_layer_sim_mean"] = off_diag.mean().item()
            stats[f"{prefix}/cross_layer_sim_min"] = off_diag.min().item()
            stats[f"{prefix}/cross_layer_sim_max"] = off_diag.max().item()

        used_mask = mat > 0
        layer_count = used_mask.float().sum(dim=0)
        coverage = layer_count / max(1, num_layers)
        coverage_list = coverage.cpu().tolist()
        stats[f"{prefix}/expert_depth_coverage_mean"] = coverage.mean().item()
        stats[f"{prefix}/expert_depth_coverage_max"] = coverage.max().item()
        stats[f"_hist/{prefix}/expert_depth_coverage"] = coverage_list
        if prefix == "routing":
            stats["_hist/expert_depth_coverage"] = coverage_list

        pool_counts = mat.sum(dim=0)
        pool_total = pool_counts.sum().item()
        num_experts = pool_counts.shape[0]
        pool_ideal = pool_total / max(1, num_experts)

        if pool_ideal > 0:
            stats[f"{prefix}/global_pool_load_imbalance"] = (pool_counts.max() / pool_ideal).item()
        else:
            stats[f"{prefix}/global_pool_load_imbalance"] = 0.0
        stats[f"{prefix}/global_pool_load_cv"] = (
            (pool_counts.std() / pool_counts.mean()).item() if pool_counts.mean() > 0 else 0.0
        )
        stats[f"{prefix}/global_pool_num_active"] = int((pool_counts > 0).sum().item())
        stats[f"{prefix}/global_pool_top_expert"] = int(pool_counts.argmax().item())

        pool_frac = (pool_counts / max(pool_total, 1.0)).cpu()
        stats[f"_hist/{prefix}/global_pool_expert_load_frac"] = pool_frac.tolist()
        if prefix == "routing":
            stats["_hist/global_pool_expert_load_frac"] = pool_frac.tolist()
            stats["_hist/global_expert_layer_count"] = layer_count.cpu().tolist()

        pool_entropy = -(pool_frac * (pool_frac + 1e-10).log()).sum().item()
        stats[f"{prefix}/global_pool_entropy"] = pool_entropy / math.log(num_experts) if num_experts > 1 else 0.0

    return stats
