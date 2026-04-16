"""Routing stats collection and expert bias updates during training."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .distributed import is_distributed, unwrap_model


def update_expert_biases(
    model,
    *,
    bias_rate: float,
    distributed: bool = False,
    per_proj_rates: dict[str, float] | None = None,
) -> None:
    """Update expert biases using DeepSeek V3-style load balancing.

    For models with global load balancing (SpeedrunMoEGPT-style), delegates
    to model.update_global_bias(). For standard models with per-router biases,
    updates each DeepSeekRouter directly.
    """
    raw_model = unwrap_model(model)

    # Global bias update path (MoE-Everything with global load balancing)
    if hasattr(raw_model, 'update_global_bias') and getattr(raw_model, 'global_load_balancing', False):
        if per_proj_rates is None:
            per_proj_rates = {}
        defaults = {
            "q": bias_rate, "k": bias_rate, "v": bias_rate,
            "o": bias_rate, "mlp": bias_rate, "branch": bias_rate,
        }
        defaults.update(per_proj_rates)
        raw_model.update_global_bias(bias_rate, bias_rates=defaults)
        return

    # Per-router bias update path
    from src.models.router import DeepSeekRouter

    if hasattr(raw_model, 'get_all_routers'):
        for router in raw_model.get_all_routers():
            if isinstance(router, DeepSeekRouter):
                _update_single_router_bias(router, bias_rate, distributed)

    # Branch router biases
    if hasattr(raw_model, 'branch_routers'):
        for br in raw_model.branch_routers:
            if hasattr(br, 'branch_bias') and hasattr(br, 'local_counts'):
                _update_branch_router_bias(br, bias_rate, distributed)


def _update_single_router_bias(router, bias_rate: float, distributed: bool) -> None:
    """Update expert bias for a single DeepSeek router."""
    with torch.no_grad():
        counts = router.local_tokens_per_expert
        if distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if counts.sum() > 0:
            total = counts.sum()
            loads = counts / total
            expected = 1.0 / counts.shape[0]
            s = torch.sign(loads - expected)
            router.expert_bias -= (s - s.mean()) * bias_rate
            router.expert_bias.clamp_(-16.0, 16.0)
        router.local_tokens_per_expert.zero_()


def _update_branch_router_bias(br, bias_rate: float, distributed: bool) -> None:
    """Update bias for a branch router."""
    with torch.no_grad():
        if distributed:
            dist.all_reduce(br.local_counts, op=dist.ReduceOp.SUM)
        if br.local_counts.sum() > 0:
            total = br.local_counts.sum()
            loads = br.local_counts / total
            expected = 1.0 / br.local_counts.shape[0]
            s = torch.sign(loads - expected)
            br.branch_bias -= (s - s.mean()) * bias_rate
            br.branch_bias.clamp_(-16.0, 16.0)
        br.local_counts.zero_()


def get_bias_rate(
    model,
    step: int,
    base_rate: float,
    warmup_start: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """Get the current bias update rate, with optional linear warmup."""
    raw_model = unwrap_model(model)
    if hasattr(raw_model, 'get_bias_rate'):
        return raw_model.get_bias_rate(step, base_rate, warmup_start, warmup_steps)
    if warmup_steps > 0 and step < warmup_steps:
        frac = step / warmup_steps
        return warmup_start + (base_rate - warmup_start) * frac
    return base_rate
