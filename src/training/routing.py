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


def exploration_rate_schedule(
    step: int,
    target: float,
    warmup_start: float,
    warmup_steps: int,
) -> float:
    """Linear schedule from `warmup_start` to `target` over `warmup_steps`.

    After `warmup_steps` the rate stays at `target`. `warmup_steps <= 0` means
    no schedule (the rate is `target` for every step). Used to implement the
    "early-step router exploration warmup" technique from
    docs/research/external_moe_techniques.md: start at a lower exploration rate
    so random expert picks do not destabilize the router before it has learned,
    then ramp to the configured target over the first N steps.
    """
    if warmup_steps <= 0:
        return target
    if step >= warmup_steps:
        return target
    frac = max(0.0, step) / warmup_steps
    return warmup_start + (target - warmup_start) * frac


def apply_router_exploration_rate(model, rate: float) -> int:
    """Push `rate` onto every expert-router submodule with an `exploration_rate`.

    Returns the number of router modules updated (useful for asserting
    coverage in tests). Safe for DDP/FSDP-wrapped models — the iteration goes
    through the unwrapped model and mutates the shared tensor-free attribute
    directly.

    `BranchRouter` instances are deliberately skipped: MoE-Everything configs
    can set `branch_router_exploration_rate` independently of the model-wide
    `router_exploration_rate`, so the training-time warmup must not clobber
    that separate value. Branch-router rate control remains a model-config
    knob, not a trainer-side schedule.
    """
    # Lazy import — the router module pulls torch dependencies and we do not
    # want to force those on code paths that only inspect this helper.
    try:
        from src.models.routing.routers import BranchRouter
    except Exception:  # pragma: no cover - defensive
        BranchRouter = None  # type: ignore[assignment]

    raw_model = unwrap_model(model)
    count = 0
    for module in raw_model.modules():
        if BranchRouter is not None and isinstance(module, BranchRouter):
            continue
        if hasattr(module, "exploration_rate"):
            module.exploration_rate = float(rate)
            count += 1
    return count
