"""Global expert bias management: per-projection rates, zero-sum update, clamp, warmup, broadcast.

Extracted from speedrun_moe_gpt.py. This module manages the global bias system
used for DeepSeek-style aux-loss-free load balancing. The bias is maintained
per projection type (Q, K, V, O, MLP, branch) and broadcast to per-depth routers.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def get_bias_rate(
    step: int,
    base_rate: float,
    warmup_start: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """Compute the current bias update rate with optional linear warmup.

    Args:
        step: Current training step.
        base_rate: Target bias rate after warmup.
        warmup_start: Initial bias rate at step 0.
        warmup_steps: Number of steps to linearly ramp from warmup_start to base_rate.

    Returns:
        Current bias rate.
    """
    if warmup_steps > 0 and step < warmup_steps:
        frac = step / warmup_steps
        return warmup_start + (base_rate - warmup_start) * frac
    return base_rate


def update_bias_from_counts(
    bias: torch.Tensor,
    counts: torch.Tensor,
    rate: float,
    clamp_range: float = 16.0,
    distributed: bool = False,
    *,
    zero_sum: bool = True,
) -> None:
    """Update expert bias in-place using the configured DeepSeek-V3-style
    sign update. DEC-2 (RESOLVED → AC-6) selects between two reference
    formulations via the `zero_sum` flag.

    `zero_sum=True` (default, mirrors `nmoe.Router.update_bias`):
        s     = sign(load - 1/E)
        delta = (s - s.mean()) * rate
        bias -= delta
      The mean-subtraction pins the cumulative bias mean at zero so the
      bias does not drift unboundedly under asymmetric loads.

    `zero_sum=False` (mirrors Megatron-LM's plain-sign update):
        delta = sign(avg_load - load) * rate
        bias += delta
      The mean is allowed to drift up to ±rate per step under asymmetric
      loads — still bounded by `clamp_range` but otherwise unconstrained.

    Both modes target the same intuition: underloaded experts get a
    small positive bias bump, overloaded experts get a small negative
    bump. Both clamp to ±`clamp_range` after the update.

    Args:
        bias: (num_experts,) expert bias buffer to update in-place.
        counts: (num_experts,) token counts per expert for this step.
        rate: Bias update rate.
        clamp_range: Maximum absolute bias value (DeepSeek-V3 default 16).
        distributed: Whether to all-reduce counts across workers.
        zero_sum: True for nmoe / DeepSeek-V3 zero-sum; False for the
                  Megatron-LM plain-sign reference. Plumbed from
                  `TrainingConfig.bias_update_zero_sum` (default True).
    """
    with torch.no_grad():
        if distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if counts.sum() > 0:
            total = counts.sum()
            loads = counts / total
            expected = 1.0 / counts.shape[0]
            s = torch.sign(loads - expected)
            if zero_sum:
                bias -= (s - s.mean()) * rate
            else:
                bias -= s * rate
            bias.clamp_(-clamp_range, clamp_range)


def broadcast_bias_to_routers(
    global_bias: torch.Tensor,
    routers: list,
    bias_attr: str = "expert_bias",
) -> None:
    """Broadcast a global bias tensor to a list of per-depth routers.

    Each router's bias is set to the global bias value. This is used when
    global load balancing is enabled to synchronize all per-depth routers
    with a single shared bias.

    Args:
        global_bias: (num_experts,) shared bias tensor.
        routers: List of router modules with a bias buffer attribute.
        bias_attr: Name of the bias attribute on each router.
    """
    for router in routers:
        if hasattr(router, bias_attr):
            getattr(router, bias_attr).copy_(global_bias)
