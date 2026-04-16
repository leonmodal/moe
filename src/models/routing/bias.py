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
) -> None:
    """Update expert bias in-place using zero-sum sign update.

    The update rule from DeepSeek V3:
      s = sign(load - expected)
      bias -= (s - s.mean()) * rate
      bias = clamp(bias, -clamp_range, clamp_range)

    The zero-sum property (s - s.mean()) ensures biases don't drift collectively.

    Args:
        bias: (num_experts,) expert bias buffer to update in-place.
        counts: (num_experts,) token counts per expert for this step.
        rate: Bias update rate.
        clamp_range: Maximum absolute bias value.
        distributed: Whether to all-reduce counts across workers.
    """
    with torch.no_grad():
        if distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if counts.sum() > 0:
            total = counts.sum()
            loads = counts / total
            expected = 1.0 / counts.shape[0]
            s = torch.sign(loads - expected)
            bias -= (s - s.mean()) * rate
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
