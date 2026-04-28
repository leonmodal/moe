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
    sign update. The `zero_sum` flag selects between two reference
    formulations:

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
        if distributed and dist.is_available() and dist.is_initialized():
            # Only call all_reduce when a process group is actually
            # live. Treating `distributed=True` as a hard requirement
            # crashes during local debugging, importable tests, and
            # the `update_bias_from_counts()` unit suite where callers
            # may propagate `distributed=True` from a config without setting
            # up DDP first.
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


def update_bias_from_quantile(
    bias: torch.Tensor,
    ema: torch.Tensor,
    scores: torch.Tensor,
    *,
    target_q: float,
    eta: float,
    clamp_range: float = 16.0,
    distributed: bool = False,
) -> None:
    """Pure fp32 quantile-balancing update. Updates `bias` and `ema`
    in place from the per-expert raw scores accumulated over the
    last training window.

    The contract is:
      * Each expert's per-batch top score becomes the input
        observation. We summarize the score distribution by the
        target quantile (default 0.5 = median) per expert.
      * The EMA tracks the per-expert quantile across windows:
            ema = (1 - eta) * ema + eta * q_e
      * The bias derivation pulls each expert toward the GLOBAL
        median of those EMA values:
            bias[e] = clamp(ema.median() - ema[e], -clamp_range, +clamp_range)
        Underloaded experts (low scores -> low ema) get a positive
        bias; overloaded experts get a negative one. The clamp
        bounds match the DeepSeek-V3 ±16 envelope.

    Args:
      bias: (num_experts,) — bias buffer to update in place.
      ema:  (num_experts,) — persistent EMA of per-expert quantile.
      scores: (n_active_tokens, num_experts) — concatenated detached
              fp32 raw scores from the active-token forward(s).
              Empty `n_active_tokens` is a no-op (returns without
              touching `bias` or `ema`).
      target_q: Quantile to track per expert, in [0, 1].
                Defaults to 0.5 (median).
      eta: EMA learning rate, in (0, 1].
      clamp_range: Absolute bias clamp.
      distributed: When True AND `dist` is initialized, the global
                   quantile estimate is computed by `all_gather`-ing
                   raw scores across ranks before quantile reduction.
                   When `dist` is not initialized, falls back to the
                   local-rank concatenation.

    The function is a pure update — it does NOT touch model state
    or autograd graphs. The caller is responsible for clearing the
    score accumulator after this update fires.
    """
    with torch.no_grad():
        if scores.numel() == 0:
            return
        if not (0.0 <= target_q <= 1.0):
            raise ValueError(
                f"quantile target_q must be in [0, 1]; got {target_q}"
            )
        if not (0.0 < eta <= 1.0):
            raise ValueError(
                f"quantile eta must be in (0, 1]; got {eta}"
            )
        if distributed and dist.is_available() and dist.is_initialized():
            # Ragged-safe all_gather: per-rank token counts may differ
            # (e.g. branch-masked paths give different active-token
            # counts on different ranks). Gather row-counts first,
            # pad each rank's scores to the max row count, all_gather
            # the padded tensors, then slice each rank back to its
            # original row count and concatenate. The fp32 quantile
            # is computed on the resulting ragged-but-correct global
            # concatenation.
            world_size = dist.get_world_size()
            local_rows = torch.tensor(
                [scores.shape[0]], dtype=torch.long, device=scores.device,
            )
            gathered_counts = [
                torch.zeros_like(local_rows) for _ in range(world_size)
            ]
            dist.all_gather(gathered_counts, local_rows)
            row_counts = [int(t.item()) for t in gathered_counts]
            max_rows = max(row_counts)
            num_experts = scores.shape[1] if scores.ndim > 1 else 0
            padded = scores.new_zeros((max_rows, num_experts))
            padded[: scores.shape[0]] = scores
            gathered_padded = [
                torch.zeros_like(padded) for _ in range(world_size)
            ]
            dist.all_gather(gathered_padded, padded)
            scores = torch.cat(
                [g[:n] for g, n in zip(gathered_padded, row_counts) if n > 0],
                dim=0,
            )
        # Per-expert target quantile across all observed tokens.
        # `quantile` operates on dim=0 (the n_active_tokens axis) and
        # returns a (num_experts,) tensor.
        q_per_expert = torch.quantile(
            scores.float(), q=float(target_q), dim=0,
        )
        # EMA in-place update.
        ema.mul_(1.0 - eta).add_(eta * q_per_expert)
        # Bias = pull each expert toward the global median of EMAs.
        target = ema.median()
        new_bias = (target - ema).clamp_(-clamp_range, clamp_range)
        bias.copy_(new_bias)


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
