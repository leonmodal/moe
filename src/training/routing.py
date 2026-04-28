"""Routing stats collection and expert bias updates during training."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .distributed import is_distributed, unwrap_model


_DEFAULT_BIAS_LABELS = ("q", "k", "v", "o", "mlp", "branch")
# AC-1: methods that drive the post-step bias-update walker. `quantile` will
# join this set in Milestone D when `_update_single_router_quantile_bias`
# lands; until then, calling `update_expert_biases` for a `quantile` model is
# a no-op so we don't accidentally run the wrong update path.
_BIAS_UPDATE_METHODS = frozenset({"deepseek_bias"})


def update_expert_biases(
    model,
    *,
    bias_rate: float,
    distributed: bool = False,
    per_proj_rates: dict[str, float] | None = None,
    zero_sum: bool = True,
) -> None:
    """Update expert biases using DeepSeek V3-style load balancing.

    Walks every load-balancing owner exposed by `raw_model.get_all_balancing_owners()`
    and runs the unified `_update_single_router_bias` on each. Owners expose the
    canonical interface — `expert_bias` (persistent fp32) and
    `local_tokens_per_expert` (non-persistent fp32) — so MLP routers, attention
    routers, and branch routers all share the same code path (DEC-18). The
    label yielded alongside each owner (`mlp`/`q`/`k`/`v`/`o`/`branch`) selects
    a per-projection bias rate from `per_proj_rates`.

    AC-1: when `model._load_balancing_method` is stamped (by `build_model`),
    the function consults it and no-ops for any method outside
    `_BIAS_UPDATE_METHODS` (`aux_loss`, `seq_aux_loss`, `quantile`, `none`).
    This is enforcement-by-default: callers that don't go through the trainer
    (ad-hoc test fixtures, downstream tools, future code paths) still get the
    correct method-gated behavior. When the method attribute is absent, the
    function falls back to the legacy unconditional update for back-compat.

    The legacy `update_global_bias` early-return path was dead code (no model
    in this repo defined either `update_global_bias` or `global_load_balancing`)
    and has been removed.
    """
    raw_model = unwrap_model(model)

    # AC-1: respect the stamped method if present. `None` means the caller
    # didn't set a method — preserve legacy unconditional update for
    # back-compat. Set values must be in the bias-update set.
    method = getattr(raw_model, "_load_balancing_method", None)
    if method is not None and method not in _BIAS_UPDATE_METHODS:
        return

    get_owners = getattr(raw_model, "get_all_balancing_owners", None)
    if get_owners is None:
        return

    rates = {label: bias_rate for label in _DEFAULT_BIAS_LABELS}
    if per_proj_rates:
        rates.update(per_proj_rates)

    use_dist = distributed and dist.is_available() and dist.is_initialized()

    for owner, label in get_owners():
        rate = rates.get(label, bias_rate)
        _update_single_router_bias(owner, rate, use_dist, zero_sum=zero_sum)


def _update_single_router_bias(
    router,
    bias_rate: float,
    distributed: bool,
    *,
    zero_sum: bool = True,
) -> None:
    """Update expert bias for a single load-balancing owner.

    Per DEC-2 (RESOLVED → AC-6) two reference modes are supported, selected
    by the `zero_sum` flag plumbed from `TrainingConfig.bias_update_zero_sum`:

      `zero_sum=True` (default, matches `nmoe.Router.update_bias` — see
      `nmoe/nmoe/model.py:92-97`):
          s     = sign(loads - 1/E)
          delta = (s - s.mean()) * rate
          bias -= delta             # cumulative bias mean is pinned at 0

      `zero_sum=False` (matches Megatron-LM's
      `get_updated_expert_bias` — see Megatron-LM/megatron/core/transformer/
      moe/moe_utils.py):
          delta = sign(avg_load - load) * rate
          bias += delta             # cumulative mean drifts up to ±rate per step
                                    # under asymmetric loads (still bounded by clamp)

    Both modes target the same intuition: underloaded experts get a small
    positive bias bump, overloaded experts get a small negative bump.
    Both clamp the cumulative bias to ±16 after every update (DeepSeek-V3
    scale guard).

    Implementation note: `counts.sum()` was being computed twice — once
    for the host-side `> 0` check and again for the normaliser — each
    call a CPU↔GPU sync. Compute it once and branch on a device-side mask
    so there's no host barrier (the mask yields a zero update when no
    tokens were seen, which is equivalent to the previous early-return
    behaviour).
    """
    with torch.no_grad():
        counts = router.local_tokens_per_expert
        if distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total = counts.sum()
        loads = counts / total.clamp_min(1.0)
        expected = 1.0 / counts.shape[0]
        s = torch.sign(loads - expected)
        nonzero = (total > 0).to(s.dtype)
        if zero_sum:
            # nmoe / DeepSeek-V3 zero-sum (mean-subtracted) update.
            router.expert_bias -= (s - s.mean()) * bias_rate * nonzero
        else:
            # Megatron-LM plain-sign update; equivalent to flipping the sign
            # since `s = sign(load - 1/E)` and Megatron uses `sign(avg - load)`.
            router.expert_bias -= s * bias_rate * nonzero
        router.expert_bias.clamp_(-16.0, 16.0)
        router.local_tokens_per_expert.zero_()


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


def collect_router_z_loss(model) -> torch.Tensor | None:
    """Sum the most recent per-router z-loss contributions and clear them.

    Each `ExplorationTopKRouter` / `DeepSeekRouter` caches its per-call z-loss
    contribution on `_last_z_loss` (a scalar tensor when `router_z_loss_coef > 0`,
    `None` otherwise). This walker gathers those scalars across the whole model
    and returns their sum with autograd intact, so the trainer can add it to
    the total loss before `backward()`.

    After collecting, every consumed `_last_z_loss` is reset to `None`. Without
    the reset, a router whose forward is skipped on a later micro-batch
    (e.g., a depth that the branch router routes around in MoE-Everything)
    would contribute its stale tensor from the previous step on every
    subsequent call until its own forward runs again — silently double-counting
    z-loss into the total. Routers that did run this micro-batch re-populate
    `_last_z_loss` on the *next* forward, so the clear is safe.

    Returns `None` when every router has `_last_z_loss is None` — i.e. the
    feature is disabled; callers should treat `None` as "add nothing" so
    models configured without z-loss pay no extra Python work beyond the
    module-tree walk.
    """
    raw_model = unwrap_model(model)
    acc: torch.Tensor | None = None
    for module in raw_model.modules():
        z = getattr(module, "_last_z_loss", None)
        if z is None:
            continue
        acc = z if acc is None else acc + z
        # Consume: prevent the same tensor from being summed again on a later
        # call if this router's forward doesn't run next micro-batch.
        module._last_z_loss = None
    return acc


def apply_router_exploration_rate(model, rate: float) -> int:
    """Push `rate` onto every router submodule with an `exploration_rate`.

    Returns the number of router modules updated (useful for asserting
    coverage in tests). Safe for DDP/FSDP-wrapped models — the iteration goes
    through the unwrapped model and mutates the shared tensor-free attribute
    directly.

    BranchRouter handling: MoE-Everything configs can set
    `branch_router_exploration_rate` independently of the model-wide
    `router_exploration_rate`. `MoEverythingConfig.__init__` also defaults
    the branch value to the model-wide value when left unset, so in that
    inherited / explicit-match case the BranchRouter should follow the
    warmup schedule alongside the expert routers (AC-10). The applier
    therefore includes every `BranchRouter` iff the unwrapped model's
    `config.branch_router_exploration_rate == config.router_exploration_rate`;
    when the two diverge, the user has opted into an independent branch
    rate and the applier leaves every `BranchRouter` alone. A model
    without a `config` attribute (unit-test stubs) keeps the skip
    behaviour — the trainer only calls this with real MoE models whose
    config is always present.
    """
    # Lazy import — the router module pulls torch dependencies and we do not
    # want to force those on code paths that only inspect this helper.
    try:
        from src.models.routing.routers import BranchRouter
    except Exception:  # pragma: no cover - defensive
        BranchRouter = None  # type: ignore[assignment]

    raw_model = unwrap_model(model)
    cfg = getattr(raw_model, "config", None)
    model_rate = getattr(cfg, "router_exploration_rate", None)
    branch_rate = getattr(cfg, "branch_router_exploration_rate", None)
    include_branch_router = (
        model_rate is not None
        and branch_rate is not None
        and model_rate == branch_rate
    )
    count = 0
    for module in raw_model.modules():
        if BranchRouter is not None and isinstance(module, BranchRouter):
            if not include_branch_router:
                continue
        if hasattr(module, "exploration_rate"):
            module.exploration_rate = float(rate)
            count += 1
    return count
