"""Routing stats collection and expert bias updates during training."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .distributed import is_distributed, unwrap_model


_DEFAULT_BIAS_LABELS = ("q", "k", "v", "o", "mlp", "branch")
# Methods that trigger the post-step bias-update walker. The
# DeepSeek-V3-style sigmoid+bias router is the only active method
# today; quantile-based balancing will be added later. Calling the
# update path for any other method is a no-op so the wrong update
# rule never runs by accident.
_BIAS_UPDATE_METHODS = frozenset({"deepseek_bias"})


def trainer_optimizer_step_and_bias_update(
    model,
    optimizer,
    scheduler,
    train_cfg,
    cfg: dict,
    *,
    distributed: bool,
    global_step: int,
) -> None:
    """The trainer's optimizer-step / scheduler-step / bias-update
    tail, extracted into a single helper so it can be tested
    end-to-end with spies on every call.

    Production call order (locked by this helper):
      1. `optimizer.step()`  — apply gradients to parameters.
      2. `scheduler.step()`  — advance learning-rate schedule.
      3. `trainer_post_optimizer_bias_update(...)` — method-gated,
         no_grad-wrapped expert bias update.

    Tests instrument this helper to verify both the call order and
    the `torch.is_grad_enabled() == False` contract at the bias-update
    call site. The production trainer also calls this helper, so a
    future regression that re-orders or unwraps the no_grad block
    would be caught.
    """
    optimizer.step()
    scheduler.step()
    trainer_post_optimizer_bias_update(
        model, train_cfg, cfg,
        distributed=distributed, global_step=global_step,
    )


def trainer_post_optimizer_bias_update(
    model,
    train_cfg,
    cfg: dict,
    *,
    distributed: bool,
    global_step: int,
) -> None:
    """Run the production post-optimizer-step bias-update block.

    The trainer invokes this once per training step, after
    `optimizer.step()` and `scheduler.step()`. The function:

      1. Reads the resolved `load_balancing_method` from the
         (unwrapped) model and gates the bias update on
         `_BIAS_UPDATE_METHODS` (currently `{deepseek_bias}`).
      2. Computes the current bias rate via `get_bias_rate(...)`
         (linear warmup from `bias_warmup_start` → `bias_update_rate`
         over `bias_warmup_steps`).
      3. Resolves per-projection rate overrides via the canonical
         `training:` block resolver.
      4. Wraps `update_expert_biases(...)` in `torch.no_grad()` so
         the function's misuse guard is satisfied.

    The helper is the canonical production call site for the
    bias-update; tests instrument it directly.
    """
    from .balancing_fields import _resolve_balancing_field

    raw_model = unwrap_model(model)
    method_for_bias_update = getattr(raw_model, "_load_balancing_method", None)
    method_allows_bias_update = (
        method_for_bias_update is None
        or method_for_bias_update in _BIAS_UPDATE_METHODS
    )
    # Per-class effective rate: when the nested-schema yaml sets
    # `model.mlp_router.balancing == "deepseek_bias"` and a
    # per-class `bias_update_rate`, the trainer must run the bias
    # update on that rate even when top-level
    # `train_cfg.bias_update_rate` is zero (the migrator strips the
    # top-level field once a per-class block exists). Read the
    # mirrored `effective_bias_update_rate` that `model_factory.py`
    # stamps onto `config` for every nested deepseek_bias build.
    raw_config = getattr(raw_model, "config", None)
    effective_rate = getattr(raw_config, "effective_bias_update_rate", None)
    bias_update_rate = train_cfg.bias_update_rate
    if (effective_rate is not None and effective_rate > 0
            and bias_update_rate <= 0):
        bias_update_rate = float(effective_rate)
    bias_warmup_start = (
        getattr(raw_config, "effective_bias_warmup_start", None)
        if train_cfg.bias_warmup_start <= 0 else train_cfg.bias_warmup_start
    )
    if bias_warmup_start is None:
        bias_warmup_start = train_cfg.bias_warmup_start
    bias_warmup_steps = (
        getattr(raw_config, "effective_bias_warmup_steps", None)
        if train_cfg.bias_warmup_steps <= 0 else train_cfg.bias_warmup_steps
    )
    if bias_warmup_steps is None:
        bias_warmup_steps = train_cfg.bias_warmup_steps
    if not (bias_update_rate > 0 and method_allows_bias_update):
        return
    rate = get_bias_rate(
        model, global_step, bias_update_rate,
        bias_warmup_start, bias_warmup_steps,
    )
    per_proj_rates = {
        "q": _resolve_balancing_field(cfg, "bias_rate_q", rate),
        "k": _resolve_balancing_field(cfg, "bias_rate_k", rate),
        "v": _resolve_balancing_field(cfg, "bias_rate_v", rate),
        "o": _resolve_balancing_field(cfg, "bias_rate_o", rate),
        "mlp": _resolve_balancing_field(cfg, "bias_rate_mlp", rate),
        "branch": _resolve_balancing_field(cfg, "bias_rate_branch", rate),
    }
    # Per-class `bias_update_zero_sum` mirror — falls back to
    # top-level when unset.
    effective_zero_sum = getattr(
        raw_config, "effective_bias_update_zero_sum", None,
    )
    if effective_zero_sum is None:
        effective_zero_sum = train_cfg.bias_update_zero_sum
    with torch.no_grad():
        update_expert_biases(
            model, bias_rate=rate, distributed=distributed,
            per_proj_rates=per_proj_rates,
            zero_sum=bool(effective_zero_sum),
        )


def update_expert_biases(
    model,
    *,
    bias_rate: float,
    distributed: bool = False,
    per_proj_rates: dict[str, float] | None = None,
    zero_sum: bool = True,
) -> None:
    """Update expert biases using DeepSeek V3-style load balancing.

    Walks every load-balancing owner exposed by
    `raw_model.get_all_balancing_owners()` and runs the unified
    `_update_single_router_bias` on each. Owners expose the canonical
    interface — `expert_bias` (persistent fp32) and
    `local_tokens_per_expert` (non-persistent fp32) — so MLP routers,
    attention routers, and branch routers all share the same code
    path. The label yielded alongside each owner
    (`mlp`/`q`/`k`/`v`/`o`/`branch`) selects a per-projection bias
    rate from `per_proj_rates`.

    Method gating: when `model._load_balancing_method` is stamped (by
    `build_model`), the function consults it and no-ops for any
    method outside `_BIAS_UPDATE_METHODS` (`aux_loss`, `seq_aux_loss`,
    `quantile`, `none`). This is enforcement-by-default: callers that
    don't go through the trainer (ad-hoc test fixtures, downstream
    tools) still get the correct method-gated behavior. When the
    method attribute is absent, the function falls back to the
    legacy unconditional update for back-compat.

    Misuse guard: when this function would actually mutate
    `expert_bias`, it must be called inside `torch.no_grad()` (i.e.
    post-optimizer-step). The check runs after the method-dispatch
    no-op so non-bias methods still no-op cleanly under the default
    grad-enabled context.
    """
    raw_model = unwrap_model(model)

    # Respect the stamped method if present. `None` means the caller
    # didn't set a method — preserve legacy unconditional update for
    # back-compat. Set values must be in the bias-update set.
    method = getattr(raw_model, "_load_balancing_method", None)
    if method is not None and method not in _BIAS_UPDATE_METHODS:
        return

    get_owners = getattr(raw_model, "get_all_balancing_owners", None)
    if get_owners is None:
        return

    # Misuse guard: only bias-update-active methods reach this point,
    # so fail fast if the caller is in a grad-enabled context (a
    # bias update inside backward / before optimizer.step would
    # mutate `expert_bias` from stale or partially-accumulated
    # counts). Placed AFTER the method-dispatch return so non-bias
    # methods continue to no-op cleanly under the default
    # grad-enabled context.
    if torch.is_grad_enabled():
        raise RuntimeError(
            "update_expert_biases must be called inside a `torch.no_grad()` "
            "context (post-optimizer-step). Wrap the call in "
            "`with torch.no_grad():` — the trainer path already does this. "
            "If you saw this from a test, your test is missing the no_grad "
            "wrapper that production code uses."
        )

    rates = {label: bias_rate for label in _DEFAULT_BIAS_LABELS}
    if per_proj_rates:
        rates.update(per_proj_rates)

    use_dist = distributed and dist.is_available() and dist.is_initialized()

    for owner, label in get_owners():
        # Skip branch routers in exploration-only mode: the mode is
        # by construction independent of the bias-update signal, so
        # nudging `expert_bias` here would do nothing routing-wise
        # AND could accumulate spurious nonzero values that surprise
        # checkpoint-resume / state-dict diff readers.
        if (
            label == "branch"
            and getattr(owner, "balancing", "none") == "exploration_only"
        ):
            continue
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

    Two reference modes are supported, selected by the `zero_sum`
    flag plumbed from `TrainingConfig.bias_update_zero_sum`:

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
        if distributed and dist.is_available() and dist.is_initialized():
            # Only all_reduce when a process group is actually live.
            # The trainer call site already checks `is_initialized()`,
            # but direct callers (unit tests, debug fixtures) may pass
            # `distributed=True` without initializing DDP. Defensive
            # guard so neither path crashes.
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


_EXPLORATION_DECAY_SCHEDULES = ("constant", "linear", "cosine")


def exploration_decay_schedule(
    step: int,
    *,
    schedule: str,
    initial_rate: float,
    decay_steps: int,
    final_rate: float = 0.0,
) -> float:
    """Decay an exploration rate from `initial_rate` to `final_rate` over
    `decay_steps`, following one of `constant` / `linear` / `cosine` shapes.

    This is the companion of `exploration_rate_schedule` for the
    "exploration_only" branch routing mode and similar use-cases that
    need a rate scheduler decaying TOWARDS zero, not warming up TO a
    target. The schedules are:

    * `constant`: returns `initial_rate` for every step (decay_steps
      ignored). Use when the rate is held flat — typically for
      diagnostic runs that want a fixed exploration mix.
    * `linear`: `initial_rate * (1 - step/decay_steps) + final_rate * step/decay_steps`
      until step == decay_steps; final_rate thereafter.
    * `cosine`: `final_rate + 0.5 * (initial_rate - final_rate) * (1 + cos(pi * step / decay_steps))`
      — smooth decay from initial to final following half a cosine.
      Returns `final_rate` after `decay_steps`.

    Returns the rate as a Python float. Args:
      step: current step count (clamped to [0, decay_steps] for the
            non-constant schedules).
      schedule: one of the shapes listed above.
      initial_rate: rate at step 0.
      decay_steps: number of steps over which to decay (>= 0).
      final_rate: rate at step >= decay_steps. Defaults to 0.0 so the
                  caller can omit it for the canonical "decay to zero"
                  case.
    """
    import math
    if schedule not in _EXPLORATION_DECAY_SCHEDULES:
        raise ValueError(
            f"exploration schedule must be one of {sorted(_EXPLORATION_DECAY_SCHEDULES)}, "
            f"got {schedule!r}"
        )
    if schedule == "constant":
        return initial_rate
    if decay_steps <= 0:
        return final_rate
    clamped = min(max(0, step), decay_steps)
    frac = clamped / decay_steps
    if schedule == "linear":
        return initial_rate * (1.0 - frac) + final_rate * frac
    # schedule == "cosine"
    return final_rate + 0.5 * (initial_rate - final_rate) * (1.0 + math.cos(math.pi * frac))


def apply_branch_schedule_pre_forward(model, global_step: int) -> float | None:
    """Apply the branch exploration_only schedule for `global_step`
    BEFORE the forward pass for that step. Returns the rate that was
    applied, or `None` if the feature is inactive on this model.

    This is the canonical pre-forward hook the trainer calls once per
    training step (and once after model construction / checkpoint
    load) to keep the BranchRouter's `exploration_only_rate` aligned
    with `p_explore(global_step)`. The rate the helper returns is
    what the trainer's logging path should report for this step:
    forward + backward + optimizer all run with exactly this rate
    active.
    """
    rate = compute_branch_exploration_only_rate(model, global_step)
    if rate is None:
        return None
    apply_branch_exploration_only_rate(model, rate)
    return rate


def apply_branch_exploration_only_rate(model, rate: float) -> int:
    """Push `rate` into every BranchRouter on `model` whose
    `balancing == "exploration_only"`.

    The trainer calls this once per training step (after the optimizer
    + bias-update tail) so the per-step `p_explore(step)` from
    `exploration_decay_schedule(...)` lands on the actual routers
    before the next forward. Returns the number of routers updated, so
    the trainer's logging path can assert at least one router was
    updated when the schedule is active (catches misconfiguration like
    "balancing field set on config but never propagated to the
    BranchRouter constructor").

    The walk is gated on `balancing == "exploration_only"` to keep
    routers that opted out of the rate-driven mode untouched —
    pushing a rate into a `balancing == "none"` router would silently
    enable exploration_only without the config saying so.
    """
    raw_model = unwrap_model(model)
    updated = 0
    for module in raw_model.modules():
        if getattr(module, "balancing", "none") != "exploration_only":
            continue
        if not hasattr(module, "exploration_only_rate"):
            continue
        module.exploration_only_rate = float(rate)
        updated += 1
    return updated


def collect_branch_attn_counts(
    model,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Token-weighted ATTN-vs-total branch-selection counts on the
    most recent forward. Returns
    `(global_attn_count, global_total_count, per_depth_list)` where
    `per_depth_list[i] == (attn_count_at_depth_i, total_count_at_depth_i)`.
    Returns `None` when the model has no branch-decision cache (so the
    trainer's logging path can skip the metric).

    These raw counts are the right shape for cross-rank token-weighted
    reduction: `dist.all_reduce(SUM)` the numerator/denominator pairs
    across ranks, then divide. Averaging already-averaged fractions
    across ranks is wrong when ranks have different per-rank token
    counts (e.g. heterogeneous batch sizes).
    """
    raw_model = unwrap_model(model)
    inner = getattr(raw_model, "model", raw_model)
    cache = getattr(inner, "_all_branch_selected_experts", None)
    if cache is None:
        cache = getattr(raw_model, "_all_branch_selected_experts", None)
    if not cache:
        return None

    per_depth: list[tuple[int, int]] = []
    total_attn = 0
    total_count = 0
    for selected in cache:
        if selected is None:
            continue
        attn_count = int((selected == 0).sum().item())
        n = int(selected.numel())
        per_depth.append((attn_count, n))
        total_attn += attn_count
        total_count += n

    if not per_depth:
        return None
    return total_attn, total_count, per_depth


def collect_branch_attn_fraction(
    model,
) -> tuple[float | None, list[float] | None]:
    """Per-step branch routing telemetry: the fraction of branch tokens
    that chose ATTN (selected_experts == 0) on the most recent forward.

    Reads the model's per-depth branch-selection cache populated by the
    most recent forward. For `MoEverythingForCausalLM`, the inner
    backbone (`model.model`) appends every depth's
    `last_selected_experts` into `_all_branch_selected_experts` on each
    `_depth_step`, so a shared branch router (the default
    `per_layer_router=False` path) is observable across all depths
    even though the module-level `last_selected_experts` is overwritten
    on every depth call. Reading from the all-depths cache is the only
    correct way to compute `% ATTN` over the full step's branch
    decisions.

    Returns `(global_mean, per_depth)`:

    * `global_mean` is TOKEN-WEIGHTED across all depths: the total
      ATTN-token count divided by the total branch-token count. For a
      step where one depth chose ATTN for every token and the next
      chose MLP for every token, the global mean is 0.5 — not 0.0 (the
      shared module's final-depth view) and not the average of two
      already-averaged fractions (which is meaningless if depths have
      different token counts).
    * `per_depth` is the list of per-depth ATTN fractions, one entry
      per populated cache slot. For `per_layer_router=True` the list
      has `num_hidden_layers` entries, one per depth; for the singular
      `branch_router` path the list also has `num_hidden_layers`
      entries (one per depth call of the SAME shared router), so
      per-depth divergence is visible regardless of routing mode.

    Returns `(None, None)` when no branch decisions are present (e.g.
    a non-moe_everything model, or a moe_everything model that has not
    been run since construction). The trainer's logging path treats
    `(None, None)` as "do not emit the metric this step", so models
    without branch routers do not pay the wandb-payload cost.

    Mask-based exploration_only fraction is exposed by a separate
    `collect_branch_explore_mask_fraction(model)` helper for diagnostic
    runs that want to see how often the random override was taken.
    """
    raw_model = unwrap_model(model)
    inner = getattr(raw_model, "model", raw_model)
    cache = getattr(inner, "_all_branch_selected_experts", None)
    if cache is None:
        cache = getattr(raw_model, "_all_branch_selected_experts", None)
    if not cache:
        return None, None

    per_depth: list[float] = []
    total_attn = 0
    total_count = 0
    for selected in cache:
        if selected is None:
            continue
        per_depth.append(float((selected == 0).float().mean().item()))
        total_attn += int((selected == 0).sum().item())
        total_count += int(selected.numel())

    if not per_depth:
        return None, None
    global_mean = total_attn / total_count
    return global_mean, per_depth


def collect_branch_explore_mask_fraction(model) -> float | None:
    """Diagnostic-only fraction of branch tokens routed via the
    exploration_only mask on the most recent forward. This is NOT
    the trainer's `% ATTN` telemetry — it is the per-step mean of
    `last_exploration_only_mask` across every active exploration_only
    router. Useful when verifying the schedule actually applied a
    nonzero rate, but unsuitable as the branch-fraction proxy.

    Returns `None` when no router has a populated mask (eval-mode
    forward, rate == 0, or feature inactive on this model).
    """
    raw_model = unwrap_model(model)
    fractions: list[float] = []
    for module in raw_model.modules():
        if getattr(module, "balancing", "none") != "exploration_only":
            continue
        mask = getattr(module, "last_exploration_only_mask", None)
        if mask is None:
            continue
        fractions.append(float(mask.float().mean().item()))
    if not fractions:
        return None
    return sum(fractions) / len(fractions)


def compute_branch_exploration_only_rate(model, global_step: int) -> float | None:
    """Resolve the current exploration_only rate from the model's config.

    Returns `None` when the feature is inactive (i.e. no
    `branch_balancing == "exploration_only"` on the model's config).
    Otherwise reads the schedule shape / initial rate / floor / decay
    length from `model.config` and dispatches to
    `exploration_decay_schedule(...)`.
    """
    raw_model = unwrap_model(model)
    config = getattr(raw_model, "config", None)
    if config is None:
        return None
    if getattr(config, "branch_balancing", "none") != "exploration_only":
        return None
    schedule = getattr(config, "branch_exploration_decay", "constant")
    initial = float(getattr(config, "branch_exploration_rate", 1.0))
    decay_steps = int(getattr(config, "branch_exploration_warmup_steps", 0))
    final = float(getattr(config, "branch_exploration_min", 0.0))
    return exploration_decay_schedule(
        global_step,
        schedule=schedule,
        initial_rate=initial,
        decay_steps=decay_steps,
        final_rate=final,
    )


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
    warmup schedule alongside the expert routers. The applier
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
