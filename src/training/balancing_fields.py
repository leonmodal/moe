"""Canonical-block resolution for balancing fields.

Both `build_training_config` (`config.py`) and `build_model`
(`model_factory.py`) need to read aux coefficients from the canonical
`training:` block while warning on legacy `model:` placement; this
module owns that resolver so neither caller has a circular-import on
the other, and tests can import the helper without dragging
`src/training/__init__.py` into scope (that path transitively imports
pandas / liger).

Every balancing-related field lives under `cfg["training"]`, NOT
`cfg["model"]`. Today's yamls have already been migrated by
`scripts/migrate_balancing_fields_to_training.py`. The resolver below
tolerates unmigrated yamls (warning + reads the legacy location) so
external configs remain bootable for one release.
"""
from __future__ import annotations

import warnings
from typing import Any


# Fields that MUST live under `cfg["training"]` per the canonical-block resolver.
_BALANCING_FIELDS_IN_TRAINING: tuple[str, ...] = (
    "router_aux_loss_coef",
    "seq_aux_loss_coef",
    "bias_update_rate",
    "bias_warmup_start",
    "bias_warmup_steps",
    "bias_update_zero_sum",  # mode selector — see TrainingConfig comment.
    "load_balancing_method",
    "bias_rate_q",
    "bias_rate_k",
    "bias_rate_v",
    "bias_rate_o",
    "bias_rate_mlp",
    "bias_rate_branch",
)


def _resolve_balancing_field(cfg: dict, name: str, default: Any) -> Any:
    """Read a balancing field from `cfg["training"]`, falling back to
    `cfg["model"]` with a deprecation warning if found there.

    Returns the resolved value (or `default` if neither block has it).

    Conflict policy:
      - Both blocks have the field: prefer `training:` (canonical), warn.
      - Only `model:` has the field: read `model:`, warn (legacy yaml).
      - Only `training:` has the field: read `training:` quietly.
      - Neither block has the field: return `default` quietly.
    """
    tcfg = cfg.get("training", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    if name in tcfg:
        if name in mcfg:
            warnings.warn(
                f"Config field {name!r} appears in BOTH `training:` and `model:`; "
                f"using the `training:` value (canonical per the canonical-block resolver). Remove the "
                f"`model:` copy to silence this warning.",
                DeprecationWarning, stacklevel=3,
            )
        return tcfg[name]
    if name in mcfg:
        warnings.warn(
            f"Config field {name!r} found under `model:` — the canonical-block resolver moved it to "
            f"`training:`. The `model:` placement is deprecated; run "
            f"`python scripts/migrate_balancing_fields_to_training.py` to migrate.",
            DeprecationWarning, stacklevel=3,
        )
        return mcfg[name]
    return default


# Coefficient gating per method: which legacy coefficients are kept active under each
# `load_balancing_method`. Anything outside the per-method "active" set is
# AUTO-ZEROED with a deprecation warning so the trainer's coefficient-driven
# code paths produce behavior consistent with the resolved method.
_VALID_LOAD_BALANCING_METHODS: tuple[str, ...] = (
    "aux_loss",
    "seq_aux_loss",
    "deepseek_bias",
    "quantile",
    "none",
)
# Detach-only telemetry: methods whose loss term needs
# `router_logits` exposed as a gradient-bearing model output. For other
# methods, the model's `forward` should NOT request loss-bearing router
# logits — non-aux methods drive routing through the router-internal
# `_last_top_k_idx` / `local_tokens_per_expert` buffers and do not need the
# autograd graph to retain `router_logits`.
_AUX_BEARING_METHODS: frozenset = frozenset({"aux_loss", "seq_aux_loss"})


def output_router_logits_for_method(method: str | None) -> bool:
    """Resolve the `output_router_logits` flag from `load_balancing_method`.

    Detach-only policy:
      - `None` (no method set; legacy back-compat): return True
        (preserves the legacy default — any caller that opts out
        can pass `None` explicitly to bypass).
      - `aux_loss` / `seq_aux_loss`: return True (router scores are
        gradient-bearing inputs to the aux loss term).
      - `deepseek_bias` / `quantile` / `none`: return False (no aux
        loss term reads `router_logits`; the router-internal
        `_last_top_k_idx` and `local_tokens_per_expert` carry the
        routing-decision state that the bias-update path consumes).
    """
    if method is None:
        return True
    return method in _AUX_BEARING_METHODS
_METHOD_ACTIVE_FIELDS: dict[str, frozenset[str]] = {
    "aux_loss": frozenset({"router_aux_loss_coef"}),
    "seq_aux_loss": frozenset({"seq_aux_loss_coef"}),
    "deepseek_bias": frozenset({
        "bias_update_rate", "bias_warmup_start", "bias_warmup_steps",
        "bias_rate_q", "bias_rate_k", "bias_rate_v", "bias_rate_o",
        "bias_rate_mlp", "bias_rate_branch",
    }),
    "quantile": frozenset(),  # quantile-specific knobs land in a later round.
    "none": frozenset(),
}


# Common balancing-method knobs every router group accepts. These
# correspond to fields the resolver stamps onto the per-class
# runtime config: aux/seq-aux coefficients, the DeepSeek-bias
# update knobs, the quantile-method knobs, and the per-class
# exploration schedule.
_COMMON_ROUTER_KNOBS = frozenset({
    "balancing",
    # Aux/seq-aux loss coefficients (per-class).
    "router_aux_loss_coef",
    "seq_aux_loss_coef",
    # DeepSeek-style bias update knobs (per-class).
    "bias_update_rate",
    "bias_update_zero_sum",
    "bias_warmup_start",
    "bias_warmup_steps",
    # Quantile-method knobs (per-class). When the quantile method
    # lands in the runtime, these fields wire into the per-class
    # quantile state owners. The validator accepts them now so the
    # nested-schema configs can be authored ahead of the runtime
    # implementation.
    "quantile_eta",
    "quantile_target_q",
    "quantile_global_state",
    # Per-class exploration schedule.
    "exploration_rate",
    "exploration_decay",
    "exploration_min",
    "exploration_warmup_steps",
})

# Branch-router accepts `none`, `exploration_only`, `aux_loss`, and
# `seq_aux_loss`. Validator stays aligned with the BranchRouter
# constructor's accepted set: aux/seq-aux flow through the same
# loss path used for MLP and attention routers (using the
# branch-specific `last_probs` / `last_selected_experts` tensors).
# `deepseek_bias` and `quantile` still need owner-state plumbing
# before they can route through the post-step walker; they remain
# rejected at validator AND constructor level. The MLP and attention
# router groups accept the broader value set because their forward
# dispatch can gate on the per-class method without new router-class
# code.
_BRANCH_ROUTER_KNOWN_KEYS = frozenset(_COMMON_ROUTER_KNOBS)

_BRANCH_BALANCING_VALID = frozenset({
    "none", "exploration_only", "aux_loss", "seq_aux_loss",
    "deepseek_bias",
})

_BRANCH_DECAY_VALID = frozenset({"constant", "linear", "cosine"})

# MLP and attention nested router groups share the common knobs.
_MLP_ROUTER_KNOWN_KEYS = frozenset(_COMMON_ROUTER_KNOBS)

_ATTN_ROUTER_KNOWN_KEYS = frozenset(_COMMON_ROUTER_KNOBS) | frozenset({
    # attention-specific knobs allowed on the per-class schema:
    "scale_by_routing_weight",
})


def _validate_mlp_or_attn_router(cfg: dict, group: str) -> None:
    """Validate one of the per-class router groups
    (`model.mlp_router` or `model.attn_router`). Each group may
    define a `balancing` field with the broader balancing-method
    set, plus shared aux/seq-aux coefficients and the
    branch-style exploration schedule fields. Unknown keys are
    rejected with the allowed-keys list.
    """
    mcfg = cfg.get("model")
    if not isinstance(mcfg, dict):
        return
    nested = mcfg.get(group)
    if nested is None:
        return
    if not isinstance(nested, dict):
        raise ValueError(
            f"model.{group} must be a mapping; got {type(nested).__name__}"
        )

    if group == "mlp_router":
        allowed = _MLP_ROUTER_KNOWN_KEYS
    elif group == "attn_router":
        allowed = _ATTN_ROUTER_KNOWN_KEYS
    else:
        raise ValueError(f"unsupported router group: {group}")

    unknown = set(nested.keys()) - allowed
    if unknown:
        raise ValueError(
            f"model.{group} has unknown keys: {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )

    if "balancing" in nested:
        bal = nested["balancing"]
        if bal not in _VALID_LOAD_BALANCING_METHODS:
            raise ValueError(
                f"model.{group}.balancing={bal!r} is invalid; "
                f"must be one of {sorted(_VALID_LOAD_BALANCING_METHODS)}"
            )

    if "exploration_decay" in nested:
        decay = nested["exploration_decay"]
        if decay not in _BRANCH_DECAY_VALID:
            raise ValueError(
                f"model.{group}.exploration_decay={decay!r} is invalid; "
                f"must be one of {sorted(_BRANCH_DECAY_VALID)}"
            )

    for key in ("exploration_rate", "exploration_min"):
        if key in nested:
            v = nested[key]
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"model.{group}.{key}={v} must be in [0.0, 1.0]"
                )
    if "exploration_min" in nested and "exploration_rate" in nested:
        if nested["exploration_min"] > nested["exploration_rate"]:
            raise ValueError(
                f"model.{group}.exploration_min ({nested['exploration_min']}) "
                f"exceeds exploration_rate ({nested['exploration_rate']}); "
                f"the floor cannot exceed the initial rate."
            )
    if "exploration_warmup_steps" in nested:
        steps = nested["exploration_warmup_steps"]
        if not isinstance(steps, int) or steps < 0:
            raise ValueError(
                f"model.{group}.exploration_warmup_steps={steps} must be "
                f"a non-negative int"
            )

    for key in ("router_aux_loss_coef", "seq_aux_loss_coef"):
        if key in nested:
            v = nested[key]
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError(
                    f"model.{group}.{key}={v} must be a non-negative number"
                )

    # Illegal-combination rejection. Each balancing method has a
    # well-defined "active knobs" set; specifying a knob outside
    # that set is rejected so a yaml can't silently mix methods.
    # The active sets cover EVERY field the runtime consumes for
    # the method:
    #   * `aux_loss`        -> {router_aux_loss_coef}
    #   * `seq_aux_loss`    -> {seq_aux_loss_coef}
    #   * `deepseek_bias`   -> {bias_update_rate, bias_update_zero_sum,
    #                           bias_warmup_start, bias_warmup_steps}
    #   * `quantile`        -> {quantile_eta, quantile_target_q,
    #                           quantile_global_state}
    #   * `none`            -> {} (every active knob is rejected)
    aux_fields = {"router_aux_loss_coef"}
    seq_fields = {"seq_aux_loss_coef"}
    bias_fields = {
        "bias_update_rate", "bias_update_zero_sum",
        "bias_warmup_start", "bias_warmup_steps",
    }
    quantile_fields = {
        "quantile_eta", "quantile_target_q", "quantile_global_state",
    }
    all_active = aux_fields | seq_fields | bias_fields | quantile_fields

    method_to_allowed = {
        "aux_loss": aux_fields,
        "seq_aux_loss": seq_fields,
        "deepseek_bias": bias_fields,
        "quantile": quantile_fields,
        "none": set(),
    }
    method_summary = {
        "aux_loss": "aux_loss only uses router_aux_loss_coef.",
        "seq_aux_loss": "seq_aux_loss only uses seq_aux_loss_coef.",
        "deepseek_bias": "deepseek_bias only uses bias_update_* knobs.",
        "quantile": "quantile only uses quantile_* knobs.",
        "none": "the `none` method disables every balancing knob by definition.",
    }
    bal = nested.get("balancing")
    if bal in method_to_allowed:
        allowed = method_to_allowed[bal]
        for field in sorted(all_active - allowed):
            if field in nested and nested[field]:
                raise ValueError(
                    f"model.{group}.balancing={bal} is incompatible "
                    f"with {field}={nested[field]!r}; "
                    f"{method_summary[bal]} Drop the conflicting "
                    f"field or change `balancing`."
                )


def validate_branch_router_config(cfg: dict) -> None:
    """Validate the `model.branch_router` nested block (and the
    legacy flat-bridge fields) for unknown keys, illegal values, and
    conflicting nested-vs-flat assignments. Also validates the
    per-class `model.mlp_router` and `model.attn_router` nested
    blocks when present. Raises `ValueError` on the first issue
    found.

    Validation rules:
      * Unknown keys under `model.branch_router` are rejected with a
        list of accepted keys, so a typo'd field name does not silently
        revert to the constructor default.
      * `balancing` must be one of `{"none", "exploration_only"}`.
      * `exploration_decay` must be one of
        `{"constant", "linear", "cosine"}`.
      * `exploration_rate`, `exploration_min` must be in [0.0, 1.0].
      * `exploration_warmup_steps` must be a non-negative int.
      * If both nested and flat (`branch_<key>`) forms are present
        with conflicting values for the same field, raise. Equal
        values are accepted (the migrator can leave both during a
        partial migration).

    The validator is a pure check — it does not mutate `cfg`. The
    flat-bridge form is not rewritten into nested form here; that
    is the migrator's job.
    """
    mcfg = cfg.get("model")
    if not isinstance(mcfg, dict):
        return

    nested = mcfg.get("branch_router")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError(
                f"model.branch_router must be a mapping; got "
                f"{type(nested).__name__}"
            )
        unknown = set(nested.keys()) - _BRANCH_ROUTER_KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"model.branch_router has unknown keys: {sorted(unknown)}. "
                f"Allowed keys: {sorted(_BRANCH_ROUTER_KNOWN_KEYS)}."
            )

    def _resolve(key: str, default):
        if isinstance(nested, dict) and key in nested:
            return nested[key], "branch_router"
        flat_key = f"branch_{key}"
        if flat_key in mcfg:
            return mcfg[flat_key], flat_key
        return default, None

    balancing, src = _resolve("balancing", "none")
    if balancing not in _BRANCH_BALANCING_VALID:
        raise ValueError(
            f"model.{src}.balancing={balancing!r} is invalid; "
            f"must be one of {sorted(_BRANCH_BALANCING_VALID)}"
        )
    decay, src = _resolve("exploration_decay", "constant")
    if decay not in _BRANCH_DECAY_VALID:
        raise ValueError(
            f"model.{src}.exploration_decay={decay!r} is invalid; "
            f"must be one of {sorted(_BRANCH_DECAY_VALID)}"
        )
    rate, src = _resolve("exploration_rate", 0.0)
    if not (0.0 <= rate <= 1.0):
        raise ValueError(
            f"model.{src}.exploration_rate={rate} must be in [0.0, 1.0]"
        )
    floor, src = _resolve("exploration_min", 0.0)
    if not (0.0 <= floor <= 1.0):
        raise ValueError(
            f"model.{src}.exploration_min={floor} must be in [0.0, 1.0]"
        )
    if floor > rate:
        raise ValueError(
            f"model.branch_router.exploration_min ({floor}) exceeds "
            f"exploration_rate ({rate}); the floor is reached at "
            f"warmup_steps and so cannot exceed the initial rate."
        )
    warmup, src = _resolve("exploration_warmup_steps", 0)
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError(
            f"model.{src}.exploration_warmup_steps={warmup} must be "
            f"a non-negative int"
        )

    # Method-specific knob rejection on branch_router (mirrors the
    # check `_validate_mlp_or_attn_router` runs on MLP / attention
    # groups). For example, `balancing: aux_loss` with
    # `seq_aux_loss_coef: 0.5` is rejected because the runtime
    # ignores the non-active coefficient — silently allowing it
    # creates a misleading config where the operator thinks the
    # value is being used.
    if isinstance(nested, dict):
        bal = nested.get("balancing")
        # Branch router currently only supports `none`,
        # `exploration_only`, `aux_loss`, `seq_aux_loss` at
        # runtime; the validator already rejected anything outside
        # that set above. The active-knob rule per method:
        branch_method_to_allowed = {
            "aux_loss": {"router_aux_loss_coef"},
            "seq_aux_loss": {"seq_aux_loss_coef"},
            "deepseek_bias": {
                "bias_update_rate", "bias_update_zero_sum",
                "bias_warmup_start", "bias_warmup_steps",
            },
            # exploration_only consumes the exploration schedule
            # knobs; aux/seq coefs are not active here.
            "exploration_only": {
                "exploration_rate", "exploration_decay",
                "exploration_min", "exploration_warmup_steps",
            },
            # `none` accepts only schedule fields if the operator
            # leaves them in (no balancing knob is active here).
            "none": {
                "exploration_rate", "exploration_decay",
                "exploration_min", "exploration_warmup_steps",
            },
        }
        # Every active balancing knob the validator knows about,
        # used as the universe-of-discourse for the per-method
        # reject set.
        all_branch_active = {
            "router_aux_loss_coef", "seq_aux_loss_coef",
            "bias_update_rate", "bias_update_zero_sum",
            "bias_warmup_start", "bias_warmup_steps",
            "quantile_eta", "quantile_target_q",
            "quantile_global_state",
        }
        if bal in branch_method_to_allowed:
            allowed = branch_method_to_allowed[bal]
            for field in sorted(all_branch_active - allowed):
                if field in nested and nested[field]:
                    raise ValueError(
                        f"model.branch_router.balancing={bal} is "
                        f"incompatible with {field}={nested[field]!r}; "
                        f"the runtime ignores this knob under {bal}. "
                        f"Drop the conflicting field or change "
                        f"`balancing`."
                    )

    # Apply the same branch method×knob rejection to flat-bridge
    # form (`branch_<key>`). Some external configs and pre-migration
    # yamls still use the flat form; the validator must catch the
    # same incompatibilities there. Resolves the active method via
    # `_resolve(...)` (which already prefers nested but falls back
    # to flat) so a yaml carrying ONLY flat fields is still checked.
    bal_resolved, _ = _resolve("balancing", "none")
    if bal_resolved in {"aux_loss", "seq_aux_loss", "deepseek_bias", "exploration_only", "none"}:
        flat_branch_method_to_allowed = {
            "aux_loss": {"router_aux_loss_coef"},
            "seq_aux_loss": {"seq_aux_loss_coef"},
            "deepseek_bias": {
                "bias_update_rate", "bias_update_zero_sum",
                "bias_warmup_start", "bias_warmup_steps",
            },
            "exploration_only": {
                "exploration_rate", "exploration_decay",
                "exploration_min", "exploration_warmup_steps",
            },
            "none": {
                "exploration_rate", "exploration_decay",
                "exploration_min", "exploration_warmup_steps",
            },
        }
        flat_all_active = {
            "router_aux_loss_coef", "seq_aux_loss_coef",
            "bias_update_rate", "bias_update_zero_sum",
            "bias_warmup_start", "bias_warmup_steps",
            "quantile_eta", "quantile_target_q",
            "quantile_global_state",
        }
        flat_allowed = flat_branch_method_to_allowed[bal_resolved]
        for field in sorted(flat_all_active - flat_allowed):
            flat_field_name = f"branch_{field}"
            if flat_field_name in mcfg and mcfg[flat_field_name]:
                raise ValueError(
                    f"branch_router.balancing={bal_resolved!r} is "
                    f"incompatible with model.{flat_field_name}="
                    f"{mcfg[flat_field_name]!r}; the runtime ignores "
                    f"this knob under {bal_resolved}. Drop the "
                    f"conflicting field or change `balancing`."
                )

    # Reject conflicting nested-vs-flat assignments on the SAME field.
    if isinstance(nested, dict):
        for key in _BRANCH_ROUTER_KNOWN_KEYS:
            if key not in nested:
                continue
            flat_key = f"branch_{key}"
            if flat_key not in mcfg:
                continue
            if nested[key] != mcfg[flat_key]:
                raise ValueError(
                    f"model.branch_router.{key}={nested[key]!r} "
                    f"conflicts with model.{flat_key}={mcfg[flat_key]!r}. "
                    f"Both forms are present with different values; the "
                    f"nested form takes precedence at runtime, but a "
                    f"production yaml should not carry conflicting "
                    f"settings. Either drop the legacy flat field or "
                    f"reconcile the values."
                )

    # Per-class router groups (mlp_router, attn_router). Each is
    # validated independently so a typo or invalid value in one group
    # surfaces immediately. The flat-bridge form does not exist for
    # these two groups (they are nested-only since they are part of
    # the nested-schema deliverable, not a legacy migration).
    _validate_mlp_or_attn_router(cfg, "mlp_router")
    _validate_mlp_or_attn_router(cfg, "attn_router")


def normalize_balancing_config(cfg: dict) -> dict:
    """Apply AUTO-ZERO + warn for legacy coefficients when
    `load_balancing_method` is set.

    Mutates `cfg` in place AND returns it for fluent chaining. Reads
    `load_balancing_method` via `_resolve_balancing_field`. When the method is
    absent (no `load_balancing_method` in either block), the config is left
    alone (back-compat for yamls that pre-date the dispatch knob — the
    coefficient-driven runtime sees its original values). When the method is
    EXPLICITLY set — including `none` — every legacy coefficient outside the
    method's active set is auto-zeroed with a deprecation warning. `none`'s
    active set is empty, so it zeroes everything (Per the `none` method contract: `balancing: none`
    disables all balancing).

    Repository configs are stricter (rejection, via the future config
    validator); external/legacy configs go through AUTO-ZERO at runtime.
    """
    method = _resolve_balancing_field(cfg, "load_balancing_method", None)
    if method is None:
        return cfg
    if method not in _VALID_LOAD_BALANCING_METHODS:
        raise ValueError(
            f"load_balancing_method={method!r} is not a valid value. "
            f"Pick one of {sorted(_VALID_LOAD_BALANCING_METHODS)}."
        )

    active = _METHOD_ACTIVE_FIELDS[method]
    # Every balancing field outside the method's "active" set must be zero.
    # We zero in the canonical `training:` block.
    tcfg = cfg.setdefault("training", {})
    mcfg = cfg.get("model", {}) or {}
    conflicts: list[tuple[str, str, float]] = []
    # Mode-selector knobs are NOT coefficients; auto-zeroing them would
    # silently flip a behavior switch (e.g. bias_update_zero_sum=True →
    # False under load_balancing_method=aux_loss). They live in the
    # canonical block so they survive yaml migration, but the auto-zero
    # pass leaves them untouched — the runtime simply ignores them when
    # the method doesn't use them.
    _NON_COEFFICIENT_KNOBS = frozenset({"bias_update_zero_sum"})
    for field in _BALANCING_FIELDS_IN_TRAINING:
        if field == "load_balancing_method" or field in active or field in _NON_COEFFICIENT_KNOBS:
            continue
        # Look in both blocks (the resolver returns the first non-default
        # value, but for the auto-zero we need to know where the value LIVES
        # so we know where to overwrite it).
        for src_block_name, src_block in (("training", tcfg), ("model", mcfg)):
            if field in src_block:
                value = src_block[field]
                if value not in (0, 0.0, None):
                    conflicts.append((field, src_block_name, value))
                # Drop / zero the legacy coefficient so coefficient-driven
                # forward / trainer paths see zero.
                src_block[field] = 0.0 if isinstance(value, float) else 0
    if conflicts:
        msg_lines = [
            f"AUTO-ZERO: load_balancing_method={method!r} is incompatible with "
            f"the following non-zero coefficients (zeroed at runtime; please remove "
            f"them from the yaml to silence this warning):"
        ]
        for field, block, value in conflicts:
            msg_lines.append(f"  - {block}.{field} = {value!r}")
        warnings.warn("\n".join(msg_lines), DeprecationWarning, stacklevel=2)
    return cfg
