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


# Fields that MUST live under `cfg["training"]` per the canonical-block resolver rule.
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
                f"using the `training:` value (canonical per the canonical-block resolver rule). Remove the "
                f"`model:` copy to silence this warning.",
                DeprecationWarning, stacklevel=3,
            )
        return tcfg[name]
    if name in mcfg:
        warnings.warn(
            f"Config field {name!r} found under `model:` — the canonical-block resolver rule moved it to "
            f"`training:`. The `model:` placement is deprecated; run "
            f"`python scripts/migrate_balancing_fields_to_training.py` to migrate.",
            DeprecationWarning, stacklevel=3,
        )
        return mcfg[name]
    return default


# the load-balancing-method gating rule / the canonical-block coefficient normalizer: which legacy coefficients are kept active under each
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
# the DETACH-ONLY policy: methods whose loss term needs
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

    the load-balancing-method coefficient normalizer: repository configs are stricter
    (rejection, via the future config validator); external/legacy configs go
    through AUTO-ZERO at runtime.
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
