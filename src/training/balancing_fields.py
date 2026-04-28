"""Canonical-block resolution for DEC-3b balancing fields.

Round 2 introduced this resolver in `model_factory.py` so `build_model` could
read aux coefficients from the canonical `training:` block while warning on
legacy `model:` placement. Round 3 promotes it to its own module so both
`config.py` (`build_training_config`) and `model_factory.py` (`build_model`)
can import it without circular-init pain — and so tests can import the
helper without dragging the rest of `src/training/__init__.py` into scope
(that path transitively imports pandas / liger).

Per `docs/plan.md` DEC-3b (RESOLVED 2026-04-27 → AC-3): every balancing-
related field lives under `cfg["training"]`, NOT `cfg["model"]`. Today's
yamls have already been migrated by `scripts/migrate_balancing_fields_to_training.py`.
The resolver below tolerates unmigrated yamls (warning + reads the legacy
location) so external configs remain bootable for one release.
"""
from __future__ import annotations

import warnings
from typing import Any


# Fields that MUST live under `cfg["training"]` per DEC-3b.
_BALANCING_FIELDS_IN_TRAINING: tuple[str, ...] = (
    "router_aux_loss_coef",
    "seq_aux_loss_coef",
    "bias_update_rate",
    "bias_warmup_start",
    "bias_warmup_steps",
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
                f"using the `training:` value (canonical per DEC-3b). Remove the "
                f"`model:` copy to silence this warning.",
                DeprecationWarning, stacklevel=3,
            )
        return tcfg[name]
    if name in mcfg:
        warnings.warn(
            f"Config field {name!r} found under `model:` — DEC-3b moved it to "
            f"`training:`. The `model:` placement is deprecated; run "
            f"`python scripts/migrate_balancing_fields_to_training.py` to migrate.",
            DeprecationWarning, stacklevel=3,
        )
        return mcfg[name]
    return default


# AC-1 / DEC-3a: which legacy coefficients are kept active under each
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
    """Apply DEC-3a AUTO-ZERO + warn for legacy coefficients when
    `load_balancing_method` is set.

    Mutates `cfg` in place AND returns it for fluent chaining. Reads
    `load_balancing_method` via `_resolve_balancing_field`. When the method is
    absent (no `load_balancing_method` in either block), the config is left
    alone (back-compat for yamls that pre-date the dispatch knob — the
    coefficient-driven runtime sees its original values). When the method is
    EXPLICITLY set — including `none` — every legacy coefficient outside the
    method's active set is auto-zeroed with a deprecation warning. `none`'s
    active set is empty, so it zeroes everything (AC-16: `balancing: none`
    disables all balancing).

    Per DEC-3a (RESOLVED 2026-04-27 → AC-1): repository configs are stricter
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
    for field in _BALANCING_FIELDS_IN_TRAINING:
        if field == "load_balancing_method" or field in active:
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
            f"DEC-3a AUTO-ZERO: load_balancing_method={method!r} is incompatible with "
            f"the following non-zero coefficients (zeroed at runtime; please remove "
            f"them from the yaml to silence this warning):"
        ]
        for field, block, value in conflicts:
            msg_lines.append(f"  - {block}.{field} = {value!r}")
        warnings.warn("\n".join(msg_lines), DeprecationWarning, stacklevel=2)
    return cfg
