"""AC-3: balancing fields live under `training:` per DEC-3b.

Pre-migration, every yaml put `bias_update_rate`, `seq_aux_loss_coef`,
`router_aux_loss_coef`, and `load_balancing_method` under `model:`, but
`TrainingConfig` reads `bias_update_rate` (and friends) from `cfg["training"]`.
The result was a silent zero — the trainer never actually exercised the
deepseek bias-update path no matter what the yaml said.

These tests pin the post-migration contract:

  1. Every shipped yaml under `configs/` has its balancing fields in `training:`.
  2. `_resolve_balancing_field` returns the `training:` value.
  3. A yaml with the field still under `model:` (unmigrated) emits a
     `DeprecationWarning` and the field still resolves to its `model:` value
     (so the runtime is forgiving).
  4. A yaml with the field in BOTH blocks emits the warning and uses the
     `training:` value (canonical).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, ".")

# Import the resolver directly from its dependency-light module.
# `src/training/__init__.py` would transitively pull pandas (not available in
# every test env), so we go directly to `balancing_fields.py`.
import importlib.util
_bf_spec = importlib.util.spec_from_file_location(
    "_balancing_fields",
    Path(__file__).resolve().parent.parent / "src" / "training" / "balancing_fields.py",
)
_bf = importlib.util.module_from_spec(_bf_spec)
_bf_spec.loader.exec_module(_bf)
_BALANCING_FIELDS_IN_TRAINING = _bf._BALANCING_FIELDS_IN_TRAINING
_resolve_balancing_field = _bf._resolve_balancing_field


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _REPO_ROOT / "configs"


def _list_yamls() -> list[Path]:
    return sorted(_CONFIGS_DIR.rglob("*.yaml"))


def test_no_balancing_fields_left_under_model_block():
    """AC-3 positive: every shipped yaml has balancing fields in `training:`,
    not `model:`. Run the migrator if this fails."""
    leftovers: list[tuple[str, str]] = []
    for yaml_file in _list_yamls():
        with yaml_file.open() as f:
            cfg = yaml.safe_load(f) or {}
        mcfg = cfg.get("model", {}) or {}
        for field in _BALANCING_FIELDS_IN_TRAINING:
            if field in mcfg:
                leftovers.append((str(yaml_file.relative_to(_REPO_ROOT)), field))
    assert leftovers == [], (
        f"Found {len(leftovers)} balancing field(s) still under `model:`. "
        f"Run `python scripts/migrate_balancing_fields_to_training.py` to migrate. "
        f"Leftovers: {leftovers}"
    )


def test_resolver_reads_training_block_canonically():
    """`_resolve_balancing_field` returns the `training:` value when present."""
    cfg = {
        "model": {},
        "training": {"bias_update_rate": 0.001, "load_balancing_method": "deepseek_bias"},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rate = _resolve_balancing_field(cfg, "bias_update_rate", 0.0)
        method = _resolve_balancing_field(cfg, "load_balancing_method", None)
    assert rate == 0.001
    assert method == "deepseek_bias"
    # No warning when the field is canonically placed.
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"Resolver emitted a warning for canonical placement: {[str(w.message) for w in caught]}"
    )


def test_resolver_warns_when_field_still_under_model_block():
    """Unmigrated config: field under `model:` only — resolver returns the
    value AND emits a `DeprecationWarning` instructing the user to migrate."""
    cfg = {
        "model": {"bias_update_rate": 0.002},  # legacy placement
        "training": {},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rate = _resolve_balancing_field(cfg, "bias_update_rate", 0.0)
    assert rate == 0.002
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "DEC-3b" in str(w.message)
        for w in caught
    ), f"Expected DEC-3b deprecation warning, got: {[str(w.message) for w in caught]}"


def test_resolver_prefers_training_block_when_field_in_both():
    """Conflict: field in both blocks. Canonical `training:` value wins; warning emitted."""
    cfg = {
        "model": {"bias_update_rate": 0.999},     # legacy / accidental
        "training": {"bias_update_rate": 0.001},  # canonical
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rate = _resolve_balancing_field(cfg, "bias_update_rate", 0.0)
    assert rate == 0.001, "training: value must win when conflicts"
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "BOTH" in str(w.message)
        for w in caught
    ), f"Expected BOTH-blocks deprecation warning, got: {[str(w.message) for w in caught]}"


def test_resolver_returns_default_when_field_absent():
    """Absent in both blocks: resolver returns the default with no warning."""
    cfg = {"model": {}, "training": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rate = _resolve_balancing_field(cfg, "bias_update_rate", 0.42)
    assert rate == 0.42
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def _load_build_training_config():
    """Load `src.training.config.build_training_config` without going through
    `src.training.__init__` (which imports pandas).

    Strategy: register `_balancing_fields` (already loaded above) under the
    canonical module name `src.training.balancing_fields` so the
    `from .balancing_fields import ...` line in `config.py` resolves; then
    load `config.py` with package context `src.training`.
    """
    import importlib.util
    import sys as _sys
    import types as _types

    # Build a minimal package shell for `src.training` so relative imports work.
    if "src" not in _sys.modules:
        src_pkg = _types.ModuleType("src")
        src_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "src")]
        _sys.modules["src"] = src_pkg
    if "src.training" not in _sys.modules:
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [
            str(Path(__file__).resolve().parent.parent / "src" / "training")
        ]
        _sys.modules["src.training"] = training_pkg
    _sys.modules["src.training.balancing_fields"] = _bf

    config_spec = importlib.util.spec_from_file_location(
        "src.training.config",
        Path(__file__).resolve().parent.parent / "src" / "training" / "config.py",
    )
    config_module = importlib.util.module_from_spec(config_spec)
    _sys.modules["src.training.config"] = config_module
    config_spec.loader.exec_module(config_module)
    return config_module.build_training_config


def test_build_training_config_resolves_legacy_model_block():
    """`build_training_config` must use the resolver for `bias_update_rate`,
    `bias_warmup_start`, and `bias_warmup_steps` so that an unmigrated yaml
    with these fields under `model:` still produces the correct effective
    rate (with a deprecation warning) instead of silently zero-ing them.

    This was the production-trainer regression Codex's Round 2 review
    specifically called out: Round 2 added the resolver in `model_factory.py`
    but `TrainingConfig.from_dict()` still read directly from `cfg["training"]`,
    so a legacy `model.bias_update_rate: 0.001` yaml still became
    `train_cfg.bias_update_rate == 0.0` and the trainer's
    `if train_cfg.bias_update_rate > 0` gate stayed off.
    """
    build_training_config = _load_build_training_config()

    legacy_cfg = {
        "model": {
            # Legacy placement — should still work via the resolver, with a warning.
            "bias_update_rate": 0.001,
            "bias_warmup_start": 0.0001,
            "bias_warmup_steps": 100,
        },
        "training": {},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_cfg = build_training_config(legacy_cfg)

    assert train_cfg.bias_update_rate == 0.001, (
        f"build_training_config failed to resolve legacy `model.bias_update_rate`; "
        f"got {train_cfg.bias_update_rate} instead of 0.001 — the production-trainer "
        f"regression is back."
    )
    assert train_cfg.bias_warmup_start == 0.0001
    assert train_cfg.bias_warmup_steps == 100
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("DEC-3b" in m for m in deprecation_msgs), (
        f"Expected at least one DEC-3b deprecation warning from build_training_config, "
        f"got: {deprecation_msgs}"
    )


def test_build_training_config_canonical_training_block_quiet():
    """The canonical case: fields in `training:` produce the right
    `TrainingConfig` and no deprecation warning."""
    build_training_config = _load_build_training_config()

    canonical_cfg = {
        "model": {},
        "training": {
            "bias_update_rate": 0.002,
            "bias_warmup_start": 0.0002,
            "bias_warmup_steps": 200,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_cfg = build_training_config(canonical_cfg)

    assert train_cfg.bias_update_rate == 0.002
    assert train_cfg.bias_warmup_start == 0.0002
    assert train_cfg.bias_warmup_steps == 200
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_msgs == [], (
        f"Canonical-block placement should NOT emit a deprecation warning, "
        f"got: {deprecation_msgs}"
    )


# ──────────────────────────────────────────────────────────────────────
#  Round 12: DEC-2 `bias_update_zero_sum` config plumbing.
#  (Codex Round 11 Finding 1b)
# ──────────────────────────────────────────────────────────────────────


def test_build_training_config_bias_update_zero_sum_default_true():
    """DEC-2: the default value of `bias_update_zero_sum` is `True`
    (matches nmoe / DeepSeek-V3 zero-sum). When neither block sets the
    field, `build_training_config` resolves to `True`."""
    build_training_config = _load_build_training_config()
    cfg = {"model": {}, "training": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_cfg = build_training_config(cfg)
    assert train_cfg.bias_update_zero_sum is True
    # No deprecation warning for the default-absent path.
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_msgs == []


def test_build_training_config_bias_update_zero_sum_explicit_false():
    """DEC-2: `bias_update_zero_sum: False` in `training:` flows through
    to `train_cfg.bias_update_zero_sum`. This is the path the trainer
    consumes to switch to the Megatron-LM plain-sign update mode."""
    build_training_config = _load_build_training_config()
    cfg = {
        "model": {},
        "training": {"bias_update_zero_sum": False},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_cfg = build_training_config(cfg)
    assert train_cfg.bias_update_zero_sum is False
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_msgs == []


def test_build_training_config_bias_update_zero_sum_legacy_model_block_warns():
    """DEC-3b: `bias_update_zero_sum` is a balancing field, so the
    DEC-3b canonical-block resolver applies. A yaml with the field
    under `model:` still resolves correctly via the resolver fallback,
    but emits a `DeprecationWarning` pointing the user to the
    migration."""
    build_training_config = _load_build_training_config()
    cfg = {
        "model": {"bias_update_zero_sum": False},  # legacy placement
        "training": {},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_cfg = build_training_config(cfg)
    assert train_cfg.bias_update_zero_sum is False
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("bias_update_zero_sum" in m for m in deprecation_msgs), (
        f"Expected a DEC-3b DeprecationWarning for legacy "
        f"`model.bias_update_zero_sum`; got: {deprecation_msgs}"
    )


def test_build_training_config_bias_update_zero_sum_both_blocks_uses_canonical():
    """DEC-3b conflict policy: when set in BOTH blocks, the canonical
    `training:` value wins and a `DeprecationWarning` fires. Locks the
    `False`-overrides-`True` direction so a misplaced legacy `True`
    doesn't silently re-enable zero-sum mode for a user who explicitly
    asked for Megatron-style updates."""
    build_training_config = _load_build_training_config()
    cfg = {
        "model": {"bias_update_zero_sum": True},   # legacy
        "training": {"bias_update_zero_sum": False},  # canonical wins
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_cfg = build_training_config(cfg)
    assert train_cfg.bias_update_zero_sum is False, (
        "Canonical `training:` value must override legacy `model:` value."
    )
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any(
        "bias_update_zero_sum" in m and "BOTH" in m for m in deprecation_msgs
    ), (
        f"Expected a 'both blocks' DeprecationWarning; got: {deprecation_msgs}"
    )


if __name__ == "__main__":
    test_no_balancing_fields_left_under_model_block()
    test_resolver_reads_training_block_canonically()
    test_resolver_warns_when_field_still_under_model_block()
    test_resolver_prefers_training_block_when_field_in_both()
    test_resolver_returns_default_when_field_absent()
    test_build_training_config_resolves_legacy_model_block()
    test_build_training_config_canonical_training_block_quiet()
    print("ALL OK")
