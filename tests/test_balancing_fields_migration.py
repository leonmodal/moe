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

# Import the resolver directly from its module file to avoid pulling
# `src/training/__init__.py` (which imports data utilities that depend on
# pandas / liger and aren't relevant to this lightweight migration test).
import importlib.util
_factory_spec = importlib.util.spec_from_file_location(
    "_model_factory",
    Path(__file__).resolve().parent.parent / "src" / "training" / "model_factory.py",
)
_factory = importlib.util.module_from_spec(_factory_spec)
_factory_spec.loader.exec_module(_factory)
_BALANCING_FIELDS_IN_TRAINING = _factory._BALANCING_FIELDS_IN_TRAINING
_resolve_balancing_field = _factory._resolve_balancing_field


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


if __name__ == "__main__":
    test_no_balancing_fields_left_under_model_block()
    test_resolver_reads_training_block_canonically()
    test_resolver_warns_when_field_still_under_model_block()
    test_resolver_prefers_training_block_when_field_in_both()
    test_resolver_returns_default_when_field_absent()
    print("ALL OK")
