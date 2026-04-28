"""AC-1 + DEC-3a: `load_balancing_method` is the single dispatch knob; legacy
coefficients that conflict with it are AUTO-ZEROED with a deprecation warning.

Round 2 migrated every yaml to the canonical `training:` block per DEC-3b but
left `load_balancing_method` itself a no-op string. The knock-on was that the
migrated yamls now carry `load_balancing_method: aux_loss` AND non-zero
`seq_aux_loss_coef` AND non-zero `bias_update_rate` simultaneously — which
under coefficient-driven dispatch would activate three balancing methods at
once, the exact failure mode DEC-3a was created to prevent.

Round 3's `normalize_balancing_config` resolves the method once and zeros the
non-active legacy coefficients before either `build_training_config` or
`build_model` runs. These tests pin that contract.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

# Direct import of the dependency-light module (skips
# `src/training/__init__.py` which imports pandas).
import importlib.util
_bf_spec = importlib.util.spec_from_file_location(
    "_balancing_fields",
    Path(__file__).resolve().parent.parent / "src" / "training" / "balancing_fields.py",
)
_bf = importlib.util.module_from_spec(_bf_spec)
_bf_spec.loader.exec_module(_bf)
normalize_balancing_config = _bf.normalize_balancing_config
_VALID_LOAD_BALANCING_METHODS = _bf._VALID_LOAD_BALANCING_METHODS


def test_method_aux_loss_zeroes_seq_and_bias_coefs():
    """`load_balancing_method: aux_loss` keeps `router_aux_loss_coef` non-zero
    and AUTO-ZEROS `seq_aux_loss_coef` + `bias_update_rate` (with warning)."""
    cfg = {
        "model": {},
        "training": {
            "load_balancing_method": "aux_loss",
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)

    tcfg = cfg["training"]
    assert tcfg["router_aux_loss_coef"] == 0.001, "Switch aux must remain active"
    assert tcfg["seq_aux_loss_coef"] == 0.0, "seq aux must be zeroed under aux_loss method"
    assert tcfg["bias_update_rate"] == 0.0, "bias_update_rate must be zeroed under aux_loss method"
    assert any(
        issubclass(w.category, DeprecationWarning) and "AUTO-ZERO" in str(w.message)
        for w in caught
    ), f"Expected an AUTO-ZERO deprecation warning, got: {[str(w.message) for w in caught]}"


def test_method_deepseek_bias_zeroes_aux_coefs():
    """`load_balancing_method: deepseek_bias` keeps `bias_update_rate` non-zero
    and AUTO-ZEROS aux coefficients."""
    cfg = {
        "model": {},
        "training": {
            "load_balancing_method": "deepseek_bias",
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
            "bias_warmup_start": 0.0001,
            "bias_warmup_steps": 100,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)

    tcfg = cfg["training"]
    assert tcfg["router_aux_loss_coef"] == 0.0
    assert tcfg["seq_aux_loss_coef"] == 0.0
    assert tcfg["bias_update_rate"] == 0.001, "bias_update_rate must remain active"
    # Bias warmup parameters travel WITH bias_update_rate.
    assert tcfg["bias_warmup_start"] == 0.0001
    assert tcfg["bias_warmup_steps"] == 100
    assert any(
        issubclass(w.category, DeprecationWarning) and "AUTO-ZERO" in str(w.message)
        for w in caught
    )


def test_method_seq_aux_loss_zeroes_switch_and_bias():
    cfg = {
        "model": {},
        "training": {
            "load_balancing_method": "seq_aux_loss",
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)
    tcfg = cfg["training"]
    assert tcfg["router_aux_loss_coef"] == 0.0
    assert tcfg["seq_aux_loss_coef"] == 0.0001, "seq aux must remain active"
    assert tcfg["bias_update_rate"] == 0.0


def test_method_none_zeroes_everything():
    """Per AC-16: `load_balancing_method: none` disables ALL balancing —
    every legacy coefficient is AUTO-ZEROED with a warning. Round 3 had a
    bug where this test locked the OPPOSITE behavior (none was treated as
    a back-compat sentinel that left coefs alone); Round 4 corrects both
    the runtime AND this test.
    """
    cfg = {
        "model": {},
        "training": {
            "load_balancing_method": "none",
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)
    tcfg = cfg["training"]
    # `none` zeros every legacy coefficient — AC-16 contract.
    assert tcfg["router_aux_loss_coef"] == 0.0, "none must zero router_aux_loss_coef"
    assert tcfg["seq_aux_loss_coef"] == 0.0, "none must zero seq_aux_loss_coef"
    assert tcfg["bias_update_rate"] == 0.0, "none must zero bias_update_rate"
    assert any(
        issubclass(w.category, DeprecationWarning) and "AUTO-ZERO" in str(w.message)
        for w in caught
    ), f"Expected an AUTO-ZERO deprecation warning, got: {[str(w.message) for w in caught]}"


def test_method_absent_leaves_coefs_alone_back_compat():
    """When `load_balancing_method` is ABSENT (not even set to `none`), the
    normalizer is a no-op: legacy coefficient-driven yamls retain their
    original behavior. This is the back-compat path for yamls written
    before `load_balancing_method` existed.
    """
    cfg = {
        "model": {},
        "training": {
            # No load_balancing_method set — this is what an old yaml looks like.
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)
    tcfg = cfg["training"]
    assert tcfg["router_aux_loss_coef"] == 0.001
    assert tcfg["seq_aux_loss_coef"] == 0.0001
    assert tcfg["bias_update_rate"] == 0.001
    assert not any(
        issubclass(w.category, DeprecationWarning) and "AUTO-ZERO" in str(w.message)
        for w in caught
    )


def test_method_quantile_zeroes_aux_and_bias_for_now():
    """Quantile lands in a later round; when set, all coefficient-driven legacy
    paths AUTO-ZERO so the trainer doesn't accidentally run `aux_loss` while
    waiting for `_update_single_router_quantile_bias` to ship."""
    cfg = {
        "model": {},
        "training": {
            "load_balancing_method": "quantile",
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)
    tcfg = cfg["training"]
    assert tcfg["router_aux_loss_coef"] == 0.0
    assert tcfg["seq_aux_loss_coef"] == 0.0
    assert tcfg["bias_update_rate"] == 0.0


def test_no_method_set_leaves_coefs_alone():
    """Back-compat: yamls without a `load_balancing_method` (legacy) get the
    coefficient-driven behavior they always had."""
    cfg = {
        "model": {},
        "training": {
            # No `load_balancing_method` set.
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)
    # Nothing zeroed.
    tcfg = cfg["training"]
    assert tcfg["router_aux_loss_coef"] == 0.001
    assert tcfg["seq_aux_loss_coef"] == 0.0001
    assert tcfg["bias_update_rate"] == 0.001
    # No AUTO-ZERO warning (the only warnings allowed here are unrelated).
    assert not any(
        issubclass(w.category, DeprecationWarning) and "AUTO-ZERO" in str(w.message)
        for w in caught
    )


def test_invalid_method_raises():
    cfg = {
        "model": {},
        "training": {"load_balancing_method": "not_a_real_method"},
    }
    try:
        normalize_balancing_config(cfg)
    except ValueError as e:
        assert "load_balancing_method" in str(e)
    else:
        raise AssertionError("Expected ValueError on invalid method")


def test_method_in_legacy_model_block_warns_then_normalizes():
    """If `load_balancing_method` is in the legacy `model:` block, the resolver
    finds it (with deprecation warning), and the rest of normalize still runs."""
    cfg = {
        "model": {
            "load_balancing_method": "aux_loss",  # legacy placement
            "bias_update_rate": 0.001,            # legacy placement
        },
        "training": {
            "router_aux_loss_coef": 0.001,
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_balancing_config(cfg)
    # `bias_update_rate` zeroed in `model:` (where it actually lives);
    # `router_aux_loss_coef` left alone in `training:` (active for aux_loss).
    assert cfg["model"]["bias_update_rate"] == 0.0
    assert cfg["training"]["router_aux_loss_coef"] == 0.001
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    # Two warnings expected: one for legacy `model:` placement of
    # `load_balancing_method`, one for the AUTO-ZERO of `bias_update_rate`.
    assert any("deprecated" in m and "model:" in m for m in deprecation_msgs)
    assert any("AUTO-ZERO" in m for m in deprecation_msgs)


def test_valid_methods_set_complete():
    """Lock the canonical method enum so future additions are explicit."""
    assert set(_VALID_LOAD_BALANCING_METHODS) == {
        "aux_loss", "seq_aux_loss", "deepseek_bias", "quantile", "none",
    }


if __name__ == "__main__":
    test_method_aux_loss_zeroes_seq_and_bias_coefs()
    test_method_deepseek_bias_zeroes_aux_coefs()
    test_method_seq_aux_loss_zeroes_switch_and_bias()
    test_method_none_zeroes_everything()
    test_method_absent_leaves_coefs_alone_back_compat()
    test_method_quantile_zeroes_aux_and_bias_for_now()
    test_no_method_set_leaves_coefs_alone()
    test_invalid_method_raises()
    test_method_in_legacy_model_block_warns_then_normalizes()
    test_valid_methods_set_complete()
    print("ALL OK")
