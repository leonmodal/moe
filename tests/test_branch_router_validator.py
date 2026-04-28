"""Tests for `validate_branch_router_config(cfg)` — the AC-13/17
nested-schema validator that catches typos, illegal values, and
conflicting nested-vs-flat assignments at config-load time. The
validator runs as part of `load_config(...)` so misconfigured
yamls fail fast instead of silently using constructor defaults.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_balancing_fields_module():
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
        import types as _types
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = training_pkg
    spec = importlib.util.spec_from_file_location(
        "src.training.balancing_fields",
        str(repo / "src" / "training" / "balancing_fields.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.training.balancing_fields"] = mod
    spec.loader.exec_module(mod)
    return mod


def _validator():
    return _load_balancing_fields_module().validate_branch_router_config


def test_validator_accepts_nested_form_with_known_keys():
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_router": {
                "balancing": "exploration_only",
                "exploration_rate": 0.5,
                "exploration_decay": "linear",
                "exploration_min": 0.0,
                "exploration_warmup_steps": 100,
            },
        }
    }
    _validator()(cfg)


def test_validator_accepts_flat_form():
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_balancing": "exploration_only",
            "branch_exploration_rate": 0.7,
            "branch_exploration_decay": "cosine",
            "branch_exploration_min": 0.05,
            "branch_exploration_warmup_steps": 500,
        }
    }
    _validator()(cfg)


def test_validator_accepts_default_unset():
    """A config that never mentions branch_router or branch_* fields
    is valid; everything defaults to `none` / `0.0` / `constant`."""
    cfg = {"model": {"type": "moe_everything"}}
    _validator()(cfg)


def test_validator_accepts_no_model_block():
    """Configs without a `model:` block (e.g. a partial config used
    only for a training-side check) are accepted as no-op."""
    cfg = {"training": {"learning_rate": 1e-4}}
    _validator()(cfg)


def test_validator_rejects_unknown_key_under_branch_router():
    cfg = {
        "model": {
            "branch_router": {
                "balancing": "exploration_only",
                "exploration_warmpu_steps": 100,  # typo
            },
        }
    }
    with pytest.raises(ValueError, match="unknown keys"):
        _validator()(cfg)


def test_validator_rejects_invalid_balancing_value():
    cfg = {"model": {"branch_router": {"balancing": "explore"}}}
    with pytest.raises(ValueError, match="balancing="):
        _validator()(cfg)


def test_validator_rejects_invalid_decay_shape():
    cfg = {
        "model": {
            "branch_router": {"balancing": "exploration_only", "exploration_decay": "exp"},
        }
    }
    with pytest.raises(ValueError, match="exploration_decay="):
        _validator()(cfg)


def test_validator_rejects_rate_out_of_range():
    cfg = {"model": {"branch_router": {"exploration_rate": 1.5}}}
    with pytest.raises(ValueError, match="exploration_rate="):
        _validator()(cfg)
    cfg = {"model": {"branch_router": {"exploration_rate": -0.1}}}
    with pytest.raises(ValueError, match="exploration_rate="):
        _validator()(cfg)


def test_validator_rejects_min_above_rate():
    cfg = {
        "model": {
            "branch_router": {
                "exploration_rate": 0.3,
                "exploration_min": 0.5,
            }
        }
    }
    with pytest.raises(ValueError, match="exploration_min .* exceeds"):
        _validator()(cfg)


def test_validator_rejects_negative_warmup_steps():
    cfg = {"model": {"branch_router": {"exploration_warmup_steps": -1}}}
    with pytest.raises(ValueError, match="exploration_warmup_steps="):
        _validator()(cfg)


def test_validator_rejects_warmup_steps_non_int():
    cfg = {"model": {"branch_router": {"exploration_warmup_steps": 100.5}}}
    with pytest.raises(ValueError, match="exploration_warmup_steps="):
        _validator()(cfg)


def test_validator_rejects_conflicting_nested_and_flat():
    cfg = {
        "model": {
            "branch_router": {"balancing": "exploration_only"},
            "branch_balancing": "none",
        }
    }
    with pytest.raises(ValueError, match="conflicts with"):
        _validator()(cfg)


def test_validator_accepts_redundant_nested_and_flat_with_equal_values():
    """The migrator can leave both forms during a partial migration,
    as long as values agree."""
    cfg = {
        "model": {
            "branch_router": {"balancing": "exploration_only", "exploration_rate": 0.5},
            "branch_balancing": "exploration_only",
            "branch_exploration_rate": 0.5,
        }
    }
    _validator()(cfg)


def test_validator_rejects_branch_router_not_a_mapping():
    cfg = {"model": {"branch_router": "exploration_only"}}
    with pytest.raises(ValueError, match="must be a mapping"):
        _validator()(cfg)


def test_validator_accepts_mlp_router_aux_loss():
    cfg = {
        "model": {
            "mlp_router": {
                "balancing": "aux_loss",
                "router_aux_loss_coef": 0.001,
                "seq_aux_loss_coef": 0.0,
            }
        }
    }
    _validator()(cfg)


def test_validator_accepts_attn_router_with_specific_knob():
    cfg = {
        "model": {
            "attn_router": {
                "balancing": "deepseek_bias",
                "scale_by_routing_weight": True,
            }
        }
    }
    _validator()(cfg)


def test_validator_rejects_mlp_router_unknown_key():
    cfg = {"model": {"mlp_router": {"baalancing": "aux_loss"}}}
    with pytest.raises(ValueError, match="model.mlp_router has unknown keys"):
        _validator()(cfg)


def test_validator_rejects_attn_router_unknown_key():
    cfg = {"model": {"attn_router": {"top_p": 0.9}}}
    with pytest.raises(ValueError, match="model.attn_router has unknown keys"):
        _validator()(cfg)


def test_validator_rejects_mlp_router_invalid_balancing():
    cfg = {"model": {"mlp_router": {"balancing": "explore"}}}
    with pytest.raises(ValueError, match="model.mlp_router.balancing="):
        _validator()(cfg)


def test_validator_rejects_attn_router_invalid_balancing():
    cfg = {"model": {"attn_router": {"balancing": "freeform"}}}
    with pytest.raises(ValueError, match="model.attn_router.balancing="):
        _validator()(cfg)


def test_validator_rejects_negative_aux_coef():
    cfg = {"model": {"mlp_router": {"router_aux_loss_coef": -0.1}}}
    with pytest.raises(ValueError, match="router_aux_loss_coef="):
        _validator()(cfg)


def test_validator_rejects_negative_seq_aux_coef():
    cfg = {"model": {"attn_router": {"seq_aux_loss_coef": -0.5}}}
    with pytest.raises(ValueError, match="seq_aux_loss_coef="):
        _validator()(cfg)


def test_validator_accepts_all_three_groups_together():
    """A fully nested-schema yaml lists all three router groups; the
    validator must accept the combination.
    """
    cfg = {
        "model": {
            "branch_router": {
                "balancing": "exploration_only",
                "exploration_rate": 0.3,
                "exploration_decay": "linear",
                "exploration_warmup_steps": 1000,
                "exploration_min": 0.0,
            },
            "mlp_router": {
                "balancing": "aux_loss",
                "router_aux_loss_coef": 0.001,
                "seq_aux_loss_coef": 0.0,
            },
            "attn_router": {
                "balancing": "deepseek_bias",
                "scale_by_routing_weight": True,
            },
        }
    }
    _validator()(cfg)


def test_validator_rejects_mlp_router_min_above_rate():
    cfg = {
        "model": {
            "mlp_router": {"exploration_rate": 0.2, "exploration_min": 0.5}
        }
    }
    with pytest.raises(ValueError, match="exploration_min .* exceeds"):
        _validator()(cfg)


def test_validator_rejects_mlp_router_invalid_decay():
    cfg = {"model": {"mlp_router": {"exploration_decay": "exp"}}}
    with pytest.raises(ValueError, match="exploration_decay="):
        _validator()(cfg)


if __name__ == "__main__":
    test_validator_accepts_nested_form_with_known_keys()
    test_validator_accepts_flat_form()
    test_validator_accepts_default_unset()
    test_validator_accepts_no_model_block()
    test_validator_rejects_unknown_key_under_branch_router()
    test_validator_rejects_invalid_balancing_value()
    test_validator_rejects_invalid_decay_shape()
    test_validator_rejects_rate_out_of_range()
    test_validator_rejects_min_above_rate()
    test_validator_rejects_negative_warmup_steps()
    test_validator_rejects_warmup_steps_non_int()
    test_validator_rejects_conflicting_nested_and_flat()
    test_validator_accepts_redundant_nested_and_flat_with_equal_values()
    test_validator_rejects_branch_router_not_a_mapping()
    print("ALL OK")
