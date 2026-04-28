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


def test_validator_branch_router_accepts_aux_loss():
    """Round 32 review Finding 4: BranchRouter now accepts
    `aux_loss` and `seq_aux_loss`. The model's forward path
    computes the aux contribution from branch probabilities /
    selected indices via the same loss helpers used for MLP and
    attention routers."""
    cfg = {"model": {"branch_router": {"balancing": "aux_loss",
                                       "router_aux_loss_coef": 0.001}}}
    _validator()(cfg)


def test_validator_branch_router_accepts_seq_aux_loss():
    cfg = {"model": {"branch_router": {"balancing": "seq_aux_loss",
                                       "seq_aux_loss_coef": 0.0001}}}
    _validator()(cfg)


def test_validator_branch_router_rejects_deepseek_bias_until_runtime_lands():
    """deepseek_bias still requires owner-state plumbing
    (per-router `expert_bias` + accumulator buffers + walker
    dispatch); rejected at validator level so failures surface
    at config-load time rather than inside the build_model path.
    """
    cfg = {"model": {"branch_router": {"balancing": "deepseek_bias"}}}
    with pytest.raises(ValueError, match="branch_router.balancing="):
        _validator()(cfg)


def test_validator_branch_router_rejects_quantile():
    cfg = {"model": {"branch_router": {"balancing": "quantile"}}}
    with pytest.raises(ValueError, match="branch_router.balancing="):
        _validator()(cfg)


def test_validator_mlp_router_accepts_quantile_with_knobs():
    cfg = {
        "model": {
            "mlp_router": {
                "balancing": "quantile",
                "quantile_eta": 0.005,
                "quantile_target_q": 0.5,
                "quantile_global_state": True,
            }
        }
    }
    _validator()(cfg)


def test_validator_attn_router_accepts_quantile_with_knobs():
    cfg = {
        "model": {
            "attn_router": {
                "balancing": "quantile",
                "quantile_eta": 0.005,
                "quantile_target_q": 0.5,
                "quantile_global_state": True,
            }
        }
    }
    _validator()(cfg)


def test_validator_mlp_attn_router_groups_accept_deepseek_bias_knobs():
    """The mlp_router and attn_router groups accept the DeepSeek-bias
    update knobs (per-class). branch_router is restricted to {none,
    exploration_only} until the BranchRouter runtime supports the
    broader method set, so the deepseek_bias method on branch_router
    is rejected separately."""
    cfg = {
        "model": {
            "mlp_router": {
                "balancing": "deepseek_bias",
                "bias_update_rate": 0.001,
                "bias_update_zero_sum": True,
            },
            "attn_router": {
                "balancing": "deepseek_bias",
                "bias_update_rate": 0.001,
                "scale_by_routing_weight": True,
            },
        }
    }
    _validator()(cfg)


def test_validator_rejects_aux_loss_with_bias_update():
    """AC-17: aux_loss method must not also set bias_update_rate."""
    cfg = {"model": {"mlp_router": {
        "balancing": "aux_loss",
        "router_aux_loss_coef": 0.001,
        "bias_update_rate": 0.001,
    }}}
    with pytest.raises(ValueError, match="aux_loss is incompatible with bias_update_rate"):
        _validator()(cfg)


def test_validator_rejects_aux_loss_with_seq_aux_coef():
    cfg = {"model": {"mlp_router": {
        "balancing": "aux_loss",
        "seq_aux_loss_coef": 0.0001,
    }}}
    with pytest.raises(ValueError, match="aux_loss is incompatible with seq_aux_loss_coef"):
        _validator()(cfg)


def test_validator_rejects_aux_loss_with_quantile_knobs():
    cfg = {"model": {"mlp_router": {
        "balancing": "aux_loss",
        "quantile_eta": 0.005,
    }}}
    with pytest.raises(ValueError, match="aux_loss is incompatible with quantile_eta"):
        _validator()(cfg)


def test_validator_rejects_seq_aux_with_aux_coef():
    cfg = {"model": {"mlp_router": {
        "balancing": "seq_aux_loss",
        "router_aux_loss_coef": 0.001,
    }}}
    with pytest.raises(ValueError, match="seq_aux_loss is incompatible with router_aux_loss_coef"):
        _validator()(cfg)


def test_validator_rejects_deepseek_bias_with_aux_coef():
    cfg = {"model": {"mlp_router": {
        "balancing": "deepseek_bias",
        "bias_update_rate": 0.001,
        "router_aux_loss_coef": 0.001,
    }}}
    with pytest.raises(ValueError, match="deepseek_bias is incompatible with router_aux_loss_coef"):
        _validator()(cfg)


def test_validator_rejects_deepseek_bias_with_quantile_knobs():
    cfg = {"model": {"attn_router": {
        "balancing": "deepseek_bias",
        "quantile_eta": 0.005,
    }}}
    with pytest.raises(ValueError, match="deepseek_bias is incompatible with quantile_eta"):
        _validator()(cfg)


def test_validator_rejects_quantile_with_aux_coef():
    cfg = {"model": {"mlp_router": {
        "balancing": "quantile",
        "quantile_eta": 0.005,
        "router_aux_loss_coef": 0.001,
    }}}
    with pytest.raises(ValueError, match="quantile is incompatible with router_aux_loss_coef"):
        _validator()(cfg)


def test_validator_rejects_none_with_any_active_knob():
    """`balancing: none` disables every balancing path; specifying
    a coefficient or update knob is rejected."""
    cfg = {"model": {"mlp_router": {
        "balancing": "none",
        "router_aux_loss_coef": 0.001,
    }}}
    with pytest.raises(ValueError, match="none is incompatible with router_aux_loss_coef"):
        _validator()(cfg)
    cfg = {"model": {"attn_router": {
        "balancing": "none",
        "bias_update_rate": 0.001,
    }}}
    with pytest.raises(ValueError, match="none is incompatible with bias_update_rate"):
        _validator()(cfg)


def test_validator_rejects_aux_loss_with_bias_update_zero_sum():
    """Round 28 review Finding 3: bias_update_zero_sum is a bias-method
    knob; aux_loss must reject it."""
    cfg = {"model": {"mlp_router": {
        "balancing": "aux_loss",
        "router_aux_loss_coef": 0.001,
        "bias_update_zero_sum": True,
    }}}
    with pytest.raises(ValueError, match="aux_loss is incompatible with bias_update_zero_sum"):
        _validator()(cfg)


def test_validator_rejects_aux_loss_with_bias_warmup_steps():
    cfg = {"model": {"mlp_router": {
        "balancing": "aux_loss",
        "router_aux_loss_coef": 0.001,
        "bias_warmup_steps": 100,
    }}}
    with pytest.raises(ValueError, match="aux_loss is incompatible with bias_warmup_steps"):
        _validator()(cfg)


def test_validator_rejects_aux_loss_with_bias_warmup_start():
    cfg = {"model": {"mlp_router": {
        "balancing": "aux_loss",
        "router_aux_loss_coef": 0.001,
        "bias_warmup_start": 0.0001,
    }}}
    with pytest.raises(ValueError, match="aux_loss is incompatible with bias_warmup_start"):
        _validator()(cfg)


def test_validator_rejects_seq_aux_with_bias_warmup_steps():
    cfg = {"model": {"mlp_router": {
        "balancing": "seq_aux_loss",
        "seq_aux_loss_coef": 0.0001,
        "bias_warmup_steps": 100,
    }}}
    with pytest.raises(ValueError, match="seq_aux_loss is incompatible with bias_warmup_steps"):
        _validator()(cfg)


def test_validator_rejects_none_with_bias_update_zero_sum():
    cfg = {"model": {"mlp_router": {
        "balancing": "none",
        "bias_update_zero_sum": True,
    }}}
    with pytest.raises(ValueError, match="none is incompatible with bias_update_zero_sum"):
        _validator()(cfg)


def test_validator_rejects_none_with_quantile_global_state():
    cfg = {"model": {"mlp_router": {
        "balancing": "none",
        "quantile_global_state": True,
    }}}
    with pytest.raises(ValueError, match="none is incompatible with quantile_global_state"):
        _validator()(cfg)


def test_validator_rejects_quantile_with_bias_warmup_steps():
    cfg = {"model": {"attn_router": {
        "balancing": "quantile",
        "quantile_eta": 0.005,
        "bias_warmup_steps": 100,
    }}}
    with pytest.raises(ValueError, match="quantile is incompatible with bias_warmup_steps"):
        _validator()(cfg)


def test_validator_rejects_quantile_with_bias_update_zero_sum():
    cfg = {"model": {"attn_router": {
        "balancing": "quantile",
        "quantile_eta": 0.005,
        "bias_update_zero_sum": True,
    }}}
    with pytest.raises(ValueError, match="quantile is incompatible with bias_update_zero_sum"):
        _validator()(cfg)


def test_validator_rejects_deepseek_bias_with_quantile_global_state():
    cfg = {"model": {"mlp_router": {
        "balancing": "deepseek_bias",
        "bias_update_rate": 0.001,
        "quantile_global_state": True,
    }}}
    with pytest.raises(ValueError, match="deepseek_bias is incompatible with quantile_global_state"):
        _validator()(cfg)


def test_validator_accepts_each_method_with_only_its_active_knobs():
    """Sanity: every per-class method with ONLY its active knobs
    is accepted. Mirror of the rejection-set tests above."""
    for spec in [
        {"balancing": "aux_loss", "router_aux_loss_coef": 0.001},
        {"balancing": "seq_aux_loss", "seq_aux_loss_coef": 0.0001},
        {"balancing": "deepseek_bias", "bias_update_rate": 0.001},
        {"balancing": "quantile", "quantile_eta": 0.005,
         "quantile_target_q": 0.5, "quantile_global_state": True},
        {"balancing": "none"},
    ]:
        _validator()({"model": {"mlp_router": spec}})


def test_validator_branch_router_rejects_truly_invalid_method():
    """Even with the broader value set, an unrecognized method must
    still be rejected.
    """
    cfg = {"model": {"branch_router": {"balancing": "freeform"}}}
    with pytest.raises(ValueError, match="balancing="):
        _validator()(cfg)


def test_cli_validate_configs_rejects_invalid_nested_schema(tmp_path):
    """`scripts/validate_configs.py` must invoke the nested-schema
    validator so the CLI gate catches typos / illegal values, not
    just `load_config()`. This test writes a yaml with an unknown
    key under `model.branch_router` and runs the CLI on it; the
    invocation must exit non-zero and print the validator error.
    """
    import os
    import subprocess
    repo = Path(__file__).resolve().parent.parent

    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        """experiment_name: cli_validator_smoke
model:
  type: moe_everything
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 1
  num_experts: 2
  num_experts_per_tok: 1
  moe_intermediate_size: 16
  branch_router:
    balancing: exploration_only
    typo_field: 0.5
training:
  learning_rate: 1.0e-3
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 0
  max_steps: 1
  batch_size: 1
  gradient_accumulation: 1
  mixed_precision: ""
  output_dir: /tmp
"""
    )
    cmd = [
        sys.executable, "scripts/validate_configs.py",
        str(yaml_path),
    ]
    env = {"PYTHONPATH": str(repo), **os.environ}
    result = subprocess.run(cmd, cwd=str(repo), env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0, (
        f"CLI did not detect the nested-schema typo:\n"
        f"STDOUT={result.stdout}\nSTDERR={result.stderr}"
    )
    assert "unknown keys" in result.stdout or "nested-schema validator" in result.stdout, (
        f"CLI output did not mention the validator:\n{result.stdout}"
    )


def test_build_model_stamps_mlp_router_nested_fields_on_config(tmp_path):
    """`build_model` must read the nested `model.mlp_router` block
    and stamp every key onto the config under the
    `mlp_router_<key>` attribute namespace, so the runtime can read
    per-class methods without traversing the nested structure.
    """
    import importlib.util
    import os
    import subprocess

    repo = Path(__file__).resolve().parent.parent
    yaml_path = tmp_path / "stamp.yaml"
    yaml_path.write_text(
        """experiment_name: stamp_smoke
model:
  type: moe_everything
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 1
  head_dim: 8
  num_attention_heads: 2
  num_key_value_heads: 2
  num_experts: 4
  num_experts_per_tok: 2
  moe_intermediate_size: 32
  intermediate_size: 32
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_fully_independent
  use_deepseek_routing: true
  branch_deepseek: true
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 32
  tie_word_embeddings: true
  output_router_logits: false
  attn_implementation: eager
  mlp_router:
    balancing: aux_loss
    router_aux_loss_coef: 0.001
  attn_router:
    balancing: deepseek_bias
    bias_update_rate: 0.001
    scale_by_routing_weight: true
training:
  learning_rate: 1.0e-3
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 0
  max_steps: 1
  batch_size: 1
  gradient_accumulation: 1
  mixed_precision: ""
  output_dir: /tmp
"""
    )
    cmd = [
        sys.executable, "-c",
        f"""
import importlib.util, sys, types
from pathlib import Path
repo = Path('{str(repo)}').resolve()
if 'src.training' not in sys.modules:
    pkg = types.ModuleType('src.training')
    pkg.__path__ = [str(repo / 'src' / 'training')]
    sys.modules['src.training'] = pkg
def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, str(repo / relpath))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m
cfg_mod = _load('src.training.config', 'src/training/config.py')
factory_mod = _load('src.training.model_factory', 'src/training/model_factory.py')
cfg = cfg_mod.load_config('{str(yaml_path)}')
model, config = factory_mod.build_model(cfg)
import json
print(json.dumps({{
    'mlp_balancing': getattr(config, 'mlp_router_balancing', None),
    'mlp_aux_coef': getattr(config, 'mlp_router_router_aux_loss_coef', None),
    'mlp_quantile_eta': getattr(config, 'mlp_router_quantile_eta', None),
    'attn_balancing': getattr(config, 'attn_router_balancing', None),
    'attn_bias_rate': getattr(config, 'attn_router_bias_update_rate', None),
    'attn_scale': getattr(config, 'attn_router_scale_by_routing_weight', None),
}}))
"""
    ]
    env = {"PYTHONPATH": str(repo), **os.environ}
    result = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"build_model probe failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
    )
    import json as _json
    stamped = _json.loads(result.stdout)
    assert stamped["mlp_balancing"] == "aux_loss"
    assert stamped["mlp_aux_coef"] == 0.001
    assert stamped["mlp_quantile_eta"] is None  # not set in this fixture
    assert stamped["attn_balancing"] == "deepseek_bias"
    assert stamped["attn_bias_rate"] == 0.001
    assert stamped["attn_scale"] is True


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
