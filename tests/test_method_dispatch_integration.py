"""AC-1 end-to-end integration: `load_config()` + `build_model()` + forward +
`update_expert_biases()` honor the resolved `load_balancing_method`.

Round 4's per-family tests manually set `model._load_balancing_method` to
verify forward gating in isolation. Codex's Round 4 review correctly noted
this bypasses the real wiring — the test passes even if `build_model()`
forgets to stamp the attribute. Round 5 closes the loop:

  1. Load a yaml that sets `load_balancing_method` (canonical `training:` block).
  2. Verify `normalize_balancing_config` AUTO-ZEROED conflicting coefs.
  3. Build the model via `build_model(cfg)`.
  4. Verify the model has `_load_balancing_method` stamped (without manual
     intervention by the caller).
  5. Verify `update_expert_biases()` no-ops for non-bias methods even when
     called directly (i.e. the trainer's gate is now defense-in-depth, not
     the only enforcement point).
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, ".")

# Direct module loads to avoid `src/training/__init__.py`'s pandas dep.
def _load_modules():
    repo = Path(__file__).resolve().parent.parent
    src_path = str(repo / "src")
    import types as _types
    if "src" not in sys.modules:
        src_pkg = _types.ModuleType("src"); src_pkg.__path__ = [src_path]
        sys.modules["src"] = src_pkg
    if "src.training" not in sys.modules:
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = training_pkg

    bf_spec = importlib.util.spec_from_file_location(
        "src.training.balancing_fields",
        repo / "src" / "training" / "balancing_fields.py",
    )
    bf = importlib.util.module_from_spec(bf_spec)
    sys.modules["src.training.balancing_fields"] = bf
    bf_spec.loader.exec_module(bf)

    cfg_spec = importlib.util.spec_from_file_location(
        "src.training.config",
        repo / "src" / "training" / "config.py",
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules["src.training.config"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)
    return bf, cfg_mod


def _make_minimal_yaml(method: str, tmp: Path) -> Path:
    """Write a minimal `standard_moe` yaml with the given method to a temp file.

    The yaml deliberately includes BOTH `load_balancing_method` AND non-zero
    legacy coefficients to exercise `normalize_balancing_config`'s AUTO-ZERO.
    """
    # `deepseek_bias` requires a DeepSeek router (softmax routers have no
    # `expert_bias` buffer, so the walker yields zero owners). For other
    # methods, either router type works; pick deepseek for consistency.
    router_type = "deepseek"
    cfg = {
        "experiment_name": f"test_method_{method}",
        "model": {
            "type": "standard_moe",
            "router_type": router_type,
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "head_dim": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "moe_intermediate_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "output_router_logits": True,
            "attn_implementation": "eager",
        },
        "training": {
            "load_balancing_method": method,
            "router_aux_loss_coef": 0.001,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
        },
        "data": {"data_dir": "./fake"},
    }
    out = tmp / f"test_method_{method}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


@pytest.mark.parametrize(
    "method,expected_aux_active,expected_bias_update",
    [
        ("aux_loss",      True,  False),
        ("seq_aux_loss",  False, False),
        ("deepseek_bias", False, True),
        ("quantile",      False, False),
        ("none",          False, False),
    ],
)
def test_load_config_normalizes_then_build_model_stamps_method(
    method, expected_aux_active, expected_bias_update,
):
    """End-to-end: `load_config(yaml)` runs normalize, then `build_model(cfg)`
    stamps `_load_balancing_method` on the model. After this round-trip:
      - Aux/seq-aux coefs are auto-zeroed when not in the method's active set.
      - The model has `_load_balancing_method` stamped (not by the caller).
      - `update_expert_biases()` consults the stamped method.
    """
    bf, cfg_mod = _load_modules()

    # Load `model_factory` via the package shell so its relative imports work.
    repo = Path(__file__).resolve().parent.parent
    mf_spec = importlib.util.spec_from_file_location(
        "src.training.model_factory",
        repo / "src" / "training" / "model_factory.py",
    )
    mf = importlib.util.module_from_spec(mf_spec)
    sys.modules["src.training.model_factory"] = mf
    mf_spec.loader.exec_module(mf)

    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_minimal_yaml(method, Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # AUTO-ZERO is expected
            cfg = cfg_mod.load_config(str(yaml_path))

    # 1) `load_config` ran `normalize_balancing_config` — verify the AUTO-ZERO.
    if method == "aux_loss":
        assert cfg["training"]["router_aux_loss_coef"] == 0.001
        assert cfg["training"]["seq_aux_loss_coef"] == 0.0
        assert cfg["training"]["bias_update_rate"] == 0.0
    elif method == "deepseek_bias":
        assert cfg["training"]["router_aux_loss_coef"] == 0.0
        assert cfg["training"]["seq_aux_loss_coef"] == 0.0
        assert cfg["training"]["bias_update_rate"] == 0.001
    elif method in ("seq_aux_loss", "quantile", "none"):
        # Non-aux + non-bias coefs all zeroed (or only the relevant one kept).
        if method == "seq_aux_loss":
            assert cfg["training"]["seq_aux_loss_coef"] == 0.0001
            assert cfg["training"]["router_aux_loss_coef"] == 0.0
        else:
            assert cfg["training"]["router_aux_loss_coef"] == 0.0
            assert cfg["training"]["seq_aux_loss_coef"] == 0.0
        assert cfg["training"]["bias_update_rate"] == 0.0

    # 2) `build_model(cfg)` stamps `_load_balancing_method` on the model.
    model, model_cfg = mf.build_model(cfg)
    assert getattr(model, "_load_balancing_method", None) == method, (
        f"build_model failed to stamp _load_balancing_method={method!r} on the model"
    )
    assert getattr(model_cfg, "load_balancing_method", None) == method, (
        f"build_model failed to stamp load_balancing_method on the config"
    )

    # 3) `update_expert_biases` is method-gated AT THE FUNCTION (not just at
    # the trainer call site). Calling it directly with a non-`deepseek_bias`
    # method must no-op.
    from src.training.routing import update_expert_biases
    # Snapshot the bias buffers before calling.
    pre_biases = {
        id(o): o.expert_bias.clone()
        for o, _ in (model.get_all_balancing_owners() if hasattr(model, "get_all_balancing_owners") else [])
    }
    # Inject a non-zero count so a NON-no-op update WOULD shift the bias.
    for o, _ in (model.get_all_balancing_owners() if hasattr(model, "get_all_balancing_owners") else []):
        o.local_tokens_per_expert.zero_()
        o.local_tokens_per_expert[0] = 100.0  # heavy skew

    update_expert_biases(model, bias_rate=0.01, distributed=False)

    if expected_bias_update:
        # method=deepseek_bias: bias buffers MUST shift.
        any_changed = False
        for o, _ in model.get_all_balancing_owners():
            if not torch.equal(o.expert_bias, pre_biases[id(o)]):
                any_changed = True
                break
        assert any_changed, (
            f"method={method} should run bias update but expert_bias buffers "
            f"did not change."
        )
    else:
        # method != deepseek_bias: bias buffers must stay frozen.
        for o, _ in model.get_all_balancing_owners():
            assert torch.equal(o.expert_bias, pre_biases[id(o)]), (
                f"method={method} should NOT run bias update but expert_bias "
                f"changed (the function-level gate is missing)."
            )


if __name__ == "__main__":
    cases = [
        ("aux_loss",      True,  False),
        ("seq_aux_loss",  False, False),
        ("deepseek_bias", False, True),
        ("quantile",      False, False),
        ("none",          False, False),
    ]
    for case in cases:
        test_load_config_normalizes_then_build_model_stamps_method(*case)
    print("ALL OK")
