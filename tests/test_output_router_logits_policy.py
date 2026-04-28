"""DEC-15 / task14: `output_router_logits` is resolved from the method.

Per DEC-15:
  - aux_loss / seq_aux_loss → True (router scores are gradient-bearing inputs
    to the aux loss; the model output must carry them).
  - deepseek_bias / quantile / none → False (no aux loss reads
    `router_logits`; the routing-decision state lives in router-internal
    `_last_top_k_idx` / `local_tokens_per_expert` buffers).
  - None (no method set; legacy back-compat) → True (preserve pre-DEC-15
    default).

These tests verify the policy holds end-to-end through `load_config()` +
`build_model()`, AND that the helper function returns the correct values for
each method.
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

    mf_spec = importlib.util.spec_from_file_location(
        "src.training.model_factory",
        repo / "src" / "training" / "model_factory.py",
    )
    mf = importlib.util.module_from_spec(mf_spec)
    sys.modules["src.training.model_factory"] = mf
    mf_spec.loader.exec_module(mf)
    return bf, cfg_mod, mf


def _make_yaml(method: str | None, tmp: Path) -> Path:
    """Minimal `standard_moe` yaml with the given method (or no method)."""
    cfg = {
        "experiment_name": f"test_orl_{method or 'absent'}",
        "model": {
            "type": "standard_moe",
            "router_type": "deepseek" if method == "deepseek_bias" else "softmax",
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
            "attn_implementation": "eager",
        },
        "training": {} if method is None else {"load_balancing_method": method},
        "data": {"data_dir": "./fake"},
    }
    out = tmp / f"test_orl_{method or 'absent'}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


@pytest.mark.parametrize(
    "method,expected",
    [
        ("aux_loss",      True),
        ("seq_aux_loss",  True),
        ("deepseek_bias", False),
        ("quantile",      False),
        ("none",          False),
        (None,            True),  # legacy back-compat
    ],
)
def test_helper_resolves_orl_from_method(method, expected):
    """`output_router_logits_for_method` returns the right value per DEC-15."""
    bf, _, _ = _load_modules()
    assert bf.output_router_logits_for_method(method) is expected, (
        f"DEC-15 policy violation: method={method!r} should produce "
        f"output_router_logits={expected}, got {bf.output_router_logits_for_method(method)}"
    )


@pytest.mark.parametrize(
    "method,expected",
    [
        ("aux_loss",      True),
        ("seq_aux_loss",  True),
        ("deepseek_bias", False),
        ("quantile",      False),
        ("none",          False),
    ],
)
def test_build_model_sets_output_router_logits_per_method(method, expected):
    """End-to-end: `load_config()` + `build_model()` produces a model whose
    config has `output_router_logits` set per the DEC-15 policy."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml(method, Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, model_cfg = mf.build_model(cfg)
    assert model_cfg.output_router_logits is expected, (
        f"build_model failed to honor DEC-15 for method={method!r}: "
        f"expected output_router_logits={expected}, got "
        f"{model_cfg.output_router_logits}"
    )


def test_legacy_no_method_keeps_orl_true_for_back_compat():
    """Legacy yaml without `load_balancing_method` keeps
    `output_router_logits=True` (pre-DEC-15 default). Round 5+ unmigrated
    yamls still work."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml(None, Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, model_cfg = mf.build_model(cfg)
    assert model_cfg.output_router_logits is True, (
        f"Legacy back-compat broken: yaml without load_balancing_method should "
        f"keep output_router_logits=True, got {model_cfg.output_router_logits}"
    )


def test_aux_methods_keep_grad_bearing_router_logits_in_forward():
    """For aux methods, the model output's `router_logits` must be a tuple of
    tensors that require grad — proves the loss term can backprop through.
    For non-aux methods, `router_logits` is None (or the model skips returning
    it entirely)."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("aux_loss", Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, model_cfg = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (2, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.router_logits is not None, (
        "aux_loss model forward must return router_logits"
    )
    # Each per-layer router_logits tensor should require grad.
    for t in out.router_logits:
        assert t.requires_grad, (
            "aux_loss model's router_logits must be gradient-bearing"
        )


def test_non_aux_methods_skip_router_logits_in_forward():
    """For non-aux methods, calling forward with `output_router_logits=False`
    (the default the model_factory now resolves) should yield None
    router_logits — the model didn't allocate them."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("deepseek_bias", Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, model_cfg = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (2, 8), dtype=torch.long)
    # Pass the same flag the trainer would derive (False for deepseek_bias).
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=False)
    assert out.router_logits is None, (
        f"deepseek_bias model with output_router_logits=False should return "
        f"None for router_logits, got {type(out.router_logits)}"
    )


if __name__ == "__main__":
    cases = [
        ("aux_loss",      True),
        ("seq_aux_loss",  True),
        ("deepseek_bias", False),
        ("quantile",      False),
        ("none",          False),
        (None,            True),
    ]
    for c in cases:
        test_helper_resolves_orl_from_method(*c)
    for method, expected in cases[:-1]:  # skip the None case (different signature)
        test_build_model_sets_output_router_logits_per_method(method, expected)
    test_legacy_no_method_keeps_orl_true_for_back_compat()
    test_aux_methods_keep_grad_bearing_router_logits_in_forward()
    test_non_aux_methods_skip_router_logits_in_forward()
    print("ALL OK")
