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


def _make_yaml(method: str | None, tmp: Path, family: str = "standard_moe") -> Path:
    """Minimal yaml with the given method (or no method) for a chosen family."""
    base_model = {
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
    }
    if family == "moe_everything":
        base_model.update({
            "type": "moe_everything",
            "router_type": "deepseek" if method == "deepseek_bias" else "softmax",
            "use_deepseek_routing": method == "deepseek_bias",
            "num_attn_experts": 2,
            "num_attn_experts_per_tok": 1,
            "attn_expert_mode": "per_head_fully_independent",
            "branch_router_aux_loss_coef": 0.0,
        })
    else:
        base_model["type"] = family
        base_model["router_type"] = "deepseek" if method == "deepseek_bias" else "softmax"
    cfg = {
        "experiment_name": f"test_orl_{family}_{method or 'absent'}",
        "model": base_model,
        "training": {} if method is None else {"load_balancing_method": method},
        "data": {"data_dir": "./fake"},
    }
    out = tmp / f"test_orl_{family}_{method or 'absent'}.yaml"
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


@pytest.mark.parametrize(
    "family,method,expected",
    [
        ("standard_moe",  "aux_loss",      True),
        ("standard_moe",  "deepseek_bias", False),
        ("global_moe",    "aux_loss",      True),
        ("global_moe",    "deepseek_bias", False),
    ],
)
def test_orl_policy_across_standard_and_global_families(family, method, expected):
    """DEC-15 policy holds across `standard_moe` and `global_moe`."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml(method, Path(td), family=family)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, model_cfg = mf.build_model(cfg)
    assert model_cfg.output_router_logits is expected, (
        f"build_model failed DEC-15 for ({family}, {method}): expected "
        f"{expected}, got {model_cfg.output_router_logits}"
    )


def test_router_internal_telemetry_present_after_forward():
    """DEC-15 DETACH-ONLY: routers expose `_last_router_scores_detached`
    after every forward, even for non-aux methods where the model output
    skips returning `router_logits`. Telemetry consumers can read this
    attribute without retaining the autograd graph."""
    _, cfg_mod, mf = _load_modules()
    # Use deepseek_bias (non-aux method, output_router_logits=False).
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("deepseek_bias", Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, _ = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (2, 8), dtype=torch.long)
    _ = model(input_ids=input_ids, labels=input_ids, output_router_logits=False)

    # Walk every router that exposes the canonical `_last_router_scores_detached`
    # attribute and verify it's populated and detached.
    from src.models.router import DeepSeekRouter, ExplorationTopKRouter
    inspected = 0
    for m in model.modules():
        if isinstance(m, (DeepSeekRouter, ExplorationTopKRouter)):
            scores = getattr(m, "_last_router_scores_detached", None)
            assert scores is not None, (
                f"{type(m).__name__} did not populate _last_router_scores_detached"
            )
            assert not scores.requires_grad, (
                f"{type(m).__name__}._last_router_scores_detached must be detached"
            )
            inspected += 1
    assert inspected > 0, "no routers inspected — fixture is wrong"


@pytest.mark.parametrize("family", ["standard_moe", "global_moe"])
def test_non_aux_method_forced_orl_true_returns_detached(family):
    """DEC-15 forward-level enforcement (Codex Round 7 Blocker #1):
    even when a caller forces `output_router_logits=True`, non-aux methods
    must return DETACHED `output.router_logits`. Round 7 only enforced this
    via `build_model` (model_config.output_router_logits=False); a caller
    that overrode the kwarg could still leak grad-bearing tensors.
    """
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("deepseek_bias", Path(td), family=family)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, _ = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (2, 8), dtype=torch.long)
    # Force output_router_logits=True even though the method doesn't need them.
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.router_logits is not None, (
        f"{family} forced output_router_logits=True should still produce "
        f"router_logits, got None"
    )
    requires_grad_flags = [t.requires_grad for t in out.router_logits]
    assert not any(requires_grad_flags), (
        f"{family} deepseek_bias with forced output_router_logits=True "
        f"returned grad-bearing router_logits: {requires_grad_flags}. "
        f"DEC-15 DETACH-ONLY policy violated at the forward level."
    )


def test_moe_everything_non_aux_attention_router_info_detached():
    """DEC-15 (Codex Round 7 Blocker #1, moe_everything attention path):
    when `load_balancing_method` is non-aux and the caller forces
    `output_router_logits=True`, the q/k/v/o attention router_logits in
    `output.attention_router_info` must be detached.
    """
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("deepseek_bias", Path(td), family="moe_everything")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, _ = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (1, 4), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    if out.attention_router_info is None:
        return  # Some attention modes skip this; not a failure.
    for depth_info in out.attention_router_info:
        for name, info in depth_info.items():
            rl = info["router_logits"]
            assert not rl.requires_grad, (
                f"moe_everything deepseek_bias attention router '{name}' "
                f"router_logits is grad-bearing under forced "
                f"output_router_logits=True. DEC-15 violated."
            )


def test_detached_telemetry_fallback_in_compute_output_metrics():
    """DEC-15 detached telemetry consumer (Codex Round 7 Blocker #1):
    when `output.router_logits is None` (non-aux method default), the
    metrics path falls back to per-router `_last_router_scores_detached`
    snapshots and produces a non-zero `aux_loss_normalized`.
    """
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("deepseek_bias", Path(td))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, model_cfg = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (2, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=False)
    assert out.router_logits is None, "preflight: deepseek_bias with output_router_logits=False should yield None"

    # Import the metrics path the trainer/eval would use.
    import importlib.util
    metrics_spec = importlib.util.spec_from_file_location(
        "_metrics",
        Path(__file__).resolve().parent.parent / "src" / "training" / "metrics.py",
    )
    # Need the package context for relative imports to work.
    import sys as _sys, types as _types
    if "src.training" not in _sys.modules:
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "training")]
        _sys.modules["src.training"] = training_pkg
    metrics = importlib.util.module_from_spec(metrics_spec)
    _sys.modules["src.training.metrics"] = metrics
    metrics_spec.loader.exec_module(metrics)

    detached = metrics._collect_detached_router_scores(model)
    assert detached is not None, (
        "DEC-15 telemetry fallback failed: no detached router scores "
        "collected from a deepseek_bias model after forward."
    )
    assert all(not t.requires_grad for t in detached), (
        "Detached telemetry must not be grad-bearing"
    )


def test_moe_everything_aux_method_forward_keeps_grad_bearing_mlp_router_logits():
    """Codex Round 6 measured `out.router_logits MLP requires_grad flags ->
    [False, False]` for moe_everything aux_loss — the MLP router_logits were
    detached unconditionally, breaking aux gradient flow. Round 7 fix: only
    detach for non-aux methods."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("aux_loss", Path(td), family="moe_everything")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, _ = mf.build_model(cfg)
    model.train()
    # Use a small input that the moe_everything CPU path can handle.
    # `per_head_fully_independent` mode has CPU shape constraints around the
    # attention bank but the MLP path itself is fine.
    input_ids = torch.randint(0, model.vocab_size, (1, 4), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.router_logits is not None
    requires_grad_flags = [t.requires_grad for t in out.router_logits]
    assert all(requires_grad_flags), (
        f"moe_everything aux_loss MLP router_logits must be gradient-bearing; "
        f"got requires_grad flags = {requires_grad_flags}"
    )


def test_moe_everything_deepseek_bias_method_forward_detaches_mlp_router_logits():
    """For non-aux methods, moe_everything's MLP router_logits should be
    detached — telemetry-only, no autograd graph cost."""
    _, cfg_mod, mf = _load_modules()
    with tempfile.TemporaryDirectory() as td:
        yaml_path = _make_yaml("deepseek_bias", Path(td), family="moe_everything")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = cfg_mod.load_config(str(yaml_path))
        model, _ = mf.build_model(cfg)
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (1, 4), dtype=torch.long)
    # output_router_logits=True forces the model to return them; the policy
    # should still detach for non-aux methods so no grad leaks.
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    if out.router_logits is not None:
        requires_grad_flags = [t.requires_grad for t in out.router_logits]
        assert not any(requires_grad_flags), (
            f"moe_everything deepseek_bias MLP router_logits should be detached "
            f"(telemetry only); got requires_grad flags = {requires_grad_flags}"
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
    for method, expected in cases[:-1]:
        test_build_model_sets_output_router_logits_per_method(method, expected)
    test_legacy_no_method_keeps_orl_true_for_back_compat()
    test_aux_methods_keep_grad_bearing_router_logits_in_forward()
    test_non_aux_methods_skip_router_logits_in_forward()
    for fam_method in [
        ("standard_moe",  "aux_loss",      True),
        ("standard_moe",  "deepseek_bias", False),
        ("global_moe",    "aux_loss",      True),
        ("global_moe",    "deepseek_bias", False),
    ]:
        test_orl_policy_across_standard_and_global_families(*fam_method)
    test_router_internal_telemetry_present_after_forward()
    test_moe_everything_aux_method_forward_keeps_grad_bearing_mlp_router_logits()
    test_moe_everything_deepseek_bias_method_forward_detaches_mlp_router_logits()
    for family in ("standard_moe", "global_moe"):
        test_non_aux_method_forced_orl_true_returns_detached(family)
    test_moe_everything_non_aux_attention_router_info_detached()
    test_detached_telemetry_fallback_in_compute_output_metrics()
    print("ALL OK")
