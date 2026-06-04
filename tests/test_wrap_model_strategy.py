"""Pin the per-model-family distributed wrapping policy in `wrap_model`.

The supported model families diverge in how they interact with FSDP's
flatten-params backward hooks:

- `dense`, `standard_moe`, `global_moe` work cleanly under `FULL_SHARD`.
- `moe_everything` uses branch routing + grouped GEMM; it defaults to
  FSDP `NO_SHARD` plus an auto-wrap policy so sparse branch activity does
  not share one flat-param hook with unrelated always-active modules.
- Launchers may explicitly request `NO_SHARD` or `HYBRID_SHARD` for any
  family via the FSDP sharding override.

These tests exercise `wrap_model` under a stubbed FSDP / DDP so they can
run without distributed init, and pin the FSDP strategy selection for
each model family.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.training.distributed as dist_mod
from src.training.distributed import (
    _FSDP_SHARDING_BY_MODEL_TYPE,
    _FSDP_SKIP_MIXED_PRECISION,
    _fsdp_sharding_for,
    _fsdp_use_orig_params_for,
    _resolve_fsdp_sharding_name,
    wrap_model,
)


def test_fsdp_sharding_table_covers_every_supported_family():
    assert set(_FSDP_SHARDING_BY_MODEL_TYPE.keys()) == {
        "dense", "standard_moe", "global_moe", "moe_everything",
    }


def test_fsdp_sharding_for_supported_families_is_full_shard():
    for family in ("dense", "standard_moe", "global_moe"):
        enum_val = _fsdp_sharding_for(family)
        assert enum_val is not None, f"ShardingStrategy unavailable for {family}"
        assert enum_val.name == "FULL_SHARD", f"{family} -> {enum_val.name}"


def test_fsdp_sharding_for_moe_everything_is_no_shard():
    enum_val = _fsdp_sharding_for("moe_everything")
    assert enum_val is not None
    assert enum_val.name == "NO_SHARD"


def test_moe_everything_skips_fsdp_mixed_precision():
    # NO_SHARD + bf16 MixedPrecision triggers a `setStorage ... storage of
    # size 0` error on the embedding flat-param for this family in torch 2.10.
    # Keeping params in fp32 (the trainer still autocasts activations) avoids
    # that. If a future torch/FSDP release fixes the NO_SHARD master/shard
    # interaction, this skip can be lifted.
    assert "moe_everything" in _FSDP_SKIP_MIXED_PRECISION


def test_use_orig_params_is_true_for_all_current_strategies():
    # Pinned by _fsdp_use_orig_params_for; flip this test if the policy changes.
    for s in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"):
        assert _fsdp_use_orig_params_for(s) is True


def test_fsdp_sharding_override_resolves_aliases():
    assert _resolve_fsdp_sharding_name("standard_moe", None) == "FULL_SHARD"
    assert _resolve_fsdp_sharding_name("moe_everything", "auto") == "NO_SHARD"
    assert _resolve_fsdp_sharding_name("standard_moe", "no_shard") == "NO_SHARD"
    assert _resolve_fsdp_sharding_name("standard_moe", "hybrid_shard") == "HYBRID_SHARD"
    with pytest.raises(ValueError, match="Invalid FSDP sharding strategy"):
        _resolve_fsdp_sharding_name("standard_moe", "zero3")


def test_moe_everything_auto_wrap_policy_keeps_tied_embedding_head_together(monkeypatch):
    captured: dict[str, set[type]] = {}

    class _FakeModuleWrapPolicy:
        def __init__(self, module_classes):
            captured["module_classes"] = set(module_classes)

    monkeypatch.setattr(dist_mod, "ModuleWrapPolicy", _FakeModuleWrapPolicy)
    policy = dist_mod._moe_everything_auto_wrap_policy()

    assert policy is not None
    import torch.nn as nn
    from src.models.modeling_qwen3_moe import Qwen3MoeRMSNorm
    from src.models.moe_everything.attention_bank import AttentionExpertBank
    from src.models.moe_everything.mlp_bank import MlpExpertBank
    from src.models.routing.routers import BranchRouter

    module_classes = captured["module_classes"]
    assert AttentionExpertBank in module_classes
    assert MlpExpertBank in module_classes
    assert BranchRouter in module_classes
    assert Qwen3MoeRMSNorm in module_classes
    assert nn.Embedding not in module_classes
    assert nn.Linear not in module_classes


def test_no_ddp_masquerade_under_fsdp_strategy():
    # After the Round 7 → Round 8 fix, `--dist-strategy fsdp` must NOT
    # silently return a DDP wrapper for any supported family. The module
    # should not expose a DDP-fallback set anymore.
    assert not hasattr(dist_mod, "_USE_DDP_INSTEAD_OF_FSDP"), (
        "_USE_DDP_INSTEAD_OF_FSDP was removed in Round 8; it should not "
        "come back. moe_everything now uses real FSDP with auto_wrap_policy."
    )


class _FakeModel:
    """Minimal stand-in so `wrap_model` gets something to pass through."""


@pytest.fixture
def fake_world():
    """Make `wrap_model` think it's running distributed with 2 ranks."""
    with patch.object(dist_mod, "dist_world_size", return_value=2):
        yield


def _capture_fsdp_kwargs():
    """Return a (fake_fsdp_cls, calls_list) pair that records each call."""
    calls: list[dict] = []

    class _FakeFSDP:
        def __init__(self, model, **kwargs):
            calls.append({"model": model, **kwargs})
            self.model = model

    return _FakeFSDP, calls


def _capture_ddp_kwargs():
    calls: list[dict] = []

    class _FakeDDP:
        def __init__(self, model, **kwargs):
            calls.append({"model": model, **kwargs})
            self.model = model

    return _FakeDDP, calls


@pytest.mark.parametrize("family", ["dense", "standard_moe", "global_moe"])
def test_wrap_model_fsdp_uses_full_shard_for_sharded_families(fake_world, family):
    fake_fsdp, fsdp_calls = _capture_fsdp_kwargs()
    fake_ddp, ddp_calls = _capture_ddp_kwargs()
    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        wrapped = wrap_model(
            _FakeModel(), strategy="fsdp", local_rank=0,
            mixed_precision_name="bf16", model_type=family,
        )
    assert len(fsdp_calls) == 1 and len(ddp_calls) == 0, (
        f"{family} should have gone through FSDP, not DDP"
    )
    kwargs = fsdp_calls[0]
    assert kwargs["sharding_strategy"].name == "FULL_SHARD"
    assert kwargs["use_orig_params"] is True
    assert kwargs["sync_module_states"] is True
    # Mixed precision is populated for these families.
    assert kwargs["mixed_precision"] is not None


def test_wrap_model_fsdp_uses_real_fsdp_for_moe_everything(fake_world):
    """moe_everything under `--dist-strategy fsdp` must construct a real
    FSDP wrapper with NO_SHARD + a non-None auto_wrap_policy and no FSDP-level
    MixedPrecision (params stay in fp32; outer autocast handles activations).
    """
    fake_fsdp, fsdp_calls = _capture_fsdp_kwargs()
    fake_ddp, ddp_calls = _capture_ddp_kwargs()
    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        wrap_model(
            _FakeModel(), strategy="fsdp", local_rank=0,
            mixed_precision_name="bf16", model_type="moe_everything",
        )
    assert len(ddp_calls) == 0, (
        "moe_everything with --dist-strategy fsdp must use a real FSDP "
        "wrapper, not DDP. Round 8 removed the DDP masquerade."
    )
    assert len(fsdp_calls) == 1
    kwargs = fsdp_calls[0]
    assert kwargs["sharding_strategy"].name == "NO_SHARD"
    assert kwargs["use_orig_params"] is True
    assert kwargs["sync_module_states"] is True
    assert kwargs["mixed_precision"] is None, (
        "moe_everything FSDP path must not pass a MixedPrecision policy — "
        "see _FSDP_SKIP_MIXED_PRECISION for the rationale."
    )
    assert kwargs["auto_wrap_policy"] is not None, (
        "moe_everything needs an auto_wrap_policy so AttentionExpertBank / "
        "MlpExpertBank / BranchRouter become their own FSDP units."
    )


def test_wrap_model_fsdp_no_shard_override_for_standard_moe(fake_world):
    fake_fsdp, fsdp_calls = _capture_fsdp_kwargs()
    fake_ddp, ddp_calls = _capture_ddp_kwargs()
    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        wrap_model(
            _FakeModel(), strategy="fsdp", local_rank=0,
            mixed_precision_name="bf16", model_type="standard_moe",
            fsdp_sharding_strategy="no_shard",
        )
    assert len(fsdp_calls) == 1 and len(ddp_calls) == 0
    kwargs = fsdp_calls[0]
    assert kwargs["sharding_strategy"].name == "NO_SHARD"
    assert kwargs["mixed_precision"] is None, (
        "NO_SHARD skips FSDP-level MixedPrecision; trainer autocast still "
        "handles bf16 activations."
    )


def test_wrap_model_fsdp_hybrid_shard_override(fake_world):
    fake_fsdp, fsdp_calls = _capture_fsdp_kwargs()
    fake_ddp, _ = _capture_ddp_kwargs()
    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        wrap_model(
            _FakeModel(), strategy="fsdp", local_rank=0,
            mixed_precision_name="bf16", model_type="standard_moe",
            fsdp_sharding_strategy="hybrid_shard",
        )
    assert fsdp_calls[0]["sharding_strategy"].name == "HYBRID_SHARD"
    assert fsdp_calls[0]["mixed_precision"] is not None


def test_wrap_model_ddp_strategy_always_uses_ddp(fake_world):
    fake_fsdp, fsdp_calls = _capture_fsdp_kwargs()
    fake_ddp, ddp_calls = _capture_ddp_kwargs()
    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        for family in ("dense", "standard_moe", "global_moe", "moe_everything"):
            wrap_model(
                _FakeModel(), strategy="ddp", local_rank=0,
                mixed_precision_name="bf16", model_type=family,
            )
    assert len(fsdp_calls) == 0
    assert len(ddp_calls) == 4


def test_wrap_model_none_strategy_returns_model_unchanged(fake_world):
    m = _FakeModel()
    assert wrap_model(m, strategy="none", local_rank=0, model_type="dense") is m


def test_wrap_model_raises_on_unknown_strategy(fake_world):
    with pytest.raises(ValueError, match="Unknown strategy"):
        wrap_model(_FakeModel(), strategy="zero-3", local_rank=0, model_type="dense")


def test_wrap_model_moe_everything_fails_fast_when_auto_wrap_policy_unavailable(fake_world, monkeypatch):
    """If `_moe_everything_auto_wrap_policy()` returns None (older torch
    without `ModuleWrapPolicy`, or the AttentionExpertBank / MlpExpertBank /
    BranchRouter classes moved), `wrap_model` must raise a clear
    RuntimeError at setup rather than hand FSDP a `None` auto_wrap_policy
    and let the single-unit NO_SHARD wrapper trip the `TrainingState.IDLE`
    post-backward-hook assertion deep inside the first backward pass.
    """
    fake_fsdp, fsdp_calls = _capture_fsdp_kwargs()
    fake_ddp, _ = _capture_ddp_kwargs()

    # Stub the auto-wrap-policy builder to simulate the unavailable case.
    monkeypatch.setattr(dist_mod, "_moe_everything_auto_wrap_policy", lambda: None)

    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        with pytest.raises(RuntimeError, match="auto_wrap policy could not be built"):
            wrap_model(
                _FakeModel(), strategy="fsdp", local_rank=0,
                mixed_precision_name="bf16", model_type="moe_everything",
            )

    # Other families must not be affected by the failed-policy branch — the
    # top-level fsdp wrap path is still OK when auto_wrap policy is None
    # (they don't need one).
    fsdp_calls.clear()
    monkeypatch.setattr(dist_mod, "_moe_everything_auto_wrap_policy", lambda: None)
    with patch.object(dist_mod, "FSDP", fake_fsdp), \
         patch.object(dist_mod, "DDP", fake_ddp):
        wrap_model(
            _FakeModel(), strategy="fsdp", local_rank=0,
            mixed_precision_name="bf16", model_type="dense",
        )
    assert len(fsdp_calls) == 1
    # dense uses no auto_wrap policy, so its call proceeds normally.
    assert fsdp_calls[0]["auto_wrap_policy"] is None
