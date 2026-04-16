"""Pin the per-model-family distributed wrapping policy in `wrap_model`.

The supported model families diverge in how they interact with FSDP's
flatten-params backward hooks:

- `dense`, `standard_moe`, `global_moe` work cleanly under `FULL_SHARD`.
- `moe_everything` uses branch routing + grouped GEMM; every FSDP sharding
  strategy available on torch 2.10 fires the `TrainingState.IDLE`
  post-backward assertion because whole per-depth flat-params units can
  receive no gradient activity on a given step. The trainer transparently
  falls back to DDP when the user asks for `--dist-strategy fsdp` with
  that family so the CLI contract ("every supported model runs under
  `--dist-strategy fsdp`") is preserved end-to-end.

These tests exercise `wrap_model` under a stubbed FSDP / DDP so they can
run without distributed init, and pin both the FSDP strategy selection
for the sharded families and the DDP fallback for `moe_everything`.
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
    for s in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"):
        assert _fsdp_use_orig_params_for(s) is True


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
