"""Smoke tests for the unified trainer with all supported model variants.

Tests: forward pass, backward pass, optimizer step, checkpoint save/load,
loss sanity (decreasing over steps), and deprecated type rejection.

Model matrix (7 variants):
- dense
- standard_moe (softmax)
- standard_moe (deepseek)
- global_moe (softmax)
- global_moe (deepseek)
- moe_everything (per_head_fully_independent)
- moe_everything (per_head_precompute_kv)
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.model_factory import build_model, SUPPORTED_TYPES, _ARCHIVED_TYPES, _DEPRECATED_TYPES


def _dense_config():
    return {
        "model": {
            "type": "dense",
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "head_dim": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 512,
        },
        "training": {},
    }


def _moe_config(model_type="standard_moe", router_type="softmax", **extra):
    cfg = {
        "model": {
            "type": model_type,
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "head_dim": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "moe_intermediate_size": 32,
            "intermediate_size": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 512,
            "router_type": router_type,
        },
        "training": {},
    }
    cfg["model"].update(extra)
    return cfg


def _moe_everything_config(attn_expert_mode):
    return _moe_config(
        model_type="moe_everything",
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        attn_expert_mode=attn_expert_mode,
    )


# Full parametrized test matrix — all 7 required variants
MODEL_CONFIGS = [
    ("dense", _dense_config()),
    ("standard_moe_softmax", _moe_config("standard_moe", "softmax")),
    ("standard_moe_deepseek", _moe_config("standard_moe", "deepseek")),
    ("global_moe_softmax", _moe_config("global_moe", "softmax")),
    ("global_moe_deepseek", _moe_config("global_moe", "deepseek")),
    ("moe_everything_fully_independent", _moe_everything_config("per_head_fully_independent")),
    ("moe_everything_precompute_kv", _moe_everything_config("per_head_precompute_kv")),
]


@pytest.mark.parametrize("name,cfg", MODEL_CONFIGS, ids=[c[0] for c in MODEL_CONFIGS])
def test_forward_backward_step(name, cfg):
    """Test forward pass, backward pass, and optimizer step for each model type."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model, model_cfg = build_model(cfg)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, cfg["model"]["vocab_size"], (batch_size, seq_len)).cuda()
    labels = input_ids

    is_dense = cfg["model"]["type"] == "dense"
    output = model(
        input_ids=input_ids,
        labels=labels,
        **({} if is_dense else {"output_router_logits": True}),
    )

    assert output.loss is not None
    assert output.loss.requires_grad
    loss_value = output.loss.item()
    assert loss_value > 0, f"Loss should be positive, got {loss_value}"

    output.loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "No gradients computed"

    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize("name,cfg", MODEL_CONFIGS, ids=[c[0] for c in MODEL_CONFIGS])
def test_loss_decreases(name, cfg):
    """Test that loss decreases over 3 training steps (loss sanity check)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model, model_cfg = build_model(cfg)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, cfg["model"]["vocab_size"], (batch_size, seq_len)).cuda()
    labels = input_ids

    is_dense = cfg["model"]["type"] == "dense"
    losses = []

    for step in range(3):
        output = model(
            input_ids=input_ids,
            labels=labels,
            **({} if is_dense else {"output_router_logits": True}),
        )
        losses.append(output.loss.item())
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Loss should decrease: final loss should be less than initial loss
    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 3 steps: {losses}. "
        f"This may indicate a bug in the training loop or loss computation."
    )


@pytest.mark.parametrize("name,cfg", MODEL_CONFIGS, ids=[c[0] for c in MODEL_CONFIGS])
def test_checkpoint_roundtrip(name, cfg):
    """Test checkpoint save and load via checkpoint module for each model type."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from src.training.checkpoint import save_checkpoint, load_checkpoint
    from src.training.config import TrainingConfig
    from src.utils.training import build_lr_scheduler

    model, model_cfg = build_model(cfg)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_cfg = TrainingConfig()
    scheduler = build_lr_scheduler(optimizer, train_cfg)

    # Do a step so optimizer has state
    input_ids = torch.randint(0, cfg["model"]["vocab_size"], (2, 16)).cuda()
    is_dense = cfg["model"]["type"] == "dense"
    output = model(
        input_ids=input_ids, labels=input_ids,
        **({} if is_dense else {"output_router_logits": True}),
    )
    output.loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    test_dataset_state = {"file_idx": 3, "seq_idx": 42, "buffer": [100, 200]}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=1, output_dir=tmpdir, tokens_seen=500.0,
            dataset_state=test_dataset_state,
        )

        ckpt_dir = os.path.join(tmpdir, "checkpoint-1")
        assert os.path.isdir(ckpt_dir)
        assert os.path.exists(os.path.join(ckpt_dir, "model.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "training_state.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "data_state.pt"))

        model2, _ = build_model(cfg)
        model2 = model2.cuda()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        scheduler2 = build_lr_scheduler(optimizer2, train_cfg)

        step, data_state, tokens = load_checkpoint(model2, optimizer2, scheduler2, ckpt_dir)
        assert step == 1
        assert tokens == 500.0
        assert data_state is not None, "data_state should be restored"
        assert data_state["file_idx"] == 3
        assert data_state["seq_idx"] == 42
        assert data_state["buffer"] == [100, 200]

        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(input_ids=input_ids)
            out2 = model2(input_ids=input_ids)
        torch.testing.assert_close(out1.logits, out2.logits)


def test_deprecated_types_rejected():
    """Test that deprecated model types (deepseek_standard_moe, etc.) are rejected."""
    for deprecated_type, guidance in _DEPRECATED_TYPES.items():
        cfg = _moe_config(deprecated_type)
        with pytest.raises(ValueError, match="deprecated"):
            build_model(cfg)


def test_archived_types_rejected():
    """Test that archived model types produce clear error messages."""
    for archived_type in _ARCHIVED_TYPES:
        cfg = {"model": {"type": archived_type}, "training": {}}
        with pytest.raises(ValueError, match="archived"):
            build_model(cfg)


def test_unknown_type_rejected():
    """Test that unknown model types produce clear error messages."""
    cfg = {"model": {"type": "nonexistent_model"}, "training": {}}
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(cfg)


def test_bundled_attn_expert_mode_rejected():
    """`attn_expert_mode: "bundled"` is the deprecated 1-router-top-K design.

    The active moe_everything architecture uses H routers per projection, each
    top-1 (`per_head_fully_independent` / `per_head_precompute_kv`). Requesting
    the old mode must fail explicitly so a stale config doesn't silently fall
    back to a valid default and mask the drift.
    """
    cfg = _moe_everything_config("bundled")
    with pytest.raises(ValueError, match="attn_expert_mode"):
        build_model(cfg)


def test_moe_everything_default_attn_expert_mode_is_per_head_fully_independent():
    """Omitting `attn_expert_mode` must produce a valid default, not fall into
    the rejected `"bundled"` mode. This pins the model-factory fallback.
    """
    cfg = _moe_config(
        model_type="moe_everything",
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
    )  # no attn_expert_mode key set
    model, _ = build_model(cfg)
    assert model.config.attn_expert_mode == "per_head_fully_independent"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
