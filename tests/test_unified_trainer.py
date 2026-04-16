"""Smoke tests for the unified trainer with all supported model variants.

Tests: forward pass, backward pass, optimizer step, and checkpoint save/load
for each model type in the supported taxonomy.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.model_factory import build_model, SUPPORTED_TYPES, _ARCHIVED_TYPES


# Minimal config templates for each model type
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


def _moe_everything_config(attn_expert_mode="bundled"):
    return _moe_config(
        model_type="moe_everything",
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        attn_expert_mode=attn_expert_mode,
    )


# Parametrized test matrix
MODEL_CONFIGS = [
    ("dense", _dense_config()),
    ("standard_moe_softmax", _moe_config("standard_moe", "softmax")),
    ("standard_moe_deepseek", _moe_config("standard_moe", "deepseek")),
    ("global_moe_softmax", _moe_config("global_moe", "softmax")),
    ("global_moe_deepseek", _moe_config("global_moe", "deepseek")),
    ("moe_everything_bundled", _moe_everything_config("bundled")),
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

    # Create dummy input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, cfg["model"]["vocab_size"], (batch_size, seq_len)).cuda()
    labels = input_ids

    # Forward pass
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

    # Backward pass
    output.loss.backward()

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "No gradients computed"

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize("name,cfg", MODEL_CONFIGS, ids=[c[0] for c in MODEL_CONFIGS])
def test_checkpoint_roundtrip(name, cfg):
    """Test checkpoint save and load for each model type."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model, model_cfg = build_model(cfg)
    model = model.cuda()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        torch.save(model.state_dict(), os.path.join(tmpdir, "model.pt"))

        # Load into fresh model
        model2, _ = build_model(cfg)
        model2 = model2.cuda()
        state = torch.load(os.path.join(tmpdir, "model.pt"), map_location="cuda")
        model2.load_state_dict(state)

        # Verify outputs match
        input_ids = torch.randint(0, cfg["model"]["vocab_size"], (1, 16)).cuda()
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(input_ids=input_ids)
            out2 = model2(input_ids=input_ids)
        torch.testing.assert_close(out1.logits, out2.logits)


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


def test_deprecated_alias_works():
    """Test that deprecated aliases still work via the model factory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = _moe_config("standard_moe", "deepseek")
    # Simulate the old alias
    cfg["model"]["type"] = "deepseek_standard_moe"
    model, _ = build_model(cfg)
    assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
