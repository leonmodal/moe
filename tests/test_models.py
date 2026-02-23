"""
Tests for Standard MoE and Global MoE models.
Run with: uv run pytest tests/ -v
"""
import pytest
import torch
from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM
from src.models.global_moe import GlobalMoEModel


# ── Tiny config for fast CPU tests ──────────────────────────────────────────

def tiny_standard_config():
    return Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        output_router_logits=True,
        norm_topk_prob=True,
    )


def tiny_global_config():
    return GlobalMoEConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=8,          # global pool = 2 layers × 4 per-layer
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        output_router_logits=True,
        norm_topk_prob=True,
    )


# ── Model instantiation ──────────────────────────────────────────────────────

def test_standard_moe_instantiates():
    model = StandardMoEModel(tiny_standard_config())
    assert model is not None


def test_global_moe_instantiates():
    model = GlobalMoEForCausalLM(tiny_global_config())
    assert model is not None


# ── Forward pass produces correct outputs ───────────────────────────────────

@pytest.fixture
def standard_model():
    return StandardMoEModel(tiny_standard_config()).eval()


@pytest.fixture
def global_model():
    return GlobalMoEForCausalLM(tiny_global_config()).eval()


def _dummy_batch(vocab_size=256, B=2, T=16):
    ids = torch.randint(0, vocab_size, (B, T))
    return ids, ids  # input_ids, labels


def test_standard_forward(standard_model):
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = standard_model(input_ids=ids, labels=labels, output_router_logits=True)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, 256)


def test_global_forward(global_model):
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, 256)


def test_router_logits_present(standard_model, global_model):
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out_s = standard_model(input_ids=ids, labels=labels, output_router_logits=True)
        out_g = global_model(input_ids=ids, labels=labels, output_router_logits=True)

    # Should have one tensor per layer
    assert out_s.router_logits is not None
    assert len(out_s.router_logits) == 2    # num_hidden_layers=2

    assert out_g.router_logits is not None
    assert len(out_g.router_logits) == 2


def test_router_logits_shape(standard_model, global_model):
    ids, _ = _dummy_batch()
    with torch.no_grad():
        out_s = standard_model(input_ids=ids, output_router_logits=True)
        out_g = global_model(input_ids=ids, output_router_logits=True)

    T = 2 * 16  # B * seq_len (flattened)

    # Standard: [T, num_experts_per_layer] = [32, 4]
    assert out_s.router_logits[0].shape == (T, 4)

    # Global: [T, global_pool] = [32, 8]
    assert out_g.router_logits[0].shape == (T, 8)


# ── Global MoE: experts are shared (not duplicated) ─────────────────────────

def test_global_experts_are_shared():
    """All layers must route into the SAME expert pool object — not copies."""
    model = GlobalMoEForCausalLM(tiny_global_config())
    inner: GlobalMoEModel = model.model

    global_experts_id = id(inner.global_experts)

    # No layer should own a separate expert pool
    for layer in inner.layers:
        # The layer's mlp (GlobalSparseMoeBlock) has a gate but no experts attribute
        assert not hasattr(layer.mlp, "experts"), (
            "Layer MLP should NOT own experts — they must live on GlobalMoEModel"
        )

    # global_experts is registered exactly once on the model
    expert_modules = [
        (name, mod) for name, mod in model.named_modules()
        if "gate_up_proj" in name and "weight" not in name
    ]
    # All expert weight paths should go through model.global_experts, not model.layers.*
    for name, _ in model.named_parameters():
        if "gate_up_proj" in name or ("down_proj" in name and "layers" not in name.split(".")[2:3]):
            assert "global_experts" in name, (
                f"Expert param found outside global_experts: {name}"
            )


def test_global_no_duplicate_expert_params():
    """Global MoE should have exactly ONE expert pool, not L pools."""
    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg)

    # Count expert parameters
    expert_param_names = [n for n, _ in model.named_parameters()
                          if "gate_up_proj" in n or "down_proj" in n]

    # In standard MoE with L=2, E=4 experts per layer we'd have 2×4=8 gate_up_proj tensors
    # In global MoE with E=8 in one pool we should have exactly 8 (not 16)
    gate_up_names = [n for n in expert_param_names if "gate_up_proj" in n]
    assert len(gate_up_names) == cfg.num_experts, (
        f"Expected {cfg.num_experts} gate_up_proj tensors (one per expert), "
        f"got {len(gate_up_names)}"
    )


def test_standard_vs_global_expert_param_count():
    """Standard and Global MoE at matched scale should have similar expert param counts."""
    # Standard: 2 layers × 4 experts = 8 total expert instances
    std_cfg = tiny_standard_config()   # num_experts=4, layers=2
    std = StandardMoEModel(std_cfg)
    std_expert_params = sum(
        p.numel() for n, p in std.named_parameters()
        if "gate_up_proj" in n or "down_proj" in n
    )

    # Global: 1 pool × 8 experts (= 2 layers × 4)
    glb_cfg = tiny_global_config()    # num_experts=8, layers=2
    glb = GlobalMoEForCausalLM(glb_cfg)
    glb_expert_params = sum(
        p.numel() for n, p in glb.named_parameters()
        if "gate_up_proj" in n or "down_proj" in n
    )

    assert std_expert_params == glb_expert_params, (
        f"Standard expert params ({std_expert_params}) != Global ({glb_expert_params})"
    )


# ── Loss sanity ──────────────────────────────────────────────────────────────

def test_loss_decreases_with_gradient_step():
    """A single gradient step should reduce the loss."""
    model = StandardMoEModel(tiny_standard_config()).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids, labels = _dummy_batch()

    out1 = model(input_ids=ids, labels=labels, output_router_logits=True)
    loss1 = out1.loss
    loss1.backward()
    opt.step()
    opt.zero_grad()

    with torch.no_grad():
        out2 = model(input_ids=ids, labels=labels, output_router_logits=True)
    loss2 = out2.loss

    assert loss2.item() < loss1.item(), "Loss did not decrease after one gradient step"
