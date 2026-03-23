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

    # HF Qwen3MoeExperts uses a single fused gate_up_proj tensor [num_experts, H, 2*I].
    # Global MoE should have exactly 1 gate_up_proj (under model.global_experts),
    # NOT L copies (one per layer).
    gate_up_names = [n for n, _ in model.named_parameters() if "gate_up_proj" in n]
    assert len(gate_up_names) == 1, (
        f"Expected exactly 1 fused gate_up_proj tensor (in global_experts), "
        f"got {len(gate_up_names)}: {gate_up_names}"
    )
    assert "global_experts" in gate_up_names[0], (
        f"gate_up_proj should be under global_experts, got: {gate_up_names[0]}"
    )
    # Verify it has the right shape: [num_experts, hidden_size, 2*moe_intermediate_size]
    gup = dict(model.named_parameters())[gate_up_names[0]]
    assert gup.shape[0] == cfg.num_experts, (
        f"gate_up_proj dim 0 is {gup.shape[0]}, expected {cfg.num_experts}"
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


# ── Mixture-of-Everything tests ─────────────────────────────────────────────

from src.models.mixture_of_everything import MoEverythingConfig, MoEverythingForCausalLM


def tiny_moe_everything_config(mode="bundled"):
    return MoEverythingConfig(
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
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        attn_expert_mode=mode,
        norm_topk_prob=True,
        branch_router_aux_loss_coef=0.01,
        router_aux_loss_coef=0.01,
    )


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_instantiates(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode))
    assert model is not None


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_forward(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, 256)


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_all_grads(mode):
    """Every parameter must receive a gradient."""
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    # precompute_kv computes fresh per-expert KV tables at each layer,
    # so the initial KV projections from embedding aren't used.
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    if mode == "precompute_kv":
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None and n not in skip]
    else:
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_loss_decreases(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids, labels = _dummy_batch()

    out1 = model(input_ids=ids, labels=labels)
    out1.loss.backward()
    opt.step()
    opt.zero_grad()

    with torch.no_grad():
        out2 = model(input_ids=ids, labels=labels)
    assert out2.loss.item() < out1.loss.item(), f"{mode}: loss did not decrease"


# ── MoE-Everything with DeepSeek routing ───────────────────────────────

def tiny_moe_everything_deepseek_config(mode="bundled"):
    return MoEverythingConfig(
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
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        attn_expert_mode=mode,
        norm_topk_prob=True,
        branch_router_aux_loss_coef=0.01,
        router_aux_loss_coef=0.01,
        use_deepseek_routing=True,
    )


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_deepseek_forward(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config(mode)).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, 256)


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_deepseek_all_grads(mode):
    """Every parameter must receive a gradient with DeepSeek routing."""
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config(mode)).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    if mode == "precompute_kv":
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None and n not in skip]
    else:
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_deepseek_loss_decreases(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config(mode)).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids, labels = _dummy_batch()

    out1 = model(input_ids=ids, labels=labels)
    out1.loss.backward()
    opt.step()
    opt.zero_grad()

    with torch.no_grad():
        out2 = model(input_ids=ids, labels=labels)
    assert out2.loss.item() < out1.loss.item(), f"{mode}: loss did not decrease"


def test_moe_everything_deepseek_routers_found():
    """DeepSeekRouter instances should be discoverable for bias updates."""
    from src.models.router import DeepSeekRouter
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config("bundled"))
    ds_routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    # bundled: 1 attn router + 1 MLP gate = 2
    assert len(ds_routers) == 2, f"Expected 2 DeepSeekRouters, got {len(ds_routers)}"

    model_fi = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config("fully_independent"))
    ds_routers_fi = [m for m in model_fi.modules() if isinstance(m, DeepSeekRouter)]
    # fully_independent: 4 attn routers + 1 MLP gate = 5
    assert len(ds_routers_fi) == 5, f"Expected 5 DeepSeekRouters, got {len(ds_routers_fi)}"



def _expected_attn_router_keys(mode: str) -> set[str]:
    if mode in ("bundled", "precompute_kv"):
        return {"attn"}
    if mode == "kv_paired":
        return {"kv", "q", "o"}
    if mode == "qk_paired":
        return {"qk", "v", "o"}
    if mode == "fully_independent":
        return {"q", "k", "v", "o"}
    raise ValueError(mode)


@pytest.mark.parametrize("mode", ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"])
def test_moe_everything_output_router_fields(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, output_router_logits=True)

    assert out.loss is not None
    assert out.ce_loss is not None
    assert out.branch_aux_loss is not None
    assert out.router_logits is not None and len(out.router_logits) == 2
    assert out.selected_experts is not None and len(out.selected_experts) == 2
    assert out.branch_probs is not None and len(out.branch_probs) == 2
    assert out.attention_router_info is not None and len(out.attention_router_info) == 2
    assert set(out.attention_router_info[0].keys()) == _expected_attn_router_keys(mode)


def test_moe_everything_deepseek_checkpointing_counts_once():
    from src.models.router import DeepSeekRouter

    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config("bundled")).train()
    model.gradient_checkpointing_enable()
    ids, labels = _dummy_batch()

    out = model(input_ids=ids, labels=labels, output_router_logits=True)
    out.loss.backward()

    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    assert len(routers) == 2

    num_tokens = ids.numel()
    num_depths = model.model.num_depths
    mlp_topk = model.config.num_experts_per_tok
    attn_topk = model.config.num_attn_experts_per_tok

    attn_count = model.model.attn_bank.router.local_tokens_per_expert.sum().item()
    mlp_count = model.model.mlp_bank.gate.local_tokens_per_expert.sum().item()

    assert attn_count == pytest.approx(num_tokens * num_depths * attn_topk)
    assert mlp_count == pytest.approx(num_tokens * num_depths * mlp_topk)
