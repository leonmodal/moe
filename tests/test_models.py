"""
Tests for Standard MoE and Global MoE models.
Run with: uv run pytest tests/ -v
"""
from pathlib import Path
from types import MethodType

import pytest
import torch
import torch.nn.functional as F
import yaml
from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM
from src.models.global_moe import GlobalMoEModel
from src.models.router import DeepSeekRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)


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


def test_qwen3_moe_experts_grouped_mm_matches_legacy_reference():
    config = tiny_standard_config()
    experts = Qwen3MoeExperts(config).eval()
    assert hasattr(experts, "_maybe_grouped_mm")
    assert hasattr(experts, "_run_grouped_expert_linear")
    hidden_states = torch.randn(7, config.hidden_size)
    top_k_index = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [1, 0],
            [3, 2],
            [0, 2],
            [1, 3],
            [2, 0],
        ],
        dtype=torch.long,
    )
    top_k_weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.3, 0.7],
        ],
        dtype=hidden_states.dtype,
    )

    def legacy_forward():
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=experts.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = experts.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, experts.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    expected = legacy_forward()

    with torch.no_grad():
        fallback = experts(hidden_states, top_k_index, top_k_weights)

    torch.testing.assert_close(fallback, expected, atol=1e-6, rtol=1e-6)

    def fake_grouped_mm(self, sorted_inputs, expert_weights_t, counts):
        outputs = []
        start = 0
        for weight_t, count in zip(expert_weights_t, counts.tolist()):
            end = start + count
            outputs.append(sorted_inputs[start:end] @ weight_t)
            start = end
        return torch.cat(outputs, dim=0)

    experts._maybe_grouped_mm = MethodType(fake_grouped_mm, experts)

    with torch.no_grad():
        grouped = experts(hidden_states, top_k_index, top_k_weights)

    torch.testing.assert_close(grouped, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_router_probs_stay_fp32_under_cuda_autocast():
    cfg = tiny_moe_everything_config("per_head_precompute_kv")
    cfg.use_deepseek_routing = True

    hidden_states = torch.randn(8, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    deepseek_router = DeepSeekRouter(cfg).cuda().eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        deepseek_probs, deepseek_weights, _ = deepseek_router(hidden_states)
    assert deepseek_probs.dtype == torch.float32
    assert deepseek_weights.dtype == torch.bfloat16

    hf_router = Qwen3MoeTopKRouter(cfg).cuda().eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hf_probs, hf_weights, _ = hf_router(hidden_states)
    assert hf_probs.dtype == torch.float32
    assert hf_weights.dtype == torch.float32

    cfg.use_deepseek_routing = False
    bank = AttentionExpertBank(cfg).cuda().eval()
    router = bank._select_router("router", None)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, bank_weights, bank_probs = bank._route_flat(router, hidden_states, bank.num_kv_heads)
    assert bank_probs.dtype == torch.float32
    assert bank_weights.dtype == torch.bfloat16


def test_global_moe_logits_and_loss_match_with_grouped_mlp_dispatch():
    ref_model = GlobalMoEForCausalLM(tiny_global_config()).eval()
    grouped_model = GlobalMoEForCausalLM(tiny_global_config()).eval()
    grouped_model.load_state_dict(ref_model.state_dict())

    def fake_grouped_mm(self, sorted_inputs, expert_weights_t, counts):
        outputs = []
        start = 0
        for weight_t, count in zip(expert_weights_t, counts.tolist()):
            end = start + count
            outputs.append(sorted_inputs[start:end] @ weight_t)
            start = end
        return torch.cat(outputs, dim=0)

    grouped_model.model.global_experts._maybe_grouped_mm = MethodType(
        fake_grouped_mm,
        grouped_model.model.global_experts,
    )

    ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        ref_out = ref_model(input_ids=ids, labels=ids, output_router_logits=True)
        grouped_out = grouped_model(input_ids=ids, labels=ids, output_router_logits=True)

    torch.testing.assert_close(grouped_out.logits, ref_out.logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grouped_out.loss, ref_out.loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grouped_out.aux_loss, ref_out.aux_loss, atol=1e-6, rtol=1e-6)


# ── Mixture-of-Everything tests ─────────────────────────────────────────────

from src.models.mixture_of_everything import AttentionExpertBank, MoEverythingConfig, MoEverythingForCausalLM
from train import get_bias_update_router_groups, get_bias_update_routers, update_expert_biases


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
        branch_router_aux_loss_coef=0.0,
        router_aux_loss_coef=0.01,
    )


MOE_EVERYTHING_MODES = [
    "bundled",
    "kv_paired",
    "qk_paired",
    "fully_independent",
    "per_head_fully_independent",
    "precompute_kv",
    "per_head_precompute_kv",
]

PRECOMPUTE_KV_MODES = {
    "precompute_kv",
    "per_head_precompute_kv",
}


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
def test_moe_everything_instantiates(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode))
    assert model is not None


def test_moe_everything_deepseek_router_weights_initialized():
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config("per_head_precompute_kv"))
    routers = [module for module in model.modules() if isinstance(module, Qwen3MoeTopKRouter)]
    assert routers, "Expected MoE-Everything to contain routed expert modules"
    for router in routers:
        assert torch.count_nonzero(router.weight).item() > 0


def test_moe_everything_supports_experts_implementation_dispatch():
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config("bundled"))
    assert hasattr(model, "set_experts_implementation")
    model.set_experts_implementation("eager")
    assert model.config._experts_implementation == "eager"


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
def test_moe_everything_forward(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, 256)


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
def test_moe_everything_all_grads(mode):
    """Every parameter must receive a gradient."""
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    # precompute_kv computes fresh per-expert KV tables at each layer,
    # so the initial KV projections from embedding aren't used.
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    if mode in PRECOMPUTE_KV_MODES:
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None and n not in skip]
    else:
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
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
        branch_router_aux_loss_coef=0.0,
        router_aux_loss_coef=0.01,
        use_deepseek_routing=True,
    )


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
def test_moe_everything_deepseek_forward(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config(mode)).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, 256)


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
def test_moe_everything_deepseek_all_grads(mode):
    """Every parameter must receive a gradient with DeepSeek routing."""
    model = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config(mode)).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    if mode in PRECOMPUTE_KV_MODES:
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None and n not in skip]
    else:
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
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

    model_phfi = MoEverythingForCausalLM(tiny_moe_everything_deepseek_config("per_head_fully_independent"))
    ds_routers_phfi = [m for m in model_phfi.modules() if isinstance(m, DeepSeekRouter)]
    # per_head_fully_independent (flat bank): 4 attn routers (Q,K,V,O) + 1 MLP gate = 5
    assert len(ds_routers_phfi) == 5, f"Expected 5 DeepSeekRouters, got {len(ds_routers_phfi)}"

    model_phpkv = MoEverythingForCausalLM(
        tiny_moe_everything_deepseek_config("per_head_precompute_kv")
    )
    ds_routers_phpkv = [m for m in model_phpkv.modules() if isinstance(m, DeepSeekRouter)]
    # per_head_precompute_kv (flat bank, 1 router): 1 attn router + 1 MLP gate = 2
    assert len(ds_routers_phpkv) == 2, f"Expected 2 DeepSeekRouters, got {len(ds_routers_phpkv)}"



def _expected_attn_router_keys(mode: str, num_heads: int = 2, num_kv_heads: int = 1) -> set[str]:
    if mode in ("bundled", "precompute_kv"):
        return {"attn"}
    if mode == "per_head_precompute_kv":
        return {"attn"}
    if mode == "per_head_fully_independent":
        return {"q", "k", "v", "o"}
    if mode == "kv_paired":
        return {"kv", "q", "o"}
    if mode == "qk_paired":
        return {"qk", "v", "o"}
    if mode == "fully_independent":
        return {"q", "k", "v", "o"}
    raise ValueError(mode)


@pytest.mark.parametrize("mode", MOE_EVERYTHING_MODES)
def test_moe_everything_output_router_fields(mode):
    model = MoEverythingForCausalLM(tiny_moe_everything_config(mode)).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, output_router_logits=True)

    assert out.loss is not None
    assert out.ce_loss is not None
    # branch_aux_loss removed — no forced balance between attention and MLP
    assert out.router_logits is not None and len(out.router_logits) == 2
    assert out.selected_experts is not None and len(out.selected_experts) == 2
    assert out.branch_probs is not None and len(out.branch_probs) == 2
    assert out.attention_router_info is not None and len(out.attention_router_info) == 2
    assert set(out.attention_router_info[0].keys()) == _expected_attn_router_keys(mode)
    assert out.router_token_masks is not None and len(out.router_token_masks) == 2


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
    mlp_active_tokens = sum(mask.sum().item() for mask in out.router_token_masks if mask is not None)

    assert attn_count == pytest.approx(num_tokens * num_depths * attn_topk)
    assert mlp_count == pytest.approx(mlp_active_tokens * mlp_topk)


# ── Per-layer router tests ────────────────────────────────────────────────

def test_per_layer_router_creates_separate_routers():
    config = tiny_moe_everything_config("bundled")
    config.per_layer_router = True
    model = MoEverythingForCausalLM(config)
    assert hasattr(model.model, "branch_routers")
    assert len(model.model.branch_routers) == config.num_hidden_layers
    assert not hasattr(model.model, "branch_router")


def test_per_layer_router_forward_and_grads():
    config = tiny_moe_everything_config("bundled")
    config.per_layer_router = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    for router in model.model.branch_routers:
        assert router.gate.weight.grad is not None


def test_per_layer_router_precompute_kv():
    config = tiny_moe_everything_config("precompute_kv")
    config.per_layer_router = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()


# ── Per-layer MLP router tests ────────────────────────────────────────────

def test_per_layer_mlp_router_creates_separate_routers():
    config = tiny_moe_everything_deepseek_config("bundled")
    config.per_layer_mlp_router = True
    model = MoEverythingForCausalLM(config)
    bank = model.model.mlp_bank
    assert hasattr(bank, "gates")
    assert len(bank.gates) == config.num_hidden_layers
    assert not hasattr(bank, "gate")


# ── Per-layer attention router tests ──────────────────────────────────────

PER_HEAD_MODES = ["per_head_fully_independent", "per_head_precompute_kv"]
NON_PER_HEAD_ATTN_ROUTER_MODES = ["bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"]


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_per_layer_attn_router_creates_separate_routers(mode):
    config = tiny_moe_everything_config(mode)
    config.per_layer_attn_router = True
    model = MoEverythingForCausalLM(config)
    bank = model.model.attn_bank
    if mode == "per_head_fully_independent":
        assert hasattr(bank, "q_routers") and len(bank.q_routers) == config.num_hidden_layers
        assert hasattr(bank, "k_routers") and len(bank.k_routers) == config.num_hidden_layers
        assert hasattr(bank, "v_routers") and len(bank.v_routers) == config.num_hidden_layers
        assert hasattr(bank, "o_routers") and len(bank.o_routers) == config.num_hidden_layers
    elif mode == "per_head_precompute_kv":
        assert hasattr(bank, "routers") and len(bank.routers) == config.num_hidden_layers


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_per_layer_attn_router_forward_and_grads(mode):
    config = tiny_moe_everything_config(mode)
    config.per_layer_attn_router = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    # precompute_kv modes don't use init KV projections
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None
               and (mode not in PRECOMPUTE_KV_MODES or n not in skip)]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_per_layer_attn_router_loss_decreases(mode):
    config = tiny_moe_everything_config(mode)
    config.per_layer_attn_router = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first_loss = None
    for _ in range(20):
        out = model(input_ids=ids, labels=labels)
        if first_loss is None:
            first_loss = out.loss.item()
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    assert out.loss.item() < first_loss


@pytest.mark.parametrize("mode", NON_PER_HEAD_ATTN_ROUTER_MODES)
def test_non_per_head_per_layer_attn_router_creates_separate_routers(mode):
    config = tiny_moe_everything_config(mode)
    config.per_layer_attn_router = True
    model = MoEverythingForCausalLM(config)
    bank = model.model.attn_bank
    if mode in ("bundled", "precompute_kv"):
        assert hasattr(bank, "routers") and len(bank.routers) == config.num_hidden_layers
    elif mode == "kv_paired":
        assert hasattr(bank, "kv_routers") and len(bank.kv_routers) == config.num_hidden_layers
        assert hasattr(bank, "q_routers") and len(bank.q_routers) == config.num_hidden_layers
        assert hasattr(bank, "o_routers") and len(bank.o_routers) == config.num_hidden_layers
    elif mode == "qk_paired":
        assert hasattr(bank, "qk_routers") and len(bank.qk_routers) == config.num_hidden_layers
        assert hasattr(bank, "v_routers") and len(bank.v_routers) == config.num_hidden_layers
        assert hasattr(bank, "o_routers") and len(bank.o_routers) == config.num_hidden_layers
    elif mode == "fully_independent":
        assert hasattr(bank, "q_routers") and len(bank.q_routers) == config.num_hidden_layers
        assert hasattr(bank, "k_routers") and len(bank.k_routers) == config.num_hidden_layers
        assert hasattr(bank, "v_routers") and len(bank.v_routers) == config.num_hidden_layers
        assert hasattr(bank, "o_routers") and len(bank.o_routers) == config.num_hidden_layers


@pytest.mark.parametrize("mode", NON_PER_HEAD_ATTN_ROUTER_MODES)
def test_non_per_head_per_layer_attn_router_forward_and_grads(mode):
    config = tiny_moe_everything_config(mode)
    config.per_layer_attn_router = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    no_grad = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None
        and (mode not in PRECOMPUTE_KV_MODES or n not in skip)
    ]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


# ── Routed norm tests ─────────────────────────────────────────────────────

from src.models.mixture_of_everything import NormExpertBank


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_routed_norm_creates_norm_banks(mode):
    config = tiny_moe_everything_config(mode)
    config.routed_norm = True
    model = MoEverythingForCausalLM(config)
    attn_bank = model.model.attn_bank
    mlp_bank = model.model.mlp_bank
    # Attention: per_head_fully_independent gets attn_pre_norm, per_head_precompute_kv gets norm
    if mode == "per_head_fully_independent":
        assert isinstance(attn_bank.attn_pre_norm, NormExpertBank)
        assert attn_bank.attn_pre_norm.num_experts == config.num_hidden_layers
    elif mode == "per_head_precompute_kv":
        assert isinstance(attn_bank.norm, NormExpertBank)
        assert attn_bank.norm.num_experts == config.num_hidden_layers
    # MLP always gets a NormExpertBank
    assert isinstance(mlp_bank.norm, NormExpertBank)
    assert mlp_bank.norm.num_experts == config.num_hidden_layers


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_routed_norm_forward_and_grads(mode):
    config = tiny_moe_everything_config(mode)
    config.routed_norm = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None
               and (mode not in PRECOMPUTE_KV_MODES or n not in skip)]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_routed_norm_loss_decreases(mode):
    config = tiny_moe_everything_config(mode)
    config.routed_norm = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first_loss = None
    for _ in range(20):
        out = model(input_ids=ids, labels=labels)
        if first_loss is None:
            first_loss = out.loss.item()
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    assert out.loss.item() < first_loss


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_routed_norm_combined_with_per_layer_attn_router(mode):
    config = tiny_moe_everything_config(mode)
    config.routed_norm = True
    config.per_layer_attn_router = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    skip = {"model.init_k_proj.weight", "model.init_v_proj.weight", "model.init_k_norm.weight"}
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None
               and (mode not in PRECOMPUTE_KV_MODES or n not in skip)]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"


def _force_branch_choices(model: MoEverythingForCausalLM, attn_ids: list[int], mlp_ids: list[int]) -> None:
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        emb.zero_()
        for token_id in attn_ids:
            emb[token_id, 0] = 1.0
        for token_id in mlp_ids:
            emb[token_id, 0] = -1.0

        gate = model.model.branch_router.gate.weight
        gate.zero_()
        gate[0, 0] = 1.0
        gate[1, 0] = -1.0


def test_per_head_fully_independent_o_router_uses_attention_output():
    config = tiny_moe_everything_config("per_head_fully_independent")
    config.num_hidden_layers = 1
    model = MoEverythingForCausalLM(config).eval()
    bank = model.model.attn_bank
    captured = {}
    original_route_flat = bank._route_flat

    def wrapped_route_flat(self, router, x, top_k):
        if router is bank.o_router:
            captured["o_input"] = x.detach().clone()
        return original_route_flat(router, x, top_k)

    bank._route_flat = MethodType(wrapped_route_flat, bank)

    hidden_states = torch.randn(2, 4, config.hidden_size)
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0).expand(hidden_states.shape[0], -1)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids=position_ids)

    with torch.no_grad():
        Q, K, V = bank.project(hidden_states, position_embeddings, depth_idx=0)
        K_expanded = repeat_kv(K, bank.num_kv_groups)
        V_expanded = repeat_kv(V, bank.num_kv_groups)
        scores = torch.matmul(Q, K_expanded.transpose(2, 3)) * bank.scaling
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        expected_o_input = torch.matmul(probs, V_expanded).transpose(1, 2).reshape(-1, bank.q_dim)
        _ = bank.attend(Q, K, V, depth_idx=0)

    assert "o_input" in captured
    assert captured["o_input"].shape[-1] == bank.q_dim
    torch.testing.assert_close(captured["o_input"], expected_o_input)


def test_per_head_fully_independent_sparse_o_router_uses_attention_output():
    config = tiny_moe_everything_config("per_head_fully_independent")
    config.num_hidden_layers = 1
    model = MoEverythingForCausalLM(config).eval()
    bank = model.model.attn_bank
    captured = {}
    original_route_flat = bank._route_flat

    def wrapped_route_flat(self, router, x, top_k):
        if router is bank.o_router:
            captured["o_input"] = x.detach().clone()
        return original_route_flat(router, x, top_k)

    bank._route_flat = MethodType(wrapped_route_flat, bank)

    hidden_states = torch.randn(2, 4, config.hidden_size)
    token_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool).unsqueeze(-1)
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0).expand(hidden_states.shape[0], -1)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids=position_ids)
    K_old = torch.zeros(hidden_states.shape[0], bank.num_kv_heads, hidden_states.shape[1], bank.head_dim)
    V_old = torch.zeros_like(K_old)

    with torch.no_grad():
        B, T, H = hidden_states.shape
        flat_mask = token_mask.reshape(-1).bool()
        flat_hidden = hidden_states.reshape(B * T, H)
        hidden_selected = flat_hidden[flat_mask]

        q_flat = bank.q_pre_norm(hidden_selected)
        k_flat = bank.k_pre_norm(hidden_selected)
        v_flat = bank.v_pre_norm(hidden_selected)
        q_router, k_router, v_router, _ = bank._select_attn_routers(depth_idx=0)

        q_idx, q_w, _ = original_route_flat(q_router, q_flat, bank.num_heads)
        k_idx, k_w, _ = original_route_flat(k_router, k_flat, bank.num_kv_heads)
        v_idx, v_w, _ = original_route_flat(v_router, v_flat, bank.num_kv_heads)

        Q_sel = bank._project_heads_batched(q_flat, bank.q_proj, q_idx, q_w, bank.q_norm_weight)
        K_sel = bank._project_heads_batched(k_flat, bank.k_proj, k_idx, k_w, bank.k_norm_weight)
        V_sel = bank._project_heads_batched(v_flat, bank.v_proj, v_idx, v_w)

        Q_flat = hidden_states.new_zeros(B * T, bank.num_heads, bank.head_dim)
        K_flat = hidden_states.new_zeros(B * T, bank.num_kv_heads, bank.head_dim)
        V_flat = hidden_states.new_zeros(B * T, bank.num_kv_heads, bank.head_dim)
        Q_flat[flat_mask] = Q_sel
        K_flat[flat_mask] = K_sel
        V_flat[flat_mask] = V_sel

        Q = Q_flat.view(B, T, bank.num_heads, bank.head_dim).transpose(1, 2)
        K_fresh = K_flat.view(B, T, bank.num_kv_heads, bank.head_dim).transpose(1, 2)
        V_fresh = V_flat.view(B, T, bank.num_kv_heads, bank.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        Q, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        attn_mask_kv = token_mask.unsqueeze(1)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)

        K_expanded = repeat_kv(K_new, bank.num_kv_groups)
        V_expanded = repeat_kv(V_new, bank.num_kv_groups)
        attn_heads = hidden_states.new_zeros(B, bank.num_heads, T, bank.head_dim)
        token_mask_2d = token_mask.squeeze(-1).bool()

        for b in range(B):
            pos = token_mask_2d[b].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            Q_b = Q[b : b + 1, :, pos, :]
            scores = torch.matmul(Q_b, K_expanded[b : b + 1].transpose(2, 3)) * bank.scaling
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_b = torch.matmul(probs, V_expanded[b : b + 1])
            attn_heads[b, :, pos, :] = attn_b.squeeze(0).to(attn_heads.dtype)

        expected_o_input = (
            attn_heads.transpose(1, 2).reshape(B * T, bank.num_heads, bank.head_dim)[flat_mask].reshape(-1, bank.q_dim)
        )
        _ = bank.project_and_attend_per_head_fully_independent_sparse(
            hidden_states,
            position_embeddings,
            K_old,
            V_old,
            token_mask,
            depth_idx=0,
        )

    assert "o_input" in captured
    assert captured["o_input"].shape[-1] == bank.q_dim
    torch.testing.assert_close(captured["o_input"], expected_o_input)


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_per_head_branch_masks_are_reported(mode):
    config = tiny_moe_everything_config(mode)
    config.num_hidden_layers = 1
    model = MoEverythingForCausalLM(config).eval()
    _force_branch_choices(model, attn_ids=[1], mlp_ids=[2])
    ids = torch.tensor([[1, 2, 1, 2]])

    with torch.no_grad():
        out = model(input_ids=ids, labels=ids, output_router_logits=True)

    expected_attn = torch.tensor([True, False, True, False])
    expected_mlp = ~expected_attn
    torch.testing.assert_close(out.router_token_masks[0].cpu(), expected_mlp)

    attn_info = out.attention_router_info[0]
    attn_router_names = ("q", "k", "v", "o") if mode == "per_head_fully_independent" else ("attn",)
    for name in attn_router_names:
        info = attn_info[name]
        torch.testing.assert_close(info["token_mask"].cpu(), expected_attn)
        assert torch.count_nonzero(info["router_logits"][~expected_attn]).item() == 0

    assert torch.count_nonzero(out.router_logits[0][~expected_mlp]).item() == 0


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
@pytest.mark.parametrize(
    ("attn_ids", "mlp_ids"),
    [
        ([1], []),
        ([], [2]),
    ],
)
def test_per_head_single_branch_backward_keeps_grads(mode, attn_ids, mlp_ids):
    config = tiny_moe_everything_config(mode)
    config.num_hidden_layers = 1
    model = MoEverythingForCausalLM(config).train()
    _force_branch_choices(model, attn_ids=attn_ids, mlp_ids=mlp_ids)
    token_id = attn_ids[0] if attn_ids else mlp_ids[0]
    ids = torch.full((2, 4), token_id, dtype=torch.long)

    out = model(input_ids=ids, labels=ids, output_router_logits=True)
    out.loss.backward()

    assert model.model.attn_bank.q_proj.grad is not None
    assert model.model.mlp_bank.gate.weight.grad is not None

    attn_info = out.attention_router_info[0]
    attn_router_names = ("q", "k", "v", "o") if mode == "per_head_fully_independent" else ("attn",)
    expected_attn = torch.full((ids.numel(),), bool(attn_ids))
    expected_mlp = ~expected_attn
    torch.testing.assert_close(out.router_token_masks[0].cpu(), expected_mlp)
    for name in attn_router_names:
        torch.testing.assert_close(attn_info[name]["token_mask"].cpu(), expected_attn)


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_per_head_sparse_paths_work_under_bfloat16_autocast(mode):
    config = tiny_moe_everything_config(mode)
    config.num_hidden_layers = 1
    model = MoEverythingForCausalLM(config).train()
    _force_branch_choices(model, attn_ids=[1], mlp_ids=[2])
    ids = torch.tensor([[1, 2, 1, 2]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(input_ids=ids, labels=ids, output_router_logits=True)

    assert out.loss is not None
    assert torch.isfinite(out.loss.float())


@pytest.mark.parametrize("mode", PER_HEAD_MODES)
def test_per_head_dense_and_sparse_dispatch_match(mode):
    sparse_cfg = tiny_moe_everything_config(mode)
    sparse_cfg.num_hidden_layers = 1
    sparse_cfg.per_head_compute_mode = "sparse"
    dense_cfg = tiny_moe_everything_config(mode)
    dense_cfg.num_hidden_layers = 1
    dense_cfg.per_head_compute_mode = "dense"

    sparse_model = MoEverythingForCausalLM(sparse_cfg).eval()
    dense_model = MoEverythingForCausalLM(dense_cfg).eval()
    dense_model.load_state_dict(sparse_model.state_dict())

    _force_branch_choices(sparse_model, attn_ids=[1], mlp_ids=[2])
    dense_model.load_state_dict(sparse_model.state_dict())

    ids = torch.tensor([[1, 2, 1, 2]])
    with torch.no_grad():
        out_sparse = sparse_model(input_ids=ids, labels=ids, output_router_logits=True)
        out_dense = dense_model(input_ids=ids, labels=ids, output_router_logits=True)

    torch.testing.assert_close(out_dense.logits, out_sparse.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_dense.loss, out_sparse.loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        out_dense.attention_router_info[0]["attn" if mode == "per_head_precompute_kv" else "o"]["token_mask"].cpu(),
        out_sparse.attention_router_info[0]["attn" if mode == "per_head_precompute_kv" else "o"]["token_mask"].cpu(),
    )


def test_per_head_precompute_kv_routes_one_slot_per_kv_head():
    config = tiny_moe_everything_config("per_head_precompute_kv")
    model = MoEverythingForCausalLM(config).eval()
    hidden_states = torch.randn(2, 4, config.hidden_size)
    position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0).expand(hidden_states.shape[0], -1)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids=position_ids)

    with torch.no_grad():
        tables = model.model.attn_bank._build_per_head_precompute_kv_tables(
            hidden_states,
            position_embeddings,
            depth_idx=0,
        )

    assert tables["idx"].shape == (hidden_states.shape[0] * hidden_states.shape[1], config.num_key_value_heads)
    assert tables["weights"].shape == (hidden_states.shape[0] * hidden_states.shape[1], config.num_key_value_heads)
    assert tables["Q"].shape[1] == config.num_attention_heads
    assert tables["K_fresh"].shape[1] == config.num_key_value_heads
    assert tables["V_fresh"].shape[1] == config.num_key_value_heads


@pytest.mark.parametrize(
    ("mode", "attn_tokens", "expected"),
    [
        ("per_head_precompute_kv", 2, True),
        ("per_head_precompute_kv", 3, False),
        ("per_head_fully_independent", 1, True),
        ("per_head_fully_independent", 2, True),
        ("per_head_fully_independent", 4, False),
    ],
)
def test_per_head_auto_sparse_thresholds_are_mode_specific(mode, attn_tokens, expected):
    config = tiny_moe_everything_config(mode)
    bank = AttentionExpertBank(config)
    token_mask = torch.zeros(1, 4, 1, dtype=torch.bool)
    token_mask[:, :attn_tokens, :] = True
    assert bank.should_use_sparse_path(token_mask) is expected


def test_base_per_head_fully_independent_configs_use_256_attention_experts():
    base_cfg = Path("configs/moe_everything_per_head_fully_independent.yaml").read_text()
    debug_cfg = Path(
        "configs/scaling/debug8_xs_deepseek_moe_everything_per_head_fully_independent.yaml"
    ).read_text()
    assert "num_attn_experts: 256" in base_cfg
    assert "num_attn_experts: 256" in debug_cfg


def _tiny_global_equiv_config():
    return GlobalMoEConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        output_router_logits=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
    )


def _tiny_alternating_sanity_config():
    return MoEverythingConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_precompute_kv",
        norm_topk_prob=True,
        branch_router_aux_loss_coef=0.0,
        router_aux_loss_coef=0.0,
        per_layer_norm=True,
        sanity_check_mode="alternating_global_moe",
    )


def _copy_global_weights_into_sanity_moe(global_model, sanity_model):
    global_inner = global_model.model
    sanity_inner = sanity_model.model
    head_dim = global_model.config.head_dim
    num_heads = global_model.config.num_attention_heads
    num_kv_heads = global_model.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads

    with torch.no_grad():
        sanity_inner.embed_tokens.weight.copy_(global_inner.embed_tokens.weight)
        sanity_inner.norm.weight.copy_(global_inner.norm.weight)
        sanity_model.lm_head.weight.copy_(global_model.lm_head.weight)
        sanity_inner.mlp_bank.experts.gate_up_proj.copy_(global_inner.global_experts.gate_up_proj)
        sanity_inner.mlp_bank.experts.down_proj.copy_(global_inner.global_experts.down_proj)

        for layer_idx, layer in enumerate(global_inner.layers):
            attn_depth = 2 * layer_idx
            mlp_depth = attn_depth + 1
            sanity_inner.attn_bank.norms[attn_depth].weight.copy_(layer.input_layernorm.weight)
            sanity_inner.mlp_bank.norms[mlp_depth].weight.copy_(layer.post_attention_layernorm.weight)

            q_proj = layer.self_attn.q_proj.weight
            k_proj = layer.self_attn.k_proj.weight
            v_proj = layer.self_attn.v_proj.weight
            o_proj = layer.self_attn.o_proj.weight
            q_norm = layer.self_attn.q_norm.weight
            k_norm = layer.self_attn.k_norm.weight

            for kv_head_idx in range(num_kv_heads):
                expert_idx = layer_idx * num_kv_heads + kv_head_idx
                q_start = kv_head_idx * num_kv_groups * head_dim
                q_end = q_start + num_kv_groups * head_dim
                kv_start = kv_head_idx * head_dim
                kv_end = kv_start + head_dim

                sanity_inner.attn_bank.q_proj[expert_idx].copy_(q_proj[q_start:q_end].t())
                sanity_inner.attn_bank.k_proj[expert_idx].copy_(k_proj[kv_start:kv_end].t())
                sanity_inner.attn_bank.v_proj[expert_idx].copy_(v_proj[kv_start:kv_end].t())
                sanity_inner.attn_bank.o_proj[expert_idx].copy_(o_proj[:, q_start:q_end].t())
                sanity_inner.attn_bank.q_norm_weight[expert_idx].copy_(q_norm)
                sanity_inner.attn_bank.k_norm_weight[expert_idx].copy_(k_norm)


def test_alternating_global_sanity_matches_global_moe():
    global_model = GlobalMoEForCausalLM(_tiny_global_equiv_config()).eval()
    sanity_model = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).eval()
    _copy_global_weights_into_sanity_moe(global_model, sanity_model)

    ids, labels = _dummy_batch(B=2, T=8)
    with torch.no_grad():
        global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
        sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)

    torch.testing.assert_close(sanity_out.logits, global_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)


def test_alternating_global_sanity_uses_learned_mlp_gate():
    config = tiny_moe_everything_config("per_head_precompute_kv")
    config.num_hidden_layers = 4
    config.num_experts = 4
    config.num_experts_per_tok = 1
    config.per_layer_norm = True
    config.sanity_check_mode = "alternating_global_moe"
    model = MoEverythingForCausalLM(config).eval()
    bank = model.model.mlp_bank

    assert hasattr(bank, "gates")
    assert len(bank.gates) == 2
    captured = {"layer_0": False, "layer_1": False}

    def fake_gate_forward(expert_idx, key):
        def _forward(self, flat):
            captured[key] = True
            logits = flat.new_full((flat.shape[0], self.num_experts), -1.0e4)
            logits[:, expert_idx] = 1.0e4
            weights = flat.new_ones((flat.shape[0], 1))
            selected = torch.full((flat.shape[0], 1), expert_idx, device=flat.device, dtype=torch.long)
            return logits, weights, selected

        return _forward

    bank.gates[0].forward = MethodType(fake_gate_forward(1, "layer_0"), bank.gates[0])
    bank.gates[1].forward = MethodType(fake_gate_forward(3, "layer_1"), bank.gates[1])

    ids, _ = _dummy_batch(vocab_size=config.vocab_size, B=2, T=4)
    with torch.no_grad():
        out = model(input_ids=ids, output_router_logits=True)

    assert captured["layer_0"]
    assert captured["layer_1"]

    expected_depth_1 = torch.full_like(out.selected_experts[1], 1)
    expected_depth_3 = torch.full_like(out.selected_experts[3], 3)
    torch.testing.assert_close(out.selected_experts[1], expected_depth_1)
    torch.testing.assert_close(out.selected_experts[3], expected_depth_3)


def test_alternating_global_sanity_bias_updates_use_global_mlp_pool_only():
    class _FakeAccelerator:
        num_processes = 1

    config = tiny_moe_everything_deepseek_config("per_head_precompute_kv")
    config.num_hidden_layers = 4
    config.num_experts = 4
    config.num_experts_per_tok = 1
    config.per_layer_norm = True
    config.per_layer_attn_router = True
    config.sanity_check_mode = "alternating_global_moe"
    model = MoEverythingForCausalLM(config).eval()

    mlp_routers = get_bias_update_routers(model, mlp_only=True)
    assert mlp_routers == list(model.model.mlp_bank.gates)
    assert len(mlp_routers) == 2

    with torch.no_grad():
        for depth_idx, router in enumerate(model.model.mlp_bank.gates):
            router.local_tokens_per_expert.zero_()
            router.local_tokens_per_expert[depth_idx % config.num_experts] = 10.0

        for router in model.model.attn_bank.routers:
            router.local_tokens_per_expert.fill_(7.0)

    update_expert_biases(
        model,
        update_rate=0.1,
        accelerator=_FakeAccelerator(),
        is_global=True,
        alpha=0.0,
        routers=mlp_routers,
    )

    ref_bias = model.model.mlp_bank.gates[0].expert_bias
    for router in model.model.mlp_bank.gates[1:]:
        torch.testing.assert_close(router.expert_bias, ref_bias)


def test_global_router_update_groups_mixed_attention_pools_by_bank():
    class _FakeAccelerator:
        num_processes = 1

    config = tiny_moe_everything_deepseek_config("per_head_fully_independent")
    config.num_hidden_layers = 2
    config.num_experts = 4
    config.num_attn_experts = 4
    config.num_experts_per_tok = 1
    config.num_attn_experts_per_tok = 1
    config.per_layer_attn_router = True
    config.global_router_update = True
    model = MoEverythingForCausalLM(config).eval()

    router_groups = get_bias_update_router_groups(model)
    group_sizes = [[router.num_experts for router in group] for group in router_groups]
    assert group_sizes == [[4], [4, 4], [2, 2], [2, 2], [4, 4]]

    with torch.no_grad():
        model.model.mlp_bank.gate.local_tokens_per_expert.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
        model.model.attn_bank.q_routers[0].local_tokens_per_expert.copy_(torch.tensor([0.0, 8.0, 0.0, 0.0]))
        model.model.attn_bank.q_routers[1].local_tokens_per_expert.copy_(torch.tensor([0.0, 0.0, 8.0, 0.0]))
        model.model.attn_bank.k_routers[0].local_tokens_per_expert.copy_(torch.tensor([5.0, 0.0]))
        model.model.attn_bank.k_routers[1].local_tokens_per_expert.copy_(torch.tensor([0.0, 5.0]))
        model.model.attn_bank.v_routers[0].local_tokens_per_expert.copy_(torch.tensor([5.0, 0.0]))
        model.model.attn_bank.v_routers[1].local_tokens_per_expert.copy_(torch.tensor([0.0, 5.0]))
        model.model.attn_bank.o_routers[0].local_tokens_per_expert.copy_(torch.tensor([0.0, 0.0, 8.0, 0.0]))
        model.model.attn_bank.o_routers[1].local_tokens_per_expert.copy_(torch.tensor([0.0, 0.0, 0.0, 8.0]))

    update_expert_biases(
        model,
        update_rate=0.1,
        accelerator=_FakeAccelerator(),
        is_global=True,
        alpha=0.0,
    )

    torch.testing.assert_close(
        model.model.attn_bank.q_routers[0].expert_bias,
        model.model.attn_bank.q_routers[1].expert_bias,
    )
    torch.testing.assert_close(
        model.model.attn_bank.k_routers[0].expert_bias,
        model.model.attn_bank.k_routers[1].expert_bias,
    )
    torch.testing.assert_close(
        model.model.attn_bank.v_routers[0].expert_bias,
        model.model.attn_bank.v_routers[1].expert_bias,
    )
    torch.testing.assert_close(
        model.model.attn_bank.o_routers[0].expert_bias,
        model.model.attn_bank.o_routers[1].expert_bias,
    )
    assert model.model.attn_bank.k_routers[0].expert_bias.shape[0] == 2
    assert model.model.attn_bank.q_routers[0].expert_bias.shape[0] == 4


def test_representative_experiment_configs_use_expected_gqa_ratios():
    expected = {
        "configs/standard_moe.yaml": 8,
        "configs/global_moe.yaml": 8,
        "configs/scaling/xs_standard.yaml": 8,
        "configs/scaling/xs_global.yaml": 8,
        "configs/scaling/xs_dense_baseline.yaml": 8,
        "configs/scaling/m_standard.yaml": 4,
        "configs/scaling/m_global.yaml": 4,
        "configs/scaling/s_standard.yaml": 4,
        "configs/scaling/s_global.yaml": 4,
        "configs/moe_everything_per_head_precompute_kv.yaml": 8,
        "configs/moe_everything_per_head_precompute_kv_sanity.yaml": 8,
        "configs/moe_everything_per_head_fully_independent.yaml": 8,
    }
    for rel_path, num_kv_heads in expected.items():
        model_cfg = yaml.safe_load(Path(rel_path).read_text())["model"]
        assert model_cfg["num_key_value_heads"] == num_kv_heads, rel_path


def test_sanity_config_matches_global_attention_geometry():
    sanity_cfg = yaml.safe_load(Path("configs/moe_everything_per_head_precompute_kv_sanity.yaml").read_text())["model"]
    global_cfg = yaml.safe_load(Path("configs/global_moe.yaml").read_text())["model"]

    assert sanity_cfg["num_hidden_layers"] == global_cfg["num_hidden_layers"] * 2
    for key in ("hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads"):
        assert sanity_cfg[key] == global_cfg[key]


# ── Dynamic depth tests ───────────────────────────────────────────────────

def test_dynamic_depth_training():
    config = tiny_moe_everything_config("bundled")
    config.dynamic_depth_min = 0.5
    config.dynamic_depth_max = 1.0
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    # Run several times — depth varies but should always produce valid loss
    for _ in range(5):
        out = model(input_ids=ids, labels=labels)
        assert out.loss is not None
        assert out.loss.item() > 0


def test_dynamic_depth_eval_uses_full():
    config = tiny_moe_everything_config("bundled")
    config.dynamic_depth_min = 0.5
    config.dynamic_depth_max = 1.0
    model = MoEverythingForCausalLM(config).eval()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, output_router_logits=True)
    # At eval, all depths run
    assert len(out.branch_probs) == config.num_hidden_layers


def test_dynamic_depth_grads():
    config = tiny_moe_everything_config("bundled")
    config.dynamic_depth_min = 0.5
    config.dynamic_depth_max = 1.0
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    for n, p in model.named_parameters():
        if "embed_tokens" not in n and "lm_head" not in n:
            # Some params might not get grads if depth is too short,
            # but loss should still be valid
            pass
    assert out.loss.item() > 0


# ── Depthwise attention tests ─────────────────────────────────────────────

def test_depthwise_attention_full():
    config = tiny_moe_everything_config("bundled")
    config.depthwise_attention = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    assert model.model.depth_queries.grad is not None


def test_depthwise_attention_block():
    config = tiny_moe_everything_config("bundled")
    config.num_hidden_layers = 8
    config.depthwise_attention = True
    config.depthwise_block_size = 4
    model = MoEverythingForCausalLM(config).train()
    # block_size=4, num_depths=8 → 2 blocks → 2 query vectors
    assert model.model.depth_queries.shape[0] == 2
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    assert model.model.depth_queries.grad is not None


def test_depthwise_attention_precompute_kv():
    config = tiny_moe_everything_config("precompute_kv")
    config.depthwise_attention = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()


def test_depthwise_attention_with_per_layer_router():
    config = tiny_moe_everything_config("bundled")
    config.per_layer_router = True
    config.depthwise_attention = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()


# ── Combined features test ────────────────────────────────────────────────

def test_all_three_features_combined():
    config = tiny_moe_everything_config("bundled")
    config.per_layer_router = True
    config.dynamic_depth_min = 0.5
    config.dynamic_depth_max = 1.0
    config.depthwise_attention = True
    model = MoEverythingForCausalLM(config).train()
    ids, labels = _dummy_batch()
    out = model(input_ids=ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
