"""Prelude / coda hybrid: standard-MoE-style decoder blocks wrapped
around the recurrent MoE-Everything bank.

Coverage:
- Boundary blocks build with the expected per-layer MLP pool geometry.
- Forward + backward run cleanly with both branch_balancing modes used
  in the hybrid configs (fixed_alternating, exploration_only).
- The MoEverythingForCausalLM balancing-owner walker yields every
  prelude/coda MLP gate (so the trainer's deepseek_bias update hits them).
- Defaults (prelude_layers=0, coda_layers=0) keep the model identical to
  a pure MoE-Everything build: empty boundary ModuleLists, no boundary
  config, no boundary owners.
"""

import pytest
import torch

from src.models.moe_everything.config import MoEverythingConfig
from src.models.moe_everything.model import MoEverythingForCausalLM
from src.models.modeling_qwen3_moe import Qwen3MoeDecoderLayer
from src.models.router import DeepSeekRouter
from src.utils.load_balance_artifacts import build_load_balance_artifacts


def _build_hybrid_config(
    *,
    branch_balancing: str,
    prelude_layers: int = 2,
    coda_layers: int = 2,
    recurrent_depths: int = 4,
    **extra,
) -> MoEverythingConfig:
    return MoEverythingConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=recurrent_depths,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_recompute_kv",
        attn_routing_bundle="qkvo",
        per_layer_router=True,
        per_layer_mlp_router=True,
        per_layer_attn_router=True,
        per_layer_norm=True,
        per_layer_qk_norm=True,
        use_deepseek_routing=True,
        branch_balancing=branch_balancing,
        output_router_logits=True,
        tie_word_embeddings=False,
        prelude_layers=prelude_layers,
        coda_layers=coda_layers,
        boundary_num_experts=4,
        boundary_num_experts_per_tok=2,
        boundary_moe_intermediate_size=16,
        boundary_num_groups=2,
        boundary_group_topk=1,
        boundary_router_type="deepseek",
        **extra,
    )


@pytest.mark.parametrize("branch_balancing", ["fixed_alternating", "exploration_only"])
def test_prelude_coda_hybrid_builds_with_expected_geometry(branch_balancing):
    extra = (
        dict(branch_exploration_rate=0.5, branch_exploration_decay="constant")
        if branch_balancing == "exploration_only"
        else {}
    )
    cfg = _build_hybrid_config(branch_balancing=branch_balancing, **extra)
    model = MoEverythingForCausalLM(cfg).eval()
    inner = model.model

    assert len(inner.prelude_blocks) == 2
    assert len(inner.coda_blocks) == 2

    for block in (*inner.prelude_blocks, *inner.coda_blocks):
        # Boundary block is the Qwen3 decoder layer reused as-is.
        assert isinstance(block, Qwen3MoeDecoderLayer)
        # Boundary MLP pool size + top-K come from boundary_* fields,
        # not from the bank's 8/2 settings.
        gate = block.mlp.gate
        assert isinstance(gate, DeepSeekRouter)
        assert gate.num_experts == 4
        assert gate.top_k == 2
        # Boundary attention reuses main config's GQA geometry.
        assert block.self_attn.q_proj.out_features == cfg.num_attention_heads * cfg.head_dim
        assert block.self_attn.k_proj.out_features == cfg.num_key_value_heads * cfg.head_dim


@pytest.mark.parametrize("branch_balancing", ["fixed_alternating", "exploration_only"])
def test_prelude_coda_hybrid_forward_backward_finite(branch_balancing):
    extra = (
        dict(branch_exploration_rate=0.5, branch_exploration_decay="constant")
        if branch_balancing == "exploration_only"
        else {}
    )
    cfg = _build_hybrid_config(branch_balancing=branch_balancing, **extra)
    model = MoEverythingForCausalLM(cfg).train()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(out.loss).item()

    out.loss.backward()
    # Gradients reach the first prelude block AND the last coda block.
    first_prelude_param = next(model.model.prelude_blocks[0].parameters())
    last_coda_param = next(model.model.coda_blocks[-1].parameters())
    assert first_prelude_param.grad is not None
    assert last_coda_param.grad is not None
    assert torch.isfinite(first_prelude_param.grad).all()
    assert torch.isfinite(last_coda_param.grad).all()


def test_prelude_coda_routers_registered_as_balancing_owners():
    cfg = _build_hybrid_config(branch_balancing="fixed_alternating")
    model = MoEverythingForCausalLM(cfg).eval()

    owner_ids_mlp = {
        id(owner) for owner, label in model.get_all_balancing_owners() if label == "mlp"
    }
    # Each of the 4 boundary blocks owns its own gate; all four MUST be
    # in the walker's "mlp" bucket so the trainer's deepseek-bias update
    # hits them.
    for block in (*model.model.prelude_blocks, *model.model.coda_blocks):
        assert id(block.mlp.gate) in owner_ids_mlp


def test_prelude_coda_per_layer_load_balancing_is_independent():
    """Each prelude/coda router must own its own expert_bias buffer; a
    skew applied to one layer's local_tokens_per_expert must not
    propagate into any other layer's bias after the update walker runs.
    """
    from src.training.routing import update_expert_biases

    cfg = _build_hybrid_config(branch_balancing="fixed_alternating")
    # Force the trainer to use the deepseek-bias path for owners labeled
    # "mlp" — matches what the new yamls do via load_balancing_method.
    cfg.load_balancing_method = "deepseek_bias"
    model = MoEverythingForCausalLM(cfg).eval()
    model._load_balancing_method = "deepseek_bias"

    boundary_gates = [
        block.mlp.gate
        for block in (*model.model.prelude_blocks, *model.model.coda_blocks)
    ]
    # Sanity: 4 boundary layers = 4 distinct router instances with
    # distinct bias buffers.
    assert len(boundary_gates) == 4
    assert len({id(g.expert_bias) for g in boundary_gates}) == 4

    # Seed prelude[0] with a heavily skewed load (expert 0 gets all the
    # tokens). All other boundary layers stay at uniform load.
    target_gate = boundary_gates[0]
    other_gates = boundary_gates[1:]
    skew = torch.zeros_like(target_gate.local_tokens_per_expert)
    skew[0] = 1024.0
    target_gate.local_tokens_per_expert.copy_(skew)
    for g in other_gates:
        g.local_tokens_per_expert.copy_(torch.zeros_like(g.local_tokens_per_expert))

    pre_biases = [g.expert_bias.clone() for g in boundary_gates]
    with torch.no_grad():
        update_expert_biases(model, bias_rate=0.1)
    post_biases = [g.expert_bias for g in boundary_gates]

    # The skewed router's bias must have moved.
    target_delta = (post_biases[0] - pre_biases[0]).abs().max().item()
    assert target_delta > 0.0, "prelude[0] bias should update under a skewed load"

    # Every other boundary router's bias must be untouched (uniform load
    # — the zero-sum sign-update is identically zero).
    for idx, (before, after) in enumerate(zip(pre_biases[1:], post_biases[1:]), start=1):
        delta = (after - before).abs().max().item()
        assert delta == 0.0, (
            f"boundary layer {idx} bias must not move when only prelude[0] saw a "
            f"skewed load (got delta={delta})"
        )


def test_zero_boundaries_match_pure_moe_everything_shape():
    # With prelude_layers=0 and coda_layers=0, the model must build
    # without any boundary state: empty ModuleLists, no boundary config.
    cfg = _build_hybrid_config(
        branch_balancing="fixed_alternating",
        prelude_layers=0,
        coda_layers=0,
    )
    model = MoEverythingForCausalLM(cfg).eval()

    assert len(model.model.prelude_blocks) == 0
    assert len(model.model.coda_blocks) == 0
    assert model.model._boundary_config is None

    # No boundary owners in the balancing walker.
    inner = model.model
    boundary_gate_ids = {
        id(getattr(getattr(b, "mlp", None), "gate", None))
        for b in (*inner.prelude_blocks, *inner.coda_blocks)
    }
    boundary_gate_ids.discard(id(None))
    assert boundary_gate_ids == set()


def test_recurrent_moe_everything_fixed_alternating_loops_middle_bank():
    from src.training.routing import update_expert_biases

    cfg = _build_hybrid_config(
        branch_balancing="fixed_alternating",
        prelude_layers=1,
        coda_layers=1,
        recurrent_depths=4,
        recurrent=True,
        recurrent_adapter=True,
        recurrent_adapter_intermediate_size=64,
        recurrent_adapter_activation="gelu",
        recurrent_adapter_bias=False,
        global_router_update=True,
    )
    cfg.model_type = "recurrent_moe_everything"
    cfg.mlp_router_balancing = "deepseek_bias"
    cfg.attn_router_balancing = "deepseek_bias"

    model = MoEverythingForCausalLM(cfg).train()
    model._load_balancing_method = "deepseek_bias"
    model.gradient_checkpointing_enable()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(
        input_ids=input_ids,
        labels=input_ids,
        output_router_logits=True,
        num_steps=torch.tensor([1, 2], dtype=torch.long),
    )

    assert torch.isfinite(out.loss).item()
    assert out.num_steps_no_grad == 1
    assert out.num_steps_with_grad == 2
    assert model.model._last_num_steps_no_grad == 1
    assert model.model._last_num_steps_with_grad == 2
    assert len(model.model._all_mlp_selected_experts) == cfg.num_hidden_layers * 3

    payload = build_load_balance_artifacts(model, step=50)
    assert payload is not None
    assert "branch" not in payload["pools"]
    assert payload["bias_pools"]["mlp_recurrent"]["summary"]["num_routers"] == 1

    mlp_depths = payload["pools"]["mlp_recurrent"]["per_depth"]
    assert len(mlp_depths) == cfg.num_hidden_layers * 3
    assert [row["_loop"] for row in mlp_depths] == [0] * 4 + [1] * 4 + [2] * 4
    assert [row["_block"] for row in mlp_depths] == [0, 1, 2, 3] * 3
    for idx, row in enumerate(mlp_depths):
        if idx % 2 == 1:
            assert row["active_tokens"] == input_ids.numel()
        else:
            assert row["active_tokens"] == 0

    out.loss.backward()
    assert model.model.recurrent_adapter.up_proj.weight.grad is not None
    assert next(model.model.prelude_blocks[0].parameters()).grad is not None
    assert next(model.model.coda_blocks[0].parameters()).grad is not None

    mlp_gates = list(model.model.mlp_bank.gates)
    assert any(g.local_tokens_per_expert.sum().item() > 0 for g in mlp_gates)
    with torch.no_grad():
        update_expert_biases(model, bias_rate=0.1, distributed=False)
    first_bias = mlp_gates[0].expert_bias
    for gate in mlp_gates[1:]:
        torch.testing.assert_close(gate.expert_bias, first_bias)
    assert all(g.local_tokens_per_expert.sum().item() == 0 for g in mlp_gates)


def test_recurrent_moe_everything_model_factory_wires_recurrence():
    from src.training.model_factory import build_model

    cfg = {
        "model": {
            "type": "recurrent_moe_everything",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "prelude_layers": 1,
            "coda_layers": 1,
            "head_dim": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_attn_experts": 4,
            "num_attn_experts_per_tok": 1,
            "attn_expert_mode": "per_head_recompute_kv",
            "attn_routing_bundle": "qkvo",
            "per_layer_router": True,
            "per_layer_mlp_router": True,
            "per_layer_attn_router": True,
            "per_layer_norm": True,
            "per_layer_qk_norm": True,
            "router_type": "deepseek",
            "use_deepseek_routing": True,
            "tie_word_embeddings": False,
            "prelude_coda": {
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "num_groups": 2,
                "group_topk": 1,
                "router_type": "deepseek",
            },
            "mlp_router": {"balancing": "deepseek_bias"},
            "attn_router": {"balancing": "deepseek_bias"},
            "branch_router": {"balancing": "fixed_alternating"},
        },
        "recurrence": {
            "mean_backprop_depth": 3,
            "eval_recurrence": 7,
        },
        "training": {},
    }

    model, model_cfg = build_model(cfg)
    assert model_cfg.model_type == "recurrent_moe_everything"
    assert model_cfg.recurrent is True
    assert model_cfg.eval_recurrence == 7
    assert model_cfg.mean_backprop_depth == 3
    assert model.model.recurrent is True

    input_ids = torch.randint(0, model_cfg.vocab_size, (2, 6))
    out = model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.isfinite(out.loss).item()
    assert out.num_steps_no_grad == 0
    assert out.num_steps_with_grad == 1
