import torch

from src.models.moe_everything.config import MoEverythingConfig
from src.models.moe_everything.model import MoEverythingForCausalLM
from src.models.routing.routers import BranchRouterRecorder
from src.training.routing import update_expert_biases
from src.utils.load_balance_artifacts import build_load_balance_artifacts


def test_fixed_alternating_branch_policy_preserves_qkvo_recompute_kv():
    cfg = MoEverythingConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
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
        branch_balancing="fixed_alternating",
        output_router_logits=True,
        tie_word_embeddings=False,
    )
    model = MoEverythingForCausalLM(cfg).eval()

    inner = model.model
    assert inner.sanity_check_mode is None
    assert inner.attn_bank.mode == "per_head_recompute_kv"
    assert inner.attn_bank.routing_bundle == "qkvo"
    assert all(isinstance(router, BranchRouterRecorder) for router in inner.branch_routers)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        model(input_ids, return_logits=False)

    selected = inner._all_branch_selected_experts
    assert len(selected) == cfg.num_hidden_layers
    assert [int(depth_choice[0, 0, 0].item()) for depth_choice in selected] == [0, 1, 0, 1]


def test_fixed_alternating_32_depth_runs_exactly_sixteen_attn_and_mlp_slots():
    cfg = MoEverythingConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=1,
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
        branch_balancing="fixed_alternating",
        global_router_update=True,
        output_router_logits=True,
        tie_word_embeddings=False,
    )
    model = MoEverythingForCausalLM(cfg).train()
    inner = model.model

    assert not hasattr(inner.attn_bank, "v_routers")
    assert not hasattr(inner.attn_bank, "o_routers")
    assert len(inner.attn_bank.qk_routers) == 32
    assert len(inner.mlp_bank.gates) == 32

    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    hidden_states = inner(input_ids)
    hidden_states.sum().backward()

    branch_selected = [
        int(depth_choice[0, 0, 0].item())
        for depth_choice in inner._all_branch_selected_experts
    ]
    assert branch_selected == [0, 1] * 16

    tokens = input_ids.numel()
    active_q_counts = torch.zeros(cfg.num_attn_experts)
    for depth_idx, per_head_routers in enumerate(inner.attn_bank.qk_routers):
        for router in per_head_routers:
            count_sum = int(router.local_tokens_per_expert.sum().item())
            if depth_idx % 2 == 0:
                assert count_sum == tokens
                active_q_counts += router.local_tokens_per_expert.detach().float()
            else:
                assert count_sum == 0

    active_mlp_counts = torch.zeros(cfg.num_experts)
    for depth_idx, gate in enumerate(inner.mlp_bank.gates):
        count_sum = int(gate.local_tokens_per_expert.sum().item())
        if depth_idx % 2 == 1:
            assert count_sum == tokens
            active_mlp_counts += gate.local_tokens_per_expert.detach().float()
        else:
            assert count_sum == 0

    assert int(active_q_counts.sum().item()) == tokens * cfg.num_key_value_heads * 16
    assert int(active_mlp_counts.sum().item()) == tokens * 16

    q_loads = active_q_counts / active_q_counts.sum().clamp_min(1.0)
    q_sign = torch.sign(q_loads - (1.0 / cfg.num_attn_experts))
    expected_q_bias = -(q_sign - q_sign.mean()) * 0.001

    with torch.no_grad():
        update_expert_biases(
            model,
            bias_rate=0.0,
            distributed=False,
            zero_sum=True,
            per_proj_rates={
                "q": 0.001,
                "k": 0.0,
                "v": 0.0,
                "o": 0.0,
                "mlp": 0.0,
                "branch": 0.0,
            },
        )

    for per_head_routers in inner.attn_bank.qk_routers:
        for router in per_head_routers:
            torch.testing.assert_close(router.expert_bias, expected_q_bias)
            assert int(router.local_tokens_per_expert.sum().item()) == 0


def test_fixed_alternating_load_balance_keeps_attention_depth_snapshots():
    cfg = MoEverythingConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=1,
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
        branch_balancing="fixed_alternating",
        output_router_logits=True,
        tie_word_embeddings=False,
    )
    model = MoEverythingForCausalLM(cfg).train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    model.model(input_ids)

    payload = build_load_balance_artifacts(model, step=50)
    assert payload is not None
    assert "branch" not in payload["pools"]

    attn_depths = payload["pools"]["attn:qkvo"]["per_depth"]
    mlp_depths = payload["pools"]["mlp"]["per_depth"]
    assert len(attn_depths) == cfg.num_hidden_layers
    assert len(mlp_depths) == cfg.num_hidden_layers

    for depth, row in enumerate(attn_depths):
        if depth % 2 == 0:
            assert row["active_tokens"] > 0
            assert row["assignments"] > 0
        else:
            assert row["active_tokens"] == 0
            assert row["assignments"] == 0

    for depth, row in enumerate(mlp_depths):
        if depth % 2 == 1:
            assert row["active_tokens"] == input_ids.numel()
            assert row["assignments"] == input_ids.numel() * cfg.num_experts_per_tok
        else:
            assert row["active_tokens"] == 0
            assert row["assignments"] == 0
