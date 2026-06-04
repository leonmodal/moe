"""Execution-shape tests for dense MoE-Everything recompute attention."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.moe_everything import AttentionExpertBank, MoEverythingConfig


def _make_config(mode: str, **kwargs) -> MoEverythingConfig:
    routing_bundle = "q_k_v_o" if mode == "per_head_no_recompute" else "qk_v_o"
    return MoEverythingConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        head_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=3,
        num_experts_per_tok=1,
        num_attn_experts=3,
        num_attn_experts_per_tok=1,
        attn_expert_mode=mode,
        attn_routing_bundle=routing_bundle,
        **kwargs,
    )


def _make_bank(mode: str) -> AttentionExpertBank:
    return AttentionExpertBank(_make_config(mode)).eval()


def test_ema_attention_context_doubles_qk_v_router_input_only():
    bank = AttentionExpertBank(
        _make_config(
            "per_head_recompute_k",
            attn_router_context="ema_qk_v",
            attn_router_context_decay=0.5,
        )
    )

    assert bank.qk_routers[0].weight.shape[1] == 32
    assert bank.v_routers[0].weight.shape[1] == 32
    assert bank.o_routers[0].weight.shape[1] == bank.q_group_dim


def test_ema_attention_context_is_causal_prefix_only():
    bank = AttentionExpertBank(
        _make_config(
            "per_head_recompute_k",
            attn_router_context="ema_qk_v",
            attn_router_context_decay=0.5,
        )
    )
    x = torch.tensor(
        [
            [
                [2.0, 0.0] + [0.0] * 14,
                [0.0, 4.0] + [0.0] * 14,
                [8.0, 8.0] + [0.0] * 14,
            ]
        ]
    )

    ema = bank._causal_prefix_ema(x)

    assert torch.allclose(ema[:, 0], torch.zeros_like(ema[:, 0]))
    assert torch.allclose(ema[:, 1, :2], torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(ema[:, 2, :2], torch.tensor([[0.5, 2.0]]))


def test_ema_recompute_table_build_runs_with_router_context():
    bank = AttentionExpertBank(
        _make_config(
            "per_head_recompute_kv",
            attn_router_context="ema_qk_v",
            attn_router_context_decay=0.95,
        )
    ).eval()
    hidden_states = torch.randn(1, 4, 16)
    position_embeddings = (
        torch.ones(1, 4, 4),
        torch.zeros(1, 4, 4),
    )

    tables = bank._build_per_head_recompute_tables(hidden_states, position_embeddings)

    assert tables["qk_idx"].shape == (4, 2)
    assert tables["v_idx"].shape == (4, 2)
    assert tables["Q"].shape == (1, 4, 4, 4)


def _recompute_tables(mode: str) -> dict[str, torch.Tensor]:
    batch_size, seq_len, hidden_size = 1, 5, 16
    num_heads, num_kv_heads, head_dim = 4, 2, 4
    flat_tokens = batch_size * seq_len

    qk_idx = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 2],
            [2, 2],
        ],
        dtype=torch.long,
    )
    v_idx = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 1],
        ],
        dtype=torch.long,
    )
    tables = {
        "flat": torch.randn(flat_tokens, hidden_size),
        "qk_idx": qk_idx,
        "v_idx": v_idx,
        "Q": torch.randn(batch_size, num_heads, seq_len, head_dim),
    }
    if mode == "per_head_recompute_k":
        tables["V_fresh"] = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    return tables


@pytest.mark.parametrize("mode", ["per_head_recompute_k", "per_head_recompute_kv"])
def test_recompute_attention_uses_dense_full_query_tables(mode):
    bank = _make_bank(mode)
    tables = _recompute_tables(mode)
    _, _, seq_len, head_dim = tables["Q"].shape
    position_embeddings = (
        torch.ones(1, seq_len, head_dim),
        torch.zeros(1, seq_len, head_dim),
    )
    attention_mask = torch.zeros(1, 1, seq_len, seq_len)
    calls = []

    def fake_attention(Q, K, V, attention_mask=None):
        calls.append(
            {
                "q_batch": Q.shape[0],
                "q_heads": Q.shape[1],
                "q_len": Q.shape[2],
                "mask_q_len": None if attention_mask is None else attention_mask.shape[-2],
            }
        )
        return Q.new_zeros(Q.shape)

    bank._run_attention = fake_attention

    out = bank._run_per_head_recompute_expert_tables(
        tables,
        position_embeddings,
        attention_mask=attention_mask,
    )

    assert out.shape == tables["Q"].shape
    assert calls
    assert all(call["q_batch"] == tables["Q"].shape[0] for call in calls)
    assert all(call["q_heads"] == tables["Q"].shape[1] for call in calls)
    assert all(call["q_len"] == seq_len for call in calls)
    assert all(call["mask_q_len"] == seq_len for call in calls)


def test_recompute_dense_execution_preserves_attention_gradients():
    bank = _make_bank("per_head_recompute_k")
    tables = _recompute_tables("per_head_recompute_k")
    tables["Q"].requires_grad_()
    _, _, seq_len, head_dim = tables["Q"].shape
    position_embeddings = (
        torch.ones(1, seq_len, head_dim),
        torch.zeros(1, seq_len, head_dim),
    )

    def fake_attention(Q, K, V, attention_mask=None):
        return Q * 2.0

    bank._run_attention = fake_attention
    out = bank._run_per_head_recompute_expert_tables(tables, position_embeddings)
    out.sum().backward()

    assert tables["Q"].grad is not None
    assert torch.count_nonzero(tables["Q"].grad).item() == tables["Q"].numel()
