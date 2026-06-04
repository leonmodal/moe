from __future__ import annotations

import torch.nn.functional as F
import pytest
import torch

from src.models.moe_everything import MoEverythingConfig, MoEverythingForCausalLM


def _tiny_config() -> MoEverythingConfig:
    return MoEverythingConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_attn_experts=4,
        num_attn_experts_per_tok=1,
        max_position_embeddings=64,
        attn_expert_mode="per_head_recompute_kv",
        attn_routing_bundle="qkvo",
        output_router_logits=True,
        tie_word_embeddings=True,
    )


def test_moe_everything_ce_matches_logits_path_without_returning_logits():
    torch.manual_seed(0)
    model = MoEverythingForCausalLM(_tiny_config()).train()
    input_ids = torch.randint(0, model.vocab_size, (2, 7))

    with_logits = model(
        input_ids=input_ids,
        labels=input_ids,
        output_router_logits=True,
        return_logits=True,
    )
    no_logits = model(
        input_ids=input_ids,
        labels=input_ids,
        output_router_logits=True,
        return_logits=False,
    )

    assert with_logits.logits is not None
    assert no_logits.logits is None
    assert "logits" not in no_logits
    torch.testing.assert_close(no_logits.ce_loss, with_logits.ce_loss)
    torch.testing.assert_close(no_logits.loss, with_logits.loss)


def test_moe_everything_ce_is_next_token_prediction():
    torch.manual_seed(0)
    model = MoEverythingForCausalLM(_tiny_config()).eval()
    input_ids = torch.randint(0, model.vocab_size, (2, 7))

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            labels=input_ids,
            output_router_logits=False,
            return_logits=True,
        )

    explicit_ntp = F.cross_entropy(
        out.logits[:, :-1, :].contiguous().view(-1, model.vocab_size),
        input_ids[:, 1:].contiguous().view(-1),
    )
    torch.testing.assert_close(out.ce_loss, explicit_ntp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Liger fused CE path")
def test_cuda_liger_fused_ce_is_next_token_prediction():
    pytest.importorskip("liger_kernel.transformers")
    torch.manual_seed(0)
    model = MoEverythingForCausalLM(_tiny_config()).cuda().to(torch.bfloat16).eval()
    input_ids = torch.randint(0, model.vocab_size, (2, 7), device="cuda")

    with torch.no_grad():
        fused = model(
            input_ids=input_ids,
            labels=input_ids,
            output_router_logits=False,
            return_logits=False,
        )
        logits = model(
            input_ids=input_ids,
            labels=None,
            output_router_logits=False,
            return_logits=True,
        ).logits

    explicit_ntp = F.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, model.vocab_size),
        input_ids[:, 1:].contiguous().view(-1),
    )
    torch.testing.assert_close(fused.ce_loss.float(), explicit_ntp, atol=2e-2, rtol=2e-2)
