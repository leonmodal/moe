from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models import DeepSeekStandardMoEModel, Qwen3MoeConfig


def _tiny_standard_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        intermediate_size=64,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        router_aux_loss_coef=0.0,
        output_router_logits=False,
        norm_topk_prob=True,
    )


def test_deepseek_standard_moe_ce_is_next_token_prediction():
    torch.manual_seed(0)
    model = DeepSeekStandardMoEModel(_tiny_standard_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            labels=input_ids,
            output_router_logits=False,
            return_logits=True,
        )

    explicit_ntp = F.cross_entropy(
        out.logits[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
        input_ids[:, 1:].contiguous().view(-1),
    )
    torch.testing.assert_close(out.loss, explicit_ntp)
