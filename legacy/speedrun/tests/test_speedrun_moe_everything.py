import pytest
import torch

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from src.models.speedrun_mixture_of_everything import (
    FunctionalRMSNorm,
    SpeedrunMoEverythingConfig,
    SpeedrunMoEverythingForCausalLM,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only tests")


def _tiny_config(mode: str) -> SpeedrunMoEverythingConfig:
    return SpeedrunMoEverythingConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        num_attn_experts=8,
        num_attn_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        attn_expert_mode=mode,
        output_router_logits=True,
        per_layer_norm=False,
        post_norm=False,
    )


@pytest.mark.parametrize("mode", ["per_head_precompute_kv", "per_head_fully_independent"])
def test_speedrun_moe_everything_forward_uses_functional_norms(mode):
    model = SpeedrunMoEverythingForCausalLM(_tiny_config(mode)).cuda().eval()
    ids = torch.randint(0, model.config.vocab_size, (2, 16), device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids, output_router_logits=True)

    assert torch.isfinite(out.loss)
    assert any(isinstance(m, FunctionalRMSNorm) for m in model.modules())
    assert not any(isinstance(m, Qwen3MoeRMSNorm) for m in model.modules())
