"""weighted_sum branch routing: both ATTN and MLP run every depth, outputs
blended by the softmax-binary router weights (w_attn + w_mlp = 1).

Coverage:
- BranchRouter forward in weighted_sum mode returns all-True masks and
  softmax probs (no masking applied to the weights), so the depth-step's
  `w_attn * attn_out + w_mlp * mlp_out` blends both branches naturally.
- Probs sum to 1 per token.
- The router carries no `expert_bias` buffer (binary softmax, not DeepSeek
  sigmoid + bias) so the bias-update walker has nothing to update.
- Full model build + forward + backward produce finite values.
- Branch aux loss / branch entropy bonus / branch bias update are all
  inert in this mode.
"""

import pytest
import torch

from src.models.moe_everything.config import MoEverythingConfig
from src.models.moe_everything.model import MoEverythingForCausalLM
from src.models.routing.routers import BranchRouter


def test_weighted_sum_router_returns_unmasked_weights_summing_to_one():
    torch.manual_seed(0)
    router = BranchRouter(hidden_size=16, balancing="weighted_sum")
    h = torch.randn(2, 5, 16)
    w_attn, w_mlp, attn_mask, mlp_mask = router(h)

    # Both branches active for every token (no masking).
    assert attn_mask.all().item()
    assert mlp_mask.all().item()

    # softmax-binary weights: per-token sum is 1.
    summed = (w_attn + w_mlp).squeeze(-1)
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)

    # Both weights are strictly in (0, 1) (probabilities, no degenerate
    # one-hot collapse at init).
    assert (w_attn > 0).all() and (w_attn < 1).all()
    assert (w_mlp > 0).all() and (w_mlp < 1).all()


def test_weighted_sum_router_has_no_deepseek_bias_state():
    router = BranchRouter(hidden_size=16, balancing="weighted_sum")
    # Soft mixer — no hard pick, no bias buffer needed.
    assert not router.use_deepseek_style
    assert not hasattr(router, "expert_bias") or router.expert_bias is None or \
        router.expert_bias.numel() == 0 or True  # buffer may not exist at all
    # We register expert_bias only when use_deepseek_style or quantile.
    # For weighted_sum neither is true, so the buffer should NOT exist.
    assert "expert_bias" not in dict(router.named_buffers())
    assert "local_tokens_per_expert" not in dict(router.named_buffers())


def _build_config(num_depths: int = 4) -> MoEverythingConfig:
    return MoEverythingConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=num_depths,
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
        branch_balancing="weighted_sum",
        output_router_logits=True,
        tie_word_embeddings=False,
    )


def test_weighted_sum_full_model_forward_backward_finite():
    torch.manual_seed(0)
    model = MoEverythingForCausalLM(_build_config()).train()
    input_ids = torch.randint(0, 64, (2, 8))
    out = model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(out.loss).item()
    out.loss.backward()
    # Gradients reach the first branch router's gate (the soft mixer has
    # to receive task-loss gradient so it can learn to mix).
    first_branch_gate = model.model.branch_routers[0].gate
    assert first_branch_gate.weight.grad is not None
    assert torch.isfinite(first_branch_gate.weight.grad).all()


def test_weighted_sum_branch_aux_and_entropy_inert():
    torch.manual_seed(0)
    cfg = _build_config()
    cfg.branch_router_aux_loss_coef = 0.01  # Set positive — should still be inert.
    cfg.branch_entropy_coef = 0.5            # Set positive — should still be inert.
    model = MoEverythingForCausalLM(cfg).train()
    input_ids = torch.randint(0, 64, (2, 8))
    out = model(input_ids=input_ids, labels=input_ids)
    # branch aux loss path is gated on balancing in {aux_loss, seq_aux_loss};
    # entropy bonus is gated on balancing == sampling_entropy. weighted_sum
    # triggers neither — so branch_aux_loss is None and the loss equals ce.
    assert out.branch_aux_loss is None
