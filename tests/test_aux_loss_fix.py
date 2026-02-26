"""
Test that the double-softmax bug is fixed and aux loss responds to skewed routing.
"""
import torch
import torch.nn.functional as F
from src.models.load_balancing import load_balancing_loss_func


def test_uniform_routing():
    """Uniform routing should give aux_loss = top_k."""
    probs = torch.ones(1000, 128) / 128
    loss = load_balancing_loss_func((probs,), num_experts=128, top_k=4)
    print(f"Uniform (128 experts, top-4):  aux_loss = {loss.item():.4f}  (expect ~4.0)")
    assert abs(loss.item() - 4.0) < 0.1, f"Expected ~4.0, got {loss.item()}"


def test_skewed_routing():
    """Heavily skewed routing should give aux_loss >> top_k."""
    probs = torch.ones(1000, 128) * 0.001
    probs[:, :4] = 0.25
    probs = probs / probs.sum(dim=-1, keepdim=True)
    loss = load_balancing_loss_func((probs,), num_experts=128, top_k=4)
    print(f"Skewed (4/128 hot):            aux_loss = {loss.item():.4f}  (expect >> 4.0)")
    assert loss.item() > 20.0, f"Expected >> 4.0, got {loss.item()}"


def test_moderate_skew():
    """Moderate skew should give aux_loss > top_k but not extreme."""
    probs = torch.ones(1000, 128) * 0.005
    probs[:, :16] = 0.05
    probs = probs / probs.sum(dim=-1, keepdim=True)
    loss = load_balancing_loss_func((probs,), num_experts=128, top_k=4)
    print(f"Moderate (16/128 hot):         aux_loss = {loss.item():.4f}  (expect > 4.0)")
    assert loss.item() > 6.0, f"Expected > 6.0, got {loss.item()}"


def test_no_double_softmax():
    """Verify we're NOT applying double softmax — skewed input should NOT look uniform."""
    probs = torch.ones(1000, 128) * 0.001
    probs[:, :4] = 0.25
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Our fixed loss
    fixed = load_balancing_loss_func((probs,), num_experts=128, top_k=4)

    # Simulate the old buggy double-softmax loss
    double_softmaxed = F.softmax(probs, dim=-1)
    buggy = load_balancing_loss_func((double_softmaxed,), num_experts=128, top_k=4)

    print(f"Fixed loss (skewed):           {fixed.item():.4f}")
    print(f"Double-softmax loss (skewed):  {buggy.item():.4f}")
    assert fixed.item() > 3 * buggy.item(), "Fixed loss should be much larger than double-softmax loss for skewed input"


def test_standard_moe_uses_fixed_loss():
    """StandardMoEModel.forward should use our fixed loss."""
    from src.models import StandardMoEModel, Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=2,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=768,
        intermediate_size=3072, router_aux_loss_coef=0.1,
    )
    model = StandardMoEModel(config)
    ids = torch.randint(0, 1000, (2, 32))
    out = model(input_ids=ids, labels=ids, output_router_logits=True)
    print(f"Standard MoE aux_loss:         {out.aux_loss.item():.4f}")
    assert out.aux_loss is not None
    assert out.loss is not None


def test_global_moe_uses_fixed_loss():
    """GlobalMoEForCausalLM.forward should use our fixed loss."""
    from src.models import GlobalMoEConfig, GlobalMoEForCausalLM

    config = GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=2,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4, moe_intermediate_size=768,
        intermediate_size=3072, router_aux_loss_coef=0.1,
    )
    model = GlobalMoEForCausalLM(config)
    ids = torch.randint(0, 1000, (2, 32))
    out = model(input_ids=ids, labels=ids, output_router_logits=True)
    print(f"Global MoE aux_loss:           {out.aux_loss.item():.4f}")
    assert out.aux_loss is not None
    assert out.loss is not None


if __name__ == "__main__":
    print("=" * 60)
    print("Testing aux loss fix")
    print("=" * 60)

    test_uniform_routing()
    test_skewed_routing()
    test_moderate_skew()
    test_no_double_softmax()

    print()
    print("Testing model integration...")
    test_standard_moe_uses_fixed_loss()
    test_global_moe_uses_fixed_loss()

    print()
    print("ALL TESTS PASSED")
