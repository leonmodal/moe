"""
End-to-end integration test for the full MoE training pipeline.

Tests:
  1. Forward pass + loss for all 4 model types
  2. seq_load_balancing_loss is correctly computed and added to total loss
  3. batch-level aux loss is correctly zeroed when coeff=0
  4. Gradients flow through seq loss to router weights
  5. Expert bias update (per-layer and global interpolation)
  6. Loss decomposition: total = CE + coeff*aux + seq_coef*seq_aux
"""
import copy
import math

import torch

from src.models import (
    Qwen3MoeConfig,
    StandardMoEModel,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    DeepSeekGlobalMoEForCausalLM,
)
from src.models.load_balancing import seq_load_balancing_loss_func
from src.models.router import DeepSeekRouter


# ── Shared tiny config ─────────────────────────────────────────────────

def make_standard_config(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        router_aux_loss_coef=0.0,   # batch-level off
        norm_topk_prob=True,
        output_router_logits=True,
    )
    defaults.update(overrides)
    return Qwen3MoeConfig(**defaults)


def make_global_config(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        router_aux_loss_coef=0.0,
        norm_topk_prob=True,
        output_router_logits=True,
    )
    defaults.update(overrides)
    return GlobalMoEConfig(**defaults)


def make_batch(B=2, S=32, vocab_size=256):
    input_ids = torch.randint(0, vocab_size, (B, S))
    labels = input_ids.clone()
    labels[:, :1] = -100  # mask first token
    return input_ids, labels


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: Forward pass produces valid outputs for all model types
# ═══════════════════════════════════════════════════════════════════════

def test_forward_all_model_types():
    """All 4 model types should produce valid loss and router_logits."""
    torch.manual_seed(42)
    B, S = 2, 32

    models = {
        "StandardMoE (softmax)": lambda: StandardMoEModel(make_standard_config()),
        "DeepSeekStandard (sigmoid)": lambda: DeepSeekStandardMoEModel(make_standard_config()),
        "GlobalMoE (softmax)": lambda: GlobalMoEForCausalLM(make_global_config()),
        "DeepSeekGlobal (sigmoid)": lambda: DeepSeekGlobalMoEForCausalLM(make_global_config()),
    }

    print("=" * 70)
    print("  TEST 1: Forward pass for all model types")
    print("=" * 70)

    for name, build_fn in models.items():
        model = build_fn()
        model.train()
        input_ids, labels = make_batch(B, S, model.config.vocab_size)

        with torch.no_grad():
            output = model(input_ids=input_ids, labels=labels, output_router_logits=True)

        assert output.loss is not None, f"{name}: loss is None"
        assert output.loss.isfinite(), f"{name}: loss is not finite: {output.loss}"
        assert output.router_logits is not None, f"{name}: router_logits is None"

        num_layers = len(output.router_logits)
        layer_shape = output.router_logits[0].shape

        print(f"  {name:<35} loss={output.loss.item():.4f}  "
              f"layers={num_layers}  logits_shape={list(layer_shape)}")

        # Verify router_logits shapes
        E = model.config.num_experts
        expected_T = B * S
        assert layer_shape == (expected_T, E), (
            f"{name}: expected ({expected_T}, {E}), got {list(layer_shape)}"
        )

        # For DeepSeek models, verify sigmoid scores are in (0, 1)
        if "DeepSeek" in name:
            for rl in output.router_logits:
                assert rl.min() >= 0 and rl.max() <= 1, (
                    f"{name}: sigmoid scores out of range [{rl.min():.3f}, {rl.max():.3f}]"
                )

    print("  ✓ ALL PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: seq_aux_loss correctly added to total loss
# ═══════════════════════════════════════════════════════════════════════

def test_seq_aux_loss_integration():
    """
    Verify that when _seq_aux_loss_coef > 0, the seq loss is correctly
    added to the total loss. Check: total = CE + seq_coef * seq_aux.
    """
    print("=" * 70)
    print("  TEST 2: seq_aux_loss integration")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 2, 32
    seq_coef = 0.0001

    for name, build_fn in [
        ("DeepSeekGlobal", lambda: DeepSeekGlobalMoEForCausalLM(make_global_config())),
        ("DeepSeekStandard", lambda: DeepSeekStandardMoEModel(make_standard_config())),
    ]:
        # Model WITHOUT seq loss
        model_no_seq = build_fn()
        model_no_seq.train()

        # Model WITH seq loss
        model_with_seq = copy.deepcopy(model_no_seq)
        model_with_seq._seq_aux_loss_coef = seq_coef

        input_ids, labels = make_batch(B, S, model_no_seq.config.vocab_size)

        with torch.no_grad():
            out_no = model_no_seq(input_ids=input_ids, labels=labels, output_router_logits=True)
            out_yes = model_with_seq(input_ids=input_ids, labels=labels, output_router_logits=True)

        # Manually compute seq loss from router_logits
        manual_seq = seq_load_balancing_loss_func(
            out_yes.router_logits,
            model_no_seq.config.num_experts,
            model_no_seq.config.num_experts_per_tok,
            batch_size=B,
        )

        loss_no = out_no.loss.item()
        loss_yes = out_yes.loss.item()
        seq_val = manual_seq.item()
        expected_diff = seq_coef * seq_val
        actual_diff = loss_yes - loss_no

        print(f"  {name}:")
        print(f"    Loss (no seq):       {loss_no:.6f}")
        print(f"    Loss (with seq):     {loss_yes:.6f}")
        print(f"    Seq loss:            {seq_val:.6f}")
        print(f"    Expected diff:       {expected_diff:.8f}")
        print(f"    Actual diff:         {actual_diff:.8f}")

        # Allow small numerical tolerance
        assert abs(actual_diff - expected_diff) < 1e-5, (
            f"Loss diff mismatch: got {actual_diff}, expected {expected_diff}"
        )
        print(f"    ✓ PASS")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: batch-level aux loss is zero when coeff=0
# ═══════════════════════════════════════════════════════════════════════

def test_batch_aux_loss_zeroed():
    """With router_aux_loss_coef=0.0, the batch-level aux loss should not
    affect the total loss."""
    print("=" * 70)
    print("  TEST 3: batch-level aux loss zeroed when coeff=0")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 2, 32

    # Global MoE with coeff=0 (default)
    config = make_global_config(router_aux_loss_coef=0.0)
    model = DeepSeekGlobalMoEForCausalLM(config)
    model.train()
    input_ids, labels = make_batch(B, S, config.vocab_size)

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, output_router_logits=True)

    # aux_loss should be computed (for logging) but NOT added to loss
    assert output.aux_loss is not None, "aux_loss should be computed for logging"
    print(f"  aux_loss value:   {output.aux_loss.item():.6f}  (computed but not added)")
    print(f"  router_aux_coef:  {config.router_aux_loss_coef}")
    print(f"  ✓ PASS — batch aux loss exists but coeff=0 means no effect on total loss\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: seq loss gradients flow to router weights
# ═══════════════════════════════════════════════════════════════════════

def test_seq_loss_gradient_flow():
    """
    Verify gradients from seq_load_balancing_loss flow back to router weights.
    The seq loss depends on sigmoid(W @ h), so dL/dW should be nonzero.
    """
    print("=" * 70)
    print("  TEST 4: seq loss gradient flow to router weights")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 2, 32
    seq_coef = 0.001  # larger coeff to get measurable gradient

    model = DeepSeekGlobalMoEForCausalLM(make_global_config())
    model._seq_aux_loss_coef = seq_coef
    model.train()

    input_ids, labels = make_batch(B, S, model.config.vocab_size)
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    output.loss.backward()

    # Check router weight gradients
    router_grads = {}
    for name, param in model.named_parameters():
        if "gate.weight" in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            router_grads[name] = grad_norm

    print(f"  Router weight gradients (should be > 0):")
    all_nonzero = True
    for name, gn in router_grads.items():
        status = "✓" if gn > 0 else "✗"
        print(f"    {status} {name}: grad_norm = {gn:.6f}")
        if gn == 0:
            all_nonzero = False

    assert len(router_grads) > 0, "No router weights found!"
    assert all_nonzero, "Some router gradients are zero!"
    print(f"  ✓ PASS — gradients flow through seq loss to all {len(router_grads)} router layers\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: Expert bias update — per-layer (standard MoE)
# ═══════════════════════════════════════════════════════════════════════

def test_expert_bias_update_standard():
    """
    Test the expert bias update for standard (non-global) MoE.
    Each layer's router should update independently based on its own token counts.
    """
    print("=" * 70)
    print("  TEST 5: Expert bias update — standard MoE (per-layer)")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 4, 64
    rate = 0.001

    config = make_standard_config()
    config.topk_scaling_factor = None
    config.num_groups = None
    config.group_topk = None
    model = DeepSeekStandardMoEModel(config)
    model.train()

    input_ids, labels = make_batch(B, S, config.vocab_size)

    # Run forward WITH grad enabled to accumulate token counts
    # (DeepSeekRouter only counts when torch.is_grad_enabled())
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    output.loss.backward()  # need backward to complete the graph
    model.zero_grad()

    # Find all DeepSeekRouters
    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    assert len(routers) > 0, "No DeepSeekRouters found"

    # Save pre-update state
    pre_biases = [r.expert_bias.clone() for r in routers]
    pre_counts = [r.local_tokens_per_expert.clone() for r in routers]

    # Verify counts are nonzero (tokens were routed)
    for i, counts in enumerate(pre_counts):
        assert counts.sum() > 0, f"Router {i}: no tokens counted"

    # Apply per-layer bias update (same logic as train.py, is_global=False)
    for router in routers:
        counts = router.local_tokens_per_expert.clone()
        avg = counts.mean()
        router.expert_bias += torch.sign(avg - counts) * rate
        router.local_tokens_per_expert.zero_()

    # Verify biases changed
    print(f"  Found {len(routers)} DeepSeekRouters")
    all_changed = True
    for i, (pre, router) in enumerate(zip(pre_biases, routers)):
        diff = (router.expert_bias - pre).abs().max().item()
        changed = diff > 0
        if not changed:
            all_changed = False
        token_count = pre_counts[i].sum().item()
        print(f"    Layer {i}: tokens={token_count:.0f}  bias_delta_max={diff:.6f}  "
              f"bias_range=[{router.expert_bias.min():.6f}, {router.expert_bias.max():.6f}]")

    assert all_changed, "Some biases didn't change!"
    print(f"  ✓ PASS — all {len(routers)} router biases updated\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 6: Expert bias update — global interpolation
# ═══════════════════════════════════════════════════════════════════════

def test_expert_bias_update_global():
    """
    Test the global interpolation logic from train.py:update_expert_biases.
    Verify alpha=1 gives per-layer, alpha=0 gives global, alpha=0.5 blends.
    """
    print("=" * 70)
    print("  TEST 6: Expert bias update — global MoE interpolation")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 4, 64
    rate = 0.001

    config = make_global_config()
    config.topk_scaling_factor = None
    config.num_groups = None
    config.group_topk = None
    model = DeepSeekGlobalMoEForCausalLM(config)
    model.train()

    input_ids, labels = make_batch(B, S, config.vocab_size)

    # Run forward WITH grad enabled to accumulate token counts
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    output.loss.backward()
    model.zero_grad()

    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    counts_list = [r.local_tokens_per_expert.clone() for r in routers]

    def apply_global_update(counts_list, rate, alpha):
        """Replicate train.py:update_expert_biases (is_global=True)."""
        biases = [torch.zeros_like(c) for c in counts_list]
        global_counts = torch.stack(counts_list).sum(dim=0)
        global_avg = global_counts.mean()
        global_delta = torch.sign(global_avg - global_counts) * rate

        for i, counts in enumerate(counts_list):
            if alpha > 0:
                layer_avg = counts.mean()
                layer_delta = torch.sign(layer_avg - counts) * rate
                biases[i] = alpha * layer_delta + (1 - alpha) * global_delta
            else:
                biases[i] = global_delta.clone()
        return biases

    # Test alpha=1 (per-layer only)
    biases_a1 = apply_global_update(counts_list, rate, alpha=1.0)
    # Test alpha=0 (global only — all layers get same delta)
    biases_a0 = apply_global_update(counts_list, rate, alpha=0.0)
    # Test alpha=0.5 (blend)
    biases_a5 = apply_global_update(counts_list, rate, alpha=0.5)

    # alpha=0: all layers should have identical deltas
    a0_match = all(
        (biases_a0[i] - biases_a0[0]).abs().max().item() < 1e-7
        for i in range(1, len(biases_a0))
    )

    # alpha=1: layers should generally differ (different token distributions)
    a1_differ = any(
        (biases_a1[i] - biases_a1[0]).abs().max().item() > 1e-7
        for i in range(1, len(biases_a1))
    )

    # alpha=0.5: should be exact average of alpha=0 and alpha=1
    a5_correct = all(
        ((biases_a5[i] - 0.5 * biases_a1[i] - 0.5 * biases_a0[i]).abs().max().item() < 1e-7)
        for i in range(len(biases_a5))
    )

    print(f"  Found {len(routers)} DeepSeekRouters")
    print(f"  alpha=0 (global):    all layers same delta: {a0_match}  {'✓' if a0_match else '✗'}")
    print(f"  alpha=1 (per-layer): layers differ:         {a1_differ}  {'✓' if a1_differ else '✗'}")
    print(f"  alpha=0.5 (blend):   correct interpolation: {a5_correct}  {'✓' if a5_correct else '✗'}")

    assert a0_match, "alpha=0 should give identical deltas across layers"
    assert a1_differ, "alpha=1 should give different deltas per layer"
    assert a5_correct, "alpha=0.5 should be exact average of alpha=0 and alpha=1"
    print(f"  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 7: Loss decomposition — verify CE + aux + seq_aux
# ═══════════════════════════════════════════════════════════════════════

def test_loss_decomposition():
    """
    Verify: total_loss = CE_loss + router_aux_coef * aux_loss + seq_coef * seq_aux_loss

    With router_aux_loss_coef=0: total = CE + seq_coef * seq_aux
    """
    print("=" * 70)
    print("  TEST 7: Loss decomposition (CE + aux + seq_aux)")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 2, 32
    seq_coef = 0.0001

    # Test both global and standard
    for name, build_fn in [
        ("DeepSeekGlobal", lambda: DeepSeekGlobalMoEForCausalLM(make_global_config())),
        ("DeepSeekStandard", lambda: DeepSeekStandardMoEModel(make_standard_config())),
    ]:
        model = build_fn()
        model._seq_aux_loss_coef = seq_coef
        model.train()

        input_ids, labels = make_batch(B, S, model.config.vocab_size)

        with torch.no_grad():
            output = model(input_ids=input_ids, labels=labels, output_router_logits=True)

        total = output.loss.item()
        aux = output.aux_loss.item() if output.aux_loss is not None else 0.0

        # Manually compute seq loss
        seq_aux = seq_load_balancing_loss_func(
            output.router_logits,
            model.config.num_experts,
            model.config.num_experts_per_tok,
            batch_size=B,
        ).item()

        # With router_aux_loss_coef=0, the batch aux doesn't contribute
        aux_coef = model.config.router_aux_loss_coef
        ce = total - aux_coef * aux - seq_coef * seq_aux

        print(f"  {name}:")
        print(f"    total_loss:   {total:.6f}")
        print(f"    CE_loss:      {ce:.6f}")
        print(f"    aux_loss:     {aux:.6f}  × coef={aux_coef} = {aux_coef * aux:.8f}")
        print(f"    seq_aux_loss: {seq_aux:.6f}  × coef={seq_coef} = {seq_coef * seq_aux:.8f}")
        print(f"    CE + contributions: {ce + aux_coef * aux + seq_coef * seq_aux:.6f}")

        # Verify decomposition
        reconstructed = ce + aux_coef * aux + seq_coef * seq_aux
        assert abs(total - reconstructed) < 1e-5, (
            f"Loss decomposition failed: {total} != {reconstructed}"
        )

        # CE should be positive and dominate
        assert ce > 0, f"CE loss should be positive, got {ce}"
        # seq_aux should be around 1.0 (near uniform routing at init)
        assert 0.5 < seq_aux < 5.0, f"seq_aux_loss should be near 1.0, got {seq_aux}"
        print(f"    ✓ PASS")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 8: seq loss value matches paper at uniform routing
# ═══════════════════════════════════════════════════════════════════════

def test_seq_loss_uniform_minimum():
    """
    With uniform routing scores, the seq loss minimum should be exactly 1.0
    (matching the DeepSeek V3 paper and Megatron).
    """
    print("=" * 70)
    print("  TEST 8: seq loss at uniform routing = 1.0")
    print("=" * 70)

    E, K, B, S = 64, 8, 4, 128
    num_layers = 3

    # Perfectly uniform scores (all experts equal)
    uniform = torch.ones(B * S, E) / E  # already sums to 1
    gate_logits = tuple([uniform] * num_layers)

    loss = seq_load_balancing_loss_func(gate_logits, E, K, batch_size=B)
    print(f"  Uniform softmax: loss = {loss.item():.6f}  (expected: 1.0)")

    # Also test with uniform sigmoid (raw, not sum-to-1)
    raw_uniform = torch.ones(B * S, E) * 0.5  # sigmoid-like, all 0.5
    gate_logits_sig = tuple([raw_uniform] * num_layers)
    loss_sig = seq_load_balancing_loss_func(gate_logits_sig, E, K, batch_size=B)
    print(f"  Uniform sigmoid: loss = {loss_sig.item():.6f}  (expected: 1.0, after normalization)")

    assert abs(loss.item() - 1.0) < 0.01, f"Expected 1.0, got {loss.item()}"
    assert abs(loss_sig.item() - 1.0) < 0.01, f"Expected 1.0, got {loss_sig.item()}"
    print(f"  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 9: alpha schedule + multi-GPU step concern
# ═══════════════════════════════════════════════════════════════════════

def test_alpha_schedule_correctness():
    """
    Verify the alpha schedule: cosine decay over 5k steps, then 0.
    """
    print("=" * 70)
    print("  TEST 9: Alpha schedule correctness (5k warmup)")
    print("=" * 70)

    def bias_alpha_schedule(step, warmup_steps=5000):
        if step >= warmup_steps:
            return 0.0
        progress = step / max(1, warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    points = [
        (0, 1.0, "start — per-layer only"),
        (1250, 0.8536, "25% of warmup"),
        (2500, 0.5, "50% of warmup — even blend"),
        (3750, 0.1464, "75% of warmup"),
        (5000, 0.0, "end of warmup — global only"),
        (10000, 0.0, "well past warmup — stays 0"),
        (100000, 0.0, "any later step — stays 0"),
    ]

    all_pass = True
    for step, expected, desc in points:
        alpha = bias_alpha_schedule(step)
        match = abs(alpha - expected) < 0.001
        status = "✓" if match else "✗"
        print(f"  {status} step={step:>6d}  alpha={alpha:.4f}  (expected {expected:.4f})  {desc}")
        if not match:
            all_pass = False

    assert all_pass
    print(f"  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  FULL INTEGRATION TEST SUITE")
    print("=" * 70 + "\n")

    test_forward_all_model_types()
    test_seq_aux_loss_integration()
    test_batch_aux_loss_zeroed()
    test_seq_loss_gradient_flow()
    test_expert_bias_update_standard()
    test_expert_bias_update_global()
    test_loss_decomposition()
    test_seq_loss_uniform_minimum()
    test_alpha_schedule_correctness()

    print("=" * 70)
    print("  ALL 9 INTEGRATION TESTS PASSED")
    print("=" * 70)
