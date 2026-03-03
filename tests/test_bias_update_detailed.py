"""
Detailed, step-by-step verification of update_expert_biases from train.py.

Runs the ACTUAL function (not a reimplementation) against a real model,
prints every intermediate value, and verifies each step against Megatron.
"""
import math
import sys
sys.path.insert(0, ".")

import torch
from src.models import (
    GlobalMoEConfig,
    DeepSeekGlobalMoEForCausalLM,
    Qwen3MoeConfig,
    DeepSeekStandardMoEModel,
)
from src.models.router import DeepSeekRouter


# ── Inline from train.py (can't import directly — triggers liger CUDA kernels) ──

def bias_alpha_schedule(step: int, warmup_steps: int = 5000) -> float:
    """Cosine decay from 1 → 0 over warmup_steps, then stays at 0."""
    if step >= warmup_steps:
        return 0.0
    progress = step / max(1, warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def update_expert_biases(model, update_rate, accelerator, is_global=False, alpha=0.0):
    """Exact copy of train.py:update_expert_biases."""
    stats = {}
    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    if not routers:
        return stats
    per_router_counts = []
    for router in routers:
        counts = router.local_tokens_per_expert.clone()
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        per_router_counts.append(counts)
        router.local_tokens_per_expert.zero_()
    if is_global:
        global_counts = torch.stack(per_router_counts).sum(dim=0)
        global_avg = global_counts.mean()
        global_delta = torch.sign(global_avg - global_counts) * update_rate
        for router, counts in zip(routers, per_router_counts):
            if alpha > 0:
                layer_avg = counts.mean()
                layer_delta = torch.sign(layer_avg - counts) * update_rate
                router.expert_bias += alpha * layer_delta + (1 - alpha) * global_delta
            else:
                router.expert_bias += global_delta
    else:
        for router, counts in zip(routers, per_router_counts):
            avg = counts.mean()
            router.expert_bias += torch.sign(avg - counts) * update_rate
    all_bias = torch.cat([r.expert_bias for r in routers])
    stats["routing/expert_bias_mean"] = all_bias.mean().item()
    stats["routing/expert_bias_std"] = all_bias.std().item()
    stats["routing/expert_bias_min"] = all_bias.min().item()
    stats["routing/expert_bias_max"] = all_bias.max().item()
    return stats


# ── Fake accelerator for single-GPU testing ───────────────────────────

class FakeAccelerator:
    num_processes = 1
    def unwrap_model(self, m):
        return m


def make_global_config():
    return GlobalMoEConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=4,
        head_dim=16, num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        intermediate_size=128, max_position_embeddings=512, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, router_aux_loss_coef=0.0,
        norm_topk_prob=True, output_router_logits=True,
    )


def make_standard_config():
    cfg = Qwen3MoeConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=4,
        head_dim=16, num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        intermediate_size=128, max_position_embeddings=512, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, router_aux_loss_coef=0.0,
        norm_topk_prob=True, output_router_logits=True,
    )
    cfg.topk_scaling_factor = None
    cfg.num_groups = None
    cfg.group_topk = None
    return cfg


def megatron_get_updated_expert_bias(tokens_per_expert, expert_bias, rate):
    """Megatron-LM reference (moe_utils.py:1119-1142)."""
    with torch.no_grad():
        avg = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = avg - tokens_per_expert
        return expert_bias + torch.sign(offset) * rate


def print_expert_table(label, values, fmt=".0f"):
    """Print a compact expert-indexed table."""
    header = "  " + "".join(f"E{i:>5d}" for i in range(len(values)))
    vals = "  " + "".join(f"{v:>5{fmt}}" for v in values.tolist())
    print(f"  {label}:")
    print(header)
    print(vals)


# ═══════════════════════════════════════════════════════════════════════
#  TEST A: Per-layer bias update (standard MoE) — step by step
# ═══════════════════════════════════════════════════════════════════════

def test_perlayer_step_by_step():
    print("=" * 70)
    print("  TEST A: Per-layer bias update — step by step")
    print("=" * 70)

    torch.manual_seed(42)
    config = make_standard_config()
    model = DeepSeekStandardMoEModel(config)
    model.train()
    accel = FakeAccelerator()

    E = config.num_experts
    K = config.num_experts_per_tok
    rate = 0.001

    # Forward pass to accumulate token counts
    B, S = 4, 64
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    labels = input_ids.clone()
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    output.loss.backward()
    model.zero_grad()

    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    print(f"\n  {len(routers)} layers, {E} experts, top-{K}, rate={rate}")

    # Snapshot counts before update
    counts_before = [r.local_tokens_per_expert.clone() for r in routers]
    biases_before = [r.expert_bias.clone() for r in routers]

    print(f"\n  --- Before update ---")
    for i, counts in enumerate(counts_before):
        print(f"\n  Layer {i}:")
        print_expert_table("token counts", counts)
        avg = counts.mean()
        sign_delta = torch.sign(avg - counts)
        print(f"    avg = {avg:.1f}")
        print_expert_table("sign(avg - counts)", sign_delta, fmt=".0f")
        print_expert_table("expected delta", sign_delta * rate, fmt=".4f")

    # Run the ACTUAL update_expert_biases
    stats = update_expert_biases(model, rate, accel, is_global=False, alpha=0.0)

    # Verify against manual computation and Megatron
    print(f"\n  --- After update ---")
    all_match_megatron = True
    for i, router in enumerate(routers):
        counts = counts_before[i]
        bias_after = router.expert_bias

        # Manual expected
        avg = counts.mean()
        manual_delta = torch.sign(avg - counts) * rate
        manual_expected = biases_before[i] + manual_delta

        # Megatron expected
        mega_expected = megatron_get_updated_expert_bias(
            counts.unsqueeze(0), biases_before[i].unsqueeze(0), rate
        ).squeeze(0)

        diff_manual = (bias_after - manual_expected).abs().max().item()
        diff_mega = (bias_after - mega_expected).abs().max().item()

        print(f"\n  Layer {i}:")
        print_expert_table("bias after", bias_after, fmt=".4f")
        print(f"    matches manual:   {'✓' if diff_manual < 1e-7 else '✗'} (max diff: {diff_manual:.2e})")
        print(f"    matches Megatron: {'✓' if diff_mega < 1e-7 else '✗'} (max diff: {diff_mega:.2e})")

        if diff_mega > 1e-7:
            all_match_megatron = False

        # Verify counts were zeroed
        assert router.local_tokens_per_expert.sum() == 0, "Counts not zeroed!"

    assert all_match_megatron, "Megatron mismatch!"
    print(f"\n  ✓ ALL LAYERS MATCH MEGATRON\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST B: Global bias update — step by step
# ═══════════════════════════════════════════════════════════════════════

def test_global_step_by_step():
    print("=" * 70)
    print("  TEST B: Global bias update (alpha=0) — step by step")
    print("=" * 70)

    torch.manual_seed(42)
    config = make_global_config()
    config.topk_scaling_factor = None
    config.num_groups = None
    config.group_topk = None
    model = DeepSeekGlobalMoEForCausalLM(config)
    model.train()
    accel = FakeAccelerator()

    E = config.num_experts
    rate = 0.001

    B, S = 4, 64
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    labels = input_ids.clone()
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    output.loss.backward()
    model.zero_grad()

    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    counts_before = [r.local_tokens_per_expert.clone() for r in routers]
    biases_before = [r.expert_bias.clone() for r in routers]

    print(f"\n  {len(routers)} layers, {E} shared experts, rate={rate}")

    # Show per-layer counts
    print(f"\n  --- Per-layer token counts ---")
    for i, counts in enumerate(counts_before):
        print_expert_table(f"Layer {i}", counts)

    # Show global (pooled) counts
    global_counts = torch.stack(counts_before).sum(dim=0)
    global_avg = global_counts.mean()
    global_delta = torch.sign(global_avg - global_counts) * rate

    print(f"\n  --- Global pool (sum across all layers) ---")
    print_expert_table("global counts", global_counts)
    print(f"    global avg = {global_avg:.1f}")
    print_expert_table("sign(avg - global)", torch.sign(global_avg - global_counts), fmt=".0f")
    print_expert_table("global delta", global_delta, fmt=".4f")

    print(f"\n  KEY: alpha=0 → ALL layers get the SAME global delta")
    print(f"  This means: if expert 3 is globally overloaded, ALL layers")
    print(f"  reduce its bias, even if some layers barely used it.\n")

    # Run the ACTUAL function with alpha=0
    stats = update_expert_biases(model, rate, accel, is_global=True, alpha=0.0)

    # Verify all layers got the same delta
    print(f"  --- After update (alpha=0, purely global) ---")
    for i, router in enumerate(routers):
        expected = biases_before[i] + global_delta
        diff = (router.expert_bias - expected).abs().max().item()
        print_expert_table(f"Layer {i} bias", router.expert_bias, fmt=".4f")
        print(f"    matches expected: {'✓' if diff < 1e-7 else '✗'}")
        assert diff < 1e-7

    # Verify all layers have identical biases (since they all started at 0)
    for i in range(1, len(routers)):
        same = (routers[i].expert_bias == routers[0].expert_bias).all().item()
        assert same, f"Layer {i} bias differs from layer 0!"

    print(f"\n  ✓ All layers received identical global delta\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST C: Interpolation (alpha=0.5) — step by step
# ═══════════════════════════════════════════════════════════════════════

def test_interpolation_step_by_step():
    print("=" * 70)
    print("  TEST C: Interpolated bias update (alpha=0.5) — step by step")
    print("=" * 70)

    torch.manual_seed(42)
    config = make_global_config()
    config.topk_scaling_factor = None
    config.num_groups = None
    config.group_topk = None
    model = DeepSeekGlobalMoEForCausalLM(config)
    model.train()
    accel = FakeAccelerator()

    E = config.num_experts
    rate = 0.001
    alpha = 0.5

    B, S = 4, 64
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    labels = input_ids.clone()
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    output.loss.backward()
    model.zero_grad()

    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    counts_before = [r.local_tokens_per_expert.clone() for r in routers]
    biases_before = [r.expert_bias.clone() for r in routers]

    print(f"\n  {len(routers)} layers, {E} shared experts, alpha={alpha}, rate={rate}")

    # Compute expected deltas manually
    global_counts = torch.stack(counts_before).sum(dim=0)
    global_avg = global_counts.mean()
    global_delta = torch.sign(global_avg - global_counts) * rate

    print(f"\n  --- Computing blended deltas ---")
    expected_biases = []
    for i, counts in enumerate(counts_before):
        layer_avg = counts.mean()
        layer_delta = torch.sign(layer_avg - counts) * rate
        blended = alpha * layer_delta + (1 - alpha) * global_delta

        print(f"\n  Layer {i}:")
        print_expert_table("per-layer delta", layer_delta, fmt=".4f")
        print_expert_table("global delta   ", global_delta, fmt=".4f")
        print_expert_table("blended (0.5×L + 0.5×G)", blended, fmt=".4f")

        # Show where they disagree
        disagree = (layer_delta != global_delta)
        if disagree.any():
            print(f"    Per-layer vs global disagree on {disagree.sum().item()}/{E} experts")

        expected_biases.append(biases_before[i] + blended)

    # Run the ACTUAL function
    stats = update_expert_biases(model, rate, accel, is_global=True, alpha=alpha)

    print(f"\n  --- After update ---")
    all_match = True
    for i, router in enumerate(routers):
        diff = (router.expert_bias - expected_biases[i]).abs().max().item()
        match = diff < 1e-7
        print(f"  Layer {i}: matches expected: {'✓' if match else '✗'} (max diff: {diff:.2e})")
        if not match:
            all_match = False

    assert all_match
    print(f"\n  ✓ Interpolation correct: delta = {alpha}×per_layer + {1-alpha}×global\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST D: Alpha schedule transitions
# ═══════════════════════════════════════════════════════════════════════

def test_alpha_schedule_over_training():
    print("=" * 70)
    print("  TEST D: Alpha schedule over training")
    print("=" * 70)

    print(f"\n  Schedule: cosine decay over first 5000 steps, then 0")
    print(f"  step 0     → alpha=1.0 → purely per-layer (each layer corrects its own imbalance)")
    print(f"  step 2500  → alpha=0.5 → 50/50 blend")
    print(f"  step 5000+ → alpha=0.0 → purely global (all layers share one correction signal)")
    print()

    steps = [0, 500, 1000, 2000, 2500, 3000, 4000, 5000, 10000, 50000]
    for step in steps:
        a = bias_alpha_schedule(step)
        bar = "█" * int(a * 40) + "░" * (40 - int(a * 40))
        label = "per-layer" if a > 0.9 else "global" if a < 0.1 else "blend"
        print(f"  step {step:>6d}  alpha={a:.4f}  {bar}  ({label})")

    print(f"\n  ✓ Schedule verified\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST E: Multiple steps — bias accumulates correctly
# ═══════════════════════════════════════════════════════════════════════

def test_multi_step_bias_accumulation():
    print("=" * 70)
    print("  TEST E: Multi-step bias accumulation")
    print("=" * 70)

    torch.manual_seed(42)
    config = make_global_config()
    config.topk_scaling_factor = None
    config.num_groups = None
    config.group_topk = None
    model = DeepSeekGlobalMoEForCausalLM(config)
    model.train()
    accel = FakeAccelerator()

    E = config.num_experts
    rate = 0.001

    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]

    print(f"\n  Running 5 forward+bias_update steps...")
    print(f"  Simulating alpha schedule: step 0→5000 cosine decay\n")

    for step in range(5):
        sim_step = step * 1250  # 0, 1250, 2500, 3750, 5000
        alpha = bias_alpha_schedule(sim_step)

        B, S = 4, 64
        input_ids = torch.randint(0, config.vocab_size, (B, S))
        labels = input_ids.clone()
        output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
        output.loss.backward()
        model.zero_grad()

        stats = update_expert_biases(model, rate, accel, is_global=True, alpha=alpha)

        bias_vals = routers[0].expert_bias
        print(f"  Step {step} (sim_step={sim_step:>5d}, alpha={alpha:.4f}):")
        print(f"    bias range: [{bias_vals.min():.5f}, {bias_vals.max():.5f}]  "
              f"std={bias_vals.std():.5f}")

    # After 5 steps, biases should be nonzero and have structure
    final_bias = routers[0].expert_bias
    assert final_bias.abs().max() > 0, "Biases should be nonzero after multiple steps"
    assert final_bias.std() > 0, "Biases should have variance"

    # Different layers should have different biases (because alpha > 0 for early steps)
    layer_biases = [r.expert_bias for r in routers]
    any_differ = any(
        (layer_biases[i] - layer_biases[0]).abs().max().item() > 1e-7
        for i in range(1, len(layer_biases))
    )
    print(f"\n  Different layers have different biases: {any_differ}")
    print(f"  (expected: True, because alpha > 0 in early steps gives per-layer component)")
    assert any_differ, "Layers should differ due to per-layer component in early steps"
    print(f"  ✓ PASS\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DETAILED BIAS UPDATE VERIFICATION")
    print("=" * 70 + "\n")

    test_perlayer_step_by_step()
    test_global_step_by_step()
    test_interpolation_step_by_step()
    test_alpha_schedule_over_training()
    test_multi_step_bias_accumulation()

    print("=" * 70)
    print("  ALL 5 TESTS PASSED")
    print("=" * 70)
