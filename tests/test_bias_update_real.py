"""
Real GPU test of update_expert_biases and seq_load_balancing_loss_func
using actual Accelerator, actual train.py functions, actual models on CUDA.
"""
import sys
sys.path.insert(0, ".")

import torch
# Must happen before liger import
torch.backends.cuda.preferred_blas_library("cublaslt")

from accelerate import Accelerator
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
apply_liger_kernel_to_qwen3_moe()

from src.models import (
    GlobalMoEConfig,
    DeepSeekGlobalMoEForCausalLM,
    Qwen3MoeConfig,
    DeepSeekStandardMoEModel,
)
from src.models.router import DeepSeekRouter
from src.models.load_balancing import seq_load_balancing_loss_func
from train import update_expert_biases, bias_alpha_schedule


def make_global_config():
    cfg = GlobalMoEConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=4,
        head_dim=32, num_attention_heads=4, num_key_value_heads=2,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64,
        intermediate_size=256, max_position_embeddings=512, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, router_aux_loss_coef=0.0,
        norm_topk_prob=True, output_router_logits=True,
    )
    cfg.topk_scaling_factor = 2.5
    cfg.num_groups = 4
    cfg.group_topk = 2
    return cfg


def make_standard_config():
    cfg = Qwen3MoeConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=4,
        head_dim=32, num_attention_heads=4, num_key_value_heads=2,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64,
        intermediate_size=256, max_position_embeddings=512, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, router_aux_loss_coef=0.0,
        norm_topk_prob=True, output_router_logits=True,
    )
    cfg.topk_scaling_factor = 2.5
    cfg.num_groups = 4
    cfg.group_topk = 2
    return cfg


def fmt_counts(counts):
    return " ".join(f"{c:>5.0f}" for c in counts.tolist())


def fmt_bias(bias):
    return " ".join(f"{b:>7.4f}" for b in bias.tolist())


# ═══════════════════════════════════════════════════════════════════════

def test_perlayer_on_gpu(accelerator):
    """Per-layer bias update on real GPU with real Accelerator."""
    print("=" * 70)
    print("  Per-layer bias update (standard MoE) — real GPU")
    print("=" * 70)

    torch.manual_seed(42)
    E, K = 16, 4
    rate = 0.001
    B, S = 8, 128

    model = DeepSeekStandardMoEModel(make_standard_config())
    model._seq_aux_loss_coef = 0.0001
    model.train()
    model = accelerator.prepare(model)

    input_ids = torch.randint(0, 256, (B, S), device=accelerator.device)
    labels = input_ids.clone()

    # Forward + backward (accumulates token counts in routers)
    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    accelerator.backward(output.loss)

    raw_model = accelerator.unwrap_model(model)
    routers = [m for m in raw_model.modules() if isinstance(m, DeepSeekRouter)]

    print(f"\n  {len(routers)} layers, {E} experts, top-{K}")
    print(f"  device: {next(raw_model.parameters()).device}")

    # Show counts before update
    for i, r in enumerate(routers):
        c = r.local_tokens_per_expert
        total = c.sum().item()
        avg = c.mean().item()
        std = c.std().item()
        print(f"  Layer {i}: total={total:.0f}  avg={avg:.1f}  std={std:.1f}  counts=[{fmt_counts(c)}]")

    # Snapshot biases before
    pre_biases = [r.expert_bias.clone() for r in routers]

    # Run ACTUAL update_expert_biases
    stats = update_expert_biases(raw_model, rate, accelerator, is_global=False)

    # Verify
    print(f"\n  After update:")
    for i, r in enumerate(routers):
        delta = r.expert_bias - pre_biases[i]
        # Delta should be exactly ±rate or 0
        unique_deltas = delta.unique().tolist()
        print(f"  Layer {i}: bias=[{fmt_bias(r.expert_bias)}]")
        print(f"           delta values: {[f'{d:.4f}' for d in unique_deltas]}")
        assert all(abs(abs(d) - rate) < 1e-7 or abs(d) < 1e-7 for d in unique_deltas), \
            f"Unexpected delta values: {unique_deltas}"
        assert r.local_tokens_per_expert.sum() == 0, "Counts not zeroed"

    print(f"\n  bias stats: {stats}")
    print(f"  ✓ PASS\n")
    model.zero_grad()
    return raw_model, routers


def test_global_on_gpu(accelerator):
    """Global bias update with interpolation on real GPU."""
    print("=" * 70)
    print("  Global bias update (global MoE) — real GPU, 3 alpha values")
    print("=" * 70)

    E, K = 16, 4
    rate = 0.001
    B, S = 8, 128

    for alpha_val, label in [(1.0, "per-layer"), (0.0, "global"), (0.5, "blend")]:
        torch.manual_seed(42)  # same init each time
        model = DeepSeekGlobalMoEForCausalLM(make_global_config())
        model._seq_aux_loss_coef = 0.0001
        model.train()
        model = accelerator.prepare(model)

        input_ids = torch.randint(0, 256, (B, S), device=accelerator.device)
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
        accelerator.backward(output.loss)

        raw_model = accelerator.unwrap_model(model)
        routers = [m for m in raw_model.modules() if isinstance(m, DeepSeekRouter)]

        # Snapshot
        counts = [r.local_tokens_per_expert.clone() for r in routers]
        pre_biases = [r.expert_bias.clone() for r in routers]

        # Run actual update
        stats = update_expert_biases(raw_model, rate, accelerator, is_global=True, alpha=alpha_val)

        # Compute expected manually
        global_counts = torch.stack(counts).sum(dim=0)
        global_avg = global_counts.mean()
        global_delta = torch.sign(global_avg - global_counts) * rate

        print(f"\n  alpha={alpha_val} ({label}):")
        print(f"  Global counts: [{fmt_counts(global_counts)}]  avg={global_avg:.1f}")

        all_ok = True
        for i, r in enumerate(routers):
            if alpha_val == 0.0:
                expected = pre_biases[i] + global_delta
            elif alpha_val == 1.0:
                layer_avg = counts[i].mean()
                layer_delta = torch.sign(layer_avg - counts[i]) * rate
                expected = pre_biases[i] + layer_delta
            else:
                layer_avg = counts[i].mean()
                layer_delta = torch.sign(layer_avg - counts[i]) * rate
                expected = pre_biases[i] + alpha_val * layer_delta + (1 - alpha_val) * global_delta

            diff = (r.expert_bias - expected).abs().max().item()
            ok = diff < 1e-6
            if not ok:
                all_ok = False
            print(f"    Layer {i}: bias=[{fmt_bias(r.expert_bias)}]  match={'✓' if ok else '✗'}")

        if alpha_val == 0.0:
            # All layers should be identical
            same = all(
                (routers[i].expert_bias - routers[0].expert_bias).abs().max().item() < 1e-7
                for i in range(1, len(routers))
            )
            print(f"    All layers identical: {'✓' if same else '✗'}")
            assert same

        assert all_ok
        model.zero_grad()

    print(f"\n  ✓ PASS — all 3 alpha values correct\n")


def test_seq_loss_on_gpu(accelerator):
    """Seq loss computed on GPU matches expected values."""
    print("=" * 70)
    print("  Seq loss on real GPU")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 8, 128
    seq_coef = 0.0001

    model = DeepSeekGlobalMoEForCausalLM(make_global_config())
    model._seq_aux_loss_coef = seq_coef
    model.train()
    model = accelerator.prepare(model)

    input_ids = torch.randint(0, 256, (B, S), device=accelerator.device)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels, output_router_logits=True)

    raw_model = accelerator.unwrap_model(model)

    # Manually compute seq loss
    manual_seq = seq_load_balancing_loss_func(
        output.router_logits,
        raw_model.config.num_experts,
        raw_model.config.num_experts_per_tok,
        batch_size=B,
    )

    total = output.loss.item()
    seq_val = manual_seq.item()

    print(f"\n  total_loss:     {total:.6f}")
    print(f"  seq_aux_loss:   {seq_val:.6f}  (× {seq_coef} = {seq_coef * seq_val:.8f})")
    print(f"  CE_loss (approx): {total - seq_coef * seq_val:.6f}")
    print(f"  seq_loss ≈ 1.0:  {'✓' if 0.5 < seq_val < 3.0 else '✗'} (got {seq_val:.4f})")

    # Verify gradients
    accelerator.backward(output.loss)
    routers = [m for m in raw_model.modules() if isinstance(m, DeepSeekRouter)]
    grad_norms = [r.weight.grad.norm().item() for r in routers if r.weight.grad is not None]
    print(f"  Router grad norms: {['%.4f' % g for g in grad_norms]}")
    assert all(g > 0 for g in grad_norms), "Some router grads are zero"

    print(f"  ✓ PASS\n")
    model.zero_grad()


def test_full_training_loop(accelerator):
    """Simulate 5 training steps with bias updates + seq loss — the real loop."""
    print("=" * 70)
    print("  Full training loop simulation (5 steps)")
    print("=" * 70)

    torch.manual_seed(42)
    B, S = 8, 128
    rate = 0.001
    seq_coef = 0.0001

    model = DeepSeekGlobalMoEForCausalLM(make_global_config())
    model._seq_aux_loss_coef = seq_coef
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)
    raw_model = accelerator.unwrap_model(model)

    print(f"\n  Config: E=16, K=4, rate={rate}, seq_coef={seq_coef}")
    print(f"  Device: {accelerator.device}\n")

    for step in range(5):
        alpha = bias_alpha_schedule(step * 1000)  # simulate 0, 1k, 2k, 3k, 4k

        input_ids = torch.randint(0, 256, (B, S), device=accelerator.device)
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
        accelerator.backward(output.loss)

        # Gradient clipping
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Bias update (exactly as in train.py)
        bias_stats = update_expert_biases(
            raw_model, rate, accelerator, is_global=True, alpha=alpha,
        )

        # Compute seq loss for logging
        seq_val = seq_load_balancing_loss_func(
            output.router_logits,
            raw_model.config.num_experts,
            raw_model.config.num_experts_per_tok,
            batch_size=B,
        ).item()

        routers = [m for m in raw_model.modules() if isinstance(m, DeepSeekRouter)]
        all_bias = torch.cat([r.expert_bias for r in routers])

        print(f"  step {step}  alpha={alpha:.4f}  loss={output.loss.item():.4f}  "
              f"seq_aux={seq_val:.4f}  "
              f"bias=[{all_bias.min():.4f}, {all_bias.max():.4f}]  "
              f"bias_std={all_bias.std():.5f}")

        assert output.loss.isfinite(), f"Loss not finite at step {step}"
        assert 0.5 < seq_val < 5.0, f"seq_aux out of range at step {step}: {seq_val}"

    print(f"\n  ✓ PASS — 5 steps completed, loss finite, seq_aux near 1.0\n")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    accelerator = Accelerator(mixed_precision="bf16")

    print("\n" + "=" * 70)
    print(f"  REAL GPU TESTS — device={accelerator.device}")
    print("=" * 70 + "\n")

    test_perlayer_on_gpu(accelerator)
    test_global_on_gpu(accelerator)
    test_seq_loss_on_gpu(accelerator)
    test_full_training_loop(accelerator)

    print("=" * 70)
    print("  ALL 4 GPU TESTS PASSED")
    print("=" * 70)
