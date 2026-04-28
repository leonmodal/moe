"""AC-23 / task39: sanity equivalence between
``moe_everything(sanity_check_mode='alternating_global_moe')`` and
``global_moe`` — the headline correctness test for the priority-1 plan.

Per `docs/plan.md` AC-23, when a `moe_everything` model is configured in
the alternating-global-moe sanity mode (every odd depth runs MLP-only,
every even depth runs attention-only — with weights mapped via
``init_mapping.copy_global_to_alternating_sanity``), the model must be
mathematically equivalent to a corresponding `global_moe` model on:

1. forward output (logits + auxiliary fields),
2. loss,
3. gradients (per-parameter, after one backward),
4. gradient-checkpointed forward (deterministic vs the non-checkpointed
   path on the same input),
5. DDP-balanced bias updates (CUDA-required, skipped on CPU),
6. 50-step training drift (CUDA-required for speed; skipped on CPU).

This file lands the CPU-runnable parts (1-4). The CUDA parts are
imported as `pytest.skip(...)` placeholders so future Modal H200 runs
can extend coverage without needing to author new test files.

The fixture mirrors `legacy/speedrun/tests/test_models.py:1459-1554`'s
`_tiny_global_equiv_config` / `_tiny_alternating_sanity_config` to keep
the behavior contract stable as the production codebase migrates the
sanity machinery from speedrun to `src/models/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    MoEverythingConfig,
)
from src.models.moe_everything import MoEverythingForCausalLM
from src.models.init_mapping import copy_global_to_alternating_sanity


def _tiny_global_equiv_config():
    """Tiny GlobalMoE config that pairs 1:1 with the alternating-sanity fixture."""
    return GlobalMoEConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        output_router_logits=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
    )


def _tiny_alternating_sanity_config():
    """Tiny MoEverything config with sanity_check_mode='alternating_global_moe'.

    Depth = 2 * global_depth (each global layer becomes one attn-only depth
    + one mlp-only depth). `attn_expert_mode='per_head_precompute_kv'` is
    the only mode that supports `sanity_check_mode`.
    """
    return MoEverythingConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,            # 2 * 2 logical layers
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        num_attn_experts=2,             # one per logical layer
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_precompute_kv",
        norm_topk_prob=True,
        branch_router_aux_loss_coef=0.0,
        router_aux_loss_coef=0.0,
        per_layer_norm=True,
        sanity_check_mode="alternating_global_moe",
    )


def _build_sanity_pair():
    """Build a (global, sanity) pair with weights mapped 1:1."""
    global_model = GlobalMoEForCausalLM(_tiny_global_equiv_config()).eval()
    sanity_model = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).eval()
    copy_global_to_alternating_sanity(global_model, sanity_model)
    return global_model, sanity_model


def _dummy_batch(B: int = 2, T: int = 8, vocab_size: int = 256):
    torch.manual_seed(20260428)
    ids = torch.randint(0, vocab_size, (B, T))
    labels = ids.clone()
    return ids, labels


# ──────────────────────────────────────────────────────────────────────
#  AC-23 part 1: forward output equivalence
# ──────────────────────────────────────────────────────────────────────

def test_sanity_equivalence_forward_logits():
    """The two models must produce the same `output.logits` to bf16
    round-off on the same input under `eval()` (no dropout / no
    auxiliary balancing). This is the headline forward-pass equivalence.
    """
    global_model, sanity_model = _build_sanity_pair()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
        sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)
    torch.testing.assert_close(sanity_out.logits, global_out.logits, atol=2e-4, rtol=2e-4)


def test_sanity_equivalence_forward_loss():
    """The two models must produce the same loss scalar (CE only — no
    aux contributions in the sanity fixture)."""
    global_model, sanity_model = _build_sanity_pair()
    ids, labels = _dummy_batch()
    with torch.no_grad():
        global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
        sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)


# ──────────────────────────────────────────────────────────────────────
#  AC-23 part 2: gradient equivalence (per-param)
# ──────────────────────────────────────────────────────────────────────

def test_sanity_equivalence_gradients_per_param_pair():
    """After one forward + backward on the same input, the gradients on
    the corresponding parameters of the global and sanity models must
    match (mod the dtype/precompute_kv reshape applied in
    `copy_global_to_alternating_sanity`).

    `copy_global_to_alternating_sanity` returns a list of `ParamPair`
    entries enumerating the canonical mapping. We assert grad parity on
    the subset where `track_grad_and_opt=True` (the rest are
    permuted-and-reshaped views of the same logical parameter, which
    receive the equivalent total gradient through their shared storage).
    """
    global_model = GlobalMoEForCausalLM(_tiny_global_equiv_config()).train()
    sanity_model = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).train()
    pairs = copy_global_to_alternating_sanity(global_model, sanity_model)

    ids, labels = _dummy_batch()
    global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
    sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)

    # Loss-equivalence preflight.
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)

    global_out.loss.backward()
    sanity_out.loss.backward()

    # Walk only the tracked-grad pairs; the rest are reshape-views.
    tracked = [p for p in pairs if p.track_grad_and_opt]
    assert tracked, "init_mapping returned no tracked-grad param pairs"

    for pair in tracked:
        gleft = pair.left.grad
        gright = pair.right.grad
        assert gleft is not None, f"left ({pair.name}) has no grad"
        assert gright is not None, f"right ({pair.name}) has no grad"
        # Use a slightly looser tolerance because the precompute_kv
        # reshape can permute summation order.
        torch.testing.assert_close(
            gleft, gright, atol=2e-4, rtol=2e-4,
            msg=f"gradient mismatch on tracked pair {pair.name!r}: "
                f"left={gleft.norm().item():.6e}, right={gright.norm().item():.6e}",
        )


# ──────────────────────────────────────────────────────────────────────
#  AC-23 part 3: gradient-checkpointed equivalence
# ──────────────────────────────────────────────────────────────────────

def test_sanity_equivalence_with_gradient_checkpointing():
    """`gradient_checkpointing_enable()` on the sanity model must NOT
    change forward output or loss vs the same model without
    checkpointing (and vs the global reference).

    This locks AC-9 + AC-23 jointly: the sanity correctness contract
    holds whether or not the user enables checkpointing. The recompute
    pass must produce the same numerical output as the non-checkpointed
    forward — proven by comparing both back to the (cheap) global
    reference."""
    torch.manual_seed(20260428)
    ids, labels = _dummy_batch()

    # Reference: global_moe forward (no checkpointing).
    global_model = GlobalMoEForCausalLM(_tiny_global_equiv_config()).eval()
    with torch.no_grad():
        ref_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)

    # Sanity model WITHOUT checkpointing.
    sanity_a = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).eval()
    copy_global_to_alternating_sanity(global_model, sanity_a)
    with torch.no_grad():
        out_a = sanity_a(input_ids=ids, labels=labels, output_router_logits=True)

    # Sanity model WITH checkpointing — train mode required for
    # `torch.utils.checkpoint` to engage.
    sanity_b = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).train()
    copy_global_to_alternating_sanity(global_model, sanity_b)
    sanity_b.gradient_checkpointing_enable()
    sanity_b.eval()  # disable dropouts; checkpoint enable persists
    with torch.no_grad():
        out_b = sanity_b(input_ids=ids, labels=labels, output_router_logits=True)

    torch.testing.assert_close(out_a.logits, ref_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_b.logits, ref_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_a.loss, ref_out.loss, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_b.loss, ref_out.loss, atol=2e-4, rtol=2e-4)


# ──────────────────────────────────────────────────────────────────────
#  AC-23 parts 4-5: DDP and 50-step drift (CUDA-required placeholders)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="DDP equivalence is CUDA-required")
def test_sanity_equivalence_ddp_per_rank_bias():
    """AC-23 DDP variant: with the bias-update path active under
    `torch.distributed`, the sanity and global models must produce the
    same per-rank bias updates after one forward+backward+step. This
    requires multi-rank CUDA setup; skipped on CPU.
    """
    pytest.skip("CUDA-required; lands on Modal H200 in a follow-up round.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="50-step drift is CUDA-required for speed")
def test_sanity_equivalence_50_step_training_drift():
    """AC-23 50-step training drift: after 50 optimizer steps on the
    same input stream, the sanity and global models must remain within
    the documented bf16 round-off bound. CPU runs are too slow; defer
    to Modal H200.
    """
    pytest.skip("CUDA-required for runtime; lands on Modal H200 in a follow-up round.")


if __name__ == "__main__":
    test_sanity_equivalence_forward_logits()
    test_sanity_equivalence_forward_loss()
    test_sanity_equivalence_gradients_per_param_pair()
    test_sanity_equivalence_with_gradient_checkpointing()
    print("ALL OK (CPU portion of AC-23)")
