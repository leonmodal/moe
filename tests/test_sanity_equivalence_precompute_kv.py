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

from unittest.mock import patch

from src.models import (
    DeepSeekGlobalMoEForCausalLM,
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
    """AC-9 + AC-23 jointly: in TRAIN mode with grad enabled, the sanity
    model with `gradient_checkpointing_enable()` actually invokes
    `torch.utils.checkpoint` (the model gates the path on
    `gradient_checkpointing AND training`), produces the same forward
    logits and loss as the non-checkpointed sanity model AND the
    global reference, AND produces the same per-parameter gradients
    against BOTH peers.

    Round 12 (Codex Round 11 Finding 3b/3c/3d):
    - Spies on `src.models.moe_everything.model.checkpoint` to assert
      the checkpoint path is actually invoked.
    - Compares checkpointed gradients against the global reference,
      not just sanity-no-ckpt vs sanity-ckpt.
    """
    torch.manual_seed(20260428)
    ids, labels = _dummy_batch()

    # Reference: global_moe forward, train mode + backward — we'll
    # compare sanity gradients against this reference per init_mapping
    # ParamPair.
    global_model = GlobalMoEForCausalLM(_tiny_global_equiv_config()).train()
    ref_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
    ref_out.loss.backward()

    # Sanity model WITHOUT checkpointing, train mode.
    sanity_a = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).train()
    pairs_a = copy_global_to_alternating_sanity(global_model, sanity_a)
    out_a = sanity_a(input_ids=ids, labels=labels, output_router_logits=True)
    out_a.loss.backward()

    # Sanity model WITH checkpointing, TRAIN mode + grad enabled (the
    # gating condition on `torch.utils.checkpoint` in
    # MoEverythingModel._depth_step is `gradient_checkpointing AND
    # training`).
    sanity_b = MoEverythingForCausalLM(_tiny_alternating_sanity_config()).train()
    pairs_b = copy_global_to_alternating_sanity(global_model, sanity_b)
    sanity_b.gradient_checkpointing_enable()

    # Spy on the moe_everything model's `checkpoint` import to verify
    # the recompute path actually fires — without this, a future change
    # that flips the gating condition off could pass the test silently.
    from src.models.moe_everything import model as me_model
    spy_calls = {"count": 0}
    real_checkpoint = me_model.checkpoint

    def _spy_checkpoint(*args, **kwargs):
        spy_calls["count"] += 1
        return real_checkpoint(*args, **kwargs)

    with patch.object(me_model, "checkpoint", side_effect=_spy_checkpoint) as _:
        out_b = sanity_b(input_ids=ids, labels=labels, output_router_logits=True)
        out_b.loss.backward()

    assert spy_calls["count"] > 0, (
        "torch.utils.checkpoint was NOT invoked during the gradient_"
        "checkpointing test — the gating condition must be enabled but "
        f"never fired (got {spy_calls['count']} calls). A future "
        "regression that bypasses checkpointing would slip through."
    )

    # Forward equivalence vs the global reference.
    torch.testing.assert_close(out_a.logits, ref_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_b.logits, ref_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_a.loss, ref_out.loss, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_b.loss, ref_out.loss, atol=2e-4, rtol=2e-4)

    # Checkpointed-vs-non-checkpointed forward equivalence (AC-9).
    torch.testing.assert_close(out_a.logits, out_b.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_a.loss, out_b.loss, atol=1e-5, rtol=1e-5)

    # Gradient equivalence: every tracked-grad parameter pair must
    # have matching gradients across all three runs:
    #   sanity-no-ckpt vs sanity-ckpt   (AC-9 recompute parity)
    #   sanity-ckpt    vs global ref    (AC-23 sanity equivalence
    #                                   under checkpointing)
    tracked_a = [p for p in pairs_a if p.track_grad_and_opt]
    tracked_b = [p for p in pairs_b if p.track_grad_and_opt]
    assert len(tracked_a) == len(tracked_b)
    for pa, pb in zip(tracked_a, tracked_b):
        ga = pa.right.grad
        gb = pb.right.grad
        gref = pa.left.grad  # `left` is the global-reference param.
        assert ga is not None and gb is not None and gref is not None, (
            f"missing grad on {pa.name!r}"
        )
        # AC-9: checkpointed recompute matches non-checkpointed forward.
        torch.testing.assert_close(
            ga, gb, atol=2e-4, rtol=2e-4,
            msg=f"AC-9 gradient mismatch (no-ckpt vs ckpt) on tracked pair "
                f"{pa.name!r}: |ga|={ga.norm().item():.3e}, "
                f"|gb|={gb.norm().item():.3e}",
        )
        # AC-23: checkpointed sanity matches global reference.
        torch.testing.assert_close(
            gref, gb, atol=2e-4, rtol=2e-4,
            msg=f"AC-23 gradient mismatch (ckpt sanity vs global ref) on "
                f"tracked pair {pa.name!r}: |gref|={gref.norm().item():.3e}, "
                f"|gb|={gb.norm().item():.3e}",
        )


# ──────────────────────────────────────────────────────────────────────
#  Larger AC-23 fixture with multi-expert routing (Codex Round 10
#  Finding 3 follow-up). Single-expert configs bypass the bias /
#  routing path entirely; the plan's headline contract is for the
#  full deepseek routing pair.
# ──────────────────────────────────────────────────────────────────────


def _multi_expert_global_config():
    """Multi-expert GlobalMoE config: 2 logical layers, 4 MLP experts,
    top-2 routing. Pairs with `_multi_expert_alternating_sanity_config`.
    """
    return GlobalMoEConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,                  # MULTI-EXPERT (was 1)
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        output_router_logits=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
    )


def _multi_expert_alternating_sanity_config():
    """Multi-expert MoEverything sanity config matching the global
    fixture's expert count.
    """
    return MoEverythingConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,                  # MULTI-EXPERT (was 1)
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        num_attn_experts=2,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_precompute_kv",
        norm_topk_prob=True,
        branch_router_aux_loss_coef=0.0,
        router_aux_loss_coef=0.0,
        per_layer_norm=True,
        sanity_check_mode="alternating_global_moe",
    )


def test_sanity_equivalence_multi_expert_forward_logits():
    """AC-23 multi-expert: with 4 MLP experts and top-2 routing, forward
    logits of the sanity model still match the global reference within
    bf16 round-off. Non-trivial routing (multiple experts in play)
    exercises the gate / expert-selection paths the 1-expert smoke
    fixture skips entirely."""
    torch.manual_seed(20260428)
    global_model = GlobalMoEForCausalLM(_multi_expert_global_config()).eval()
    sanity_model = MoEverythingForCausalLM(_multi_expert_alternating_sanity_config()).eval()
    copy_global_to_alternating_sanity(global_model, sanity_model)

    ids, labels = _dummy_batch(B=2, T=8)
    with torch.no_grad():
        global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
        sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)
    torch.testing.assert_close(sanity_out.logits, global_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)


def test_sanity_equivalence_multi_expert_gradients():
    """AC-23 multi-expert: gradient equivalence at the per-parameter-
    pair level for the multi-expert fixture."""
    torch.manual_seed(20260428)
    global_model = GlobalMoEForCausalLM(_multi_expert_global_config()).train()
    sanity_model = MoEverythingForCausalLM(_multi_expert_alternating_sanity_config()).train()
    pairs = copy_global_to_alternating_sanity(global_model, sanity_model)

    ids, labels = _dummy_batch(B=2, T=8)
    global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
    sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)

    global_out.loss.backward()
    sanity_out.loss.backward()

    tracked = [p for p in pairs if p.track_grad_and_opt]
    assert tracked
    for pair in tracked:
        gleft = pair.left.grad
        gright = pair.right.grad
        assert gleft is not None and gright is not None
        torch.testing.assert_close(
            gleft, gright, atol=2e-4, rtol=2e-4,
            msg=f"multi-expert gradient mismatch on {pair.name!r}",
        )


# ──────────────────────────────────────────────────────────────────────
#  DeepSeek-routing fixture (Codex Round 11 Finding 3a)
#
#  AC-23's headline contract is the DeepSeek routing pair. The fixtures
#  above use ExplorationTopKRouter (softmax, no bias path); they cover
#  the structural attn-vs-mlp alternation but skip the
#  sigmoid + biased-topk-with-unbiased-gather routing recipe DEC-2 /
#  AC-7 are designed around. The tests below build the equivalent pair
#  with `DeepSeekGlobalMoEForCausalLM` + `MoEverythingConfig(
#  use_deepseek_routing=True)` and assert forward/loss equivalence.
# ──────────────────────────────────────────────────────────────────────


def _deepseek_global_config():
    """DeepSeek-routing GlobalMoE config: 2 logical layers, 4 MLP
    experts, top-2 routing, sigmoid + biased-topk recipe."""
    return GlobalMoEConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        output_router_logits=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
    )


def _deepseek_alternating_sanity_config():
    """DeepSeek-routing MoEverything sanity config; pairs with
    `_deepseek_global_config`. `use_deepseek_routing=True` swaps the
    MLP gate to `DeepSeekRouter` (sigmoid + biased-topk + unbiased
    gather)."""
    return MoEverythingConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=128,
        max_position_embeddings=128,
        num_attn_experts=2,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_precompute_kv",
        use_deepseek_routing=True,        # DEEPSEEK ROUTING
        norm_topk_prob=True,
        branch_router_aux_loss_coef=0.0,
        router_aux_loss_coef=0.0,
        per_layer_norm=True,
        sanity_check_mode="alternating_global_moe",
    )


def test_sanity_equivalence_deepseek_routing_forward():
    """AC-23 headline DEEPSEEK contract: with sigmoid + biased-topk +
    unbiased-gather routing on both sides, the sanity model and the
    global reference produce equivalent forward logits and loss within
    bf16 round-off."""
    torch.manual_seed(20260428)
    global_model = DeepSeekGlobalMoEForCausalLM(_deepseek_global_config()).eval()
    sanity_model = MoEverythingForCausalLM(_deepseek_alternating_sanity_config()).eval()
    copy_global_to_alternating_sanity(global_model, sanity_model)

    ids, labels = _dummy_batch(B=2, T=8)
    with torch.no_grad():
        global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
        sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)
    torch.testing.assert_close(sanity_out.logits, global_out.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)


def test_sanity_equivalence_deepseek_routing_gradients():
    """AC-23 headline DEEPSEEK contract: gradient equivalence at the
    per-parameter-pair level for the DeepSeek routing fixture.

    Skips pairs whose `.left.grad` is None — those entries pair
    persistent buffers (e.g. `expert_bias`), not learnable
    parameters; buffers have no autograd grad even though they are
    weight-mapped 1:1 by `copy_global_to_alternating_sanity`.
    """
    torch.manual_seed(20260428)
    global_model = DeepSeekGlobalMoEForCausalLM(_deepseek_global_config()).train()
    sanity_model = MoEverythingForCausalLM(_deepseek_alternating_sanity_config()).train()
    pairs = copy_global_to_alternating_sanity(global_model, sanity_model)

    ids, labels = _dummy_batch(B=2, T=8)
    global_out = global_model(input_ids=ids, labels=labels, output_router_logits=True)
    sanity_out = sanity_model(input_ids=ids, labels=labels, output_router_logits=True)
    torch.testing.assert_close(sanity_out.loss, global_out.loss, atol=2e-4, rtol=2e-4)

    global_out.loss.backward()
    sanity_out.loss.backward()

    tracked = [p for p in pairs if p.track_grad_and_opt]
    assert tracked
    checked = 0
    for pair in tracked:
        gleft = pair.left.grad
        gright = pair.right.grad
        if gleft is None and gright is None:
            # Buffer pair (e.g. expert_bias) — no autograd grad to compare.
            continue
        assert gleft is not None and gright is not None, (
            f"asymmetric grad on DeepSeek pair {pair.name!r}: "
            f"left={None if gleft is None else 'tensor'}, "
            f"right={None if gright is None else 'tensor'}"
        )
        torch.testing.assert_close(
            gleft, gright, atol=2e-4, rtol=2e-4,
            msg=f"DeepSeek-routing gradient mismatch on {pair.name!r}",
        )
        checked += 1
    assert checked > 0, "No DeepSeek param pairs had gradients to check"


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
