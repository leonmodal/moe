"""AC-9 partial coverage: BranchRouter is checkpoint-recompute safe.

Today `BranchRouter.forward` increments `local_tokens_per_expert` on every
training-mode call. Without an `is_checkpoint_recompute()` guard, the gradient-
checkpointing backward replay would double-count the same tokens, biasing the
post-step bias update.

This test exercises the guard directly by:
  1. Running one forward pass with `is_grad_enabled()=True` and the recompute
     context flag OFF (the "real" forward) — expect counts to advance.
  2. Running another forward pass with the recompute context flag ON
     (simulating a gradient-checkpoint replay) — expect counts to stay frozen.

We test the BranchRouter in isolation (no full model wrap) so the assertion is
deterministic and doesn't depend on PyTorch's `torch.utils.checkpoint`
internals or model-wide RNG state.
"""
import sys
sys.path.insert(0, ".")

import torch

from src.models.routing.routers import BranchRouter
from src.models.router import checkpoint_recompute_context


def test_branch_router_real_forward_increments_counts():
    """First (real) forward pass must increment `local_tokens_per_expert`."""
    torch.manual_seed(0)
    router = BranchRouter(hidden_size=8, use_deepseek_style=True).train()
    x = torch.randn(2, 4, 8)  # (B, T, D) = (2, 4, 8)
    pre = router.local_tokens_per_expert.clone()
    with torch.enable_grad():
        router(x)
    post = router.local_tokens_per_expert
    # 2 sequences * 4 tokens = 8 routing decisions; counts must sum to 8.
    assert int(post.sum().item()) == 8, f"expected sum=8, got {post.sum().item()}"
    assert (post != pre).any(), "counts must advance on real forward"


def test_branch_router_recompute_does_not_double_count():
    """Forward pass under checkpoint-recompute context must NOT mutate counts.

    This locks the AC-9 contract that gradient-checkpointing recompute does not
    duplicate count-buffer increments.
    """
    torch.manual_seed(0)
    router = BranchRouter(hidden_size=8, use_deepseek_style=True).train()
    x = torch.randn(2, 4, 8)
    # Real forward (sets the baseline).
    with torch.enable_grad():
        router(x)
    snapshot = router.local_tokens_per_expert.clone()
    # Recompute pass: same inputs, but the recompute context flag is on.
    with torch.enable_grad():
        with checkpoint_recompute_context(True):
            router(x)
    # Counts must not have advanced from the snapshot.
    assert torch.equal(router.local_tokens_per_expert, snapshot), (
        f"counts changed during recompute: {snapshot} -> {router.local_tokens_per_expert}"
    )


def test_branch_router_no_grad_does_not_count():
    """Inference path (`torch.no_grad`) must not advance counts either."""
    torch.manual_seed(0)
    router = BranchRouter(hidden_size=8, use_deepseek_style=True).train()
    x = torch.randn(2, 4, 8)
    pre = router.local_tokens_per_expert.clone()
    with torch.no_grad():
        router(x)
    assert torch.equal(router.local_tokens_per_expert, pre), (
        "counts must not advance under torch.no_grad()"
    )


def test_branch_router_buffer_names_are_canonical():
    """DEC-18 invariant: BranchRouter uses canonical buffer names.

    Legacy `branch_bias` / `local_counts` are replaced by the canonical
    `expert_bias` / `local_tokens_per_expert` so the unified
    `_update_single_router_bias` walker handles every owner uniformly.
    """
    router = BranchRouter(hidden_size=8, use_deepseek_style=True)
    assert hasattr(router, "expert_bias"), "BranchRouter must expose canonical `expert_bias`"
    assert router.expert_bias.shape == (2,)
    assert router.expert_bias.dtype == torch.float32
    assert hasattr(router, "local_tokens_per_expert"), (
        "BranchRouter must expose canonical `local_tokens_per_expert`"
    )
    assert router.local_tokens_per_expert.shape == (2,)
    assert router.local_tokens_per_expert.dtype == torch.float32
    # Legacy buffers gone.
    assert not hasattr(router, "branch_bias"), "legacy `branch_bias` buffer must be removed"
    assert not hasattr(router, "local_counts"), "legacy `local_counts` buffer must be removed"


def test_branch_router_exploration_mask_reused_under_recompute():
    """AC-9 (Round 3 fill): with `exploration_rate > 0` and the non-deepseek
    softmax branch path, the exploration mask + random-override values must
    be CACHED on the first (real) forward and REUSED on the
    `is_checkpoint_recompute()` replay. Otherwise the recompute would
    re-sample fresh exploration and change branch choices, breaking gradient
    consistency.

    Round 2's full-integration test left `exploration_rate=0.0`, so the cache
    was never exercised. This test explicitly walks the cache path.
    """
    from src.models.router import checkpoint_recompute_context

    torch.manual_seed(2026)
    router = BranchRouter(
        hidden_size=8,
        use_deepseek_style=False,  # softmax branch path
        exploration_rate=0.5,       # half the tokens trip exploration
    ).train()
    x = torch.randn(2, 4, 8)

    # Real forward — pass with the recompute context flag OFF, sample mask + overrides.
    with torch.enable_grad():
        with checkpoint_recompute_context(False):
            real_out = router(x)
    cached_mask = router._last_exploration_mask
    cached_overrides = router._last_exploration_random
    assert cached_mask is not None, (
        "real forward must populate `_last_exploration_mask` for recompute reuse"
    )
    assert cached_overrides is not None, (
        "real forward must populate `_last_exploration_random` for recompute reuse"
    )
    assert cached_mask.shape == (2, 4)
    # Some tokens trip the explore mask (probability 0.5, 8 tokens, seed-fixed).
    assert cached_mask.any(), "exploration_rate=0.5 should produce at least one masked token"

    real_choice = router.last_selected_experts.clone()
    cached_mask_snapshot = cached_mask.clone()
    cached_overrides_snapshot = cached_overrides.clone()

    # Recompute pass — flag ON. The cache must drive the branch decision; if
    # the recompute were to fresh-sample `torch.rand`, the choice could
    # diverge.
    with torch.enable_grad():
        with checkpoint_recompute_context(True):
            recompute_out = router(x)

    # The mask + overrides must be EXACTLY what they were on the real forward
    # (the cache wasn't refreshed).
    assert torch.equal(router._last_exploration_mask, cached_mask_snapshot), (
        "recompute must NOT refresh `_last_exploration_mask`"
    )
    assert torch.equal(router._last_exploration_random, cached_overrides_snapshot), (
        "recompute must NOT refresh `_last_exploration_random`"
    )
    # Branch choices must match between real and recompute.
    assert torch.equal(router.last_selected_experts, real_choice), (
        "recompute branch decision diverged from real forward — exploration "
        "cache is not being reused (AC-9 regression)."
    )


def test_branch_router_under_torch_utils_checkpoint_matches_no_checkpoint():
    """Full AC-9 integration: a forward+backward through `torch.utils.checkpoint`
    must produce the same `local_tokens_per_expert` as a forward+backward with
    checkpointing disabled, on the same seed.

    `torch.utils.checkpoint` re-executes the forward during backward; without
    the `is_checkpoint_recompute()` guard introduced by Round 1 + the
    exploration-mask cache introduced by Round 2, the recompute would
    double-count or change the branch decision and the comparison would
    diverge. This test exercises the same end-to-end path that
    `MoEverythingModel.gradient_checkpointing_enable` lights up in production.
    """
    from src.models.router import checkpoint_recompute_context

    def _checkpoint_context_fn():
        # Mirrors `MoEverythingModel._checkpoint_context_fn`: real forward
        # runs with the recompute flag OFF, recompute runs with it ON.
        return checkpoint_recompute_context(False), checkpoint_recompute_context(True)

    def _make_router_and_input(seed: int):
        torch.manual_seed(seed)
        router = BranchRouter(hidden_size=8, use_deepseek_style=True).train()
        x = torch.randn(2, 4, 8, requires_grad=True)
        return router, x

    def _ckpt_call(router, x):
        # `torch.utils.checkpoint` requires a callable taking + returning tensors.
        # We sum the four returned tensors so backward has something to chain to.
        def fn(inp):
            w_attn, w_mlp, attn_mask, mlp_mask = router(inp)
            return w_attn.sum() + w_mlp.sum() + attn_mask.float().sum() + mlp_mask.float().sum()

        return torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=_checkpoint_context_fn,
        )

    # Run A: checkpointing disabled.
    router_a, x_a = _make_router_and_input(seed=42)
    out_a = router_a(x_a)
    (out_a[0].sum() + out_a[1].sum()).backward()
    counts_a = router_a.local_tokens_per_expert.clone()

    # Run B: checkpointing enabled (single block).
    router_b, x_b = _make_router_and_input(seed=42)
    loss = _ckpt_call(router_b, x_b)
    loss.backward()
    counts_b = router_b.local_tokens_per_expert

    # The count buffer must be identical between the two runs — recompute did
    # NOT double-count (AC-9).
    assert torch.equal(counts_a, counts_b), (
        f"counts diverged between checkpointed and non-checkpointed runs: "
        f"{counts_a} vs {counts_b}. AC-9 regression."
    )


def test_branch_router_use_sampling_choice_reused_under_recompute():
    """AC-9 (Round 4 fill): when `use_sampling=True`, the multinomial draw on
    the real forward must be CACHED and REUSED on recompute. Without the
    cache, recompute would re-sample (potentially with a different RNG state
    if `preserve_rng_state` is ever disabled), breaking gradient consistency.
    """
    from src.models.router import checkpoint_recompute_context

    torch.manual_seed(2026)
    router = BranchRouter(
        hidden_size=8,
        use_deepseek_style=True,
        use_sampling=True,
    ).train()
    x = torch.randn(2, 4, 8)

    # Real forward.
    with torch.enable_grad():
        with checkpoint_recompute_context(False):
            router(x)
    cached_choice = router._last_sampling_choice
    assert cached_choice is not None, (
        "real forward must populate `_last_sampling_choice` for recompute reuse"
    )
    snapshot = cached_choice.clone()
    real_branch_decision = router.last_selected_experts.clone()

    # Recompute pass — DELIBERATELY change torch's RNG state to prove the
    # cache is what determines the recompute branch decision (not RNG).
    torch.manual_seed(9999)
    with torch.enable_grad():
        with checkpoint_recompute_context(True):
            router(x)

    # Cache must be unchanged.
    assert torch.equal(router._last_sampling_choice, snapshot), (
        "recompute must NOT refresh `_last_sampling_choice`"
    )
    # Branch decision must match the real forward (cache-driven).
    assert torch.equal(router.last_selected_experts, real_branch_decision), (
        "recompute branch decision diverged from real forward — multinomial "
        "cache is not being reused (AC-9 regression)."
    )


def test_deepseek_router_exploration_mask_reused_under_recompute():
    """AC-9 (Round 4 fill): `DeepSeekRouter` (not just BranchRouter) caches its
    exploration mask on real forward and reuses it on recompute. Round 3 only
    covered BranchRouter; Codex's Round 3 review specifically asks for this.
    """
    from src.models.router import DeepSeekRouter, checkpoint_recompute_context

    class _MiniCfg:
        hidden_size = 8
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        topk_scaling_factor = None
        num_groups = None
        group_topk = None
        router_exploration_rate = 0.5  # half the tokens trip exploration
        router_z_loss_coef = 0.0

    torch.manual_seed(2026)
    cfg = _MiniCfg()
    router = DeepSeekRouter(cfg).train()
    x = torch.randn(8, 8)  # (T, hidden)

    # Real forward.
    with torch.enable_grad():
        with checkpoint_recompute_context(False):
            router(x)
    cached_mask = router._last_exploration_mask
    assert cached_mask is not None
    assert cached_mask.any(), "exploration_rate=0.5 should produce some masked tokens"
    snapshot = cached_mask.clone()
    real_top_k_idx = router._last_top_k_idx.clone()

    # Recompute pass — change RNG state to prove cache-driven.
    torch.manual_seed(9999)
    with torch.enable_grad():
        with checkpoint_recompute_context(True):
            router(x)

    assert torch.equal(router._last_exploration_mask, snapshot), (
        "recompute must NOT refresh `DeepSeekRouter._last_exploration_mask`"
    )
    assert torch.equal(router._last_top_k_idx, real_top_k_idx), (
        "recompute top-k decision diverged from real forward — DeepSeekRouter "
        "exploration cache is not being reused (AC-9 regression)."
    )


if __name__ == "__main__":
    test_branch_router_real_forward_increments_counts()
    test_branch_router_recompute_does_not_double_count()
    test_branch_router_no_grad_does_not_count()
    test_branch_router_buffer_names_are_canonical()
    test_branch_router_exploration_mask_reused_under_recompute()
    test_branch_router_under_torch_utils_checkpoint_matches_no_checkpoint()
    test_branch_router_use_sampling_choice_reused_under_recompute()
    test_deepseek_router_exploration_mask_reused_under_recompute()
    print("ALL OK")
