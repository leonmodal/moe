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


def _checkpoint_ctx_fn():
    """Mirrors `MoEverythingModel._checkpoint_context_fn`: real forward runs
    with the recompute flag OFF, recompute runs with it ON. Used by every
    AC-9 stochastic test below to wire `torch.utils.checkpoint` correctly."""
    from src.models.router import checkpoint_recompute_context
    return checkpoint_recompute_context(False), checkpoint_recompute_context(True)


def _run_branch_router(
    *,
    seed: int,
    use_deepseek_style: bool,
    exploration_rate: float = 0.0,
    use_sampling: bool = False,
    use_checkpoint: bool,
    preserve_rng_state: bool = True,
):
    """Build a fresh `BranchRouter`, run forward+backward, return
    `(selected_experts, local_tokens_per_expert, input.grad, loss.item())`.

    The same callable runs both checkpointed and non-checkpointed (when
    `use_checkpoint=False`), so AC-9 equality assertions can compare the two.
    """
    torch.manual_seed(seed)
    router = BranchRouter(
        hidden_size=8,
        use_deepseek_style=use_deepseek_style,
        exploration_rate=exploration_rate,
        use_sampling=use_sampling,
    ).train()
    torch.manual_seed(seed)  # re-seed so the input tensor is identical regardless of router init RNG cost
    x = torch.randn(2, 4, 8, requires_grad=True)

    def fn(inp):
        w_attn, w_mlp, attn_mask, mlp_mask = router(inp)
        return w_attn.sum() + w_mlp.sum()

    if use_checkpoint:
        loss = torch.utils.checkpoint.checkpoint(
            fn, x,
            use_reentrant=False,
            context_fn=_checkpoint_ctx_fn,
            preserve_rng_state=preserve_rng_state,
        )
    else:
        loss = fn(x)
    loss.backward()
    return (
        router.last_selected_experts.clone() if router.last_selected_experts is not None else None,
        router.local_tokens_per_expert.clone()
            if hasattr(router, "local_tokens_per_expert") and router.local_tokens_per_expert is not None
            else None,
        x.grad.clone(),
        loss.item(),
    )


def test_branch_router_softmax_exploration_checkpoint_matches_no_checkpoint():
    """AC-9 (Round 6): BranchRouter softmax exploration produces identical
    selected experts AND identical input gradients under
    `torch.utils.checkpoint(use_reentrant=False, preserve_rng_state=True)` as
    without checkpointing. Locks the gradient-consistency contract Codex
    Round 5 review specifically asked for."""
    sel_no_ckpt, _, grad_no_ckpt, _ = _run_branch_router(
        seed=2027, use_deepseek_style=False, exploration_rate=0.5,
        use_checkpoint=False,
    )
    sel_ckpt, _, grad_ckpt, _ = _run_branch_router(
        seed=2027, use_deepseek_style=False, exploration_rate=0.5,
        use_checkpoint=True,
    )
    assert torch.equal(sel_no_ckpt, sel_ckpt), (
        f"Selected experts diverged between checkpointed and non-checkpointed runs. "
        f"AC-9 regression."
    )
    assert torch.allclose(grad_no_ckpt, grad_ckpt, atol=1e-6), (
        f"Input gradients diverged: max diff = "
        f"{(grad_no_ckpt - grad_ckpt).abs().max().item()}. AC-9 regression."
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


def test_branch_router_use_sampling_checkpoint_matches_no_checkpoint():
    """AC-9 (Round 6): `use_sampling=True` (`torch.multinomial` path) produces
    identical multinomial draws and identical gradients under checkpoint as
    without, courtesy of `preserve_rng_state=True`. Compares selections,
    count buffers, AND gradients to lock the full equivalence."""
    sel_no_ckpt, counts_no_ckpt, grad_no_ckpt, _ = _run_branch_router(
        seed=2028, use_deepseek_style=True, use_sampling=True,
        use_checkpoint=False,
    )
    sel_ckpt, counts_ckpt, grad_ckpt, _ = _run_branch_router(
        seed=2028, use_deepseek_style=True, use_sampling=True,
        use_checkpoint=True,
    )
    assert torch.equal(sel_no_ckpt, sel_ckpt), (
        f"use_sampling: selections diverged between checkpointed and non-"
        f"checkpointed. AC-9 regression."
    )
    assert torch.equal(counts_no_ckpt, counts_ckpt), (
        f"use_sampling: count buffers diverged: {counts_no_ckpt} vs {counts_ckpt}. "
        f"AC-9 regression (count guard not preserving across checkpoint)."
    )
    assert torch.allclose(grad_no_ckpt, grad_ckpt, atol=1e-6), (
        f"use_sampling: gradients diverged: max diff = "
        f"{(grad_no_ckpt - grad_ckpt).abs().max().item()}. AC-9 regression."
    )


def test_branch_router_use_sampling_diverges_when_rng_state_not_preserved():
    """AC-9 (Round 6) negative test: with `preserve_rng_state=False`, the
    `use_sampling` path SHOULD produce different multinomial draws on real
    forward and recompute (because RNG state isn't restored). This locks the
    contract that the AC-9 stochastic-path determinism requires PyTorch's
    default `preserve_rng_state=True`. Codex's Round 5 probe demonstrated
    this divergence (`maxdiff = 0.123`) and asked for an explicit test that
    captures it.

    If a future change made `preserve_rng_state=False` deterministic somehow
    (e.g. by re-introducing a cache layer that survives the saved-tensor
    count check), this test would start failing — at which point the test
    should be re-evaluated (the underlying configuration is unsupported by
    AC-9 unless that hypothetical change explicitly demonstrates safety).
    """
    sel_no_ckpt, _, grad_no_ckpt, _ = _run_branch_router(
        seed=2029, use_deepseek_style=True, use_sampling=True,
        use_checkpoint=False,
    )
    sel_ckpt_no_rng, _, grad_ckpt_no_rng, _ = _run_branch_router(
        seed=2029, use_deepseek_style=True, use_sampling=True,
        use_checkpoint=True, preserve_rng_state=False,
    )
    selections_diverge = not torch.equal(sel_no_ckpt, sel_ckpt_no_rng)
    grads_diverge = not torch.allclose(grad_no_ckpt, grad_ckpt_no_rng, atol=1e-6)
    assert selections_diverge or grads_diverge, (
        "Expected stochastic divergence with preserve_rng_state=False (the "
        "recompute should re-sample multinomial with a different RNG state). "
        "If this assertion fires it means PyTorch's checkpoint behavior "
        "changed and the AC-9 documented assumption no longer holds — "
        "re-evaluate the test before passing."
    )


def test_branch_router_real_checkpoint_stochastic_matches_no_checkpoint():
    """Real `torch.utils.checkpoint(...).backward()` integration test for
    stochastic paths (Codex Round 4 Blocker #3).

    Round 4's checkpoint-context-manager test only exercised the non-stochastic
    count guard. Codex correctly noted that stochastic paths need real
    checkpoint backward to prove the cache survives the autograd replay.

    Setup: BranchRouter with `exploration_rate=0.5` (softmax path).
    Run A: forward+backward without checkpointing.
    Run B: forward through `torch.utils.checkpoint(...)` + backward.
    Both runs use the same seed.
    Assert: both runs produce identical `last_selected_experts` AND identical
    `local_tokens_per_expert` AND identical input.grad on the same seed.
    """
    from src.models.router import checkpoint_recompute_context

    def _checkpoint_context_fn():
        return checkpoint_recompute_context(False), checkpoint_recompute_context(True)

    def _make_router_and_input(seed: int):
        torch.manual_seed(seed)
        # Use the deepseek_style branch (which has the count buffer); the
        # non-deepseek path also has exploration sampling.
        router = BranchRouter(
            hidden_size=8,
            use_deepseek_style=False,
            exploration_rate=0.5,
        ).train()
        x = torch.randn(2, 4, 8, requires_grad=True)
        return router, x

    def _do_run(router, x, *, use_checkpoint: bool):
        def fn(inp):
            w_attn, w_mlp, attn_mask, mlp_mask = router(inp)
            return w_attn.sum() + w_mlp.sum()

        if use_checkpoint:
            loss = torch.utils.checkpoint.checkpoint(
                fn, x, use_reentrant=False, context_fn=_checkpoint_context_fn,
            )
        else:
            loss = fn(x)
        loss.backward()
        return loss

    # Run A: no checkpointing.
    router_a, x_a = _make_router_and_input(seed=2027)
    _do_run(router_a, x_a, use_checkpoint=False)
    sel_a = router_a.last_selected_experts.clone()
    grad_a = x_a.grad.clone()

    # Run B: with `torch.utils.checkpoint`. PyTorch's `preserve_rng_state=True`
    # default + the cache make the recompute use the same exploration outcomes.
    router_b, x_b = _make_router_and_input(seed=2027)
    _do_run(router_b, x_b, use_checkpoint=True)
    sel_b = router_b.last_selected_experts
    grad_b = x_b.grad

    # Branch decisions match.
    assert torch.equal(sel_a, sel_b), (
        f"Real checkpoint backward changed branch decisions on the same seed. "
        f"AC-9 regression."
    )
    # Gradients match (the headline AC-9 contract: gradient consistency).
    assert torch.allclose(grad_a, grad_b, atol=1e-5), (
        f"Real checkpoint backward changed input gradients on the same seed. "
        f"max diff = {(grad_a - grad_b).abs().max().item()}. AC-9 regression."
    )


def _run_deepseek_router(*, seed: int, use_checkpoint: bool):
    """Same shape as `_run_branch_router` but for `DeepSeekRouter` with
    `router_exploration_rate=0.5`."""
    from src.models.router import DeepSeekRouter

    class _MiniCfg:
        hidden_size = 8
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        topk_scaling_factor = None
        num_groups = None
        group_topk = None
        router_exploration_rate = 0.5
        router_z_loss_coef = 0.0

    torch.manual_seed(seed)
    router = DeepSeekRouter(_MiniCfg()).train()
    torch.manual_seed(seed)
    x = torch.randn(8, 8, requires_grad=True)

    def fn(inp):
        scores, weights, idx = router(inp)
        return scores.sum() + weights.sum()

    if use_checkpoint:
        loss = torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=_checkpoint_ctx_fn,
        )
    else:
        loss = fn(x)
    loss.backward()
    return (
        router._last_top_k_idx.clone() if router._last_top_k_idx is not None else None,
        router.local_tokens_per_expert.clone(),
        x.grad.clone(),
    )


def test_deepseek_router_exploration_checkpoint_matches_no_checkpoint():
    """AC-9 (Round 6): `DeepSeekRouter.forward` with `router_exploration_rate=0.5`
    produces identical `top_k_idx`, `local_tokens_per_expert`, and input
    gradients under real checkpoint vs no-checkpoint, on the same seed.

    Replaces the Round 5 finite-gradient test (insufficient per Codex
    Round 5 review)."""
    idx_no_ckpt, counts_no_ckpt, grad_no_ckpt = _run_deepseek_router(
        seed=2030, use_checkpoint=False,
    )
    idx_ckpt, counts_ckpt, grad_ckpt = _run_deepseek_router(
        seed=2030, use_checkpoint=True,
    )
    assert torch.equal(idx_no_ckpt, idx_ckpt), (
        f"DeepSeekRouter top_k_idx diverged between checkpointed and non-"
        f"checkpointed runs. AC-9 regression."
    )
    assert torch.equal(counts_no_ckpt, counts_ckpt), (
        f"DeepSeekRouter local_tokens_per_expert diverged: {counts_no_ckpt} "
        f"vs {counts_ckpt}. AC-9 count guard regression."
    )
    assert torch.allclose(grad_no_ckpt, grad_ckpt, atol=1e-6), (
        f"DeepSeekRouter input gradients diverged: max diff = "
        f"{(grad_no_ckpt - grad_ckpt).abs().max().item()}. AC-9 regression."
    )


if __name__ == "__main__":
    test_branch_router_real_forward_increments_counts()
    test_branch_router_recompute_does_not_double_count()
    test_branch_router_no_grad_does_not_count()
    test_branch_router_buffer_names_are_canonical()
    test_branch_router_softmax_exploration_checkpoint_matches_no_checkpoint()
    test_branch_router_under_torch_utils_checkpoint_matches_no_checkpoint()
    test_branch_router_use_sampling_checkpoint_matches_no_checkpoint()
    test_branch_router_use_sampling_diverges_when_rng_state_not_preserved()
    test_branch_router_real_checkpoint_stochastic_matches_no_checkpoint()
    test_deepseek_router_exploration_checkpoint_matches_no_checkpoint()
    print("ALL OK")
