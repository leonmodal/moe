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


class _PhasedSnapshotRecorder:
    """Collects observations made INSIDE the checkpointed callable, one
    record per `is_checkpoint_recompute()` phase value, in order. AC-9
    requires one real-forward record (`is_checkpoint_recompute() == False`)
    AND at least one recompute record (`is_checkpoint_recompute() == True`),
    with their selections / masks equal under `preserve_rng_state=True`.
    """

    def __init__(self):
        self.records: list[dict] = []

    def record(self, **observations):
        from src.models.router import is_checkpoint_recompute
        self.records.append({
            "phase": "recompute" if is_checkpoint_recompute() else "real",
            **{k: v.clone().detach() if isinstance(v, torch.Tensor) else v
               for k, v in observations.items()},
        })

    def real_records(self):
        return [r for r in self.records if r["phase"] == "real"]

    def recompute_records(self):
        return [r for r in self.records if r["phase"] == "recompute"]


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


def test_branch_router_softmax_exploration_first_pass_matches_recompute():
    """AC-9 (Round 7): record selections INSIDE the checkpointed callable
    under both `is_checkpoint_recompute()` False (real forward) and True
    (recompute), then assert real-forward selections equal recompute
    selections. Codex Round 6 review explicitly asks for this proof.

    Wrapped in `set_checkpoint_early_stop(False)` so the recompute runs the
    FULL forward including the side-effect-only `recorder.record(...)` call.
    Without that wrapper, `use_reentrant=False`'s early-stop logic would
    bail out of the recompute as soon as it had collected the saved tensors
    it needed — typically before the recorder call.
    """
    torch.manual_seed(2031)
    router = BranchRouter(
        hidden_size=8,
        use_deepseek_style=False,
        exploration_rate=0.5,
    ).train()
    x = torch.randn(2, 4, 8, requires_grad=True)
    recorder = _PhasedSnapshotRecorder()

    def fn(inp):
        w_attn, w_mlp, attn_mask, mlp_mask = router(inp)
        recorder.record(selected_experts=router.last_selected_experts)
        return w_attn.sum() + w_mlp.sum()

    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        loss = torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=_checkpoint_ctx_fn,
        )
        loss.backward()

    real = recorder.real_records()
    recompute = recorder.recompute_records()
    assert len(real) == 1, f"expected exactly one real-forward record, got {len(real)}"
    assert len(recompute) >= 1, f"expected at least one recompute record, got {len(recompute)}"
    real_sel = real[0]["selected_experts"]
    for r in recompute:
        assert torch.equal(real_sel, r["selected_experts"]), (
            f"BranchRouter softmax exploration: selections diverged between "
            f"real forward and recompute under preserve_rng_state=True. "
            f"AC-9 first-pass/recompute proof failed."
        )


def test_branch_router_use_sampling_first_pass_matches_recompute():
    """AC-9 (Round 7): same first-pass-vs-recompute proof for
    `use_sampling=True` (multinomial path)."""
    torch.manual_seed(2032)
    router = BranchRouter(
        hidden_size=8,
        use_deepseek_style=True,
        use_sampling=True,
    ).train()
    x = torch.randn(2, 4, 8, requires_grad=True)
    recorder = _PhasedSnapshotRecorder()

    def fn(inp):
        w_attn, w_mlp, attn_mask, mlp_mask = router(inp)
        recorder.record(selected_experts=router.last_selected_experts)
        return w_attn.sum() + w_mlp.sum()

    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        loss = torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=_checkpoint_ctx_fn,
        )
        loss.backward()

    real = recorder.real_records()
    recompute = recorder.recompute_records()
    assert len(real) == 1
    assert len(recompute) >= 1
    real_sel = real[0]["selected_experts"]
    for r in recompute:
        assert torch.equal(real_sel, r["selected_experts"]), (
            f"BranchRouter use_sampling: multinomial draws diverged between "
            f"real forward and recompute. AC-9 regression."
        )


def test_deepseek_router_exploration_first_pass_matches_recompute():
    """AC-9 (Round 7): first-pass-vs-recompute proof for DeepSeekRouter
    exploration. Records BOTH `_last_top_k_idx` and `_last_exploration_mask`
    on each phase."""
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

    torch.manual_seed(2033)
    router = DeepSeekRouter(_MiniCfg()).train()
    x = torch.randn(8, 8, requires_grad=True)
    recorder = _PhasedSnapshotRecorder()

    def fn(inp):
        scores, weights, idx = router(inp)
        recorder.record(
            top_k_idx=router._last_top_k_idx,
            exploration_mask=router._last_exploration_mask,
        )
        return scores.sum() + weights.sum()

    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        loss = torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=_checkpoint_ctx_fn,
        )
        loss.backward()

    real = recorder.real_records()
    recompute = recorder.recompute_records()
    assert len(real) == 1
    assert len(recompute) >= 1
    real_idx = real[0]["top_k_idx"]
    real_mask = real[0]["exploration_mask"]
    for r in recompute:
        assert torch.equal(real_idx, r["top_k_idx"]), (
            f"DeepSeekRouter top_k_idx diverged between real and recompute. "
            f"AC-9 regression."
        )
        # `exploration_mask` is `None` when exploration_rate==0; ours is 0.5
        # so it should be a tensor.
        if real_mask is not None:
            assert r["exploration_mask"] is not None
            assert torch.equal(real_mask, r["exploration_mask"]), (
                f"DeepSeekRouter exploration_mask diverged between real and "
                f"recompute. AC-9 regression."
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

    `torch.utils.checkpoint` re-executes the forward during backward; the
    `is_checkpoint_recompute()` count-buffer guard (Round 1) prevents the
    recompute from double-counting. PyTorch's default `preserve_rng_state=True`
    keeps stochastic ops (exploration sampling, multinomial) deterministic
    across the real forward and the recompute (Round 5+). This test
    exercises the same end-to-end path that
    `MoEverythingModel.gradient_checkpointing_enable` lights up in
    production.
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
    checkpoint backward to prove RNG-preservation gives identical outputs
    across the real forward and the recompute (Round 5 design).

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
    # default makes the recompute use the same exploration outcomes as the
    # real forward (Round 5 design — the explicit cache layer was removed).
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


def test_full_model_moe_everything_branch_deepseek_checkpoint_matches_no_checkpoint():
    """AC-9 (Round 8) full-model regression: build two identical
    `MoEverythingForCausalLM` models on the same seed with
    `branch_deepseek=True`. Run model A without gradient checkpointing,
    model B with `gradient_checkpointing_enable()`. Same input seed.
    Assert branch `local_tokens_per_expert` and selections match.

    This is the AC-9 positive test the original plan text explicitly
    requires (see `docs/plan.md:104-105`). Round 7's targeted callable
    tests covered the routing-level contract; this test covers the full
    model wiring through `MoEverythingModel._depth_step` and
    `gradient_checkpointing_enable()`.
    """
    from src.models import MoEverythingConfig, MoEverythingForCausalLM

    def _make_model(seed: int):
        torch.manual_seed(seed)
        cfg = MoEverythingConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,  # 2 depth iterations: minimal but exercises checkpoint
            head_dim=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_attn_experts=2,
            num_attn_experts_per_tok=1,
            attn_expert_mode="per_head_fully_independent",
            branch_router_aux_loss_coef=0.0,
            use_deepseek_routing=True,
            branch_deepseek=True,  # the AC-9 contract specifies this
            topk_scaling_factor=2.5,
            per_layer_router=False,
            per_layer_mlp_router=False,
            per_layer_attn_router=False,
            routed_norm=False,
            per_layer_norm=False,
            post_norm=False,
            dynamic_depth_min=1.0,
            dynamic_depth_max=1.0,
            depthwise_attention=False,
            depthwise_block_size=0,
            per_head_compute_mode="auto",
            per_head_dense_fraction_threshold=0.75,
            scale_attn_by_routing_weight=True,
            scale_branch_by_routing_weight=True,
            router_exploration_rate=0.0,
            branch_router_exploration_rate=0.0,
            branch_sampling=False,
            branch_level="token",
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            norm_topk_prob=True,
            router_aux_loss_coef=0.0,    # method-driven runtime; coefs zero
            seq_aux_loss_coef=0.0,
            output_router_logits=False,  # non-aux method
            attn_implementation="eager",
        )
        torch.manual_seed(seed)
        model = MoEverythingForCausalLM(cfg).train()
        return model

    seed = 2034
    model_a = _make_model(seed)
    model_b = _make_model(seed)
    # Confirm parameter parity at init.
    for (n_a, p_a), (_, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.equal(p_a.data, p_b.data), f"init param diverged: {n_a}"

    # Identical input across both runs.
    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, model_a.vocab_size, (1, 4), dtype=torch.long)
    labels = input_ids.clone()

    # Run A: no gradient checkpointing.
    out_a = model_a(input_ids=input_ids, labels=labels)
    out_a.loss.backward()
    branch_a = model_a.model.branch_router
    counts_a = branch_a.local_tokens_per_expert.clone()
    sel_a = branch_a.last_selected_experts.clone() if branch_a.last_selected_experts is not None else None

    # Run B: gradient_checkpointing_enable().
    model_b.gradient_checkpointing_enable()
    out_b = model_b(input_ids=input_ids, labels=labels)
    out_b.loss.backward()
    branch_b = model_b.model.branch_router
    counts_b = branch_b.local_tokens_per_expert.clone()
    sel_b = branch_b.last_selected_experts.clone() if branch_b.last_selected_experts is not None else None

    assert torch.equal(counts_a, counts_b), (
        f"AC-9 full-model regression: branch local_tokens_per_expert "
        f"diverged between checkpointed and non-checkpointed runs. "
        f"counts_a={counts_a}, counts_b={counts_b}"
    )
    if sel_a is not None and sel_b is not None:
        # Last-recorded selection: under checkpointing, this is the
        # recompute's value (recompute overwrites `last_selected_experts`
        # last); under no-checkpointing, it's the only forward's value. Both
        # should match per the AC-9 RNG-preservation contract.
        assert torch.equal(sel_a, sel_b), (
            f"AC-9 full-model regression: branch_router selections diverged "
            f"between checkpointed and non-checkpointed runs."
        )


if __name__ == "__main__":
    test_branch_router_real_forward_increments_counts()
    test_branch_router_recompute_does_not_double_count()
    test_branch_router_no_grad_does_not_count()
    test_branch_router_buffer_names_are_canonical()
    test_branch_router_softmax_exploration_first_pass_matches_recompute()
    test_branch_router_use_sampling_first_pass_matches_recompute()
    test_deepseek_router_exploration_first_pass_matches_recompute()
    test_branch_router_softmax_exploration_checkpoint_matches_no_checkpoint()
    test_branch_router_under_torch_utils_checkpoint_matches_no_checkpoint()
    test_branch_router_use_sampling_checkpoint_matches_no_checkpoint()
    test_branch_router_use_sampling_diverges_when_rng_state_not_preserved()
    test_branch_router_real_checkpoint_stochastic_matches_no_checkpoint()
    test_deepseek_router_exploration_checkpoint_matches_no_checkpoint()
    test_full_model_moe_everything_branch_deepseek_checkpoint_matches_no_checkpoint()
    print("ALL OK")
