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


if __name__ == "__main__":
    test_branch_router_real_forward_increments_counts()
    test_branch_router_recompute_does_not_double_count()
    test_branch_router_no_grad_does_not_count()
    test_branch_router_buffer_names_are_canonical()
    print("ALL OK")
