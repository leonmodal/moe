"""Tests for the `exploration_only` BranchRouter mode and the
companion exploration-decay schedules.

`exploration_only`: when set, the BranchRouter ignores its scoring
network entirely and routes every token to ATTN/MLP via a uniform
Bernoulli draw. The model still produces a real loss (downstream
attention + MLP get the routed tokens), so this is useful for
diagnostic runs that need a fully randomized branch distribution
while keeping the rest of the model trainable.

`exploration_decay_schedule`: a generic helper for decaying an
exploration rate from `initial_rate` to `final_rate` over
`decay_steps` following one of three shapes (`constant`, `linear`,
`cosine`).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.routing.routers import BranchRouter


def _load_routing_module():
    """Load `src.training.routing` directly without triggering the
    `src.training.__init__` package chain."""
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
        import types as _types
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = training_pkg
    spec = importlib.util.spec_from_file_location(
        "src.training.routing",
        repo / "src" / "training" / "routing.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.training.routing"] = module
    spec.loader.exec_module(module)
    return module


# ──────────────────────────────────────────────────────────────────────
#  exploration_only mode
# ──────────────────────────────────────────────────────────────────────


def test_exploration_only_routes_uniformly_at_random():
    """With `exploration_only=True`, the BranchRouter's choice
    distribution should be uniform over ATTN/MLP (each with ~50%
    probability), regardless of the gate's learned weights or the
    input.

    The test seeds a batch of T=8192 tokens. The expected count
    per branch is `T/2 = 4096`. The Bernoulli noise floor at this
    sample size is `sqrt(T*0.5*0.5) ≈ 45`, so an absolute deviation
    of 200 corresponds to ~4σ — comfortable margin without false
    flakes from RNG variance.
    """
    torch.manual_seed(20260428)
    H = 32
    T = 8192

    # Build a router with arbitrary gate weights — the test verifies
    # those weights are IGNORED in `exploration_only` mode.
    router = BranchRouter(
        hidden_size=H,
        exploration_only=True,
    ).train()
    # Bias the gate dramatically: this would normally make every
    # token pick MLP. exploration_only must override that.
    with torch.no_grad():
        torch.nn.init.constant_(router.gate.weight[1], 100.0)
        torch.nn.init.constant_(router.gate.weight[0], -100.0)

    x = torch.randn(T, H)
    _, _, _, _ = router(x)

    # Count routing decisions.
    selections = router.last_selected_experts.reshape(-1)
    n_attn = (selections == 0).sum().item()
    n_mlp = (selections == 1).sum().item()
    assert n_attn + n_mlp == T

    # Both branches must be exercised, and the split must be near 50/50.
    abs_dev = abs(n_attn - n_mlp)
    assert abs_dev < 200, (
        f"exploration_only routing is not uniform: ATTN={n_attn}, MLP={n_mlp}, "
        f"|ATTN - MLP|={abs_dev}, expected ≤ 200 (≈4σ for T={T})."
    )


def test_exploration_only_off_is_unchanged_argmax_path():
    """`exploration_only=False` (default) produces the existing
    argmax-of-softmax behavior. Locking this contract so a future
    exploration_only refactor doesn't change the default path."""
    torch.manual_seed(20260428)
    H = 16
    T = 64

    router = BranchRouter(hidden_size=H, exploration_only=False).train()
    with torch.no_grad():
        # Bias gate to always pick MLP (index 1).
        torch.nn.init.constant_(router.gate.weight[1], 100.0)
        torch.nn.init.constant_(router.gate.weight[0], -100.0)

    # Use strictly-positive input so `gate(x) @ weight[i].T` has a
    # deterministic sign (otherwise random-Gaussian inputs flip the
    # argmax row by row).
    x = torch.ones(T, H)
    _, _, _, _ = router(x)
    selections = router.last_selected_experts.reshape(-1)
    # Every token must pick MLP.
    assert (selections == 1).all(), (
        "default-mode BranchRouter with biased gate should pick MLP for "
        "every token (argmax); got distribution: "
        f"ATTN={(selections == 0).sum().item()}, MLP={(selections == 1).sum().item()}"
    )


def test_exploration_only_eval_mode_falls_back_to_argmax():
    """When the router is in `eval()` mode, `exploration_only` is
    not active — the routing is deterministic. This matches the
    semantic of the existing `exploration_rate` knob (training-time
    only) and ensures evaluation runs are reproducible."""
    torch.manual_seed(20260428)
    H = 16
    T = 64

    router = BranchRouter(hidden_size=H, exploration_only=True).eval()
    with torch.no_grad():
        torch.nn.init.constant_(router.gate.weight[1], 100.0)
        torch.nn.init.constant_(router.gate.weight[0], -100.0)

    # Strictly-positive input so the biased argmax is deterministic.
    x = torch.ones(T, H)
    _, _, _, _ = router(x)
    selections = router.last_selected_experts.reshape(-1)
    assert (selections == 1).all(), (
        f"eval-mode exploration_only should fall back to argmax; got distribution: "
        f"ATTN={(selections == 0).sum().item()}, MLP={(selections == 1).sum().item()}"
    )


def test_exploration_only_returns_loss_traceable_outputs():
    """`exploration_only` must still return weight tensors that
    flow gradient through the gate's projection. The selection
    itself is randomized (no gradient), but the routing weights
    are produced via `softmax`/`sigmoid` of the gate's output and
    must keep the autograd graph intact."""
    torch.manual_seed(20260428)
    H = 16
    T = 32

    router = BranchRouter(hidden_size=H, exploration_only=True).train()
    x = torch.randn(T, H, requires_grad=True)
    w_attn, w_mlp, _, _ = router(x)

    assert w_attn.requires_grad, "w_attn must be gradient-bearing"
    assert w_mlp.requires_grad, "w_mlp must be gradient-bearing"
    # Backprop a non-trivial loss; gate.weight must accumulate gradient.
    (w_attn.sum() + w_mlp.sum()).backward()
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.norm().item() > 0


# ──────────────────────────────────────────────────────────────────────
#  exploration_decay_schedule helper
# ──────────────────────────────────────────────────────────────────────


def test_exploration_decay_schedule_constant():
    routing = _load_routing_module()
    for step in (0, 10, 1000, 10_000):
        assert routing.exploration_decay_schedule(
            step, schedule="constant", initial_rate=0.5, decay_steps=100,
        ) == 0.5


def test_exploration_decay_schedule_linear():
    routing = _load_routing_module()
    # Linear schedule from 1.0 → 0.0 over 100 steps.
    assert routing.exploration_decay_schedule(
        0, schedule="linear", initial_rate=1.0, decay_steps=100,
    ) == 1.0
    assert routing.exploration_decay_schedule(
        50, schedule="linear", initial_rate=1.0, decay_steps=100,
    ) == pytest.approx(0.5)
    assert routing.exploration_decay_schedule(
        100, schedule="linear", initial_rate=1.0, decay_steps=100,
    ) == 0.0
    assert routing.exploration_decay_schedule(
        200, schedule="linear", initial_rate=1.0, decay_steps=100,
    ) == 0.0  # past end → final_rate


def test_exploration_decay_schedule_cosine():
    routing = _load_routing_module()
    # Cosine schedule: starts at initial_rate, ends at final_rate, smooth.
    assert routing.exploration_decay_schedule(
        0, schedule="cosine", initial_rate=1.0, decay_steps=100,
    ) == pytest.approx(1.0)
    # At midpoint the cosine schedule is at the average.
    assert routing.exploration_decay_schedule(
        50, schedule="cosine", initial_rate=1.0, decay_steps=100,
    ) == pytest.approx(0.5, abs=1e-6)
    assert routing.exploration_decay_schedule(
        100, schedule="cosine", initial_rate=1.0, decay_steps=100,
    ) == pytest.approx(0.0, abs=1e-6)
    assert routing.exploration_decay_schedule(
        300, schedule="cosine", initial_rate=1.0, decay_steps=100,
    ) == 0.0


def test_exploration_decay_schedule_with_nonzero_final():
    """Allow decaying to a non-zero floor (e.g. keep 5% exploration
    forever)."""
    routing = _load_routing_module()
    rate = routing.exploration_decay_schedule(
        100, schedule="linear", initial_rate=0.5, decay_steps=100, final_rate=0.05,
    )
    assert rate == pytest.approx(0.05)
    rate_mid = routing.exploration_decay_schedule(
        50, schedule="linear", initial_rate=0.5, decay_steps=100, final_rate=0.05,
    )
    assert rate_mid == pytest.approx(0.275, abs=1e-6)  # midway 0.5 → 0.05


def test_exploration_decay_schedule_zero_decay_steps_returns_final():
    """Edge case: `decay_steps=0` collapses to the final rate
    immediately (any step → final_rate). This is the natural
    semantics for "no decay window — start at the floor"."""
    routing = _load_routing_module()
    assert routing.exploration_decay_schedule(
        0, schedule="linear", initial_rate=1.0, decay_steps=0, final_rate=0.05,
    ) == 0.05
    assert routing.exploration_decay_schedule(
        100, schedule="cosine", initial_rate=1.0, decay_steps=0, final_rate=0.05,
    ) == 0.05


def test_exploration_decay_schedule_rejects_unknown_shape():
    routing = _load_routing_module()
    with pytest.raises(ValueError, match="exploration schedule"):
        routing.exploration_decay_schedule(
            0, schedule="quadratic", initial_rate=1.0, decay_steps=100,
        )


def test_exploration_decay_schedule_negative_step_clamps_to_zero():
    """Defensive: a negative step (caller bug) must clamp to step 0
    rather than producing a >initial_rate value via extrapolation."""
    routing = _load_routing_module()
    rate = routing.exploration_decay_schedule(
        -10, schedule="linear", initial_rate=1.0, decay_steps=100,
    )
    assert rate == 1.0
    rate_cos = routing.exploration_decay_schedule(
        -10, schedule="cosine", initial_rate=1.0, decay_steps=100,
    )
    assert rate_cos == pytest.approx(1.0)


if __name__ == "__main__":
    test_exploration_only_routes_uniformly_at_random()
    test_exploration_only_off_is_unchanged_argmax_path()
    test_exploration_only_eval_mode_falls_back_to_argmax()
    test_exploration_only_returns_loss_traceable_outputs()
    test_exploration_decay_schedule_constant()
    test_exploration_decay_schedule_linear()
    test_exploration_decay_schedule_cosine()
    test_exploration_decay_schedule_with_nonzero_final()
    test_exploration_decay_schedule_zero_decay_steps_returns_final()
    test_exploration_decay_schedule_rejects_unknown_shape()
    test_exploration_decay_schedule_negative_step_clamps_to_zero()
    print("ALL OK")
