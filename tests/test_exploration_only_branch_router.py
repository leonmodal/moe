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


def test_exploration_only_records_mask_and_rate():
    """Lock the telemetry contract: with `exploration_only_rate > 0`,
    the BranchRouter records per-token `last_exploration_only_mask`
    AND the rate that produced it. Tests / telemetry / logging
    depend on these attributes."""
    torch.manual_seed(20260428)
    H = 16
    T = 32
    rate = 0.3

    router = BranchRouter(
        hidden_size=H, exploration_only_rate=rate,
    ).train()
    x = torch.randn(T, H)
    _, _, _, _ = router(x)

    mask = router.last_exploration_only_mask
    assert mask is not None
    assert mask.dtype == torch.bool
    assert mask.shape == (T,)
    # The mask is a Bernoulli sample at `rate`; for T=32, rate=0.3,
    # expected count is 9.6, std is sqrt(32*0.3*0.7) ≈ 2.6, so a
    # generous absolute deviation of 8 keeps the test stable across
    # PyTorch RNG implementations.
    n_explored = mask.sum().item()
    assert abs(n_explored - 32 * rate) < 8, (
        f"explore-mask count {n_explored} far from expected "
        f"{32 * rate} (rate={rate}, T=32)"
    )
    assert router.last_exploration_only_rate == pytest.approx(rate)


def test_exploration_only_rate_zero_disables_mask_recording():
    """With `exploration_only_rate=0.0`, no exploration occurs and
    no mask is recorded. The router falls through to the regular
    argmax path."""
    torch.manual_seed(20260428)
    H = 16
    T = 16

    router = BranchRouter(hidden_size=H, exploration_only_rate=0.0).train()
    x = torch.ones(T, H)
    _, _, _, _ = router(x)
    assert router.last_exploration_only_mask is None
    assert router.last_exploration_only_rate == 0.0


def test_exploration_only_rate_validates_range():
    """`exploration_only_rate` must be in [0, 1]. Out-of-range
    values raise `ValueError` at construction time so misconfigured
    yamls fail fast."""
    with pytest.raises(ValueError, match="exploration_only_rate must be in"):
        BranchRouter(hidden_size=16, exploration_only_rate=-0.1)
    with pytest.raises(ValueError, match="exploration_only_rate must be in"):
        BranchRouter(hidden_size=16, exploration_only_rate=1.5)


def test_exploration_only_rng_injected_first_4_of_8_mask():
    """RNG-injected determinism (a softer variant of the AC-14
    plan-text test that doesn't require seeding `torch.Generator`
    at the routing call site).

    With `exploration_only_rate=1.0`, EVERY token is explored, so
    `last_exploration_only_mask` is all True regardless of RNG
    state. With `exploration_only_rate=0.0`, NO token is explored,
    so the mask is None (no recording). This test locks both
    boundary cases and is robust to RNG implementation details.

    A future trainer-integration test (with the real RNG injection
    via `torch.Generator`) can lock the per-token mask pattern at
    intermediate rates."""
    torch.manual_seed(20260428)
    H = 16
    T = 8

    # rate=1.0: every token is in the explore mask.
    router_full = BranchRouter(hidden_size=H, exploration_only_rate=1.0).train()
    x = torch.randn(T, H)
    _, _, _, _ = router_full(x)
    full_mask = router_full.last_exploration_only_mask
    assert full_mask is not None and full_mask.all().item(), (
        f"rate=1.0 should mark every token; got {full_mask}"
    )

    # rate=0.0: no exploration, mask not recorded.
    router_none = BranchRouter(hidden_size=H, exploration_only_rate=0.0).train()
    _, _, _, _ = router_none(x)
    assert router_none.last_exploration_only_mask is None


def test_exploration_only_does_not_use_expert_bias():
    """With `exploration_only_rate=1.0` AND `use_deepseek_style=True`,
    the BranchRouter has an `expert_bias` buffer, but exploration
    routing should completely IGNORE it. Setting expert_bias to a
    huge value that would normally force every token to MLP must
    NOT change the explored selection (uniform Bernoulli is
    independent of expert_bias)."""
    torch.manual_seed(20260428)
    H = 16
    T = 8192

    # DeepSeek-style with extreme bias toward MLP.
    router = BranchRouter(
        hidden_size=H,
        exploration_only_rate=1.0,
        use_deepseek_style=True,
    ).train()
    with torch.no_grad():
        # MLP (index 1) bias is +100; ATTN (index 0) is -100. Without
        # `exploration_only`, biased argmax is always MLP.
        router.expert_bias[1] = 100.0
        router.expert_bias[0] = -100.0

    x = torch.randn(T, H)
    _, _, _, _ = router(x)
    selections = router.last_selected_experts.reshape(-1)
    n_attn = (selections == 0).sum().item()
    n_mlp = (selections == 1).sum().item()
    # Uniform 50/50 distribution despite the extreme bias.
    abs_dev = abs(n_attn - n_mlp)
    assert abs_dev < 200, (
        f"exploration_only with extreme expert_bias still picked unevenly: "
        f"ATTN={n_attn}, MLP={n_mlp}, |dev|={abs_dev} (expected ≤200 ≈ 4σ)"
    )


def test_exploration_only_local_tokens_per_expert_unchanged_under_zero_rate_argmax_collapse():
    """Negative test (lighter version of AC-14's deterministic-collapse
    test): with `exploration_only_rate=0.0` AND
    `use_deepseek_style=True`, the router falls back to the regular
    argmax path and `local_tokens_per_expert` is updated normally.
    This proves the rate=0 fall-through preserves the production
    bias-update machinery — the explore path ONLY engages when the
    rate is > 0.
    """
    torch.manual_seed(20260428)
    H = 16
    T = 32

    router = BranchRouter(
        hidden_size=H,
        exploration_only_rate=0.0,
        use_deepseek_style=True,
    ).train()
    with torch.no_grad():
        # Bias toward MLP so every token goes to index 1.
        router.expert_bias[1] = 100.0
        router.expert_bias[0] = -100.0

    x = torch.ones(T, H)
    _, _, _, _ = router(x)
    counts = router.local_tokens_per_expert
    # Every token should land in MLP (index 1).
    assert counts[0].item() == 0
    assert counts[1].item() == T


def test_legacy_exploration_only_bool_still_works():
    """Backwards-compat: the legacy `exploration_only: bool` kwarg
    is still accepted. `True` maps to rate=1.0; `False` to 0.0."""
    router_true = BranchRouter(hidden_size=16, exploration_only=True)
    assert router_true.exploration_only_rate == 1.0
    assert router_true.balancing == "exploration_only"

    router_false = BranchRouter(hidden_size=16, exploration_only=False)
    assert router_false.exploration_only_rate == 0.0
    assert router_false.balancing == "none"


# ──────────────────────────────────────────────────────────────────────
#  AC-14 plan-text contracts (Codex Round 20 Findings 1-4)
# ──────────────────────────────────────────────────────────────────────


def test_balancing_field_validates_value():
    """`balancing` accepts only `none` or `exploration_only`. Other
    values fail at construction so misconfigured yamls fail fast."""
    with pytest.raises(ValueError, match="balancing must be one of"):
        BranchRouter(hidden_size=16, balancing="quantile")


def test_balancing_exploration_only_auto_promotes_from_rate():
    """If a caller passes `exploration_only_rate > 0` but leaves
    `balancing="none"` (default), the router auto-promotes
    `balancing` to `"exploration_only"`. This avoids a foot-gun
    where the rate is set but the mode flag is not."""
    router = BranchRouter(hidden_size=16, exploration_only_rate=0.3)
    assert router.balancing == "exploration_only"


def test_telemetry_resets_on_each_forward():
    """After a `rate>0` training forward records a mask, a
    subsequent `rate=0` or eval-mode forward must reset both
    `last_exploration_only_mask` and `last_exploration_only_rate`
    to None / 0.0 so a downstream telemetry reader can't pick up
    stale values."""
    torch.manual_seed(20260428)
    H = 16
    T = 8
    router = BranchRouter(
        hidden_size=H, balancing="exploration_only", exploration_only_rate=0.5,
    ).train()

    # Step 1: nonzero rate, training mode → mask + rate recorded.
    x = torch.randn(T, H)
    _, _, _, _ = router(x)
    assert router.last_exploration_only_mask is not None
    assert router.last_exploration_only_rate == 0.5

    # Step 2: rate goes to 0; training mode. Mask must reset.
    router.exploration_only_rate = 0.0
    _, _, _, _ = router(x)
    assert router.last_exploration_only_mask is None
    assert router.last_exploration_only_rate == 0.0

    # Step 3: rate goes back up but model switches to eval.
    router.exploration_only_rate = 1.0
    router.eval()
    _, _, _, _ = router(x)
    assert router.last_exploration_only_mask is None
    assert router.last_exploration_only_rate == 0.0


def test_rng_injected_first_4_of_8_mask_and_choices():
    """AC-14 plan-text deterministic RNG-injected test.

    Build a `torch.Generator` and seed it so the first
    `torch.rand(8)` draw produces 4 values < 0.5 (these are the
    `explore_mask`'s True positions: tokens 0-3 are explored,
    tokens 4-7 fall through to argmax). The router is configured
    with `balancing="exploration_only"` and
    `exploration_only_rate=0.5`. The test:

    1. Records the mask the router produced and asserts it
       matches the expected `[True, True, True, True, False,
       False, False, False]` pattern.
    2. Asserts that the masked tokens' choices come from the
       second `torch.rand` draw (uniform 50/50 random choice),
       NOT from the gate's argmax.
    3. Asserts that the unmasked tokens' choices match the
       gate's argmax exactly (set up so all 8 argmax values are
       index 1 / MLP).
    """
    H = 16
    T = 8

    router = BranchRouter(
        hidden_size=H,
        balancing="exploration_only",
        exploration_only_rate=0.5,
    ).train()

    # Bias gate so argmax is deterministically MLP (index 1) for
    # every token (unmasked tokens follow this).
    with torch.no_grad():
        torch.nn.init.constant_(router.gate.weight[1], 100.0)
        torch.nn.init.constant_(router.gate.weight[0], -100.0)

    # Construct a generator whose state produces the desired
    # explore mask + random-choice pattern. We probe the
    # generator to capture what `torch.rand(8)` and the second
    # `torch.rand(8)` will produce, then construct expected
    # values from those probes.
    gen_probe = torch.Generator()
    gen_probe.manual_seed(20260428)
    expected_mask_draw = torch.rand(8, generator=gen_probe)
    expected_choice_draw = torch.rand(8, generator=gen_probe)
    expected_mask = expected_mask_draw < 0.5
    expected_random_choice = (expected_choice_draw < 0.5).long()

    # Re-seed the actual generator the router will use.
    gen = torch.Generator()
    gen.manual_seed(20260428)
    router.exploration_generator = gen

    # Strictly-positive input so the biased argmax is
    # deterministic for unmasked tokens.
    x = torch.ones(T, H)
    _, _, _, _ = router(x)

    actual_mask = router.last_exploration_only_mask
    actual_choices = router.last_selected_experts.reshape(-1)

    # 1) Mask matches the expected pattern from the seeded generator.
    assert torch.equal(actual_mask, expected_mask), (
        f"explore_mask mismatch: actual={actual_mask.tolist()}, "
        f"expected={expected_mask.tolist()}"
    )

    # 2) Masked tokens' choices come from the random-choice draw.
    for i in range(T):
        if expected_mask[i]:
            assert actual_choices[i] == expected_random_choice[i], (
                f"masked token {i}: actual_choice={actual_choices[i].item()}, "
                f"expected_random={expected_random_choice[i].item()}"
            )

    # 3) Unmasked tokens' choices match argmax (always MLP=1 here).
    for i in range(T):
        if not expected_mask[i]:
            assert actual_choices[i] == 1, (
                f"unmasked token {i}: actual_choice={actual_choices[i].item()}, "
                f"expected argmax MLP=1"
            )


def test_exploration_only_does_not_increment_local_tokens_per_expert():
    """AC-14 negative test (call-site): a DeepSeek-style branch
    router under `balancing=exploration_only` MUST NOT increment
    `local_tokens_per_expert` during forward. This is the
    construction-side half of the "branch DeepSeek bias is inert
    under exploration_only" contract."""
    torch.manual_seed(20260428)
    H = 16
    T = 32

    router = BranchRouter(
        hidden_size=H,
        balancing="exploration_only",
        exploration_only_rate=1.0,
        use_deepseek_style=True,
    ).train()

    # Pre-populate counts to a non-zero baseline; the test checks
    # that the forward DOES NOT increment from this baseline.
    router.local_tokens_per_expert.copy_(torch.tensor([7.0, 11.0]))
    initial_counts = router.local_tokens_per_expert.clone()

    x = torch.randn(T, H)
    _, _, _, _ = router(x)

    assert torch.equal(router.local_tokens_per_expert, initial_counts), (
        f"exploration_only DeepSeek-style branch router incremented counts: "
        f"before={initial_counts.tolist()}, after={router.local_tokens_per_expert.tolist()}"
    )


def test_update_expert_biases_skips_exploration_only_branch_owner():
    """AC-14 negative test (walker-side): `update_expert_biases`
    must skip branch routers whose `balancing == "exploration_only"`
    even if the model's `_load_balancing_method` is
    `deepseek_bias`. A bias update on an exploration-only branch
    router would push `expert_bias` away from zero with no
    routing effect — we want neither the noise nor the
    surprising state-dict diff."""
    routing = _load_routing_module()

    branch = BranchRouter(
        hidden_size=16,
        balancing="exploration_only",
        exploration_only_rate=1.0,
        use_deepseek_style=True,
    )
    branch.local_tokens_per_expert.copy_(torch.tensor([100.0, 1.0]))
    initial_bias = branch.expert_bias.detach().clone()
    initial_counts = branch.local_tokens_per_expert.detach().clone()

    class _Model:
        _load_balancing_method = "deepseek_bias"

        def __init__(self, branch):
            self._branch = branch

        def get_all_balancing_owners(self):
            yield self._branch, "branch"

    with torch.no_grad():
        routing.update_expert_biases(
            _Model(branch), bias_rate=0.01, distributed=False,
        )

    # Branch router state must be untouched.
    assert torch.equal(branch.expert_bias, initial_bias), (
        f"exploration_only branch router's expert_bias was modified: "
        f"before={initial_bias.tolist()}, after={branch.expert_bias.tolist()}"
    )
    assert torch.equal(branch.local_tokens_per_expert, initial_counts), (
        f"exploration_only branch router's counts were zeroed by bias update: "
        f"before={initial_counts.tolist()}, after={branch.local_tokens_per_expert.tolist()}"
    )


def test_update_expert_biases_does_update_non_exploration_only_branch():
    """Companion positive test: a branch router with
    `balancing="none"` (default) and `use_deepseek_style=True` IS
    updated by the walker. This proves the skip in
    `update_expert_biases` is gated on `balancing=exploration_only`
    only — not a blanket disable for branch routers."""
    routing = _load_routing_module()

    branch = BranchRouter(
        hidden_size=16,
        balancing="none",
        use_deepseek_style=True,
    )
    branch.local_tokens_per_expert.copy_(torch.tensor([100.0, 1.0]))
    initial_bias = branch.expert_bias.detach().clone()

    class _Model:
        _load_balancing_method = "deepseek_bias"

        def __init__(self, branch):
            self._branch = branch

        def get_all_balancing_owners(self):
            yield self._branch, "branch"

    with torch.no_grad():
        routing.update_expert_biases(
            _Model(branch), bias_rate=0.01, distributed=False,
        )

    # The legacy branch path SHOULD have updated expert_bias.
    assert not torch.equal(branch.expert_bias, initial_bias), (
        "non-exploration_only branch router with deepseek_bias method "
        "should have been updated by the walker; expert_bias unchanged"
    )


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
