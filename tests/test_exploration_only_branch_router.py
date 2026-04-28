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

    Use a seeded `torch.Generator` whose first `torch.rand(8)` draw
    produces a LITERAL `[True, True, True, True, False, False,
    False, False]` mask under the `< 0.5` threshold (i.e. the first
    four values are < 0.5 and the last four are >= 0.5). This is
    the strict plan-text "first 4 of 8" assertion: the test does not
    just match whatever the generator happens to produce, it pins
    the literal mask pattern.

    Seed 48 has been verified to produce this exact pattern with
    `torch` 2.x's default RNG implementation; if a future torch
    version changes the RNG output and this seed no longer matches,
    re-derive the seed by brute-forcing
    `for s in range(N): torch.Generator().manual_seed(s); rand(8)`
    until a seed produces the desired threshold pattern.

    The router is configured with `balancing="exploration_only"`
    and `exploration_only_rate=0.5`. The test then:

    1. Asserts the mask is literally `[T,T,T,T,F,F,F,F]`.
    2. Asserts that masked tokens' choices come from the second
       `torch.rand(8)` draw (uniform 50/50 random choice),
       NOT from the gate's argmax.
    3. Asserts that unmasked tokens' choices match the gate's
       argmax exactly (the gate is biased so argmax = MLP = 1).
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

    # Seed 48 has been verified to produce the literal
    # `[T,T,T,T,F,F,F,F]` pattern under `< 0.5` threshold.
    LITERAL_FIRST_FOUR_SEED = 48

    # Probe the generator to get the random-choice draw too. The
    # probe consumes the same prefix the router will consume on its
    # forward (`rand(8)` for the explore mask, then `rand(8)` for
    # the random choice), so we can construct the expected
    # `random_choice` tensor before re-seeding the actual generator.
    gen_probe = torch.Generator()
    gen_probe.manual_seed(LITERAL_FIRST_FOUR_SEED)
    expected_mask_draw = torch.rand(8, generator=gen_probe)
    expected_choice_draw = torch.rand(8, generator=gen_probe)

    # The literal-first-four-of-eight assertion: this is what the
    # plan text and the test name promise. If this assertion ever
    # fails, the seed produced a different threshold pattern (likely
    # because torch's RNG output changed) and the seed must be
    # re-derived; the test is intentionally strict so this drift
    # surfaces immediately.
    expected_mask = torch.tensor(
        [True, True, True, True, False, False, False, False], dtype=torch.bool
    )
    assert torch.equal(expected_mask_draw < 0.5, expected_mask), (
        "RNG drift: seed 48 no longer produces "
        "[T,T,T,T,F,F,F,F] under `< 0.5`. Re-derive seed."
    )
    expected_random_choice = (expected_choice_draw < 0.5).long()

    # Re-seed the actual generator the router will use.
    gen = torch.Generator()
    gen.manual_seed(LITERAL_FIRST_FOUR_SEED)
    router.exploration_generator = gen

    # Strictly-positive input so the biased argmax is
    # deterministic for unmasked tokens.
    x = torch.ones(T, H)
    _, _, _, _ = router(x)

    actual_mask = router.last_exploration_only_mask
    actual_choices = router.last_selected_experts.reshape(-1)

    # 1) Literal first-4-of-8 mask.
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


def _tiny_moe_everything_with_branch_balancing(
    *,
    branch_balancing: str,
    branch_exploration_rate: float,
    branch_exploration_decay: str = "constant",
    branch_exploration_warmup_steps: int = 0,
    branch_exploration_min: float = 0.0,
):
    """Build a tiny MoEverythingForCausalLM end-to-end through the
    config->model construction path (the same path model_factory.py
    uses), with the new branch_router fields wired in. This exercises
    the *call site*: `MoEverythingConfig` -> `MoEverythingModel`
    constructor -> `BranchRouter(...)` instantiation.
    """
    from src.models import MoEverythingConfig, MoEverythingForCausalLM

    cfg = MoEverythingConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
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
        branch_deepseek=True,
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
        branch_balancing=branch_balancing,
        branch_exploration_rate=branch_exploration_rate,
        branch_exploration_decay=branch_exploration_decay,
        branch_exploration_min=branch_exploration_min,
        branch_exploration_warmup_steps=branch_exploration_warmup_steps,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        seq_aux_loss_coef=0.0,
        output_router_logits=False,
        attn_implementation="eager",
    )
    torch.manual_seed(20260428)
    model = MoEverythingForCausalLM(cfg)
    model._load_balancing_method = "deepseek_bias"
    cfg.load_balancing_method = "deepseek_bias"
    return model, cfg


def test_moe_everything_call_site_propagates_branch_balancing():
    """Round 22 AC-14 call-site: building a `MoEverythingForCausalLM`
    with `branch_balancing="exploration_only"` and a non-zero
    `branch_exploration_rate` MUST result in the constructed
    `BranchRouter` having both fields populated. Catches any future
    regression in the
    `MoEverythingConfig -> MoEverythingModel -> BranchRouter`
    plumbing chain.
    """
    model, cfg = _tiny_moe_everything_with_branch_balancing(
        branch_balancing="exploration_only",
        branch_exploration_rate=0.7,
    )
    branch = model.model.branch_router
    assert branch.balancing == "exploration_only", (
        f"branch.balancing not propagated: got {branch.balancing!r}, "
        f"expected 'exploration_only'"
    )
    assert branch.exploration_only_rate == 0.7, (
        f"branch.exploration_only_rate not propagated: got "
        f"{branch.exploration_only_rate!r}, expected 0.7"
    )


def test_moe_everything_call_site_default_branch_balancing_none():
    """Negative companion: default config (branch_balancing not set)
    must produce a `BranchRouter` with `balancing == "none"` and
    `exploration_only_rate == 0.0`. Otherwise, every model would
    accidentally enable exploration_only just by being built from
    defaults."""
    model, cfg = _tiny_moe_everything_with_branch_balancing(
        branch_balancing="none",
        branch_exploration_rate=0.0,
    )
    branch = model.model.branch_router
    assert branch.balancing == "none"
    assert branch.exploration_only_rate == 0.0


def test_moe_everything_call_site_trainer_helpers_apply_rate_and_skip_bias():
    """End-to-end call-site test for the AC-14 trainer hooks.

    Build a tiny moe_everything model with `branch_balancing="
    exploration_only"`, `branch_exploration_rate=1.0`,
    `branch_exploration_decay="linear"`,
    `branch_exploration_warmup_steps=10`,
    `branch_exploration_min=0.0`. Step the helpers manually:

    1. `compute_branch_exploration_only_rate(model, 0)` returns 1.0
       (start of linear decay).
    2. `compute_branch_exploration_only_rate(model, 5)` returns ~0.5
       (mid-decay).
    3. `apply_branch_exploration_only_rate(model, 0.5)` pushes the
       rate onto the branch router and returns 1 (one updated).
    4. After running a forward, the branch's
       `local_tokens_per_expert` MUST NOT be incremented by the
       count path (this is the production guard that proved the
       walker-side bias-update is inert under exploration_only).
    """
    routing = _load_routing_module()

    model, cfg = _tiny_moe_everything_with_branch_balancing(
        branch_balancing="exploration_only",
        branch_exploration_rate=1.0,
        branch_exploration_decay="linear",
        branch_exploration_warmup_steps=10,
        branch_exploration_min=0.0,
    )
    branch = model.model.branch_router

    # 1. Schedule resolution.
    rate0 = routing.compute_branch_exploration_only_rate(model, 0)
    assert rate0 == 1.0, f"linear decay step 0: expected 1.0, got {rate0}"
    rate5 = routing.compute_branch_exploration_only_rate(model, 5)
    assert math.isclose(rate5, 0.5, abs_tol=1e-6), (
        f"linear decay step 5: expected ~0.5, got {rate5}"
    )

    # 2. Apply rate hook updates the actual router.
    n = routing.apply_branch_exploration_only_rate(model, 0.42)
    assert n >= 1, (
        f"apply_branch_exploration_only_rate updated {n} routers; "
        f"expected at least 1"
    )
    assert branch.exploration_only_rate == 0.42, (
        f"branch.exploration_only_rate not pushed: got "
        f"{branch.exploration_only_rate!r}"
    )

    # 3. Pre-populate counts to a non-zero baseline; forward must
    # leave them unchanged because the count path skips
    # exploration_only.
    branch.local_tokens_per_expert.zero_()
    branch.local_tokens_per_expert += 7
    initial = branch.local_tokens_per_expert.clone()

    model.train()
    input_ids = torch.randint(0, model.vocab_size, (1, 4), dtype=torch.long)
    _ = model(input_ids=input_ids)

    assert torch.equal(branch.local_tokens_per_expert, initial), (
        f"branch local_tokens_per_expert mutated under exploration_only: "
        f"before={initial.tolist()}, after="
        f"{branch.local_tokens_per_expert.tolist()}"
    )


def test_moe_everything_call_site_apply_rate_returns_zero_on_default_config():
    """Negative companion: default-config moe_everything (no
    exploration_only) must have `apply_branch_exploration_only_rate`
    return 0 (no routers updated). This ensures the trainer's
    per-step hook is a no-op for models that opted out of the
    feature."""
    routing = _load_routing_module()
    model, _ = _tiny_moe_everything_with_branch_balancing(
        branch_balancing="none",
        branch_exploration_rate=0.0,
    )
    n = routing.apply_branch_exploration_only_rate(model, 0.42)
    assert n == 0, (
        f"apply_branch_exploration_only_rate updated {n} routers on "
        f"default-config model; expected 0"
    )
    rate = routing.compute_branch_exploration_only_rate(model, 5)
    assert rate is None, (
        f"compute_branch_exploration_only_rate returned {rate} on "
        f"default-config model; expected None"
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
