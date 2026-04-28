"""AC-6 / task11: DeepSeek bias-update lifecycle tests.

Covers the bias-update behaviors that the existing
`tests/test_bias_update_detailed.py` does NOT exercise:

* **Warmup ramp** — `get_bias_rate(model, step, base, warmup_start, warmup_steps)`
  produces the correct linearly-interpolated rate during warmup, snaps to
  `base` at `step == warmup_steps`, and stays there afterwards. The
  `warmup_steps <= 0` short-circuit returns `base` for every step.

* **Zero-sum delta** (DEC-2) — for any non-uniform load distribution, the
  bias delta from `_update_single_router_bias` sums to (numerical) zero
  so the cumulative bias does not drift unboundedly. This is the
  load-balancing invariant that the `(s - s.mean())` factor enforces.

* **Persistence** — `expert_bias` is a persistent buffer so it survives
  `state_dict()` round-trips. `local_tokens_per_expert` is non-persistent
  (per-step state) and must NOT appear in `state_dict`.

* **DDP all-reduce** — when called with `distributed=True`, the function
  invokes `dist.all_reduce(counts, op=SUM)` on `local_tokens_per_expert`
  before computing the load fractions, so unequal per-rank routing is
  resolved into a single global-balance signal.

* **±16 clamp** — the cumulative bias is clamped to `[-16, 16]` after
  every update, matching DeepSeek-V3's scale guard. Already partially
  covered in `test_router_parity_nmoe.py`; here we lock the clamp on the
  trainer-side updater specifically.
"""
from __future__ import annotations

import importlib.util
import sys
import types as _types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.router import DeepSeekRouter
from src.models.configuration_qwen3_moe import Qwen3MoeConfig


def _load_routing_module():
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
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


def _make_router(num_experts: int = 8, hidden_size: int = 16, top_k: int = 2):
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=hidden_size * 2,
        moe_intermediate_size=hidden_size * 2,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    return DeepSeekRouter(config)


# ──────────────────────────────────────────────────────────────────────
#  Warmup ramp (`get_bias_rate`)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "step,base,warmup_start,warmup_steps,expected",
    [
        # warmup_steps == 0 → no warmup, always base
        (0,    0.001, 0.0,    0,    0.001),
        (100,  0.001, 0.0,    0,    0.001),
        # warmup_start == 0 → linear ramp 0 → base over warmup_steps
        (0,    0.001, 0.0,    100,  0.0),
        (50,   0.001, 0.0,    100,  0.0005),
        (99,   0.001, 0.0,    100,  0.00099),
        (100,  0.001, 0.0,    100,  0.001),
        (200,  0.001, 0.0,    100,  0.001),
        # warmup_start > 0 → ramp from warmup_start to base
        (0,    0.005, 0.001,  100,  0.001),
        (50,   0.005, 0.001,  100,  0.003),
        (100,  0.005, 0.001,  100,  0.005),
    ],
)
def test_get_bias_rate_warmup_ramp(step, base, warmup_start, warmup_steps, expected):
    """`get_bias_rate` linearly interpolates from `warmup_start` to `base`
    over `warmup_steps`, then stays at `base`. With `warmup_steps == 0`
    the rate is always `base`.
    """
    routing = _load_routing_module()

    # `get_bias_rate` calls `unwrap_model`, which expects a model-like
    # object. A plain MagicMock without `get_bias_rate` is enough because
    # the function falls through to the linear branch when the model
    # doesn't override.
    fake_model = MagicMock(spec=[])  # spec=[] → no attributes

    rate = routing.get_bias_rate(
        fake_model,
        step=step,
        base_rate=base,
        warmup_start=warmup_start,
        warmup_steps=warmup_steps,
    )
    assert rate == pytest.approx(expected, rel=1e-6, abs=1e-9), (
        f"get_bias_rate({step=}, {base=}, {warmup_start=}, {warmup_steps=}) "
        f"= {rate}, expected {expected}"
    )


def test_get_bias_rate_respects_model_override():
    """If the model defines its own `get_bias_rate`, the trainer-side
    helper delegates to it rather than computing the linear ramp itself.
    """
    routing = _load_routing_module()

    class _ModelWithCustomRate:
        def get_bias_rate(self, step, base_rate, warmup_start, warmup_steps):
            # Custom: return double the base after step 50, else base.
            return base_rate * 2 if step >= 50 else base_rate

    rate_before = routing.get_bias_rate(_ModelWithCustomRate(), step=10, base_rate=0.001, warmup_start=0.0, warmup_steps=100)
    rate_after = routing.get_bias_rate(_ModelWithCustomRate(), step=60, base_rate=0.001, warmup_start=0.0, warmup_steps=100)
    assert rate_before == 0.001
    assert rate_after == 0.002


# ──────────────────────────────────────────────────────────────────────
#  Zero-sum delta (DEC-2)
# ──────────────────────────────────────────────────────────────────────

def test_bias_update_delta_is_zero_sum():
    """DEC-2 invariant (zero_sum=True): the bias delta has zero mean.

    After one update step in `zero_sum=True` mode, sum(new_bias -
    old_bias) ≈ 0 (within fp32 round-off). This keeps the cumulative
    bias from drifting and is the crucial difference between the
    nmoe / DeepSeek-V3 zero-sum formulation and the naive Megatron
    `sign(avg - count) * rate` (which can have nonzero mean when the
    load distribution is asymmetric)."""
    routing = _load_routing_module()

    router = _make_router(num_experts=8)
    counts = torch.tensor([10.0, 5.0, 20.0, 3.0, 15.0, 8.0, 12.0, 7.0])
    router.local_tokens_per_expert = counts.clone()
    initial_bias = router.expert_bias.detach().clone()

    routing._update_single_router_bias(
        router, bias_rate=0.001, distributed=False, zero_sum=True,
    )
    delta = router.expert_bias - initial_bias

    assert delta.sum().abs() < 1e-7, (
        f"zero_sum=True bias delta is not zero-sum: sum={delta.sum().item()}, delta={delta}"
    )


def test_bias_update_megatron_mode_can_drift_mean():
    """DEC-2 contract for `zero_sum=False`: the Megatron-LM plain-sign
    update does NOT subtract `s.mean()`, so for asymmetric loads (where
    `sign(load - 1/E)` is not balanced between +1 and -1) the bias
    mean drifts.

    Concretely, with 1 overloaded expert and 7 underloaded experts:
      `s = [+1, -1, -1, -1, -1, -1, -1, -1]`
      zero-sum delta = (s - s.mean()) * rate = (s + 0.75) * rate; sum = 0.
      Megatron delta = -s * rate                      ; sum = 6 * rate.
    This test asserts the Megatron-mode delta has the expected non-zero
    mean — a defining behavioral difference from the zero-sum mode.
    """
    routing = _load_routing_module()

    router = _make_router(num_experts=8)
    # 1 overloaded expert, 7 underloaded.
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    router.local_tokens_per_expert = counts.clone()
    initial_bias = router.expert_bias.detach().clone()

    rate = 0.001
    routing._update_single_router_bias(
        router, bias_rate=rate, distributed=False, zero_sum=False,
    )
    delta = router.expert_bias - initial_bias

    # `delta = -sign(loads - 1/E) * rate` =
    #   for the heavy expert: -1 * rate (negative)
    #   for the seven light experts: +1 * rate (positive)
    # Sum = -rate + 7 * rate = +6 * rate.
    assert delta.sum().item() == pytest.approx(6 * rate, rel=1e-6), (
        f"zero_sum=False (Megatron) delta sum = {delta.sum().item()}, expected {6*rate}"
    )
    # The heavy expert gets bias DECREASED, the light experts INCREASED —
    # same routing-direction signal as zero-sum mode.
    assert delta[0].item() < 0, "heavy expert bias must decrease"
    assert (delta[1:] > 0).all(), "light experts' biases must all increase"


@pytest.mark.parametrize("zero_sum", [True, False])
def test_both_modes_drive_overloaded_bias_negative(zero_sum):
    """Convergence-style test (DEC-2): under repeated steps with the
    same overloaded-load pattern, the heavy expert's bias must drift
    monotonically negative for BOTH modes — they target the same
    routing-correction signal even though the per-step deltas differ.
    """
    routing = _load_routing_module()
    router = _make_router(num_experts=4)
    rate = 0.05

    bias_history: list[float] = []
    for _ in range(20):
        # Same heavy/light pattern every step (simulates a stuck imbalance).
        router.local_tokens_per_expert = torch.tensor([100.0, 1.0, 1.0, 1.0])
        routing._update_single_router_bias(
            router, bias_rate=rate, distributed=False, zero_sum=zero_sum,
        )
        bias_history.append(router.expert_bias[0].item())

    # The heavy expert's bias should be monotonically non-increasing
    # (allowing equal-step plateaus due to fp32 round-off near zero), and
    # finish well below where it started.
    diffs = [bias_history[i+1] - bias_history[i] for i in range(len(bias_history)-1)]
    assert all(d <= 1e-6 for d in diffs), (
        f"zero_sum={zero_sum} heavy-expert bias is not monotonically non-increasing: {bias_history}"
    )
    assert bias_history[-1] < bias_history[0] - rate, (
        f"zero_sum={zero_sum} heavy-expert bias did not drift negative enough: "
        f"started {bias_history[0]}, ended {bias_history[-1]}, rate={rate}"
    )


# ──────────────────────────────────────────────────────────────────────
#  Persistence
# ──────────────────────────────────────────────────────────────────────

def test_expert_bias_survives_state_dict_round_trip():
    """`expert_bias` is registered as a persistent buffer. After saving
    and loading state_dict, the bias values are preserved."""
    router = _make_router(num_experts=8)
    router.expert_bias.copy_(torch.linspace(-0.5, 0.5, 8))
    saved = router.expert_bias.detach().clone()

    sd = router.state_dict()
    assert "expert_bias" in sd, "expert_bias must be in state_dict (persistent buffer)"

    # Round-trip into a fresh router instance.
    fresh = _make_router(num_experts=8)
    assert (fresh.expert_bias - saved).abs().max() > 0  # preflight: starts different
    fresh.load_state_dict(sd)
    torch.testing.assert_close(fresh.expert_bias, saved)


def test_local_tokens_per_expert_is_non_persistent():
    """`local_tokens_per_expert` is per-step transient state; it must
    NOT appear in `state_dict` (else resume would replay the last
    step's counts and double-count after the next step's bias update).
    """
    router = _make_router(num_experts=8)
    router.local_tokens_per_expert.fill_(42.0)
    sd = router.state_dict()
    assert "local_tokens_per_expert" not in sd, (
        "local_tokens_per_expert must NOT be persistent — caching transient "
        "step state into state_dict will double-count after resume."
    )


# ──────────────────────────────────────────────────────────────────────
#  DDP all-reduce
# ──────────────────────────────────────────────────────────────────────

def test_distributed_path_calls_all_reduce_sum():
    """When `distributed=True` AND a process group is initialized, the
    function calls `dist.all_reduce(counts, op=SUM)` exactly once
    before computing fractions. This is the AC-6 DDP-correctness
    contract: unequal per-rank routing must be summed across ranks
    before the bias update so all ranks compute the same
    global-balance delta. Round 12 also requires the function to
    SKIP the all_reduce call when no process group is initialized
    (Codex Round 11 Finding 1d) — covered by
    `test_distributed_path_skips_all_reduce_when_not_initialized`.
    """
    routing = _load_routing_module()
    router = _make_router(num_experts=8)
    router.local_tokens_per_expert = torch.tensor(
        [10.0, 5.0, 20.0, 3.0, 15.0, 8.0, 12.0, 7.0]
    )

    with patch.object(routing.dist, "all_reduce") as mock_ar, \
         patch.object(routing.dist, "is_initialized", return_value=True):
        routing._update_single_router_bias(router, bias_rate=0.001, distributed=True)

    # all_reduce must be called once with op=SUM on the counts tensor.
    assert mock_ar.call_count == 1, (
        f"Expected exactly 1 all_reduce call, got {mock_ar.call_count}"
    )
    args, kwargs = mock_ar.call_args
    # Permissive arg shape: positional or kwargs
    if args:
        first_arg = args[0]
    else:
        first_arg = kwargs.get("tensor")
    assert isinstance(first_arg, torch.Tensor), (
        f"all_reduce first arg must be a tensor, got {type(first_arg)}"
    )
    op = kwargs.get("op")
    if op is None and len(args) >= 2:
        op = args[1]
    # SUM op
    assert op == routing.dist.ReduceOp.SUM, (
        f"all_reduce op must be SUM, got {op}"
    )


def test_distributed_path_skips_all_reduce_when_not_initialized():
    """Round 12 (Codex Round 11 Finding 1d): with `distributed=True`
    but no process group initialized, the function MUST NOT call
    `dist.all_reduce`. The trainer's caller-side guard already does
    the same check; this test locks the defensive guard inside
    `_update_single_router_bias` so direct callers (tests, debug
    fixtures) can pass `distributed=True` safely.
    """
    routing = _load_routing_module()
    router = _make_router(num_experts=4)
    router.local_tokens_per_expert = torch.tensor([10.0, 1.0, 1.0, 1.0])
    initial_bias = router.expert_bias.detach().clone()

    with patch.object(routing.dist, "all_reduce") as mock_ar, \
         patch.object(routing.dist, "is_initialized", return_value=False):
        routing._update_single_router_bias(router, bias_rate=0.001, distributed=True)

    assert mock_ar.call_count == 0, (
        "all_reduce must NOT be called when process group is not "
        f"initialized; got {mock_ar.call_count} calls."
    )
    # The bias still updates (using rank-local counts, since there's
    # nothing to all-reduce), so the function is still useful for
    # single-rank dev runs that happen to pass distributed=True.
    assert (router.expert_bias != initial_bias).any(), (
        "Bias should still update from rank-local counts when no "
        "process group is initialized."
    )


# ──────────────────────────────────────────────────────────────────────
#  ±16 clamp lock-in
# ──────────────────────────────────────────────────────────────────────

def test_bias_clamp_at_plus_16():
    """After updates that would push above +16, the bias is clamped to
    +16 (DeepSeek-V3 scale guard)."""
    routing = _load_routing_module()
    router = _make_router(num_experts=4)
    router.expert_bias.fill_(15.99)
    # Counts that produce sign = +1 for all experts EXCEPT the heavy one.
    # (heavy one becomes -1.) So 3 experts get +rate, 1 gets -3*rate (after
    # zero-sum subtraction). For rate = 0.1: deltas = +0.075, +0.075, +0.075, -0.225.
    router.local_tokens_per_expert = torch.tensor([100.0, 1.0, 1.0, 1.0])

    # Apply many extreme updates to push above +16.
    for _ in range(200):
        router.local_tokens_per_expert = torch.tensor([100.0, 1.0, 1.0, 1.0])
        routing._update_single_router_bias(router, bias_rate=0.5, distributed=False)

    assert router.expert_bias.max().item() <= 16.0 + 1e-6, (
        f"expert_bias max {router.expert_bias.max().item()} exceeds +16 cap"
    )
    assert router.expert_bias.min().item() >= -16.0 - 1e-6, (
        f"expert_bias min {router.expert_bias.min().item()} is below -16 cap"
    )


# ──────────────────────────────────────────────────────────────────────
#  Counts reset semantics
# ──────────────────────────────────────────────────────────────────────

def test_local_tokens_zeroed_after_update():
    """`local_tokens_per_expert` must be zeroed after each update so the
    next step's accumulator starts fresh — otherwise step N's update
    would double-count step N-1's tokens."""
    routing = _load_routing_module()
    router = _make_router(num_experts=4)
    router.local_tokens_per_expert = torch.tensor([10.0, 5.0, 20.0, 3.0])
    routing._update_single_router_bias(router, bias_rate=0.001, distributed=False)
    assert router.local_tokens_per_expert.sum().item() == 0.0, (
        "local_tokens_per_expert must be zeroed after update"
    )


# ──────────────────────────────────────────────────────────────────────
#  update_bias_from_counts (src/models/routing/bias.py) coverage
#  (Codex Round 11 Finding 1c)
# ──────────────────────────────────────────────────────────────────────


def _load_bias_module():
    """Load src.models.routing.bias.update_bias_from_counts directly."""
    import importlib.util
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "src.models.routing.bias",
        repo / "src" / "models" / "routing" / "bias.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("zero_sum", [True, False])
def test_update_bias_from_counts_runs_for_both_modes(zero_sum):
    """`src/models/routing/bias.py:update_bias_from_counts` must
    accept the `zero_sum` kwarg and update bias in-place for both
    modes. Both modes target the same routing-direction signal:
    overloaded experts → bias decreases; underloaded → increases."""
    bias_mod = _load_bias_module()
    E = 4
    bias = torch.zeros(E, dtype=torch.float32)
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])  # heavy on expert 0

    bias_mod.update_bias_from_counts(
        bias, counts.clone(), rate=0.01,
        clamp_range=16.0, distributed=False, zero_sum=zero_sum,
    )
    assert bias[0].item() < 0, (
        f"zero_sum={zero_sum}: heavy expert 0 must have bias decreased, got {bias[0].item()}"
    )
    assert (bias[1:] > 0).all(), (
        f"zero_sum={zero_sum}: light experts 1-3 must have bias increased, got {bias}"
    )


def test_update_bias_from_counts_zero_sum_true_pins_mean_at_zero():
    """`zero_sum=True` (default): cumulative bias mean stays at 0.0
    (within fp32 round-off), even under asymmetric loads. This is
    the defining property of the nmoe / DeepSeek-V3 formulation."""
    bias_mod = _load_bias_module()
    E = 4
    bias = torch.zeros(E, dtype=torch.float32)
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])

    bias_mod.update_bias_from_counts(
        bias, counts.clone(), rate=0.01,
        distributed=False, zero_sum=True,
    )
    assert bias.mean().abs().item() < 1e-7, (
        f"zero_sum=True bias mean must be ≈ 0, got {bias.mean().item()}"
    )


def test_update_bias_from_counts_zero_sum_false_drifts_mean():
    """`zero_sum=False` (Megatron): cumulative bias mean drifts under
    asymmetric loads. Specifically, with 1 heavy and 3 light experts,
    `delta = -sign(load - 1/E) * rate` = `[-rate, +rate, +rate, +rate]`,
    so `sum(delta) = +2*rate ≠ 0`."""
    bias_mod = _load_bias_module()
    E = 4
    bias = torch.zeros(E, dtype=torch.float32)
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])
    rate = 0.01

    bias_mod.update_bias_from_counts(
        bias, counts.clone(), rate=rate,
        distributed=False, zero_sum=False,
    )
    # Heavy expert: -rate; 3 light experts: +rate each. Sum = +2*rate.
    expected_sum = 2 * rate
    assert bias.sum().item() == pytest.approx(expected_sum, rel=1e-6), (
        f"zero_sum=False bias sum should be {expected_sum}, got {bias.sum().item()}"
    )


def test_update_bias_from_counts_skips_all_reduce_when_not_initialized():
    """Round 12 (Codex Round 11 Finding 1d): `update_bias_from_counts`
    must guard `dist.all_reduce` with `dist.is_initialized()` so a
    direct caller (test, debug fixture) can pass `distributed=True`
    without crashing on the missing process group."""
    bias_mod = _load_bias_module()
    E = 4
    bias = torch.zeros(E, dtype=torch.float32)
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])

    with patch.object(bias_mod.dist, "all_reduce") as mock_ar, \
         patch.object(bias_mod.dist, "is_initialized", return_value=False):
        bias_mod.update_bias_from_counts(
            bias, counts.clone(), rate=0.01,
            distributed=True, zero_sum=True,
        )
    assert mock_ar.call_count == 0, (
        f"all_reduce must NOT be called when process group is not initialized; "
        f"got {mock_ar.call_count} calls."
    )


def test_update_bias_from_counts_calls_all_reduce_when_initialized():
    """Round 12: when `distributed=True` AND a process group is live,
    `update_bias_from_counts` must call `dist.all_reduce(counts, op=SUM)`
    exactly once."""
    bias_mod = _load_bias_module()
    E = 4
    bias = torch.zeros(E, dtype=torch.float32)
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])

    with patch.object(bias_mod.dist, "all_reduce") as mock_ar, \
         patch.object(bias_mod.dist, "is_initialized", return_value=True):
        bias_mod.update_bias_from_counts(
            bias, counts.clone(), rate=0.01,
            distributed=True, zero_sum=True,
        )
    assert mock_ar.call_count == 1
    args, kwargs = mock_ar.call_args
    op = kwargs.get("op")
    if op is None and len(args) >= 2:
        op = args[1]
    assert op == bias_mod.dist.ReduceOp.SUM


def test_update_bias_from_counts_clamps_to_plus_minus_16():
    """`update_bias_from_counts` clamps to ±16 after the update."""
    bias_mod = _load_bias_module()
    E = 4
    bias = torch.full((E,), 15.99, dtype=torch.float32)
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])
    for _ in range(50):
        bias_mod.update_bias_from_counts(
            bias, counts.clone(), rate=0.5,
            clamp_range=16.0, distributed=False, zero_sum=True,
        )
    assert bias.max().item() <= 16.0 + 1e-6
    assert bias.min().item() >= -16.0 - 1e-6


def test_update_bias_from_counts_zero_load_no_op():
    """With zero observed counts (e.g. rank saw no tokens), the
    function must leave bias unchanged."""
    bias_mod = _load_bias_module()
    bias = torch.tensor([0.1, -0.1, 0.05, -0.05], dtype=torch.float32)
    initial = bias.clone()
    bias_mod.update_bias_from_counts(
        bias, torch.zeros(4), rate=0.01, distributed=False, zero_sum=True,
    )
    torch.testing.assert_close(bias, initial)


# ──────────────────────────────────────────────────────────────────────
#  Real DDP all-reduce test (gloo backend, 2 spawned processes)
# ──────────────────────────────────────────────────────────────────────


def _gloo_loopback_iface() -> str:
    """Return the platform-specific loopback interface name for Gloo
    socket rendezvous. macOS uses `lo0`; Linux uses `lo`.
    """
    import platform
    return "lo0" if platform.system() == "Darwin" else "lo"


def _ddp_preflight_worker(rank: int, world_size: int, init_file: str):
    """Minimal worker that ONLY initializes a gloo process group and
    immediately destroys it. Used by `_ddp_preflight()` to detect
    environments where gloo's socket layer can't bind even a
    loopback interface — Codex's CI sandbox aborts inside
    `uv_bind` (libuv) with SIGABRT, which `mp.spawn` surfaces as
    `ProcessExitedException` rather than `ProcessRaisedException`.
    """
    import os
    import torch.distributed as _dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", _gloo_loopback_iface())
    _dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    _dist.destroy_process_group()


def _ddp_preflight(tmp_path) -> str | None:
    """Return None if a 2-rank gloo process group can be initialized
    in this environment; return a string skip-reason otherwise.

    Round 14 (Codex Round 13 Finding 1): handles BOTH
    `ProcessRaisedException` (Python exception in worker — usually
    `RuntimeError("Cannot resolve")`) and `ProcessExitedException`
    (worker SIGABRT from libuv `uv_bind: operation not permitted`).
    Without the preflight, the real test was failing in sandboxes
    that aborted gloo's socket binding.
    """
    import torch.multiprocessing as mp

    init_file = tmp_path / "ddp_preflight"
    try:
        mp.spawn(
            _ddp_preflight_worker,
            args=(2, str(init_file)),
            nprocs=2,
            join=True,
        )
    except mp.ProcessExitedException as exc:
        return f"DDP gloo preflight aborted (likely libuv/uv_bind sandbox restriction): {exc!r}"
    except mp.ProcessRaisedException as exc:
        return f"DDP gloo preflight raised in worker: {exc!r}"
    except (OSError, PermissionError, RuntimeError) as exc:
        return f"DDP gloo preflight environment error: {exc!r}"
    finally:
        # `mp.spawn` cleans up worker processes; the init file is left
        # behind on success/failure. Removing it lets the real test
        # use a fresh file-store rendezvous.
        try:
            init_file.unlink()
        except FileNotFoundError:
            pass
    return None


def _ddp_worker(rank: int, world_size: int, init_file: str, output_path: str):
    """Worker entry point for the 2-rank DDP all-reduce test.

    Round 13 (Codex Round 12 Finding 1): even with `file://`
    rendezvous, Gloo still needs a socket interface for collectives.
    Set `GLOO_SOCKET_IFNAME` to the platform-specific loopback
    interface (`lo0` on Darwin, `lo` on Linux) before calling
    `init_process_group`. If the chosen interface still cannot be
    resolved (some sandboxes scrub `/sys/class/net`), the worker
    will raise from `init_process_group` and the parent's
    `mp.spawn` will surface that as a
    `ProcessRaisedException`, which the parent catches and converts
    to `pytest.skip`.
    """
    import os
    import torch as _torch
    import torch.distributed as _dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", _gloo_loopback_iface())
    init_method = f"file://{init_file}"
    _dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    routing = _load_routing_module()
    router = _make_router(num_experts=4)

    # Asymmetric per-rank counts:
    # rank 0 saw [10, 0, 0, 0]; rank 1 saw [0, 0, 0, 10].
    # After SUM all-reduce: [10, 0, 0, 10]. Both ranks compute the SAME
    # bias delta from this aggregate, so post-update biases match.
    if rank == 0:
        router.local_tokens_per_expert = _torch.tensor([10.0, 0.0, 0.0, 0.0])
    else:
        router.local_tokens_per_expert = _torch.tensor([0.0, 0.0, 0.0, 10.0])

    routing._update_single_router_bias(
        router, bias_rate=0.01, distributed=True, zero_sum=True,
    )

    _torch.save(router.expert_bias.cpu(), output_path.format(rank=rank))
    _dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed not available in this build",
)
def test_ddp_two_rank_all_reduce_produces_identical_biases(tmp_path):
    """DEC-2 / AC-6 real-DDP correctness: with 2 spawned worker
    processes (gloo backend, asymmetric per-rank counts), the
    post-update `expert_bias` must be identical on both ranks. This
    proves the `dist.all_reduce(counts, op=SUM)` path actually
    aggregates counts before the bias update, not just that the call
    was made (which the mocked test in
    `test_distributed_path_calls_all_reduce_sum` already covers).

    Round 14 (Codex Round 13 Finding 1): preflight a tiny 2-rank
    gloo `init_process_group` + `destroy_process_group` cycle BEFORE
    running the actual test. If the preflight aborts (libuv
    `uv_bind: operation not permitted` → SIGABRT → mp
    `ProcessExitedException`), the test skips with that environment
    reason. If the preflight succeeds, the real test runs and any
    worker exception (e.g. an assertion failure) is treated as a
    real test failure rather than masked by an env-skip.
    """
    import torch.multiprocessing as mp

    skip_reason = _ddp_preflight(tmp_path)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    init_file = tmp_path / "ddp_init"
    output_template = str(tmp_path / "rank_{rank}_bias.pt")
    mp.spawn(
        _ddp_worker,
        args=(2, str(init_file), output_template),
        nprocs=2,
        join=True,
    )

    rank0_bias = torch.load(output_template.format(rank=0))
    rank1_bias = torch.load(output_template.format(rank=1))

    torch.testing.assert_close(rank0_bias, rank1_bias, atol=1e-7, rtol=1e-7)

    # Sanity: the expected aggregate is `[10, 0, 0, 10]`, so loads are
    # `[0.5, 0, 0, 0.5]` and the zero-sum update produces a non-trivial
    # symmetric delta. Bias should be non-zero somewhere.
    assert rank0_bias.abs().sum().item() > 0, (
        "post-update bias is all zeros — all-reduce probably did not run"
    )


# ──────────────────────────────────────────────────────────────────────
#  Trainer call-order: bias update must run AFTER optimizer.step
#  (Codex Round 12 Finding 2)
# ──────────────────────────────────────────────────────────────────────


def test_routing_helper_invokes_bias_update_after_optimizer_step():
    """AC-6 source-level ordering check: in
    `src/training/routing.py:trainer_optimizer_step_and_bias_update`,
    the bias-update entry point must be invoked strictly AFTER
    `optimizer.step()`. The bias update reads stale
    `local_tokens_per_expert` if it runs before the optimizer step
    has consumed the current step's gradients.

    The trainer's `run_training()` calls
    `trainer_optimizer_step_and_bias_update(...)`, which is the
    canonical extracted helper for the optimizer-step + bias-update
    sequence. This AST walk locks the helper's source-level order
    so a future refactor cannot re-arrange the calls without
    failing the test.

    The companion runtime test
    (`test_trainer_optimizer_step_and_bias_update_runtime_order`)
    additionally instruments the helper with spies and proves the
    runtime call order matches.
    """
    import ast
    repo = Path(__file__).resolve().parent.parent
    routing_src = (repo / "src" / "training" / "routing.py").read_text()
    module = ast.parse(routing_src)

    BIAS_UPDATE_ENTRY_POINTS = {
        "update_expert_biases",
        "trainer_post_optimizer_bias_update",
    }

    target_func = None
    for node in ast.walk(module):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "trainer_optimizer_step_and_bias_update"
        ):
            target_func = node
            break
    assert target_func is not None, (
        "expected `trainer_optimizer_step_and_bias_update` function in routing.py"
    )

    optimizer_step_lines: list[int] = []
    bias_update_call_lines: list[int] = []
    for child in ast.walk(target_func):
        if isinstance(child, ast.Call):
            if (
                isinstance(child.func, ast.Attribute)
                and child.func.attr == "step"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "optimizer"
            ):
                optimizer_step_lines.append(child.lineno)
            if isinstance(child.func, ast.Name) and child.func.id in BIAS_UPDATE_ENTRY_POINTS:
                bias_update_call_lines.append(child.lineno)

    assert optimizer_step_lines, (
        "expected `optimizer.step()` to appear in "
        "trainer_optimizer_step_and_bias_update"
    )
    assert bias_update_call_lines, (
        "expected one of the bias-update entry points "
        f"({sorted(BIAS_UPDATE_ENTRY_POINTS)}) to appear in "
        "trainer_optimizer_step_and_bias_update"
    )

    first_opt_step = min(optimizer_step_lines)
    first_bias_update = min(bias_update_call_lines)
    assert first_bias_update > first_opt_step, (
        f"ordering violation: bias-update call (line {first_bias_update}) "
        f"must appear AFTER `optimizer.step()` (line {first_opt_step}) "
        f"in trainer_optimizer_step_and_bias_update."
    )


def test_trainer_calls_helper_after_clip_grad_norm():
    """Round 16 source-level check: trainer.run_training()'s main step
    loop must call `trainer_optimizer_step_and_bias_update(...)`
    AFTER `clip_grad_norm_(...)`. Together with the routing-helper
    test above, this locks the full source-level ordering:
    clip_grad_norm → optimizer.step → scheduler.step → bias update.
    """
    import ast
    repo = Path(__file__).resolve().parent.parent
    trainer_src = (repo / "src" / "training" / "trainer.py").read_text()
    module = ast.parse(trainer_src)

    clip_lines: list[int] = []
    helper_lines: list[int] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            # `torch.nn.utils.clip_grad_norm_(...)` and friends.
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "clip_grad_norm_"
            ):
                clip_lines.append(node.lineno)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "trainer_optimizer_step_and_bias_update"
            ):
                helper_lines.append(node.lineno)

    assert clip_lines, "expected `clip_grad_norm_(...)` to appear in trainer.py"
    assert helper_lines, (
        "expected `trainer_optimizer_step_and_bias_update(...)` to appear in trainer.py"
    )
    assert min(helper_lines) > min(clip_lines), (
        f"ordering: trainer must call the optimizer-step helper "
        f"(line {min(helper_lines)}) AFTER clip_grad_norm_ "
        f"(line {min(clip_lines)})."
    )


# ──────────────────────────────────────────────────────────────────────
#  Real trainer call-order instrumentation (Codex Round 13 Finding 2)
# ──────────────────────────────────────────────────────────────────────


def test_trainer_optimizer_step_and_bias_update_runtime_order():
    """End-to-end runtime ordering test for the full helper:
    `trainer_optimizer_step_and_bias_update` is the canonical
    extracted helper that the production trainer calls; it
    internally invokes `optimizer.step` → `scheduler.step` →
    `trainer_post_optimizer_bias_update`. Spies on all three
    confirm the actual production code path runs them in the
    correct order, AND that `update_expert_biases` runs under
    `torch.no_grad()`.

    A future regression that re-orders or unwraps this sequence
    fails this test directly, since the trainer runs THIS helper
    (not a hand-rolled mock).
    """
    routing = _load_routing_module()

    class _TrainCfg:
        bias_update_rate = 0.01
        bias_warmup_start = 0.0
        bias_warmup_steps = 0
        bias_update_zero_sum = True

    cfg: dict = {}

    router = _make_router(num_experts=4)
    router.local_tokens_per_expert = torch.tensor([100.0, 1.0, 1.0, 1.0])

    class _Model:
        def __init__(self, gate, method):
            self._load_balancing_method = method
            self._gate = gate

        def get_all_balancing_owners(self):
            yield self._gate, "mlp"

    # Instrument: wrap optimizer.step / scheduler.step / update_expert_biases
    # with spies that record call order AND grad-enabled state at the
    # moment of each call.
    call_order: list[str] = []
    grad_enabled_at_call: list[tuple[str, bool]] = []

    class _SpyOptimizer:
        def step(self):
            call_order.append("optimizer.step")
            grad_enabled_at_call.append(("optimizer.step", torch.is_grad_enabled()))

    class _SpyScheduler:
        def step(self):
            call_order.append("scheduler.step")
            grad_enabled_at_call.append(("scheduler.step", torch.is_grad_enabled()))

    real_update = routing.update_expert_biases

    def _spy_update_expert_biases(model, **kwargs):
        call_order.append("update_expert_biases")
        grad_enabled_at_call.append(("update_expert_biases", torch.is_grad_enabled()))
        return real_update(model, **kwargs)

    model = _Model(router, "deepseek_bias")
    optimizer = _SpyOptimizer()
    scheduler = _SpyScheduler()

    routing.update_expert_biases = _spy_update_expert_biases
    try:
        # Outer context is grad-enabled (matches the trainer's main step
        # loop); the helper is responsible for the no_grad wrap.
        assert torch.is_grad_enabled()
        routing.trainer_optimizer_step_and_bias_update(
            model, optimizer, scheduler, _TrainCfg(), cfg,
            distributed=False, global_step=1,
        )
    finally:
        routing.update_expert_biases = real_update

    assert call_order == [
        "optimizer.step",
        "scheduler.step",
        "update_expert_biases",
    ], (
        f"runtime ordering violated: expected "
        f"['optimizer.step', 'scheduler.step', 'update_expert_biases'], "
        f"got {call_order}"
    )
    grad_at_bias = dict(grad_enabled_at_call)["update_expert_biases"]
    assert grad_at_bias is False, (
        f"`update_expert_biases` must be invoked under no_grad context, "
        f"got is_grad_enabled={grad_at_bias}"
    )
    # Sanity: optimizer.step and scheduler.step run with grad enabled
    # (the helper does NOT wrap them — only the bias update is
    # wrapped). This proves the no_grad wrap is scoped narrowly.
    grad_at_opt = dict(grad_enabled_at_call)["optimizer.step"]
    grad_at_sched = dict(grad_enabled_at_call)["scheduler.step"]
    assert grad_at_opt is True
    assert grad_at_sched is True


def test_trainer_post_optimizer_bias_update_runtime_order():
    """AC-6 runtime ordering test (Codex Round 14 Finding 2).

    Round 14's predecessor of this test was self-fulfilling: it
    constructed its own optimizer/scheduler/update sequence and
    verified that hand-written sequence — but didn't actually
    execute the production trainer's bias-update call site.

    Round 15 fixes this by extracting the production bias-update
    block from `src/training/trainer.py` into the helper
    `trainer_post_optimizer_bias_update(model, train_cfg, cfg, ...)`
    and testing the helper directly. Spies on:

      - `routing.update_expert_biases` (the symbol the helper
        actually calls — wrapped in `with torch.no_grad():`).

    The test asserts:
      1. The helper invokes `update_expert_biases` exactly once
         when method=`deepseek_bias` and `bias_update_rate > 0`.
      2. At the call site, `torch.is_grad_enabled()` is False
         (the production no_grad wrap is in effect).
      3. For non-bias methods (`aux_loss` / `none`) the helper
         no-ops without invoking the bias update at all.

    This exercises the EXACT production code path the trainer uses,
    so a future refactor that re-orders or unwraps the no_grad
    block would be caught. The companion AST test still locks the
    source structure of the trainer's call sequence.
    """
    routing = _load_routing_module()

    # Minimal `train_cfg` shape: only the fields the helper reads.
    class _TrainCfg:
        bias_update_rate = 0.01
        bias_warmup_start = 0.0
        bias_warmup_steps = 0
        bias_update_zero_sum = True

    # `cfg` is the raw yaml dict; the helper consults
    # `_resolve_balancing_field(cfg, "bias_rate_*", rate)` for
    # per-projection overrides. Empty dict → resolver returns the
    # default `rate` for every projection.
    cfg: dict = {}

    # Build a tiny owner-bearing model for the deepseek_bias path.
    router = _make_router(num_experts=4)
    router.local_tokens_per_expert = torch.tensor([100.0, 1.0, 1.0, 1.0])

    class _Model:
        def __init__(self, gate, method):
            self._load_balancing_method = method
            self._gate = gate

        def get_all_balancing_owners(self):
            yield self._gate, "mlp"

    captured: dict = {"calls": [], "grad_enabled_at_call": []}

    real_update = routing.update_expert_biases

    def _spy_update_expert_biases(model, **kwargs):
        captured["calls"].append(kwargs)
        captured["grad_enabled_at_call"].append(torch.is_grad_enabled())
        return real_update(model, **kwargs)

    # Path A: method=deepseek_bias → helper SHOULD invoke
    # update_expert_biases under no_grad.
    routing.update_expert_biases = _spy_update_expert_biases
    try:
        model_deepseek = _Model(router, "deepseek_bias")
        # Pre-flight: helper is invoked from grad-enabled context (the
        # trainer's main step loop is grad-enabled — the helper is
        # responsible for the no_grad wrap).
        assert torch.is_grad_enabled()
        routing.trainer_post_optimizer_bias_update(
            model_deepseek, _TrainCfg(), cfg,
            distributed=False, global_step=0,
        )
    finally:
        routing.update_expert_biases = real_update

    assert len(captured["calls"]) == 1, (
        f"helper must invoke update_expert_biases exactly once for "
        f"deepseek_bias method; got {len(captured['calls'])} calls"
    )
    assert captured["grad_enabled_at_call"] == [False], (
        f"helper must invoke update_expert_biases under no_grad; got "
        f"grad_enabled={captured['grad_enabled_at_call']}"
    )
    # The call must have used `bias_rate=train_cfg.bias_update_rate`
    # (no warmup, since warmup_steps=0) and `zero_sum=True`.
    spy_kwargs = captured["calls"][0]
    assert spy_kwargs["bias_rate"] == _TrainCfg.bias_update_rate
    assert spy_kwargs["zero_sum"] is True
    assert spy_kwargs["distributed"] is False
    # Per-projection rates should default to the global rate.
    assert spy_kwargs["per_proj_rates"]["mlp"] == _TrainCfg.bias_update_rate

    # Path B: method=aux_loss → helper SHOULD no-op.
    captured = {"calls": [], "grad_enabled_at_call": []}
    routing.update_expert_biases = _spy_update_expert_biases
    try:
        model_aux = _Model(router, "aux_loss")
        routing.trainer_post_optimizer_bias_update(
            model_aux, _TrainCfg(), cfg,
            distributed=False, global_step=0,
        )
    finally:
        routing.update_expert_biases = real_update
    assert captured["calls"] == [], (
        f"helper must no-op for non-bias method aux_loss; "
        f"got {len(captured['calls'])} calls"
    )

    # Path C: method=none → helper SHOULD no-op.
    captured = {"calls": [], "grad_enabled_at_call": []}
    routing.update_expert_biases = _spy_update_expert_biases
    try:
        model_none = _Model(router, "none")
        routing.trainer_post_optimizer_bias_update(
            model_none, _TrainCfg(), cfg,
            distributed=False, global_step=0,
        )
    finally:
        routing.update_expert_biases = real_update
    assert captured["calls"] == []

    # Path E: model wrapped in an FSDP-style wrapper. The wrapper
    # itself does NOT expose `_load_balancing_method`; only the inner
    # `_fsdp_wrapped_module` does. The helper must call
    # `unwrap_model(...)` before reading the method attribute,
    # otherwise the helper's no-op gate misses non-bias methods on
    # wrapped models.
    captured = {"calls": [], "grad_enabled_at_call": []}

    class _FSDPWrapper:
        """Minimal FSDP-style wrapper: `unwrap_model(...)` checks
        `_fsdp_wrapped_module` (see src/training/distributed.py).
        The wrapper itself intentionally does NOT have a
        `_load_balancing_method` attribute — if `getattr(wrapper, ...)`
        returned the inner attribute by accident, the test would
        vacuously pass."""
        def __init__(self, inner):
            self._fsdp_wrapped_module = inner

    inner = _Model(router, "aux_loss")  # non-bias method
    wrapped = _FSDPWrapper(inner)

    # Preflight: the wrapper does not directly expose
    # `_load_balancing_method` (so a non-unwrap-aware getattr would
    # see `None` → method-allows-bias-update branch via the legacy
    # back-compat path → unwanted bias update fires).
    assert not hasattr(wrapped, "_load_balancing_method")

    routing.update_expert_biases = _spy_update_expert_biases
    try:
        routing.trainer_post_optimizer_bias_update(
            wrapped, _TrainCfg(), cfg,
            distributed=False, global_step=0,
        )
    finally:
        routing.update_expert_biases = real_update
    assert captured["calls"] == [], (
        "wrapper-aware no-op: helper must unwrap_model(...) before "
        "reading _load_balancing_method, so a wrapped model with "
        "non-bias method (here: aux_loss on the inner module) does "
        "NOT trigger update_expert_biases. Got "
        f"{len(captured['calls'])} calls."
    )

    # Path D: bias_update_rate=0 → helper SHOULD no-op even for
    # deepseek_bias (the bias rate is the kill switch).
    class _ZeroRateCfg:
        bias_update_rate = 0.0
        bias_warmup_start = 0.0
        bias_warmup_steps = 0
        bias_update_zero_sum = True

    captured = {"calls": [], "grad_enabled_at_call": []}
    routing.update_expert_biases = _spy_update_expert_biases
    try:
        model_deepseek_zero_rate = _Model(router, "deepseek_bias")
        routing.trainer_post_optimizer_bias_update(
            model_deepseek_zero_rate, _ZeroRateCfg(), cfg,
            distributed=False, global_step=0,
        )
    finally:
        routing.update_expert_biases = real_update
    assert captured["calls"] == []


# ──────────────────────────────────────────────────────────────────────
#  Misuse-negative test for update_expert_biases (Codex Round 13 Finding 2)
# ──────────────────────────────────────────────────────────────────────


def test_update_expert_biases_rejects_grad_enabled_call():
    """AC-6 misuse guard: calling `update_expert_biases` while
    `torch.is_grad_enabled()` is True (i.e. inside a forward / before
    backward / inside backward without grad disabled) is a programming
    error — the bias update reads `local_tokens_per_expert` and
    mutates `expert_bias`, both of which should happen post-step
    with no grad context.

    Round 14 added the precondition; Round 15 (Codex Round 14 Finding 1)
    moves it AFTER the method-dispatch check so non-bias methods
    still no-op cleanly under the default grad-enabled context. This
    test specifically covers `deepseek_bias` (the only method that
    actually performs a bias update); the
    `test_update_expert_biases_no_op_for_non_bias_methods` test
    locks the no-op contract for the other methods.
    """
    routing = _load_routing_module()

    class _ModelWithOwners:
        _load_balancing_method = "deepseek_bias"

        def get_all_balancing_owners(self):
            return iter(())

    model = _ModelWithOwners()

    # Default (grad-enabled): the precondition must trigger because
    # method=`deepseek_bias` would actually mutate bias buffers.
    assert torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="update_expert_biases"):
        routing.update_expert_biases(model, bias_rate=0.001, distributed=False)

    # Within `torch.no_grad()`, the call is allowed (this is the
    # production trainer path: post-step, after the optimizer has
    # consumed the gradients).
    with torch.no_grad():
        routing.update_expert_biases(model, bias_rate=0.001, distributed=False)


@pytest.mark.parametrize(
    "method", ["aux_loss", "seq_aux_loss", "quantile", "none"]
)
def test_update_expert_biases_no_op_for_non_bias_methods(method):
    """AC-1/AC-6 (Codex Round 14 Finding 1): non-bias methods must
    no-op cleanly under the default grad-enabled context — the
    method-dispatch gate runs BEFORE the no_grad misuse guard, so
    direct callers (tests, downstream tools) calling
    `update_expert_biases` for `aux_loss`/`seq_aux_loss`/`quantile`/
    `none` get the documented no-op semantics rather than a
    `RuntimeError`.

    The misuse guard exists to catch accidental misuse on
    `deepseek_bias` (the only currently-active bias method); it
    should NEVER fire for other methods because they don't
    walk owners or mutate buffers.
    """
    routing = _load_routing_module()
    router = _make_router(num_experts=4)

    class _ModelWithOwners:
        def __init__(self, gate, method):
            self._load_balancing_method = method
            self._gate = gate

        def get_all_balancing_owners(self):
            yield self._gate, "mlp"

    model = _ModelWithOwners(router, method)

    # Inject a non-zero count so a NON-no-op call would shift the bias.
    router.local_tokens_per_expert = torch.tensor([100.0, 1.0, 1.0, 1.0])
    initial_bias = router.expert_bias.detach().clone()
    initial_counts = router.local_tokens_per_expert.detach().clone()

    # Default grad-enabled context — must NOT raise; must NOT mutate.
    assert torch.is_grad_enabled()
    routing.update_expert_biases(model, bias_rate=0.01, distributed=False)

    torch.testing.assert_close(router.expert_bias, initial_bias)
    torch.testing.assert_close(router.local_tokens_per_expert, initial_counts)


# ──────────────────────────────────────────────────────────────────────
#  500-step synthetic convergence test (Codex Round 12 Finding 2)
# ──────────────────────────────────────────────────────────────────────


def test_zero_sum_bias_drives_load_to_uniform_within_500_steps():
    """AC-6 convergence: zero-sum bias updates drive a persistently-
    skewed routing distribution toward 5% of uniform within 500
    steps, while keeping `expert_bias.sum() ≈ 0` throughout.

    Round 14 (Codex Round 13 Finding 2): tightened to the 5%
    threshold via WINDOWED AVERAGING. A single step with discrete
    top-K and finite T has an irreducible binomial-sampling noise
    floor (~13% for `T*K=2048, E=8`); the time-average over the
    last `WINDOW` steps smooths the binomial noise to well below
    5% for the same fixture (window=200 steps × T*K=2048 ≈
    400k selections → noise floor ~1%). The contract is
    "the time-averaged routing distribution is uniform within 5%",
    which is what 'convergence to uniform' means in the
    DeepSeek-V3 paper.

    Synthetic simulation:
      - 8 experts, top_k=2, T tokens per step.
      - Per-step routing simulator: each token's "preference" for an
        expert is `prefs[i] + expert_bias[i] + per-step noise`. Top-K
        picks the K biggest. Underlying preferences are FIXED across
        steps; per-step gaussian noise breaks discrete-top-K
        bistability so convergence is smooth.
      - Track `expert_bias.sum()` (must stay near zero) and the
        time-averaged load over the last 200 steps (must be within
        5% of uniform per expert).
    """
    routing = _load_routing_module()
    torch.manual_seed(20260428)

    E = 8           # num_experts
    K = 2           # top_k
    T = 1024        # tokens per step
    BIAS_RATE = 0.005
    MAX_STEPS = 500
    WINDOW = 200    # time-average window for the convergence assertion
    TOLERANCE = 0.05  # 5% of uniform per the AC-6 contract

    # Persistent token preferences: skewed toward experts 0 and 1.
    base_prefs = torch.zeros(T, E)
    base_prefs[:, 0] = 0.5
    base_prefs[:, 1] = 0.4

    router = _make_router(num_experts=E)
    expert_bias_sum_history: list[float] = []
    # Per-step expert counts so we can compute the time-averaged load.
    counts_history: list[torch.Tensor] = []

    uniform_load = 1.0 / E
    for step in range(MAX_STEPS):
        noise = torch.randn(T, E) * 0.1
        biased = base_prefs + router.expert_bias.unsqueeze(0) + noise
        _, top_k_idx = torch.topk(biased, K, dim=-1)
        counts = torch.bincount(top_k_idx.reshape(-1), minlength=E).float()
        counts_history.append(counts.clone())

        router.local_tokens_per_expert = counts.clone()
        routing._update_single_router_bias(
            router, bias_rate=BIAS_RATE, distributed=False, zero_sum=True,
        )
        expert_bias_sum_history.append(router.expert_bias.sum().item())

    # 1) Zero-sum invariant: bias.sum() must stay near zero throughout.
    max_abs_sum = max(abs(s) for s in expert_bias_sum_history)
    assert max_abs_sum < 1e-4, (
        f"AC-6 zero-sum invariant violated: max |expert_bias.sum()| = "
        f"{max_abs_sum:.6e} over {MAX_STEPS} steps"
    )

    # 2) AC-6 5% convergence: the time-average of expert load over the
    # last `WINDOW` steps must be within 5% of uniform per expert.
    window_counts = torch.stack(counts_history[-WINDOW:]).sum(dim=0)
    window_loads = window_counts / window_counts.sum()
    max_load_dev = (window_loads - uniform_load).abs().max().item() / uniform_load
    assert max_load_dev < TOLERANCE, (
        f"AC-6 5% convergence violated: time-averaged load over last "
        f"{WINDOW} of {MAX_STEPS} steps deviates from uniform by "
        f"{max_load_dev:.4f} (tolerance {TOLERANCE}). "
        f"Per-expert loads: {window_loads.tolist()} (uniform={uniform_load:.4f})."
    )

    # Convergence direction sanity: the head window must have been
    # significantly worse — proves the bias update caused the
    # convergence rather than the system already being uniform.
    head_window = min(WINDOW, MAX_STEPS // 2)
    head_counts = torch.stack(counts_history[:head_window]).sum(dim=0)
    head_loads = head_counts / head_counts.sum()
    head_dev = (head_loads - uniform_load).abs().max().item() / uniform_load
    assert head_dev > 5 * TOLERANCE, (
        f"AC-6 convergence direction: head-window load deviation should "
        f"have been at least 5x the tolerance ({5*TOLERANCE}); got "
        f"{head_dev:.4f}. The fixture's overload pattern is too weak — "
        f"the test is degenerate."
    )

    # 3) Heavy experts (0 and 1) must have negative bias by the end;
    # light experts (2..7) must have positive bias on average.
    final_bias = router.expert_bias
    assert final_bias[0].item() < 0, (
        f"heavy expert 0 must have negative bias by step {MAX_STEPS}; "
        f"got {final_bias[0].item():.4f}"
    )
    assert final_bias[1].item() < 0, (
        f"heavy expert 1 must have negative bias; got {final_bias[1].item():.4f}"
    )
    assert final_bias[2:].mean().item() > 0, (
        f"light experts (2..7) must have positive bias on average by step {MAX_STEPS}; "
        f"got {final_bias[2:].tolist()}"
    )


if __name__ == "__main__":
    # Simple smoke runner; pytest is the canonical entry point.
    cases = [
        (0, 0.001, 0.0, 0, 0.001),
        (50, 0.001, 0.0, 100, 0.0005),
        (100, 0.001, 0.0, 100, 0.001),
        (50, 0.005, 0.001, 100, 0.003),
    ]
    for c in cases:
        test_get_bias_rate_warmup_ramp(*c)
    test_get_bias_rate_respects_model_override()
    test_bias_update_delta_is_zero_sum()
    test_expert_bias_survives_state_dict_round_trip()
    test_local_tokens_per_expert_is_non_persistent()
    test_distributed_path_calls_all_reduce_sum()
    test_bias_clamp_at_plus_16()
    test_local_tokens_zeroed_after_update()
    print("ALL OK")
