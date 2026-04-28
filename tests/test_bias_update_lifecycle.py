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


def _ddp_worker(rank: int, world_size: int, init_file: str, output_path: str):
    """Worker entry point for the 2-rank DDP all-reduce test.

    Uses a file-store rendezvous (`init_method=file://...`) instead of
    a TCP/localhost rendezvous so the test runs in sandboxed
    environments that can't bind 127.0.0.1 (Codex Round 11 Finding 1a).
    The init file is created in `tmp_path` and cleaned up by pytest.
    """
    import torch as _torch
    import torch.distributed as _dist

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

    Round 12 (Codex Round 11 Finding 1a): use a `file://` rendezvous
    so this test runs in sandboxes that can't bind localhost. Skip
    cleanly if mp.spawn fails for environment reasons (e.g. some CI
    runners don't allow process forking).
    """
    import torch.multiprocessing as mp

    init_file = tmp_path / "ddp_init"
    output_template = str(tmp_path / "rank_{rank}_bias.pt")

    try:
        mp.spawn(
            _ddp_worker,
            args=(2, str(init_file), output_template),
            nprocs=2,
            join=True,
        )
    except (RuntimeError, OSError, PermissionError) as exc:
        # Some sandboxes block multiprocessing or process-group setup
        # entirely. The mocked test
        # (test_distributed_path_calls_all_reduce_sum) still locks the
        # call signature; this one verifies the cross-rank semantics
        # opportunistically.
        pytest.skip(f"DDP environment unavailable: {exc!r}")

    rank0_bias = torch.load(output_template.format(rank=0))
    rank1_bias = torch.load(output_template.format(rank=1))

    torch.testing.assert_close(rank0_bias, rank1_bias, atol=1e-7, rtol=1e-7)

    # Sanity: the expected aggregate is `[10, 0, 0, 10]`, so loads are
    # `[0.5, 0, 0, 0.5]` and the zero-sum update produces a non-trivial
    # symmetric delta. Bias should be non-zero somewhere.
    assert rank0_bias.abs().sum().item() > 0, (
        "post-update bias is all zeros — all-reduce probably did not run"
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
