"""AC-10 router-exploration warmup integration.

The trainer can linearly ramp the effective router-exploration rate from
`router_exploration_warmup_start` to the model-configured
`router_exploration_rate` over `router_exploration_warmup_steps` steps. This
implements the "Early-step capacity warmup" technique listed as a medium-
priority follow-up in `docs/research/external_moe_techniques.md`.

The tests here pin:
  1. The schedule math (`exploration_rate_schedule`).
  2. The coverage of `apply_router_exploration_rate` — every router module
     with an `exploration_rate` attribute is updated.
  3. The `TrainingConfig` wiring through `build_training_config`.
  4. A no-op default: configs without warmup fields keep their current
     exploration rate across steps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.config import TrainingConfig, build_training_config
from src.training.routing import (
    apply_router_exploration_rate,
    exploration_rate_schedule,
)


# ── Schedule math ────────────────────────────────────────────────────────────


def test_schedule_noop_when_steps_zero():
    for step in [0, 1, 100, 10_000]:
        assert exploration_rate_schedule(step, target=0.02, warmup_start=0.5, warmup_steps=0) == 0.02


def test_schedule_linear_between_start_and_target():
    # Warmup from 0.5 → 0.02 over 100 steps.
    assert exploration_rate_schedule(0, target=0.02, warmup_start=0.5, warmup_steps=100) == 0.5
    mid = exploration_rate_schedule(50, target=0.02, warmup_start=0.5, warmup_steps=100)
    assert mid == pytest.approx(0.26, abs=1e-9), f"expected 0.26 at midpoint, got {mid}"
    assert exploration_rate_schedule(100, target=0.02, warmup_start=0.5, warmup_steps=100) == 0.02
    # Past warmup, stays at target.
    assert exploration_rate_schedule(500, target=0.02, warmup_start=0.5, warmup_steps=100) == 0.02


def test_schedule_ramp_up_from_zero_also_works():
    # Matches the "start low, ramp up to target" interpretation.
    assert exploration_rate_schedule(0, target=0.1, warmup_start=0.0, warmup_steps=100) == 0.0
    mid = exploration_rate_schedule(50, target=0.1, warmup_start=0.0, warmup_steps=100)
    assert mid == pytest.approx(0.05, abs=1e-9)
    assert exploration_rate_schedule(100, target=0.1, warmup_start=0.0, warmup_steps=100) == 0.1


# ── Applier coverage ─────────────────────────────────────────────────────────


class _FakeRouter(nn.Module):
    def __init__(self, rate: float = 0.0):
        super().__init__()
        self.exploration_rate = rate
        # Trainable param so the module is not pruned by eval().
        self.weight = nn.Parameter(torch.zeros(1))


class _ModelWithRouters(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = _FakeRouter(0.0)
        self.r2 = _FakeRouter(0.0)
        self.r3 = _FakeRouter(0.0)
        self.inert = nn.Linear(2, 2)  # no `exploration_rate`


def test_apply_updates_every_router_like_submodule():
    m = _ModelWithRouters()
    n = apply_router_exploration_rate(m, 0.125)
    assert n == 3
    assert m.r1.exploration_rate == 0.125
    assert m.r2.exploration_rate == 0.125
    assert m.r3.exploration_rate == 0.125
    assert not hasattr(m.inert, "exploration_rate")


def test_apply_is_idempotent_and_accepts_zero():
    m = _ModelWithRouters()
    apply_router_exploration_rate(m, 0.25)
    apply_router_exploration_rate(m, 0.25)  # second call does not drift
    assert m.r1.exploration_rate == 0.25
    apply_router_exploration_rate(m, 0.0)
    assert m.r1.exploration_rate == 0.0


def test_apply_works_through_ddp_style_wrapper():
    """Mimic the DDP wrapper shape (`.module` attribute on wrapped model)."""
    inner = _ModelWithRouters()

    class _Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    wrapper = _Wrapper(inner)
    n = apply_router_exploration_rate(wrapper, 0.3)
    assert n == 3
    assert inner.r1.exploration_rate == 0.3


# ── Config wiring ────────────────────────────────────────────────────────────


def test_training_config_defaults_preserve_noop_behaviour():
    tc = TrainingConfig()
    assert tc.router_exploration_warmup_start == 0.0
    assert tc.router_exploration_warmup_steps == 0


def test_build_training_config_reads_warmup_fields():
    raw_cfg = {
        "training": {
            "router_exploration_warmup_start": 0.4,
            "router_exploration_warmup_steps": 250,
        }
    }
    tc = build_training_config(raw_cfg)
    assert tc.router_exploration_warmup_start == 0.4
    assert tc.router_exploration_warmup_steps == 250


def test_build_training_config_absent_fields_are_zero():
    tc = build_training_config({"training": {}})
    assert tc.router_exploration_warmup_start == 0.0
    assert tc.router_exploration_warmup_steps == 0


# ── Integration: schedule + applier drives per-step value ────────────────────


def test_per_step_application_matches_schedule():
    m = _ModelWithRouters()
    target = 0.02
    warmup_start = 0.5
    warmup_steps = 10
    for step in range(0, warmup_steps + 3):
        rate = exploration_rate_schedule(step, target, warmup_start, warmup_steps)
        apply_router_exploration_rate(m, rate)
        assert m.r1.exploration_rate == pytest.approx(rate, abs=1e-9)
