"""Regression test for the per-step exploration-rate apply cache.

`apply_router_exploration_rate` walks `model.modules()` and sets the
`exploration_rate` attribute on every non-BranchRouter module with that
attribute. Before Round 14 the trainer called that helper on every
optimizer step — even after the warmup schedule converged and the rate
stayed constant — turning a one-time config into permanent per-step
Python overhead on large MoE models.

Round 14 added a `last_applied_exploration_rate` cache: the trainer only
calls the applier when the scheduled rate differs from the previous step.
This test pins that behaviour via source-level inspection (the cached
path lives inside `run_training`'s training loop and isn't practical to
unit-test end-to-end without a live model, but the source-level pin
surfaces if a future edit drops the cache).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import run_training


def _trainer_source() -> str:
    return inspect.getsource(run_training)


def test_trainer_caches_last_applied_exploration_rate():
    src = _trainer_source()
    assert "last_applied_exploration_rate" in src, (
        "Trainer must cache the last applied exploration rate so the "
        "module-tree walk is skipped on steps where the schedule has not "
        "advanced. Drop the cache and this test should fail."
    )


def test_trainer_skips_apply_when_rate_is_unchanged():
    src = _trainer_source()
    # The cache guard must compare against the last applied value. We look
    # for the key comparison: `if current_rate != last_applied_exploration_rate:`
    # (exact wording can evolve, but the != last_applied check must exist).
    assert "current_rate != last_applied_exploration_rate" in src, (
        "Trainer must guard `apply_router_exploration_rate(...)` behind "
        "`current_rate != last_applied_exploration_rate` so unchanged rates "
        "do not trigger the full module-tree walk each step."
    )
    # And must update the cache after applying.
    assert "last_applied_exploration_rate = current_rate" in src, (
        "Trainer must update `last_applied_exploration_rate` after each "
        "apply, otherwise the guard would re-trigger every step."
    )


def test_trainer_still_applies_first_step_via_none_sentinel():
    src = _trainer_source()
    # The cache is initialized to None before the loop so the very first
    # step always applies (None != any float).
    assert "last_applied_exploration_rate: float | None = None" in src, (
        "Cache must initialise to `None` so the very first step still "
        "applies (None != any float)."
    )


def test_trainer_short_circuits_schedule_past_warmup():
    """After warmup converges the trainer must stop evaluating the
    schedule and stop walking `model.modules()`. The plateau guard
    (`exploration_plateau_applied`) is what makes the cache actually
    effective — float inequality during warmup produced a distinct
    rate every step, so the earlier `!=` check always fired.
    """
    src = _trainer_source()
    assert "exploration_plateau_applied" in src, (
        "Trainer must maintain a plateau flag that is set once the schedule "
        "has converged; without it every warmup step recomputes the rate "
        "and re-calls the module-tree walk because `frac = step/warmup` "
        "produces a fresh float each step."
    )
    assert "exploration_plateau_applied = True" in src, (
        "Trainer must set the plateau flag once past warmup so subsequent "
        "steps skip the schedule entirely."
    )
