"""Pure-function tests for `update_bias_from_quantile(...)` (the
foundation of AC-10/11/12).

The full AC-10 runtime contract has more pieces (per-bank state,
DDP `all_gather`, EMA persistence, resume parity). This file pins
the pure update function alone — the layer that every higher-level
piece sits on top of.
"""
from __future__ import annotations

import pytest
import torch

from src.models.routing.bias import update_bias_from_quantile


def test_update_bias_from_quantile_basic_shapes():
    """The update mutates `bias` and `ema` in place; shapes
    are preserved."""
    n_experts = 4
    bias = torch.zeros(n_experts)
    ema = torch.zeros(n_experts)
    scores = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [1.5, 2.5, 3.5, 4.5],
         [2.0, 3.0, 4.0, 5.0]]
    )
    update_bias_from_quantile(
        bias, ema, scores, target_q=0.5, eta=0.5,
    )
    assert bias.shape == (n_experts,)
    assert ema.shape == (n_experts,)
    # ema should track the column-wise medians (1.5, 2.5, 3.5, 4.5)
    # scaled by eta=0.5 from initial ema=0.0:
    # ema = 0.5 * 0 + 0.5 * [1.5, 2.5, 3.5, 4.5] = [0.75, 1.25, 1.75, 2.25]
    expected_ema = 0.5 * torch.tensor([1.5, 2.5, 3.5, 4.5])
    assert torch.allclose(ema, expected_ema, atol=1e-6)


def test_update_bias_from_quantile_pulls_low_ema_experts_up():
    """A heavy expert (high quantile) gets a negative bias; a
    light expert (low quantile) gets a positive bias."""
    bias = torch.zeros(4)
    ema = torch.tensor([0.1, 0.2, 0.5, 0.9])  # expert 3 heavy
    # Synthetic scores — but we want to test the bias derivation
    # specifically. Use minimal observation that doesn't move the
    # ema much: zero rows shouldn't trigger the no-op branch.
    scores = torch.tensor([[0.1, 0.2, 0.5, 0.9]])  # one row
    update_bias_from_quantile(
        bias, ema, scores, target_q=0.5, eta=0.0001,
    )
    # Bias is clamp(ema.median() - ema, -16, 16). With ema ≈
    # [0.1, 0.2, 0.5, 0.9], median = 0.35.
    # bias[0] = 0.35 - 0.1 = +0.25 (light expert gets boost)
    # bias[3] = 0.35 - 0.9 = -0.55 (heavy expert gets cut)
    assert bias[0] > 0
    assert bias[3] < 0
    assert bias[3] < bias[0]


def test_update_bias_from_quantile_clamps_to_range():
    """Bias values larger than `clamp_range` are clamped."""
    bias = torch.zeros(4)
    # Construct an EMA spread that would push bias outside ±16.
    ema = torch.tensor([0.0, 0.0, 0.0, 100.0])
    scores = torch.tensor([[0.0, 0.0, 0.0, 100.0]])
    update_bias_from_quantile(
        bias, ema, scores, target_q=0.5, eta=0.0001,
    )
    # bias[3] = clamp(median - 100, ...) — median is small, so
    # bias[3] ≈ -100, clamped to -16.
    assert bias[3] == pytest.approx(-16.0, abs=1e-6)
    # bias[0..2] = clamp(median - 0, -16, 16). median ≈ 0, so
    # bias is near zero (well within clamp).
    for i in range(3):
        assert -16.0 <= bias[i] <= 16.0


def test_update_bias_from_quantile_no_op_on_empty_scores():
    """Empty `scores` (e.g. an entirely-masked-out training step)
    is a no-op: bias and ema unchanged."""
    bias = torch.tensor([0.1, 0.2, 0.3])
    ema = torch.tensor([0.5, 0.6, 0.7])
    bias_clone = bias.clone()
    ema_clone = ema.clone()
    update_bias_from_quantile(
        bias, ema, scores=torch.empty(0, 3), target_q=0.5, eta=0.5,
    )
    assert torch.equal(bias, bias_clone)
    assert torch.equal(ema, ema_clone)


def test_update_bias_from_quantile_rejects_invalid_target_q():
    bias = torch.zeros(2)
    ema = torch.zeros(2)
    scores = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="target_q"):
        update_bias_from_quantile(bias, ema, scores, target_q=1.5, eta=0.1)
    with pytest.raises(ValueError, match="target_q"):
        update_bias_from_quantile(bias, ema, scores, target_q=-0.1, eta=0.1)


def test_update_bias_from_quantile_rejects_invalid_eta():
    bias = torch.zeros(2)
    ema = torch.zeros(2)
    scores = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="eta"):
        update_bias_from_quantile(bias, ema, scores, target_q=0.5, eta=0.0)
    with pytest.raises(ValueError, match="eta"):
        update_bias_from_quantile(bias, ema, scores, target_q=0.5, eta=1.5)


def test_update_bias_from_quantile_ema_decay_over_iterations():
    """Repeatedly calling the update with different score
    distributions should track the moving quantile via the EMA."""
    bias = torch.zeros(2)
    ema = torch.zeros(2)
    eta = 0.5

    # Iteration 1: scores favor expert 0.
    scores = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    update_bias_from_quantile(bias, ema, scores, target_q=0.5, eta=eta)
    ema_after_1 = ema.clone()
    assert ema[0] > ema[1]  # expert 0 has higher quantile

    # Iteration 2: scores favor expert 1.
    scores = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    update_bias_from_quantile(bias, ema, scores, target_q=0.5, eta=eta)
    # The EMA should now have moved partway toward the new distribution.
    # ema_after_2[1] should be > ema_after_1[1] because expert 1's
    # observed quantile jumped from 0 to 2.
    assert ema[1] > ema_after_1[1]


if __name__ == "__main__":
    print("Run via pytest")
