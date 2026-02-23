"""
Tests for routing statistics computation.
"""
import math
import pytest
import torch
from src.utils.routing_stats import compute_routing_stats


def _uniform_logits(T=32, E=8, K=2):
    """All experts equally likely — perfectly balanced."""
    probs = torch.full((T, E), 1.0 / E)
    return [probs]


def _collapsed_logits(T=32, E=8):
    """All tokens route to expert 0."""
    probs = torch.zeros(T, E)
    probs[:, 0] = 1.0
    return [probs]


def _two_layer_logits(T=32, E=8):
    """Two layers with identical uniform routing."""
    probs = torch.full((T, E), 1.0 / E)
    return [probs, probs]


# ── load_imbalance ───────────────────────────────────────────────────────────

def test_uniform_load_imbalance_is_one():
    stats = compute_routing_stats(_uniform_logits(T=64, E=8, K=2), num_experts_per_tok=2)
    assert abs(stats["routing/layer_00_load_imbalance"] - 1.0) < 0.05


def test_collapsed_load_imbalance_is_high():
    stats = compute_routing_stats(_collapsed_logits(T=32, E=8), num_experts_per_tok=1)
    # All tokens hit expert 0 → imbalance = E
    assert stats["routing/layer_00_load_imbalance"] == pytest.approx(8.0)


# ── load_cv ──────────────────────────────────────────────────────────────────

def test_uniform_load_cv_is_zero():
    stats = compute_routing_stats(_uniform_logits(T=64, E=8, K=2), num_experts_per_tok=2)
    assert stats["routing/layer_00_load_cv"] < 0.1


def test_collapsed_load_cv_is_high():
    stats = compute_routing_stats(_collapsed_logits(T=32, E=8), num_experts_per_tok=1)
    assert stats["routing/layer_00_load_cv"] > 1.0


# ── utilization ──────────────────────────────────────────────────────────────

def test_collapsed_utilization_is_low():
    stats = compute_routing_stats(_collapsed_logits(T=32, E=8), num_experts_per_tok=1)
    # Only 1 of 8 experts used → 0.125
    assert stats["routing/layer_00_utilization"] == pytest.approx(1 / 8)


def test_uniform_utilization_is_one():
    # With enough tokens and uniform probs, all experts get hit
    stats = compute_routing_stats(_uniform_logits(T=128, E=4, K=2), num_experts_per_tok=2)
    assert stats["routing/layer_00_utilization"] == pytest.approx(1.0)


# ── entropy ──────────────────────────────────────────────────────────────────

def test_uniform_entropy_is_one():
    stats = compute_routing_stats(_uniform_logits(T=32, E=8, K=2), num_experts_per_tok=2)
    assert abs(stats["routing/layer_00_entropy"] - 1.0) < 1e-4


def test_collapsed_entropy_is_zero():
    stats = compute_routing_stats(_collapsed_logits(T=32, E=8), num_experts_per_tok=1)
    assert stats["routing/layer_00_entropy"] < 0.01


# ── histogram data ───────────────────────────────────────────────────────────

def test_expert_load_frac_shape():
    E = 8
    stats = compute_routing_stats(_uniform_logits(T=32, E=E, K=2), num_experts_per_tok=2)
    load_frac = stats["_hist/routing/layer_00_expert_load_frac"]
    assert len(load_frac) == E


def test_expert_load_frac_sums_to_one():
    stats = compute_routing_stats(_uniform_logits(T=32, E=8, K=2), num_experts_per_tok=2)
    load_frac = stats["_hist/routing/layer_00_expert_load_frac"]
    assert abs(sum(load_frac) - 1.0) < 1e-5


# ── multi-layer ───────────────────────────────────────────────────────────────

def test_multi_layer_has_entry_per_layer():
    logits = _two_layer_logits(T=32, E=8)
    stats = compute_routing_stats(logits, num_experts_per_tok=2)
    assert "routing/layer_00_load_imbalance" in stats
    assert "routing/layer_01_load_imbalance" in stats


# ── global MoE cross-layer metrics ───────────────────────────────────────────

def test_cross_layer_sim_identical_layers_is_one():
    """Two layers with identical routing → cosine sim = 1."""
    logits = _two_layer_logits(T=32, E=8)
    stats = compute_routing_stats(logits, num_experts_per_tok=2, is_global=True)
    assert abs(stats["routing/cross_layer_sim_mean"] - 1.0) < 1e-4


def test_cross_layer_sim_orthogonal_layers():
    """Two layers routing to disjoint halves of experts → sim ~ 0."""
    T, E = 32, 8
    p1 = torch.zeros(T, E); p1[:, :4] = 0.25   # only experts 0-3
    p2 = torch.zeros(T, E); p2[:, 4:] = 0.25   # only experts 4-7
    stats = compute_routing_stats([p1, p2], num_experts_per_tok=2, is_global=True)
    assert stats["routing/cross_layer_sim_mean"] < 0.1


def test_global_metrics_not_computed_for_standard():
    """is_global=False should not produce cross-layer metrics."""
    stats = compute_routing_stats(_two_layer_logits(), num_experts_per_tok=2, is_global=False)
    assert "routing/cross_layer_sim_mean" not in stats
    assert "routing/expert_depth_coverage_mean" not in stats


def test_depth_coverage_keys_present_for_global():
    logits = _two_layer_logits(T=32, E=8)
    stats = compute_routing_stats(logits, num_experts_per_tok=2, is_global=True)
    assert "routing/expert_depth_coverage_mean" in stats
    assert "routing/expert_depth_coverage_max" in stats
    assert "_hist/expert_depth_coverage" in stats
    assert len(stats["_hist/expert_depth_coverage"]) == 8
