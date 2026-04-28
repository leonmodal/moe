"""DEC-17 / task38: `softmax_position` rename + top-1 guard.

Per `docs/plan.md` DEC-17, the canonical name for the score-function vs
top-k ordering knob is `softmax_position` ∈ {pre_topk, post_topk}.
The legacy field `router_topk_ordering` ∈ {post, pre} is accepted as a
deprecated alias with a DeprecationWarning. The mapping is:

    softmax_position = "pre_topk"   ⟷ legacy router_topk_ordering = "post"
    softmax_position = "post_topk"  ⟷ legacy router_topk_ordering = "pre"

The top-1 guard rejects `softmax_position="post_topk"` with `top_k=1`
because softmax of a single selected logit produces a constant `1.0`
weight — destroying the gradient signal that would otherwise route
through the routing weight back to the gate.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.router import (
    ExplorationTopKRouter,
    _resolve_softmax_position,
    _validate_softmax_position_top1_guard,
)
from src.models.configuration_qwen3_moe import Qwen3MoeConfig


def _make_config(top_k: int = 2, **overrides) -> Qwen3MoeConfig:
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class _ConfigShim:
    """Lightweight config object for the resolver unit-tests."""
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────
#  _resolve_softmax_position
# ──────────────────────────────────────────────────────────────────────


def test_resolve_default_is_pre_topk():
    """When neither field is set, default is `pre_topk` — preserves the
    pre-DEC-17 default behaviour (softmax-then-topk)."""
    assert _resolve_softmax_position(_ConfigShim()) == "pre_topk"


@pytest.mark.parametrize("canonical", ["pre_topk", "post_topk"])
def test_resolve_canonical_no_warning(canonical):
    """Setting `softmax_position` directly produces no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _resolve_softmax_position(_ConfigShim(softmax_position=canonical)) == canonical


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("post", "pre_topk"),
        ("pre", "post_topk"),
    ],
)
def test_resolve_legacy_alias_with_deprecation_warning(legacy, expected):
    """Setting `router_topk_ordering` resolves to the canonical value
    AND emits a DeprecationWarning telling the user to migrate."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = _resolve_softmax_position(_ConfigShim(router_topk_ordering=legacy))
    assert result == expected
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1, (
        f"Expected one DeprecationWarning, got {len(deprecation_warnings)}: {caught}"
    )
    assert "router_topk_ordering" in str(deprecation_warnings[0].message)
    assert "softmax_position" in str(deprecation_warnings[0].message)


def test_resolve_canonical_overrides_consistent_legacy_silently():
    """If both fields are set with CONSISTENT values, prefer the canonical
    one and emit no warning (no conflict to warn about)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _resolve_softmax_position(
            _ConfigShim(softmax_position="pre_topk", router_topk_ordering="post"),
        ) == "pre_topk"


def test_resolve_canonical_overrides_inconsistent_legacy_with_warning():
    """If both fields are set with INCONSISTENT values, the canonical
    one wins AND a DeprecationWarning fires for the conflict."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = _resolve_softmax_position(
            _ConfigShim(softmax_position="post_topk", router_topk_ordering="post"),
        )
    assert result == "post_topk"
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    assert "DEC-17 conflict" in str(deprecation_warnings[0].message)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("softmax_position", "middle"),
        ("router_topk_ordering", "middle"),
    ],
)
def test_resolve_unknown_value_raises(field, bad_value):
    """Unknown values raise `ValueError` early (not at first forward)."""
    cfg = _ConfigShim(**{field: bad_value})
    with pytest.raises(ValueError, match=field):
        _resolve_softmax_position(cfg)


# ──────────────────────────────────────────────────────────────────────
#  _validate_softmax_position_top1_guard
# ──────────────────────────────────────────────────────────────────────


def test_top1_guard_rejects_post_topk_with_topk_1():
    """The top-1 guard must reject `post_topk` + `top_k=1` because
    softmax of a single selected logit yields a constant `1.0` weight,
    destroying the gradient signal."""
    with pytest.raises(ValueError, match="DEC-17 top-1 guard"):
        _validate_softmax_position_top1_guard("post_topk", top_k=1)


@pytest.mark.parametrize(
    "position,top_k",
    [
        ("pre_topk", 1),  # OK: softmax over all experts, then gather one — non-constant.
        ("pre_topk", 2),
        ("post_topk", 2),
        ("post_topk", 4),
    ],
)
def test_top1_guard_allows_safe_combinations(position, top_k):
    """Other combinations are allowed (no error)."""
    _validate_softmax_position_top1_guard(position, top_k=top_k)  # must not raise


# ──────────────────────────────────────────────────────────────────────
#  ExplorationTopKRouter integration
# ──────────────────────────────────────────────────────────────────────


def test_exploration_topk_router_constructs_with_canonical_field():
    """End-to-end: the canonical `softmax_position` field flows through
    the router constructor without warnings."""
    cfg = _make_config(top_k=2, softmax_position="post_topk")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        router = ExplorationTopKRouter(cfg)
    assert router.softmax_position == "post_topk"
    # Back-compat: legacy `topk_ordering` attribute still set for any
    # downstream caller that hasn't migrated.
    assert router.topk_ordering == "pre"


def test_exploration_topk_router_legacy_field_still_works_with_warning():
    """End-to-end: the legacy `router_topk_ordering` field still
    constructs the router but emits a DeprecationWarning."""
    cfg = _make_config(top_k=2, router_topk_ordering="post")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        router = ExplorationTopKRouter(cfg)
    assert router.softmax_position == "pre_topk"
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("router_topk_ordering" in str(w.message) for w in deprecation_warnings), (
        f"expected a DeprecationWarning for `router_topk_ordering`; got {caught}"
    )


def test_exploration_topk_router_top1_guard_in_constructor():
    """Constructor rejects `softmax_position=post_topk` + `top_k=1`."""
    cfg = _make_config(top_k=1, softmax_position="post_topk")
    with pytest.raises(ValueError, match="DEC-17 top-1 guard"):
        ExplorationTopKRouter(cfg)


def test_exploration_topk_router_top1_guard_via_legacy_alias():
    """Constructor still applies the top-1 guard when the user sets
    the legacy alias `router_topk_ordering=pre` (which maps to
    `softmax_position=post_topk`) with `top_k=1`."""
    cfg = _make_config(top_k=1, router_topk_ordering="pre")
    with pytest.raises(ValueError, match="DEC-17 top-1 guard"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ExplorationTopKRouter(cfg)


def test_exploration_topk_router_default_no_field_set():
    """No softmax_position and no router_topk_ordering → default is
    `pre_topk` (preserves pre-DEC-17 behaviour)."""
    cfg = _make_config(top_k=2)
    # Make sure neither attribute is set (some configs may default
    # router_topk_ordering="post"; clear if present).
    if hasattr(cfg, "router_topk_ordering"):
        delattr(cfg, "router_topk_ordering")
    if hasattr(cfg, "softmax_position"):
        delattr(cfg, "softmax_position")
    router = ExplorationTopKRouter(cfg)
    assert router.softmax_position == "pre_topk"


if __name__ == "__main__":
    test_resolve_default_is_pre_topk()
    for canon in ("pre_topk", "post_topk"):
        test_resolve_canonical_no_warning(canon)
    for legacy, expected in [("post", "pre_topk"), ("pre", "post_topk")]:
        test_resolve_legacy_alias_with_deprecation_warning(legacy, expected)
    test_resolve_canonical_overrides_consistent_legacy_silently()
    test_resolve_canonical_overrides_inconsistent_legacy_with_warning()
    for field, bad_value in [
        ("softmax_position", "middle"),
        ("router_topk_ordering", "middle"),
    ]:
        try:
            test_resolve_unknown_value_raises(field, bad_value)
        except Exception:
            pass
    test_top1_guard_rejects_post_topk_with_topk_1()
    for position, top_k in [
        ("pre_topk", 1), ("pre_topk", 2),
        ("post_topk", 2), ("post_topk", 4),
    ]:
        test_top1_guard_allows_safe_combinations(position, top_k)
    test_exploration_topk_router_constructs_with_canonical_field()
    test_exploration_topk_router_legacy_field_still_works_with_warning()
    test_exploration_topk_router_top1_guard_in_constructor()
    test_exploration_topk_router_top1_guard_via_legacy_alias()
    test_exploration_topk_router_default_no_field_set()
    print("ALL OK")
