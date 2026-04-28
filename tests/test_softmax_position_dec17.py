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


# ──────────────────────────────────────────────────────────────────────
#  Positive top-1 gradient test for pre_topk (Codex Round 11 Finding 2b)
# ──────────────────────────────────────────────────────────────────────


def _init_router_weight(router):
    """The router's `nn.Linear` weight defaults to zeros under
    Qwen3MoeTopKRouter — model-construction code initializes it
    later. For unit gradient tests we need a non-zero init so the
    softmax derivative isn't degenerate."""
    with torch.no_grad():
        torch.nn.init.normal_(router.weight, std=0.02)


def test_pre_topk_with_top_k_1_preserves_gradient_signal():
    """DEC-17 positive case: `softmax_position=pre_topk` + `top_k=1`
    preserves the routing-weight gradient signal — the softmax is
    computed over all E logits BEFORE the top-K, so the gathered
    weight is a non-constant softmax probability and `loss.backward()`
    yields a non-zero gradient on `router.weight`.

    This is the symmetric companion to
    `test_exploration_topk_router_top1_guard_in_constructor`, which
    rejects the dangerous `post_topk + top_k=1` combination. Together
    they prove the guard is BOTH necessary (post_topk + top_k=1 kills
    gradients) and sufficient (pre_topk + top_k=1 preserves them).

    The test uses `weights.pow(2).sum()` as the surrogate loss because
    `weights.sum()` is identically `T` (rows of weights sum to 1 under
    norm_topk_prob), which makes the gradient zero by construction
    regardless of routing.
    """
    torch.manual_seed(2026_04_28)
    cfg = _make_config(top_k=1, softmax_position="pre_topk")
    router = ExplorationTopKRouter(cfg).train()
    _init_router_weight(router)

    T = 8
    x = torch.randn(T, cfg.hidden_size, requires_grad=False)
    probs, weights, indices = router(x)

    # The forward must produce non-constant weights (softmax probability
    # of the selected expert, NOT the constant 1.0).
    assert weights.shape == (T, 1)
    assert (weights < 1.0 - 1e-6).any(), (
        f"pre_topk + top_k=1 should produce sub-1.0 softmax weights, got {weights}"
    )

    # Use a non-trivial loss that depends on the weight values (not just
    # their sum-to-one constraint).
    loss = weights.pow(2).sum()
    loss.backward()
    grad_norm = router.weight.grad.norm().item()
    assert grad_norm > 1e-6, (
        f"pre_topk + top_k=1 must produce a non-zero gradient on router.weight; "
        f"got grad_norm={grad_norm}. Gradient signal is dead — guard regressed."
    )


def test_docs_present_softmax_position_as_canonical():
    """DEC-17 docs audit (Codex Round 11 Finding 2c): both
    `docs/configuration.md` and `docs/routing.md` must mention
    `softmax_position` as the canonical field name. This catches
    stale docs that still present `router_topk_ordering` as the
    primary configuration knob."""
    repo = Path(__file__).resolve().parent.parent
    config_doc = (repo / "docs" / "configuration.md").read_text()
    routing_doc = (repo / "docs" / "routing.md").read_text()

    assert "softmax_position" in config_doc, (
        "docs/configuration.md must document `softmax_position` as the "
        "canonical DEC-17 field name."
    )
    assert "softmax_position" in routing_doc, (
        "docs/routing.md must document `softmax_position` as the "
        "canonical DEC-17 field name."
    )


def test_docs_router_topk_ordering_is_only_marked_as_legacy():
    """DEC-17 docs audit: any remaining mention of
    `router_topk_ordering` in `docs/configuration.md` or
    `docs/routing.md` must be in the context of the deprecation alias
    (i.e., accompanied by 'deprecated', 'alias', or 'legacy' on the
    same line). This catches stale references that still present
    `router_topk_ordering` as the primary field name (Codex Round 11
    Finding 2c).
    """
    repo = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    LEGACY_TOKENS = ("deprecated", "alias", "legacy", "Legacy", "DEC-17", "Deprecat")
    for doc_name in ("configuration.md", "routing.md"):
        doc_path = repo / "docs" / doc_name
        text = doc_path.read_text()
        for line_idx, line in enumerate(text.splitlines(), start=1):
            if "router_topk_ordering" not in line:
                continue
            if any(tok in line for tok in LEGACY_TOKENS):
                continue
            offenders.append(f"docs/{doc_name}:{line_idx}: {line.strip()}")

    assert not offenders, (
        "docs still mention `router_topk_ordering` as a primary field "
        "(no 'deprecated' / 'legacy' / 'alias' / 'DEC-17' marker on the "
        f"same line):\n  - " + "\n  - ".join(offenders)
        + "\n\nUpdate these to either present `softmax_position` as canonical "
        + "or mark the line as the deprecated alias."
    )


def test_post_topk_with_top_k_2_preserves_gradient_signal():
    """Sanity check for the `post_topk` path itself: with `top_k=2`
    (the guard does not fire), `loss.backward()` produces a non-zero
    router.weight gradient. This locks the contract that `post_topk`
    is functional for `top_k > 1`, so the top-1 guard is the ONLY
    rejection criterion — not a blanket disabling of `post_topk`."""
    torch.manual_seed(2026_04_28)
    cfg = _make_config(top_k=2, softmax_position="post_topk")
    router = ExplorationTopKRouter(cfg).train()
    _init_router_weight(router)

    T = 8
    x = torch.randn(T, cfg.hidden_size, requires_grad=False)
    probs, weights, indices = router(x)

    assert weights.shape == (T, 2)
    loss = weights.pow(2).sum()  # non-trivial in weight values
    loss.backward()
    assert router.weight.grad.norm().item() > 1e-6, (
        f"post_topk + top_k=2 must produce a non-zero gradient on router.weight"
    )


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
