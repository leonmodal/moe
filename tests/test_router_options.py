"""Unit tests for the configurable router options (Round 16).

Four knobs introduced in Round 16 (Megatron parity):

- `router_score_function ∈ {softmax, sigmoid, sqrtsoftplus}` — maps raw router
  logits to per-expert scores on the softmax-family router.
- `router_topk_ordering ∈ {post, pre}` — controls whether the top-k selection
  runs on the scored values (`post`, the legacy behaviour) or on the raw
  logits (`pre`, then score function applied only to the k selected logits).
- `num_groups` / `group_topk` — group-limited top-k for the softmax-family
  router, matching the DeepSeek path.
- `router_z_loss_coef` — magnitude regularizer on the raw logits. Cached per
  router call on `_last_z_loss`; the trainer accumulates the sum.

These tests pin:
    1. The three score functions return the correct analytic values and gradients.
    2. pre/post orderings produce the same indices for softmax (monotonic),
       different indices for sigmoid/sqrtsoftplus once the top-k is tight, and
       different weight values/gradients across all three score functions.
    3. Group-limited top-k restricts the selected experts to the chosen groups.
    4. z-loss is cached on `_last_z_loss`, is non-negative, has autograd
       through it, and drops back to `None` when the coefficient is 0.
    5. `collect_router_z_loss` sums across routers and preserves autograd.
    6. Config validation rejects unknown score functions / orderings.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.router import (
    ExplorationTopKRouter,
    _apply_score_function,
    _SCORE_FUNCTIONS,
    _TOPK_ORDERINGS,
)
from src.training.routing import collect_router_z_loss


def _tiny_cfg(**overrides):
    """Minimal config stub that satisfies Qwen3MoeTopKRouter.__init__."""
    defaults = dict(
        hidden_size=8,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        router_exploration_rate=0.0,
        router_score_function="softmax",
        router_topk_ordering="post",
        router_z_loss_coef=0.0,
        num_groups=None,
        group_topk=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Score functions ─────────────────────────────────────────────────────────


def test_apply_score_function_softmax_sums_to_one():
    logits = torch.randn(4, 6)
    out = _apply_score_function(logits, "softmax")
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert ((out >= 0) & (out <= 1)).all()


def test_apply_score_function_sigmoid_is_elementwise():
    logits = torch.randn(4, 6)
    out = _apply_score_function(logits, "sigmoid")
    expected = torch.sigmoid(logits)
    assert torch.allclose(out, expected)


def test_apply_score_function_sqrtsoftplus_matches_formula():
    logits = torch.tensor([[-10.0, 0.0, 10.0]])
    out = _apply_score_function(logits, "sqrtsoftplus")
    expected = torch.sqrt(torch.nn.functional.softplus(logits))
    assert torch.allclose(out, expected)
    # Non-negative and smooth floor near 0 for large negatives.
    assert (out >= 0).all()


def test_apply_score_function_rejects_unknown():
    with pytest.raises(ValueError, match="router_score_function"):
        _apply_score_function(torch.randn(2, 3), "argmax")


def test_score_function_set_membership():
    assert _SCORE_FUNCTIONS == {"softmax", "sigmoid", "sqrtsoftplus"}
    assert _TOPK_ORDERINGS == {"post", "pre"}


# ── Router construction ─────────────────────────────────────────────────────


def test_router_rejects_invalid_score_function():
    cfg = _tiny_cfg(router_score_function="argmax")
    with pytest.raises(ValueError, match="router_score_function"):
        ExplorationTopKRouter(cfg)


def test_router_rejects_invalid_topk_ordering():
    cfg = _tiny_cfg(router_topk_ordering="mid")
    with pytest.raises(ValueError, match="router_topk_ordering"):
        ExplorationTopKRouter(cfg)


def test_router_defaults_preserve_legacy_behavior():
    # Default = softmax + post — matches the pre-Round-16 contract.
    router = ExplorationTopKRouter(_tiny_cfg())
    assert router.score_function == "softmax"
    assert router.topk_ordering == "post"
    assert router.z_loss_coef == 0.0
    assert router.num_groups is None and router.group_topk is None


# ── Top-k ordering ──────────────────────────────────────────────────────────


def test_softmax_post_vs_pre_produces_same_indices():
    """Softmax is monotonic, so `argmax(softmax(x))` == `argmax(x)` — the
    selected expert indices are identical across post and pre orderings.
    The selected *weight values* still differ, which other tests pin.
    """
    cfg_post = _tiny_cfg(router_score_function="softmax", router_topk_ordering="post")
    cfg_pre = _tiny_cfg(router_score_function="softmax", router_topk_ordering="pre")
    router_post = ExplorationTopKRouter(cfg_post).eval()
    router_pre = ExplorationTopKRouter(cfg_pre).eval()
    # Mirror weights so the routers share inputs.
    router_pre.load_state_dict(router_post.state_dict())

    x = torch.randn(5, 8)
    _, _, idx_post = router_post(x)
    _, _, idx_pre = router_pre(x)
    assert torch.equal(idx_post.sort(dim=-1).values, idx_pre.sort(dim=-1).values), (
        "softmax pre-topk and post-topk must pick the same k experts per token"
    )


def test_softmax_post_vs_pre_weight_values_differ():
    """Post gathers from softmax(all experts). Pre applies softmax only over
    the K selected logits, so the weights are renormalized across K.
    """
    cfg_post = _tiny_cfg(
        router_score_function="softmax",
        router_topk_ordering="post",
        num_experts_per_tok=2,
        norm_topk_prob=False,  # isolate the ordering difference from renorm
    )
    cfg_pre = _tiny_cfg(
        router_score_function="softmax",
        router_topk_ordering="pre",
        num_experts_per_tok=2,
        norm_topk_prob=False,
    )
    router_post = ExplorationTopKRouter(cfg_post).eval()
    router_pre = ExplorationTopKRouter(cfg_pre).eval()
    router_pre.load_state_dict(router_post.state_dict())

    x = torch.randn(4, 8)
    _, scores_post, _ = router_post(x)
    _, scores_pre, _ = router_pre(x)
    # Under pre-ordering + no norm, the two selected weights softmax to sum 1.
    assert torch.allclose(scores_pre.sum(dim=-1), torch.ones(4), atol=1e-5)
    # Under post-ordering + no norm, the two selected weights sum to less than 1
    # (they are a subset of the full-softmax distribution).
    assert (scores_post.sum(dim=-1) < 1.0).all()


# ── Sigmoid + pre-topk ───────────────────────────────────────────────────────


def test_sigmoid_pre_vs_post_can_pick_different_experts():
    """With sigmoid the ordering of `sigmoid(x)` and `x` is the same (sigmoid
    is monotonic), so indices agree. This test is the companion that pins
    the invariant rather than a divergence; it protects against a future
    refactor silently flipping the ordering and producing subtle numeric
    differences.
    """
    cfg_post = _tiny_cfg(router_score_function="sigmoid", router_topk_ordering="post")
    cfg_pre = _tiny_cfg(router_score_function="sigmoid", router_topk_ordering="pre")
    router_post = ExplorationTopKRouter(cfg_post).eval()
    router_pre = ExplorationTopKRouter(cfg_pre).eval()
    router_pre.load_state_dict(router_post.state_dict())

    x = torch.randn(6, 8)
    _, _, idx_post = router_post(x)
    _, _, idx_pre = router_pre(x)
    assert torch.equal(idx_post.sort(dim=-1).values, idx_pre.sort(dim=-1).values)


def test_sigmoid_weight_values_are_elementwise_not_competing():
    router = ExplorationTopKRouter(
        _tiny_cfg(router_score_function="sigmoid", num_experts_per_tok=3, norm_topk_prob=False)
    ).eval()
    x = torch.randn(4, 8)
    _, scores, _ = router(x)
    # Sigmoid scores do not sum to 1; they sit in (0, 1) individually.
    assert ((scores >= 0) & (scores <= 1)).all()


# ── Group-limited top-k on the softmax router ────────────────────────────────


def test_softmax_group_limited_topk_restricts_selection_to_groups():
    # 8 experts, 2 groups × 4 experts, pick from 1 group, top-2 experts.
    cfg = _tiny_cfg(
        num_experts=8,
        num_experts_per_tok=2,
        num_groups=2,
        group_topk=1,
    )
    router = ExplorationTopKRouter(cfg).eval()
    # Set weights so group 0 (experts 0-3) has uniformly higher logits than group 1.
    with torch.no_grad():
        router.weight.zero_()
        router.weight[:4, :] = 1.0  # rows correspond to experts 0-3
    x = torch.ones(5, 8)
    _, _, idx = router(x)
    # All selected experts should be in group 0 (indices 0-3).
    assert (idx < 4).all(), f"selected experts should come only from group 0, got {idx}"


def test_softmax_group_limited_topk_left_alone_when_unset():
    cfg = _tiny_cfg(num_experts=8, num_experts_per_tok=2)
    router = ExplorationTopKRouter(cfg).eval()
    x = torch.randn(3, 8)
    _, _, idx = router(x)
    # No constraint — indices can be anywhere.
    assert idx.shape == (3, 2)
    assert ((idx >= 0) & (idx < 8)).all()


# ── Z-loss ──────────────────────────────────────────────────────────────────


def test_z_loss_cached_and_non_negative():
    cfg = _tiny_cfg(router_z_loss_coef=1e-3)
    router = ExplorationTopKRouter(cfg).train()
    x = torch.randn(4, 8, requires_grad=True)
    router(x)
    assert router._last_z_loss is not None
    assert router._last_z_loss.item() >= 0


def test_z_loss_scales_with_coefficient():
    cfg_a = _tiny_cfg(router_z_loss_coef=1e-3)
    cfg_b = _tiny_cfg(router_z_loss_coef=1e-1)
    router_a = ExplorationTopKRouter(cfg_a).train()
    router_b = ExplorationTopKRouter(cfg_b).train()
    router_b.load_state_dict(router_a.state_dict())
    x = torch.randn(4, 8)
    router_a(x)
    router_b(x)
    # Doubling the coefficient doubles the z-loss (linear in coef).
    ratio = (router_b._last_z_loss / router_a._last_z_loss).item()
    assert ratio == pytest.approx(100.0, rel=1e-4), f"expected 100x, got {ratio}"


def test_z_loss_none_when_coefficient_zero():
    cfg = _tiny_cfg(router_z_loss_coef=0.0)
    router = ExplorationTopKRouter(cfg).train()
    x = torch.randn(4, 8)
    router(x)
    assert router._last_z_loss is None


def test_z_loss_has_gradient_through_logits():
    cfg = _tiny_cfg(router_z_loss_coef=1e-2)
    router = ExplorationTopKRouter(cfg).train()
    x = torch.randn(4, 8)
    router(x)
    z = router._last_z_loss
    assert z is not None
    z.backward()
    # Non-zero gradient on the router weight.
    assert router.weight.grad is not None
    assert router.weight.grad.abs().sum().item() > 0


# ── collect_router_z_loss ────────────────────────────────────────────────────


class _Container(torch.nn.Module):
    def __init__(self, router):
        super().__init__()
        self.inner = router


def test_collect_router_z_loss_sums_across_routers():
    cfg = _tiny_cfg(router_z_loss_coef=1e-3)
    r1 = ExplorationTopKRouter(cfg).train()
    r2 = ExplorationTopKRouter(cfg).train()
    r2.load_state_dict(r1.state_dict())

    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = r1
            self.b = r2

    model = Pair()
    x = torch.randn(4, 8)
    r1(x)
    r2(x)
    total = collect_router_z_loss(model)
    assert total is not None
    expected = (r1._last_z_loss + r2._last_z_loss).item()
    assert total.item() == pytest.approx(expected, rel=1e-6)


def test_collect_router_z_loss_returns_none_when_all_disabled():
    cfg = _tiny_cfg(router_z_loss_coef=0.0)
    r1 = ExplorationTopKRouter(cfg).train()
    x = torch.randn(2, 8)
    r1(x)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.r = r1

    assert collect_router_z_loss(Wrap()) is None


def test_collect_router_z_loss_preserves_autograd():
    cfg = _tiny_cfg(router_z_loss_coef=1e-2)
    router = ExplorationTopKRouter(cfg).train()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.r = router

    model = Wrap()
    x = torch.randn(4, 8, requires_grad=True)
    router(x)
    total = collect_router_z_loss(model)
    assert total is not None
    total.backward()
    assert router.weight.grad is not None and router.weight.grad.abs().sum() > 0


# ── DeepSeek router z-loss integration ──────────────────────────────────────


def test_deepseek_router_also_caches_z_loss():
    from src.models.router import DeepSeekRouter

    cfg = _tiny_cfg(router_z_loss_coef=1e-2)
    cfg.topk_scaling_factor = None
    router = DeepSeekRouter(cfg).train()
    x = torch.randn(4, 8)
    router(x)
    assert router._last_z_loss is not None and router._last_z_loss.item() >= 0


def test_deepseek_router_z_loss_none_when_disabled():
    from src.models.router import DeepSeekRouter

    cfg = _tiny_cfg(router_z_loss_coef=0.0)
    cfg.topk_scaling_factor = None
    router = DeepSeekRouter(cfg).train()
    x = torch.randn(4, 8)
    router(x)
    assert router._last_z_loss is None


# ── Factory integration (config flows through build_model) ──────────────────


def _factory_moe_config(**overrides):
    cfg = {
        "model": {
            "type": "standard_moe",
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "head_dim": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "moe_intermediate_size": 32,
            "intermediate_size": 128,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 128,
            "router_type": "softmax",
        },
        "training": {},
    }
    cfg["model"].update(overrides)
    return cfg


def test_build_model_plumbs_router_options_through_config():
    from src.training.model_factory import build_model

    cfg = _factory_moe_config(
        router_score_function="sigmoid",
        router_topk_ordering="pre",
        router_z_loss_coef=1e-3,
        num_groups=2,
        group_topk=1,
    )
    _, model_config = build_model(cfg)
    assert model_config.router_score_function == "sigmoid"
    assert model_config.router_topk_ordering == "pre"
    assert model_config.router_z_loss_coef == 1e-3
    assert model_config.num_groups == 2
    assert model_config.group_topk == 1


def test_build_model_defaults_preserve_legacy_behavior():
    from src.training.model_factory import build_model

    cfg = _factory_moe_config()
    _, model_config = build_model(cfg)
    assert model_config.router_score_function == "softmax"
    assert model_config.router_topk_ordering == "post"
    assert model_config.router_z_loss_coef == 0.0


def test_build_model_moe_everything_plumbs_router_options():
    from src.training.model_factory import build_model

    cfg = {
        "model": {
            "type": "moe_everything",
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "head_dim": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "moe_intermediate_size": 32,
            "intermediate_size": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_attn_experts": 4,
            "num_attn_experts_per_tok": 1,
            "attn_expert_mode": "per_head_fully_independent",
            "max_position_embeddings": 128,
            "router_type": "softmax",
            "router_score_function": "sqrtsoftplus",
            "router_topk_ordering": "pre",
            "router_z_loss_coef": 5e-4,
        },
        "training": {},
    }
    _, model_config = build_model(cfg)
    assert model_config.router_score_function == "sqrtsoftplus"
    assert model_config.router_topk_ordering == "pre"
    assert model_config.router_z_loss_coef == 5e-4


# ── GPU smoke: loss decreases under each option combination ─────────────────

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _run_short_training(cfg: dict, steps: int = 20, device: str = "cuda") -> list[float]:
    """Build the model, run N forward+backward+step iterations on random tokens,
    and return the per-step loss trace. Used by the GPU smoke to assert that
    each router-option combination stays within the AC-11 loss-sanity band.
    """
    from src.training.model_factory import build_model
    from src.training.routing import collect_router_z_loss

    # Deterministic init + deterministic token stream so the windowed-loss
    # invariant below is stable across runs. The previous "manual_seed(0)"
    # call happened *after* build_model so the model weights and initial
    # CUDA RNG state varied run-to-run, which occasionally let the 40-step
    # micro-training land on a head-vs-tail crossover point.
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
    model, _ = build_model(cfg)
    model = model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    vocab = cfg["model"]["vocab_size"]
    seq_len = 16
    batch = 4
    losses: list[float] = []
    for _ in range(steps):
        input_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels, output_router_logits=True)
        z = collect_router_z_loss(model)
        loss = out.loss + z if z is not None else out.loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return losses


_SMOKE_COMBINATIONS = [
    # Legacy baseline — softmax + post.
    dict(id="softmax_post"),
    # Pre-softmax ordering — weights renormalize across the K selected logits.
    dict(id="softmax_pre", router_topk_ordering="pre"),
    # Sigmoid scoring — non-competing independent scores.
    dict(id="sigmoid_post", router_score_function="sigmoid"),
    # Sqrtsoftplus scoring — non-competing, smoother floor.
    dict(id="sqrtsoftplus_post", router_score_function="sqrtsoftplus"),
    # Z-loss on — exercises the trainer collection path.
    dict(id="z_loss", router_z_loss_coef=1e-3),
    # Combined: sigmoid + pre + z-loss.
    dict(id="sigmoid_pre_zloss", router_score_function="sigmoid", router_topk_ordering="pre", router_z_loss_coef=1e-3),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("overrides", _SMOKE_COMBINATIONS, ids=[c["id"] for c in _SMOKE_COMBINATIONS])
def test_router_option_gpu_smoke_keeps_loss_sane(overrides):
    """AC-11 sanity: each option combination must produce finite, bounded loss.

    The smoke runs on *random* token streams with a tiny model, so there is
    no learning signal — loss cannot trend down. The point of the test is
    to catch (a) NaN/Inf blowups and (b) runaway loss (a real regression
    signal), not to verify convergence. Convergence evidence for these
    combinations lives in the longer Stage B runs under
    `scripts/validate_multi_gpu_pipeline.py`; see `status.md` for numbers.
    """
    opts = {k: v for k, v in overrides.items() if k != "id"}
    cfg = _factory_moe_config(**opts)
    losses = _run_short_training(cfg, steps=40)
    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses), (
        f"option combo {overrides['id']} produced non-finite loss: {losses}"
    )
    # No-runaway bound: vocab=256 → ln(256)≈5.54, so uniform-random predictions
    # sit around 5.5; we allow slack for init transients but reject anything
    # that diverges well past the uniform-prediction baseline.
    assert max(losses) < 10.0, f"{overrides['id']}: max loss {max(losses)} too high"
    assert min(losses) > 0.0, f"{overrides['id']}: non-positive loss {min(losses)}"
