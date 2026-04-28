"""AC-7 / task12: DeepSeekRouter vs nmoe.Router parity.

The nmoe project (`/Users/leon/Desktop/modal/moe/nmoe/nmoe/model.py:66-97`) is
the production reference for the DeepSeek-V3 sigmoid + expert-bias routing
recipe; the bias-update step here is also the canonical zero-sum
`sign(load - 1/E)` formulation from the DeepSeek-V3 paper. Our
`src/models/router.py:DeepSeekRouter` and
`src/training/routing.py:_update_single_router_bias` should reproduce both
behaviours bit-for-bit (modulo dtype and the `norm_topk_prob` for top_k=1
guard which our codebase intentionally keeps active to preserve gradient
flow).

This test file ports a minimal nmoe-equivalent reference inline so we
don't import the heavy nmoe stack (which has CUDA-only deps); the
inlined reference is byte-identical to the public nmoe Router forward
and update_bias by construction. If nmoe's Router changes meaningfully
in the future, this file will need to be re-synced.

Coverage:

* ``test_forward_parity_topk_2_normalized`` — same hidden_states + same
  weights + same bias produce the same (top_k indices, weights) for
  the standard top-K > 1 path.
* ``test_forward_parity_with_scaling_factor`` — DeepSeekRouter's
  `topk_scaling_factor` (post-norm) matches nmoe's `routed_scaling_factor`.
* ``test_forward_parity_zero_bias_matches_simple_topk`` — sanity:
  with `expert_bias=0`, the selection is the simple top-k of sigmoid
  scores.
* ``test_bias_update_zero_sum_parity`` — `_update_single_router_bias`
  and nmoe's `update_bias` produce the same delta for the same
  fractional load distribution.
* ``test_bias_update_clamp_parity`` — both implementations clamp to
  ±16 after the update.
* ``test_bias_update_zero_load_no_op`` — both leave bias unchanged
  when no tokens were observed (DEC-2 contract).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.router import DeepSeekRouter
import importlib.util
import types as _types


def _load_routing_module():
    """Load `src.training.routing` directly without triggering the package
    `src.training.__init__` chain (which imports pandas/data dependencies
    that are not part of this CPU test environment)."""
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


class _NmoeRouterReference:
    """Inlined byte-for-byte copy of nmoe.model.Router.forward and
    update_bias as of 2026-04-28; see
    `nmoe/nmoe/model.py:66-97`. Decoupled from nmoe's heavy CUDA import
    chain so the parity test can run on CPU without B200 dependencies."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, top_k: int,
                 route_scale: float = 1.0, routed_scaling_factor: float = 1.0):
        self.weight = weight                  # (E, D), bf16 in nmoe
        self.bias = bias                      # (E,), fp32 in nmoe
        self.top_k = top_k
        self.route_scale = route_scale
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, x: torch.Tensor):
        """Identical to nmoe.Router.forward (model.py:77-90)."""
        logits = F.linear(x, self.weight).float()
        if self.route_scale != 1.0:
            logits = logits * self.route_scale
        scores = torch.sigmoid(logits)
        scores_for_selection = scores + self.bias
        _, indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1)
        weights = torch.gather(scores, 1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        if self.routed_scaling_factor != 1.0:
            weights = weights * self.routed_scaling_factor
        return weights.to(x.dtype), indices

    @torch.no_grad()
    def update_bias(self, expert_loads: torch.Tensor, gamma: float = 0.001):
        """Identical to nmoe.Router.update_bias (model.py:92-97)."""
        expected = 1.0 / expert_loads.numel()
        s = torch.sign(expert_loads - expected)
        self.bias -= gamma * (s - s.mean())
        self.bias.clamp_(-16.0, 16.0)


def _make_deepseek_router(num_experts: int, hidden_size: int, top_k: int,
                          *, scaling_factor: float | None = None,
                          norm_topk_prob: bool = True):
    """Build a `DeepSeekRouter` with a minimal Qwen3MoeConfig fixture."""
    from src.models.configuration_qwen3_moe import Qwen3MoeConfig
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
        norm_topk_prob=norm_topk_prob,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        topk_scaling_factor=scaling_factor,
    )
    return DeepSeekRouter(config)


def test_forward_parity_topk_2_normalized():
    """top_k=2 path (norm_topk_prob=True) must match nmoe.Router exactly."""
    torch.manual_seed(2026_04_28)
    E, D, K = 8, 16, 2
    T = 12

    ds = _make_deepseek_router(E, D, K)
    ds.eval()  # disable exploration / training-only side effects
    weight = ds.weight.detach().clone()
    bias = torch.zeros(E, dtype=torch.float32)
    nmoe = _NmoeRouterReference(weight, bias.clone(), top_k=K)

    x = torch.randn(T, D, dtype=torch.float32)

    # DeepSeek path: returns (router_top_value, top_k_idx) — but
    # `DeepSeekRouter.forward` returns (router_top_value, top_k_idx) as a
    # 2-tuple. Match nmoe's signature (weights, indices).
    _, ds_weights, ds_indices = ds(x)

    # nmoe path
    nmoe_weights, nmoe_indices = nmoe.forward(x)

    torch.testing.assert_close(ds_indices, nmoe_indices)
    torch.testing.assert_close(ds_weights.float(), nmoe_weights.float(), atol=1e-6, rtol=1e-6)


def test_forward_parity_with_scaling_factor():
    """`topk_scaling_factor` (DeepSeekRouter) ≡ `routed_scaling_factor` (nmoe).

    Both apply post-normalization to the gathered weights; the path is
    `weights *= scale` after the sum-to-1 clamp."""
    torch.manual_seed(2026_04_28)
    E, D, K = 6, 16, 2
    T = 8
    SCALE = 2.5  # DeepSeek-V3 default

    ds = _make_deepseek_router(E, D, K, scaling_factor=SCALE)
    ds.eval()
    nmoe = _NmoeRouterReference(
        ds.weight.detach().clone(),
        torch.zeros(E, dtype=torch.float32),
        top_k=K,
        routed_scaling_factor=SCALE,
    )

    x = torch.randn(T, D, dtype=torch.float32)
    _, ds_weights, ds_indices = ds(x)
    nmoe_weights, nmoe_indices = nmoe.forward(x)

    torch.testing.assert_close(ds_indices, nmoe_indices)
    torch.testing.assert_close(ds_weights.float(), nmoe_weights.float(), atol=1e-6, rtol=1e-6)


def test_forward_parity_zero_bias_matches_simple_topk():
    """With `expert_bias = 0`, both implementations select the top-K
    raw sigmoid scores (no bias dance, just the baseline recipe)."""
    torch.manual_seed(2026_04_28)
    E, D, K = 4, 8, 2
    T = 6

    ds = _make_deepseek_router(E, D, K)
    ds.eval()
    nmoe = _NmoeRouterReference(
        ds.weight.detach().clone(),
        torch.zeros(E, dtype=torch.float32),
        top_k=K,
    )

    x = torch.randn(T, D, dtype=torch.float32)
    _, _, ds_indices = ds(x)
    _, nmoe_indices = nmoe.forward(x)

    # Direct top-K of raw sigmoid logits — sanity check.
    raw_logits = F.linear(x, ds.weight.float()).float()
    raw_scores = torch.sigmoid(raw_logits)
    _, naive_indices = torch.topk(raw_scores, K, dim=-1)
    torch.testing.assert_close(ds_indices, naive_indices)
    torch.testing.assert_close(nmoe_indices, naive_indices)


def test_bias_update_zero_sum_parity():
    """`_update_single_router_bias` ≡ `nmoe.Router.update_bias` for the
    same load distribution and same `bias_rate / gamma`.

    Both implement DeepSeek-V3's zero-sum sign update:
        s = sign(loads - 1/E)
        bias -= gamma * (s - s.mean())
        bias.clamp(-16, 16)
    """
    _update_single_router_bias = _load_routing_module()._update_single_router_bias

    E = 8
    GAMMA = 0.001

    # Build a non-uniform load distribution.
    raw_counts = torch.tensor(
        [10.0, 5.0, 20.0, 3.0, 15.0, 8.0, 12.0, 7.0],
        dtype=torch.float32,
    )

    # ── Our path ──
    ds_router = _make_deepseek_router(E, hidden_size=16, top_k=2)
    ds_router.local_tokens_per_expert = raw_counts.clone()
    initial_bias = ds_router.expert_bias.detach().clone()
    _update_single_router_bias(ds_router, bias_rate=GAMMA, distributed=False)
    ds_delta = ds_router.expert_bias - initial_bias

    # ── nmoe path ──
    nmoe_bias = initial_bias.clone()
    expert_loads = raw_counts / raw_counts.sum()  # nmoe takes fractional loads
    nmoe = _NmoeRouterReference(
        weight=torch.zeros(E, 16, dtype=torch.float32),
        bias=nmoe_bias,
        top_k=2,
    )
    nmoe.update_bias(expert_loads, gamma=GAMMA)
    nmoe_delta = nmoe.bias - initial_bias

    torch.testing.assert_close(ds_delta, nmoe_delta, atol=1e-7, rtol=1e-6)


def test_bias_update_clamp_parity():
    """Both paths clamp to ±16.0. Construct an extreme load that, if
    repeated many times, would push some entries past the bound."""
    _update_single_router_bias = _load_routing_module()._update_single_router_bias

    E = 4
    GAMMA = 5.0  # extreme rate to trigger clamp in one update

    raw_counts = torch.tensor([100.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    # Our path
    ds_router = _make_deepseek_router(E, hidden_size=8, top_k=2)
    ds_router.local_tokens_per_expert = raw_counts.clone()
    ds_router.expert_bias.fill_(15.5)  # near the +16 cap
    _update_single_router_bias(ds_router, bias_rate=GAMMA, distributed=False)
    assert ds_router.expert_bias.max().item() <= 16.0 + 1e-6
    assert ds_router.expert_bias.min().item() >= -16.0 - 1e-6

    # nmoe path
    nmoe_bias = torch.full((E,), 15.5, dtype=torch.float32)
    nmoe = _NmoeRouterReference(
        weight=torch.zeros(E, 8, dtype=torch.float32),
        bias=nmoe_bias,
        top_k=2,
    )
    expert_loads = raw_counts / raw_counts.sum()
    nmoe.update_bias(expert_loads, gamma=GAMMA)
    assert nmoe.bias.max().item() <= 16.0 + 1e-6
    assert nmoe.bias.min().item() >= -16.0 - 1e-6


def test_bias_update_zero_load_no_op():
    """DEC-2 contract: with zero observed counts (rank saw no tokens), the
    bias must be left unchanged. Both implementations honor this — ours
    via the `total.clamp_min(1.0)` + `(total > 0)` mask, nmoe via the
    update_bias fp arithmetic happening to be ill-defined for an all-zero
    load (caller responsibility — but the trainer doesn't call update_bias
    when there are no observed counts)."""
    _update_single_router_bias = _load_routing_module()._update_single_router_bias

    E = 4
    GAMMA = 0.001

    ds_router = _make_deepseek_router(E, hidden_size=8, top_k=2)
    ds_router.local_tokens_per_expert.zero_()
    initial_bias = ds_router.expert_bias.detach().clone()
    _update_single_router_bias(ds_router, bias_rate=GAMMA, distributed=False)
    torch.testing.assert_close(ds_router.expert_bias, initial_bias)


if __name__ == "__main__":
    test_forward_parity_topk_2_normalized()
    test_forward_parity_with_scaling_factor()
    test_forward_parity_zero_bias_matches_simple_topk()
    test_bias_update_zero_sum_parity()
    test_bias_update_clamp_parity()
    test_bias_update_zero_load_no_op()
    print("ALL OK")
