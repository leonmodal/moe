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


# ──────────────────────────────────────────────────────────────────────
#  Codex Round 10 Finding 2 follow-ups: lock the missing AC-7 contracts.
# ──────────────────────────────────────────────────────────────────────

def test_forward_parity_nonzero_bias_uses_unbiased_weights():
    """Critical AC-7 contract: with `expert_bias != 0`, both
    DeepSeekRouter and nmoe.Router select via biased scores BUT gather
    weights from UNBIASED sigmoid scores. If DeepSeekRouter accidentally
    gathered from `biased_scores` instead of `scores`, the weights
    would shift by `+expert_bias[selected]`. This test catches that.

    Construct a bias large enough to FLIP the top-K selection from the
    no-bias case so the test is sensitive to where the bias is applied
    (selection vs weight) and not just to per-expert magnitude.
    """
    torch.manual_seed(2026_04_28)
    E, D, K = 4, 16, 2
    T = 6

    ds = _make_deepseek_router(E, D, K)
    ds.eval()

    # Bias[3] >> bias[0..2] so expert 3 is always in the top-K, regardless
    # of its raw sigmoid score. Bias is symmetric for the other ranks so
    # the second top-K slot is still RNG-driven.
    bias = torch.tensor([0.0, 0.0, 0.0, 5.0], dtype=torch.float32)
    ds.expert_bias = bias.clone()

    nmoe = _NmoeRouterReference(
        ds.weight.detach().clone(),
        bias.clone(),
        top_k=K,
    )

    x = torch.randn(T, D, dtype=torch.float32)
    _, ds_weights, ds_indices = ds(x)
    nmoe_weights, nmoe_indices = nmoe.forward(x)

    # Selection must match exactly.
    torch.testing.assert_close(ds_indices, nmoe_indices)
    # Weights must match exactly (both gather from unbiased sigmoid, then
    # normalize).
    torch.testing.assert_close(ds_weights.float(), nmoe_weights.float(), atol=1e-6, rtol=1e-6)

    # Sanity: expert 3 must be in every row's selection (proves the
    # bias actually changed the selection — otherwise the test would
    # vacuously pass with a near-zero bias).
    assert (ds_indices == 3).any(dim=-1).all(), (
        "expert 3 with bias=+5 must be in every token's top-K"
    )

    # Cross-check: the gathered weights must equal `sigmoid(logits)` at
    # the selected indices, NOT `sigmoid(logits) + bias`. If we had a bug
    # gathering from biased_scores, w[..., where indices==3] would be
    # ~5.0 + (raw sigmoid in (0,1)) ≈ 5+something — clearly out of (0, 1].
    raw_logits = F.linear(x, ds.weight.float()).float()
    raw_scores = torch.sigmoid(raw_logits)
    expected_pre_norm = torch.gather(raw_scores, 1, ds_indices)
    expected_post_norm = expected_pre_norm / expected_pre_norm.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)
    torch.testing.assert_close(
        ds_weights.float(), expected_post_norm, atol=1e-6, rtol=1e-6,
    )


def test_topk_1_skips_normalization_to_preserve_gradient():
    """DeepSeekRouter intentionally skips `weights /= weights.sum()` for
    `top_k=1` because for K=1 the sum is a single value and dividing
    yields a constant `1.0` that has zero gradient w.r.t. the router
    weights — destroying any gradient signal that would route through
    the routing weight back to the gate.

    nmoe.Router does NOT have this guard; for top_k=1 it returns
    `1.0` weights everywhere. Lock this intentional divergence so a
    future "let's just match nmoe exactly" PR doesn't accidentally
    re-introduce the gradient kill.
    """
    torch.manual_seed(2026_04_28)
    E, D = 4, 16
    T = 6

    ds = _make_deepseek_router(E, D, top_k=1, norm_topk_prob=True)
    ds.eval()

    x = torch.randn(T, D, dtype=torch.float32)
    _, ds_weights, _ = ds(x)

    # Weights must NOT be all-1.0 (would mean normalization happened).
    # They should be the unbiased sigmoid score at the selected expert,
    # which lives in (0, 1).
    assert ds_weights.shape == (T, 1)
    assert (ds_weights < 1.0).any(), (
        "DeepSeekRouter top_k=1 returned all-1.0 weights — normalization "
        "guard regressed; this kills the routing-weight gradient signal."
    )
    assert (ds_weights > 0.0).all(), (
        f"top_k=1 weights must be > 0 (raw sigmoid), got {ds_weights}"
    )


def test_router_runs_in_fp32_under_bf16_autocast():
    """DEC-2 / nmoe parity: the router's logits and sigmoid must run in
    fp32 even under bf16 autocast — both nmoe.Router (which casts via
    `.float()`) and DeepSeekRouter (which uses
    `torch.autocast(enabled=False)` + `.float()`) follow this contract.
    Otherwise low-precision sigmoid saturation would silently produce
    biased routing distributions.
    """
    torch.manual_seed(2026_04_28)
    E, D, K = 4, 16, 2
    T = 6

    ds = _make_deepseek_router(E, D, K)
    ds.eval()

    x = torch.randn(T, D, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        scores, weights, indices = ds(x)

    # `scores` is the raw sigmoid output stored as a side-effect on the
    # router — it must be fp32 even under bf16 autocast.
    assert scores.dtype == torch.float32, (
        f"DeepSeekRouter scores are {scores.dtype} under bf16 autocast — "
        f"fp32 sigmoid contract violated."
    )
    # `weights` is post-normalization, cast to the input dtype on output.
    assert weights.dtype == x.dtype, (
        f"DeepSeekRouter output weights should match input dtype ({x.dtype}), "
        f"got {weights.dtype}"
    )


def test_topk_scaling_factor_yaml_source_comment_audit():
    """AC-7 config-comment audit: every yaml that sets
    `topk_scaling_factor` must include a YAML comment citing the
    DeepSeek-V3 source (default 2.5 from the V3 paper, Section 3.2).

    This locks documentation surface so a reviewer or future maintainer
    sees the provenance directly in the yaml without having to grep
    `nmoe/` or DeepSeek-V3.
    """
    repo = Path(__file__).resolve().parent.parent
    configs_root = repo / "configs"
    if not configs_root.exists():
        pytest.skip("configs/ directory not present in this checkout.")

    offenders: list[str] = []
    SOURCES = ("DeepSeek", "deepseek", "V3", "nmoe", "Section 3.2")
    for yaml_path in sorted(configs_root.rglob("*.yaml")):
        text = yaml_path.read_text()
        lines = text.splitlines()
        for line_idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "topk_scaling_factor" in stripped:
                inline_has_source = any(tok in line for tok in SOURCES)
                preceding_has_source = False
                for prev_line in reversed(lines[:line_idx-1]):
                    if not prev_line.strip():
                        break
                    if not prev_line.strip().startswith("#"):
                        break
                    if any(tok in prev_line for tok in SOURCES):
                        preceding_has_source = True
                        break
                if not (inline_has_source or preceding_has_source):
                    offenders.append(f"{yaml_path.relative_to(repo)}:{line_idx}: {stripped}")

    assert not offenders, (
        "topk_scaling_factor is set without a source comment in:\n  - "
        + "\n  - ".join(offenders)
        + "\n\nAdd a YAML comment citing DeepSeek-V3 (e.g. `# DeepSeek-V3 paper Section 3.2 default 2.5`)."
    )


def test_inlined_nmoe_reference_sync_guard():
    """The inlined `_NmoeRouterReference` is a byte-for-byte port of
    `nmoe/nmoe/model.py:Router` as of 2026-04-28. If the upstream
    `nmoe.model.Router` class changes, this test must fail so we
    re-sync — preventing the parity suite from silently passing
    against a stale reference.

    This guard hashes the load-bearing source of the upstream `class
    Router` block and pins the hash. When upstream changes, update
    both the inlined reference here AND the pinned hash in this test.
    """
    import hashlib

    repo = Path(__file__).resolve().parent.parent
    upstream = repo / "nmoe" / "nmoe" / "model.py"
    if not upstream.exists():
        pytest.skip("nmoe upstream not present in this checkout.")

    text = upstream.read_text()
    start = text.find("class Router(nn.Module):")
    assert start >= 0, "could not find `class Router(nn.Module):` in upstream"
    rest = text[start:]
    end = rest.find("\nclass ", 1)
    if end < 0:
        end = len(rest)
    router_block = rest[:end]

    digest = hashlib.sha256(router_block.encode("utf-8")).hexdigest()

    # PINNED hash for the upstream nmoe.Router as of 2026-04-28.
    # When this fails, re-sync the inlined `_NmoeRouterReference`,
    # then update the pinned hash to the new digest.
    PINNED = "00ee97ec8a37be35942bbd4dbe4135a9007558db3a25fee3221ae8ccba535c9a"

    if digest != PINNED:
        pytest.fail(
            f"nmoe.model.Router has changed upstream.\n"
            f"  computed sha256: {digest}\n"
            f"  pinned   sha256: {PINNED}\n"
            f"Re-sync the inlined `_NmoeRouterReference` in this test "
            f"file with `nmoe/nmoe/model.py`, then update PINNED."
        )


if __name__ == "__main__":
    test_forward_parity_topk_2_normalized()
    test_forward_parity_with_scaling_factor()
    test_forward_parity_zero_bias_matches_simple_topk()
    test_bias_update_zero_sum_parity()
    test_bias_update_clamp_parity()
    test_bias_update_zero_load_no_op()
    test_forward_parity_nonzero_bias_uses_unbiased_weights()
    test_topk_1_skips_normalization_to_preserve_gradient()
    test_router_runs_in_fp32_under_bf16_autocast()
    test_topk_scaling_factor_yaml_source_comment_audit()
    test_inlined_nmoe_reference_sync_guard()
    print("ALL OK")
