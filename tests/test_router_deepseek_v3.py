"""
Tests for DeepSeek V3 router fixes — validates that our router matches
the Megatron-LM DeepSeek V3 reference implementation.

Covers:
  1. FP32 gating linear (projection runs in fp32, not bf16)
  2. Epsilon guard in normalization (no NaN on near-zero scores)
  3. Scaling factor (probs * 2.5 after normalization)
  4. Group-limited top-k routing
  5. Expert bias update correctness
  6. Full forward pass integration

Run with: uv run pytest tests/test_router_deepseek_v3.py -v
"""
import pytest
import torch
from types import SimpleNamespace

from src.models.router import DeepSeekRouter, group_limited_topk


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides):
    """Minimal config that DeepSeekRouter reads from."""
    defaults = dict(
        num_experts=16,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        hidden_size=64,
        topk_scaling_factor=2.5,
        num_groups=4,
        group_topk=2,
    )
    defaults.update(overrides)

    # Qwen3MoeTopKRouter.__init__ reads these via config.X
    cfg = SimpleNamespace(**defaults)
    # Also needs num_local_experts (Qwen3MoeConfig attribute_map aliases num_experts)
    cfg.num_local_experts = cfg.num_experts
    return cfg


def make_router(**overrides):
    """Build a DeepSeekRouter from a minimal config."""
    cfg = make_config(**overrides)
    return DeepSeekRouter(cfg)


# ---------------------------------------------------------------------------
#  Test 1: FP32 gating linear
# ---------------------------------------------------------------------------

class TestFP32GatingLinear:
    """The linear projection W @ h must happen in fp32, not bf16."""

    def test_logits_are_fp32_even_with_bf16_input(self):
        router = make_router()
        # Simulate bf16 mixed precision
        hidden = torch.randn(8, 64, dtype=torch.bfloat16)

        scores, probs, indices = router(hidden)

        # The gating linear and sigmoid should have run in fp32.
        # scores returned are fp32 (sigmoid output).
        assert scores.dtype == torch.float32, f"Expected fp32 scores, got {scores.dtype}"

    def test_fp32_vs_bf16_projection_differs(self):
        """Verify that fp32 projection gives different (more precise) results than bf16."""
        router = make_router(num_experts=256, hidden_size=128, num_groups=None, group_topk=None)
        # Initialize with non-zero weights so logits aren't all ~0 (sigmoid(0)=0.5)
        torch.manual_seed(42)
        with torch.no_grad():
            router.weight.normal_(std=0.02)
        hidden = torch.randn(32, 128, dtype=torch.bfloat16)

        # Our router: fp32 projection
        scores_fp32, _, _ = router(hidden)

        # Manual bf16 projection for comparison
        with torch.no_grad():
            weight_bf16 = router.weight.to(torch.bfloat16)
            logits_bf16 = torch.nn.functional.linear(hidden, weight_bf16)  # bf16 @ bf16
            scores_bf16 = torch.sigmoid(logits_bf16.float())

        # They should NOT be bitwise identical — bf16 projection loses precision
        assert not torch.equal(scores_fp32, scores_bf16), (
            "fp32 and bf16 projections should differ for 256 experts"
        )

    def test_weight_stays_original_dtype_after_forward(self):
        """fp32 cast should not modify the stored weight dtype."""
        router = make_router()
        # Weight is initialized as float32 by Qwen3MoeTopKRouter
        original_dtype = router.weight.dtype
        hidden = torch.randn(4, 64)
        router(hidden)
        assert router.weight.dtype == original_dtype


# ---------------------------------------------------------------------------
#  Test 2: Epsilon guard in normalization
# ---------------------------------------------------------------------------

class TestEpsilonGuard:
    """Division by sum should not produce NaN even with near-zero scores."""

    def test_no_nan_with_zero_logits(self):
        router = make_router(norm_topk_prob=True, num_groups=None, group_topk=None)
        # Force logits to very negative => sigmoid ~ 0
        with torch.no_grad():
            router.weight.fill_(-100.0)
        hidden = torch.ones(4, 64)

        scores, probs, indices = router(hidden)

        assert not torch.isnan(probs).any(), "NaN in probs with near-zero sigmoid scores"
        assert not torch.isinf(probs).any(), "Inf in probs with near-zero sigmoid scores"

    def test_epsilon_does_not_affect_normal_case(self):
        """With normal scores, epsilon should be negligible."""
        router = make_router(norm_topk_prob=True, num_groups=None, group_topk=None)
        torch.manual_seed(0)
        hidden = torch.randn(8, 64)

        _, probs, _ = router(hidden)

        # probs per token should sum to scaling_factor (2.5) since norm + scale
        sums = probs.float().sum(dim=-1)
        expected = 2.5
        assert torch.allclose(sums, torch.full_like(sums, expected), atol=1e-4), (
            f"Expected probs to sum to {expected}, got {sums}"
        )


# ---------------------------------------------------------------------------
#  Test 3: Scaling factor
# ---------------------------------------------------------------------------

class TestScalingFactor:
    """After normalization, probs should be multiplied by scaling_factor."""

    def test_probs_sum_to_scaling_factor(self):
        sf = 2.5
        router = make_router(
            topk_scaling_factor=sf, norm_topk_prob=True,
            num_groups=None, group_topk=None,
        )
        torch.manual_seed(1)
        hidden = torch.randn(16, 64)

        _, probs, _ = router(hidden)

        sums = probs.float().sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, sf), atol=1e-4), (
            f"Probs should sum to {sf} (norm-to-1 * {sf}), got {sums}"
        )

    def test_no_scaling_factor(self):
        """When scaling_factor is None, probs should sum to 1."""
        router = make_router(
            topk_scaling_factor=None, norm_topk_prob=True,
            num_groups=None, group_topk=None,
        )
        torch.manual_seed(1)
        hidden = torch.randn(16, 64)

        _, probs, _ = router(hidden)

        sums = probs.float().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"Without scaling, probs should sum to 1.0, got {sums}"
        )

    def test_scaling_factor_3_point_0(self):
        """Arbitrary scaling factor."""
        sf = 3.0
        router = make_router(
            topk_scaling_factor=sf, norm_topk_prob=True,
            num_groups=None, group_topk=None,
        )
        torch.manual_seed(2)
        hidden = torch.randn(8, 64)

        _, probs, _ = router(hidden)

        sums = probs.float().sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, sf), atol=1e-4)


# ---------------------------------------------------------------------------
#  Test 4: Group-limited top-k routing
# ---------------------------------------------------------------------------

class TestGroupLimitedTopK:
    """Group-limited routing should restrict expert selection to top groups."""

    def test_basic_group_limited_topk(self):
        """Selected experts must come from at most group_topk groups."""
        torch.manual_seed(42)
        num_experts = 16
        num_groups = 4
        group_topk = 2
        topk = 4
        num_tokens = 32

        scores = torch.randn(num_tokens, num_experts).abs()
        _, top_indices = group_limited_topk(scores, topk, num_groups, group_topk)

        experts_per_group = num_experts // num_groups
        for t in range(num_tokens):
            groups_used = set()
            for idx in top_indices[t]:
                groups_used.add(idx.item() // experts_per_group)
            assert len(groups_used) <= group_topk, (
                f"Token {t} used {len(groups_used)} groups, expected <= {group_topk}: {groups_used}"
            )

    def test_group_limited_selects_correct_count(self):
        """Should always return exactly topk indices."""
        scores = torch.randn(10, 32).abs()
        top_scores, top_indices = group_limited_topk(scores, topk=8, num_groups=8, group_topk=4)
        assert top_indices.shape == (10, 8)
        assert top_scores.shape == (10, 8)

    def test_group_limited_with_expert_bias(self):
        """Group-limited routing should work with biased scores."""
        router = make_router(
            num_experts=32, num_experts_per_tok=8,
            num_groups=8, group_topk=4,
            topk_scaling_factor=2.5,
        )
        torch.manual_seed(7)
        hidden = torch.randn(16, 64)

        _, probs, indices = router(hidden)

        # Check group constraint
        experts_per_group = 32 // 8
        for t in range(16):
            groups_used = set()
            for idx in indices[t]:
                groups_used.add(idx.item() // experts_per_group)
            assert len(groups_used) <= 4, f"Token {t} used {len(groups_used)} groups"

    def test_no_groups_falls_back_to_flat_topk(self):
        """When num_groups is None, should do flat top-k (no crash)."""
        router = make_router(num_groups=None, group_topk=None)
        hidden = torch.randn(8, 64)
        scores, probs, indices = router(hidden)
        assert indices.shape == (8, 4)  # top_k = 4

    def test_group_limited_vs_flat_topk_differs(self):
        """Group-limited should give different selections than flat top-k for some inputs."""
        torch.manual_seed(123)
        num_experts = 16
        scores = torch.randn(64, num_experts).abs()

        _, flat_indices = torch.topk(scores, k=4, dim=-1)
        _, group_indices = group_limited_topk(scores, topk=4, num_groups=4, group_topk=2)

        # They shouldn't always be identical
        assert not torch.equal(flat_indices, group_indices), (
            "Group-limited and flat top-k should differ for some inputs"
        )


# ---------------------------------------------------------------------------
#  Test 5: Expert bias update
# ---------------------------------------------------------------------------

class TestExpertBiasUpdate:
    """Token counting and bias update mechanism."""

    def test_token_counts_accumulate_during_forward(self):
        router = make_router(num_experts=8, num_experts_per_tok=2, num_groups=None, group_topk=None)
        assert router.local_tokens_per_expert.sum() == 0

        hidden = torch.randn(16, 64)
        router(hidden)

        total_tokens = router.local_tokens_per_expert.sum().item()
        expected = 16 * 2  # 16 tokens, top-2
        assert total_tokens == expected, f"Expected {expected} total token-expert assignments, got {total_tokens}"

    def test_token_counts_not_accumulated_in_eval(self):
        router = make_router(num_experts=8, num_experts_per_tok=2, num_groups=None, group_topk=None)
        router.eval()
        hidden = torch.randn(16, 64)

        with torch.no_grad():
            router(hidden)

        assert router.local_tokens_per_expert.sum() == 0, "Should not count tokens when grad disabled"

    def test_bias_update_formula(self):
        """Manually verify: bias += sign(avg - counts) * rate."""
        router = make_router(num_experts=4, num_experts_per_tok=1, num_groups=None, group_topk=None)

        # Simulate: expert 0 got 10 tokens, expert 1 got 0, expert 2 got 5, expert 3 got 5
        router.local_tokens_per_expert = torch.tensor([10.0, 0.0, 5.0, 5.0])
        counts = router.local_tokens_per_expert.clone()
        avg = counts.mean()  # 5.0
        rate = 0.001

        expected_bias = torch.sign(avg - counts) * rate
        # sign(5-10)=-1, sign(5-0)=+1, sign(5-5)=0, sign(5-5)=0
        assert torch.allclose(expected_bias, torch.tensor([-0.001, 0.001, 0.0, 0.0]))

        router.expert_bias += expected_bias
        router.local_tokens_per_expert.zero_()

        assert torch.allclose(router.expert_bias, torch.tensor([-0.001, 0.001, 0.0, 0.0]))
        assert router.local_tokens_per_expert.sum() == 0

    def test_bias_persists_in_state_dict(self):
        router = make_router(num_experts=8, num_experts_per_tok=2, num_groups=None, group_topk=None)
        router.expert_bias += torch.randn(8)

        sd = router.state_dict()
        assert "expert_bias" in sd
        assert "local_tokens_per_expert" not in sd  # non-persistent


# ---------------------------------------------------------------------------
#  Test 6: Full integration — matches Megatron-LM math
# ---------------------------------------------------------------------------

class TestMegatronReference:
    """
    Reference test: manually replicate Megatron-LM's sigmoid routing math
    and compare against our DeepSeekRouter output.
    """

    def test_matches_megatron_sigmoid_routing(self):
        """Step-by-step replicate Megatron-LM moe_utils.py lines 762-775."""
        torch.manual_seed(999)
        num_experts = 16
        hidden_size = 64
        topk = 4
        scaling_factor = 2.5

        # No group routing for this test — isolate the core math
        router = make_router(
            num_experts=num_experts,
            num_experts_per_tok=topk,
            hidden_size=hidden_size,
            topk_scaling_factor=scaling_factor,
            norm_topk_prob=True,
            num_groups=None,
            group_topk=None,
        )
        hidden = torch.randn(8, hidden_size)

        # --- Our router ---
        our_scores, our_probs, our_indices = router(hidden)

        # --- Manual Megatron-LM reference ---
        with torch.no_grad():
            # moe_utils.py line 762-763: sigmoid in fp32
            logits = torch.nn.functional.linear(hidden.float(), router.weight.float())
            ref_scores = torch.sigmoid(logits.float())

            # line 764-767: biased topk, gather unbiased
            scores_for_routing = ref_scores + router.expert_bias.float()
            _, ref_indices = torch.topk(scores_for_routing, k=topk, dim=-1)
            ref_gathered = torch.gather(ref_scores, dim=1, index=ref_indices)

            # line 770: normalize with epsilon
            ref_probs = ref_gathered / (ref_gathered.sum(dim=-1, keepdim=True) + 1e-20)

            # line 774-775: scaling factor
            ref_probs = ref_probs * scaling_factor

        # Compare
        assert torch.equal(our_indices, ref_indices), (
            f"Index mismatch:\n  ours={our_indices}\n  ref={ref_indices}"
        )
        assert torch.allclose(our_probs.float(), ref_probs.float(), atol=1e-5), (
            f"Probs mismatch:\n  ours={our_probs}\n  ref={ref_probs}"
        )

    def test_matches_megatron_with_group_routing(self):
        """Full DeepSeek V3 config: sigmoid + bias + groups + scaling."""
        torch.manual_seed(42)
        num_experts = 32
        hidden_size = 64
        topk = 8
        num_groups = 8
        group_topk_val = 4
        scaling_factor = 2.5

        router = make_router(
            num_experts=num_experts,
            num_experts_per_tok=topk,
            hidden_size=hidden_size,
            topk_scaling_factor=scaling_factor,
            norm_topk_prob=True,
            num_groups=num_groups,
            group_topk=group_topk_val,
        )
        # Set some non-zero bias
        router.expert_bias.copy_(torch.randn(num_experts) * 0.01)
        hidden = torch.randn(16, hidden_size)

        our_scores, our_probs, our_indices = router(hidden)

        # Manual reference
        with torch.no_grad():
            logits = torch.nn.functional.linear(hidden.float(), router.weight.float())
            ref_scores = torch.sigmoid(logits)
            biased = ref_scores + router.expert_bias.float()

            _, ref_indices = group_limited_topk(biased, topk, num_groups, group_topk_val)
            ref_gathered = torch.gather(ref_scores, dim=1, index=ref_indices)
            ref_probs = ref_gathered / (ref_gathered.sum(dim=-1, keepdim=True) + 1e-20)
            ref_probs = ref_probs * scaling_factor

        assert torch.equal(our_indices, ref_indices)
        assert torch.allclose(our_probs.float(), ref_probs.float(), atol=1e-5)


# ---------------------------------------------------------------------------
#  Test 7: Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify config params flow through correctly."""

    def test_default_no_groups_no_scaling(self):
        """Without config attrs, should fall back to flat topk, no scaling."""
        cfg = make_config(topk_scaling_factor=None, num_groups=None, group_topk=None)
        router = DeepSeekRouter(cfg)
        assert router.scaling_factor is None
        assert router.num_groups is None
        assert router.group_topk is None

    def test_deepseek_v3_config(self):
        """Standard DeepSeek V3 params."""
        cfg = make_config(topk_scaling_factor=2.5, num_groups=8, group_topk=4)
        router = DeepSeekRouter(cfg)
        assert router.scaling_factor == 2.5
        assert router.num_groups == 8
        assert router.group_topk == 4


# ---------------------------------------------------------------------------
#  Test 8: Global pool bias update for global MoE
# ---------------------------------------------------------------------------

class _FakeAccelerator:
    """Minimal mock for single-process bias update tests."""
    num_processes = 1


class _FakeModel(torch.nn.Module):
    """Model with N DeepSeekRouters (simulates global MoE with shared experts)."""

    def __init__(self, num_layers, num_experts, hidden_size):
        super().__init__()
        cfg = make_config(
            num_experts=num_experts, hidden_size=hidden_size,
            num_experts_per_tok=2, num_groups=None, group_topk=None,
            topk_scaling_factor=None,
        )
        self.routers = torch.nn.ModuleList([DeepSeekRouter(cfg) for _ in range(num_layers)])


class TestGlobalPoolBiasUpdate:
    """Global MoE bias update should use total load across all layers."""

    def test_global_pool_sums_counts_across_layers(self):
        """With is_global=True, bias should reflect total expert load, not per-layer."""
        from train import update_expert_biases

        model = _FakeModel(num_layers=3, num_experts=4, hidden_size=64)
        acc = _FakeAccelerator()

        # Simulate: layer 0 overloads expert 0, layer 1 overloads expert 1,
        # layer 2 is balanced
        model.routers[0].local_tokens_per_expert = torch.tensor([30.0, 10.0, 10.0, 10.0])
        model.routers[1].local_tokens_per_expert = torch.tensor([10.0, 30.0, 10.0, 10.0])
        model.routers[2].local_tokens_per_expert = torch.tensor([15.0, 15.0, 15.0, 15.0])

        # Global totals: [55, 55, 35, 35] → avg=45
        # sign(45-55)=-1, sign(45-55)=-1, sign(45-35)=+1, sign(45-35)=+1
        update_expert_biases(model, update_rate=0.001, accelerator=acc, is_global=True)

        expected_delta = torch.tensor([-0.001, -0.001, 0.001, 0.001])

        # All routers get the same bias update
        for router in model.routers:
            assert torch.allclose(router.expert_bias, expected_delta), (
                f"Expected {expected_delta}, got {router.expert_bias}"
            )

    def test_standard_mode_updates_independently(self):
        """With is_global=False, each router updates from its own counts only."""
        from train import update_expert_biases

        model = _FakeModel(num_layers=2, num_experts=4, hidden_size=64)
        acc = _FakeAccelerator()

        # Layer 0: expert 0 overloaded
        model.routers[0].local_tokens_per_expert = torch.tensor([40.0, 10.0, 10.0, 10.0])
        # Layer 1: balanced
        model.routers[1].local_tokens_per_expert = torch.tensor([15.0, 15.0, 15.0, 15.0])

        update_expert_biases(model, update_rate=0.001, accelerator=acc, is_global=False)

        # Router 0: avg=17.5, sign(17.5-[40,10,10,10]) = [-1,+1,+1,+1]
        expected_0 = torch.tensor([-0.001, 0.001, 0.001, 0.001])
        assert torch.allclose(model.routers[0].expert_bias, expected_0)

        # Router 1: avg=15, sign(15-[15,15,15,15]) = [0,0,0,0]
        expected_1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(model.routers[1].expert_bias, expected_1)

    def test_global_vs_standard_differs(self):
        """Global and standard modes should give different bias updates."""
        from train import update_expert_biases

        # Create two identical models
        model_g = _FakeModel(num_layers=2, num_experts=4, hidden_size=64)
        model_s = _FakeModel(num_layers=2, num_experts=4, hidden_size=64)
        acc = _FakeAccelerator()

        # Same asymmetric counts for both
        counts = [
            torch.tensor([30.0, 10.0, 10.0, 10.0]),
            torch.tensor([10.0, 30.0, 10.0, 10.0]),
        ]
        for i in range(2):
            model_g.routers[i].local_tokens_per_expert = counts[i].clone()
            model_s.routers[i].local_tokens_per_expert = counts[i].clone()

        update_expert_biases(model_g, 0.001, acc, is_global=True)
        update_expert_biases(model_s, 0.001, acc, is_global=False)

        # Global: totals=[40,40,20,20], avg=30 → all routers get same [-1,-1,+1,+1]*0.001
        # Standard: router0 avg=15 → [-1,-1,0,0]*0.001 (different from global)
        assert not torch.equal(model_g.routers[0].expert_bias, model_s.routers[0].expert_bias)

    def test_token_counts_zeroed_after_update(self):
        """Both modes should reset token counts after the update."""
        from train import update_expert_biases

        model = _FakeModel(num_layers=2, num_experts=4, hidden_size=64)
        acc = _FakeAccelerator()

        for router in model.routers:
            router.local_tokens_per_expert = torch.tensor([10.0, 20.0, 30.0, 40.0])

        update_expert_biases(model, 0.001, acc, is_global=True)

        for router in model.routers:
            assert router.local_tokens_per_expert.sum() == 0


# ---------------------------------------------------------------------------
#  Test 9: Interpolated bias update (per-layer ↔ global blend)
# ---------------------------------------------------------------------------

class TestInterpolatedBiasUpdate:
    """Blended per-layer + global bias update with alpha interpolation."""

    def _make_model_with_counts(self):
        model = _FakeModel(num_layers=3, num_experts=4, hidden_size=64)
        # Layer 0 overloads expert 0, layer 1 overloads expert 1, layer 2 balanced
        model.routers[0].local_tokens_per_expert = torch.tensor([30.0, 10.0, 10.0, 10.0])
        model.routers[1].local_tokens_per_expert = torch.tensor([10.0, 30.0, 10.0, 10.0])
        model.routers[2].local_tokens_per_expert = torch.tensor([15.0, 15.0, 15.0, 15.0])
        return model

    def test_alpha_zero_matches_global(self):
        """alpha=0 should give identical results to the old purely-global path."""
        from train import update_expert_biases

        model = self._make_model_with_counts()
        acc = _FakeAccelerator()

        update_expert_biases(model, 0.001, acc, is_global=True, alpha=0.0)

        # Global totals: [55, 55, 35, 35] → avg=45
        expected = torch.tensor([-0.001, -0.001, 0.001, 0.001])
        for router in model.routers:
            assert torch.allclose(router.expert_bias, expected), (
                f"Expected {expected}, got {router.expert_bias}"
            )

    def test_alpha_one_matches_per_layer(self):
        """alpha=1 should give identical results to independent per-layer updates."""
        from train import update_expert_biases

        model = self._make_model_with_counts()
        acc = _FakeAccelerator()

        update_expert_biases(model, 0.001, acc, is_global=True, alpha=1.0)

        # Layer 0: avg=15, sign(15-[30,10,10,10]) = [-1,+1,+1,+1]
        expected_0 = torch.tensor([-0.001, 0.001, 0.001, 0.001])
        assert torch.allclose(model.routers[0].expert_bias, expected_0)

        # Layer 1: avg=15, sign(15-[10,30,10,10]) = [+1,-1,+1,+1]
        expected_1 = torch.tensor([0.001, -0.001, 0.001, 0.001])
        assert torch.allclose(model.routers[1].expert_bias, expected_1)

        # Layer 2: avg=15, sign(15-[15,15,15,15]) = [0,0,0,0]
        expected_2 = torch.tensor([0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(model.routers[2].expert_bias, expected_2)

    def test_alpha_half_blends(self):
        """alpha=0.5 should average the per-layer and global deltas."""
        from train import update_expert_biases

        model = self._make_model_with_counts()
        acc = _FakeAccelerator()

        update_expert_biases(model, 0.001, acc, is_global=True, alpha=0.5)

        # Global delta: [-0.001, -0.001, +0.001, +0.001]
        # Layer 0 delta: [-0.001, +0.001, +0.001, +0.001]
        # Blend: 0.5*[-0.001,+0.001,+0.001,+0.001] + 0.5*[-0.001,-0.001,+0.001,+0.001]
        #      = [-0.001, 0.0, +0.001, +0.001]
        expected_0 = torch.tensor([-0.001, 0.0, 0.001, 0.001])
        assert torch.allclose(model.routers[0].expert_bias, expected_0)

        # Layer 1 delta: [+0.001, -0.001, +0.001, +0.001]
        # Blend: 0.5*[+0.001,-0.001,+0.001,+0.001] + 0.5*[-0.001,-0.001,+0.001,+0.001]
        #      = [0.0, -0.001, +0.001, +0.001]
        expected_1 = torch.tensor([0.0, -0.001, 0.001, 0.001])
        assert torch.allclose(model.routers[1].expert_bias, expected_1)

        # Layer 2 delta: [0, 0, 0, 0]
        # Blend: 0.5*[0,0,0,0] + 0.5*[-0.001,-0.001,+0.001,+0.001]
        #      = [-0.0005, -0.0005, +0.0005, +0.0005]
        expected_2 = torch.tensor([-0.0005, -0.0005, 0.0005, 0.0005])
        assert torch.allclose(model.routers[2].expert_bias, expected_2)


class TestBiasAlphaSchedule:
    """Cosine decay schedule for bias alpha: 1 → 0."""

    def test_start_is_one(self):
        from train import bias_alpha_schedule
        assert bias_alpha_schedule(0, 1000) == pytest.approx(1.0)

    def test_end_is_zero(self):
        from train import bias_alpha_schedule
        assert bias_alpha_schedule(1000, 1000) == pytest.approx(0.0, abs=1e-9)

    def test_midpoint_is_half(self):
        from train import bias_alpha_schedule
        assert bias_alpha_schedule(500, 1000) == pytest.approx(0.5, abs=1e-9)

    def test_quarter_point(self):
        from train import bias_alpha_schedule
        import math
        expected = 0.5 * (1 + math.cos(math.pi * 0.25))
        assert bias_alpha_schedule(250, 1000) == pytest.approx(expected, abs=1e-9)

    def test_beyond_max_steps_clamps_to_zero(self):
        from train import bias_alpha_schedule
        assert bias_alpha_schedule(2000, 1000) == pytest.approx(0.0, abs=1e-9)
