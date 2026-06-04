"""AC-2 coverage tests for the unified `get_all_balancing_owners` walker.

Every load-balancing owner exposes the canonical interface (`expert_bias` and
`local_tokens_per_expert` buffers). The walker yields `(owner_module, label)`
pairs from each `…ForCausalLM` wrapper. Tests here verify:

- Standalone routers in `standard_moe` (per-layer expert pools, per-router state).
- Per-layer routers in `global_moe` and `moe_everything` (PRE-DEC-19 shape; the
  DEC-19 refactor in Milestone C will collapse these to bank-level owners).
- Branch routers (singular and plural) under DEC-18 unified buffer names.
- Wrapper-vs-inner regression: the walker works on the outer
  `…ForCausalLM` wrapper, which is what `unwrap_model(...)` returns under
  DDP/FSDP — not just on the inner `…Model`.
- Negative test: softmax-only models produce ZERO owners (no DeepSeek bias
  state to update).
"""
import sys
sys.path.insert(0, ".")

import torch

from src.models import (
    Qwen3MoeConfig,
    StandardMoEModel,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    DeepSeekGlobalMoEForCausalLM,
    MoEverythingConfig,
    MoEverythingForCausalLM,
)
from src.models.router import DeepSeekRouter


def _tiny_qwen3_moe_kwargs(num_hidden_layers: int = 2, num_experts: int = 4):
    """Smallest-possible Qwen3MoeConfig that exercises the routing path."""
    return dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=num_hidden_layers,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_experts=num_experts,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        seq_aux_loss_coef=0.0,
        output_router_logits=True,
        attn_implementation="eager",
    )


def test_standard_moe_softmax_zero_owners():
    """Negative: softmax-only standard_moe has no DeepSeek owners."""
    cfg = Qwen3MoeConfig(**_tiny_qwen3_moe_kwargs(num_hidden_layers=4))
    model = StandardMoEModel(cfg)
    owners = list(model.get_all_balancing_owners())
    assert owners == [], f"expected zero owners, got {len(owners)}: {owners}"
    # And no DeepSeekRouter instances exist in the module tree.
    assert sum(isinstance(m, DeepSeekRouter) for m in model.modules()) == 0


def test_standard_moe_deepseek_per_layer_owners():
    """Per-layer expert pools → per-layer owners (one per layer)."""
    cfg = Qwen3MoeConfig(**_tiny_qwen3_moe_kwargs(num_hidden_layers=4, num_experts=8))
    model = DeepSeekStandardMoEModel(cfg)
    owners = list(model.get_all_balancing_owners())
    assert len(owners) == 4, f"expected 4 owners (one per layer), got {len(owners)}"
    for owner, label in owners:
        assert label == "mlp", f"unexpected label {label}"
        assert isinstance(owner, DeepSeekRouter)
        assert owner.expert_bias.shape == (8,)
        assert owner.local_tokens_per_expert.shape == (8,)


def test_global_moe_softmax_zero_owners():
    """Negative: softmax-only global_moe has no DeepSeek owners."""
    common = _tiny_qwen3_moe_kwargs(num_hidden_layers=4)
    common.pop("num_experts")
    cfg = GlobalMoEConfig(num_experts=4, **common)
    model = GlobalMoEForCausalLM(cfg)
    owners = list(model.get_all_balancing_owners())
    assert owners == [], f"expected zero owners, got {len(owners)}"


def test_global_moe_deepseek_per_layer_owners_pre_dec19():
    """Pre-DEC-19: per-layer routers in global_moe each own bias state.

    After the DEC-19 refactor (Milestone C), this test will be updated to
    expect a single bank-level owner on `model.model`.
    """
    common = _tiny_qwen3_moe_kwargs(num_hidden_layers=4)
    common.pop("num_experts")
    cfg = GlobalMoEConfig(num_experts=8, **common)
    model = DeepSeekGlobalMoEForCausalLM(cfg)
    owners = list(model.get_all_balancing_owners())
    assert len(owners) == 4, f"expected 4 owners (one per layer), got {len(owners)}"
    for owner, label in owners:
        assert label == "mlp"
        assert owner.expert_bias.shape == (8,)
        assert owner.local_tokens_per_expert.shape == (8,)


def _moe_everything_cfg(per_layer_router: bool):
    """Construct a small `MoEverythingConfig` with deepseek routing on."""
    return MoEverythingConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=4,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_attn_experts=2,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_no_recompute",
        branch_router_aux_loss_coef=0.0,
        use_deepseek_routing=True,
        topk_scaling_factor=2.5,
        per_layer_router=per_layer_router,
        per_layer_mlp_router=per_layer_router,
        per_layer_attn_router=per_layer_router,
        routed_norm=False,
        per_layer_norm=False,
        post_norm=False,
        dynamic_depth_min=1.0,
        dynamic_depth_max=1.0,
        depthwise_attention=False,
        depthwise_block_size=0,
        scale_attn_by_routing_weight=True,
        scale_branch_by_routing_weight=True,
        router_exploration_rate=0.0,
        branch_router_exploration_rate=0.0,
        branch_sampling=False,
        branch_level="token",
        branch_deepseek=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        seq_aux_loss_coef=0.0,
        output_router_logits=True,
        attn_implementation="eager",
    )


def test_moe_everything_walker_yields_branch_router():
    """Branch router (DeepSeek-style) appears in walker output with `branch` label."""
    cfg = _moe_everything_cfg(per_layer_router=False)
    model = MoEverythingForCausalLM(cfg)
    owners = list(model.get_all_balancing_owners())
    branch_owners = [(o, l) for o, l in owners if l == "branch"]
    assert len(branch_owners) == 1, f"expected 1 branch owner (singular), got {len(branch_owners)}"
    branch, _ = branch_owners[0]
    # DEC-18 buffer rename: branch router exposes `expert_bias` (length-2)
    # and `local_tokens_per_expert`, NOT the legacy `branch_bias` / `local_counts`.
    assert hasattr(branch, "expert_bias") and branch.expert_bias.shape == (2,)
    assert hasattr(branch, "local_tokens_per_expert") and branch.local_tokens_per_expert.shape == (2,)
    assert not hasattr(branch, "branch_bias"), "legacy branch_bias buffer must be gone"
    assert not hasattr(branch, "local_counts"), "legacy local_counts buffer must be gone"


def test_moe_everything_walker_yields_per_layer_branch_routers():
    """When `per_layer_router=True`, walker yields one branch owner per depth."""
    cfg = _moe_everything_cfg(per_layer_router=True)
    model = MoEverythingForCausalLM(cfg)
    owners = list(model.get_all_balancing_owners())
    branch_owners = [(o, l) for o, l in owners if l == "branch"]
    # `num_hidden_layers=4` → `num_depths=4`.
    assert len(branch_owners) == 4, f"expected 4 branch owners (per-layer), got {len(branch_owners)}"


def test_walker_works_on_wrapper_not_just_inner():
    """Wrapper-vs-inner regression: walker is defined on `…ForCausalLM`.

    `update_expert_biases` calls `unwrap_model(model)` which returns the outer
    `…ForCausalLM` wrapper (under DDP/FSDP), NOT the inner `…Model`. The walker
    must therefore be reachable from the wrapper.
    """
    cfg = _moe_everything_cfg(per_layer_router=False)
    model = MoEverythingForCausalLM(cfg)
    # Calling on the outer wrapper works.
    outer_owners = list(model.get_all_balancing_owners())
    # The inner `MoEverythingModel` does NOT have its own `get_all_balancing_owners`
    # — the walker only lives on the wrapper. This is the regression case.
    assert not hasattr(model.model, "get_all_balancing_owners"), (
        "Walker must live on the wrapper (so unwrap_model() finds it), "
        "not on the inner Model. If this assertion fires, the walker has "
        "been silently moved to the inner — fix `update_expert_biases` to "
        "match the new location."
    )
    # And the wrapper's walker yields a non-empty set under deepseek mode.
    assert len(outer_owners) > 0


def test_negative_softmax_only_moe_everything_zero_owners():
    """Negative: softmax-only moe_everything has no DeepSeek owners."""
    cfg = _moe_everything_cfg(per_layer_router=False)
    cfg.use_deepseek_routing = False
    cfg.branch_deepseek = False
    model = MoEverythingForCausalLM(cfg)
    owners = list(model.get_all_balancing_owners())
    assert owners == [], f"expected zero owners under softmax-only, got {len(owners)}: {[(o.__class__.__name__, l) for o, l in owners]}"


if __name__ == "__main__":
    test_standard_moe_softmax_zero_owners()
    test_standard_moe_deepseek_per_layer_owners()
    test_global_moe_softmax_zero_owners()
    test_global_moe_deepseek_per_layer_owners_pre_dec19()
    test_moe_everything_walker_yields_branch_router()
    test_moe_everything_walker_yields_per_layer_branch_routers()
    test_walker_works_on_wrapper_not_just_inner()
    test_negative_softmax_only_moe_everything_zero_owners()
    print("ALL OK")
