"""Model factory: build models from config dicts.

Supported model types:
- dense: Qwen3-based dense transformer
- standard_moe: Standard MoE with softmax or DeepSeek routing (router_type config)
- global_moe: Global MoE with shared expert pool and softmax or DeepSeek routing
- moe_everything: MoE-Everything with per-head attention routing
"""

from __future__ import annotations

import os

from src.models import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    StandardMoEModel,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    DeepSeekGlobalMoEForCausalLM,
    MoEverythingConfig,
    MoEverythingForCausalLM,
)


# Model types that have been archived to legacy/
_ARCHIVED_TYPES = {
    "speedrun_gpt",
    "speedrun_moe_fully_independent",
    "speedrun_moe_precompute_kv",
    "speedrun_moe_everything",
    "gpt2_dense",
}

# Supported model types
SUPPORTED_TYPES = {
    "dense",
    "standard_moe",
    "global_moe",
    "moe_everything",
}

# Deprecated model types that must be rejected with clear guidance
_DEPRECATED_TYPES = {
    "deepseek_standard_moe": "Use type: standard_moe with router_type: deepseek",
    "deepseek_global_moe": "Use type: global_moe with router_type: deepseek",
}


def configure_liger_kernels(cfg: dict) -> str:
    """Configure liger kernels for the given model config.

    Returns a string describing what was enabled.
    """
    training_cfg = cfg.get("training", {})
    if training_cfg.get("disable_liger", False) or os.environ.get("MOE_DISABLE_LIGER", "0") == "1":
        return "disabled"

    mtype = cfg["model"]["type"]
    if mtype in _ARCHIVED_TYPES or mtype == "gpt2_dense":
        return "disabled (unsupported model type)"

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
    except ImportError:
        return "disabled (liger_kernel not installed)"

    if mtype == "moe_everything":
        apply_liger_kernel_to_qwen3_moe(
            rope=True,
            rms_norm=True,
            swiglu=False,
            fused_linear_cross_entropy=False,
            cross_entropy=False,
        )
        return "partial (rope+rms_norm only; swiglu/fused CE disabled for moe_everything)"

    apply_liger_kernel_to_qwen3_moe()
    return "full"


def _set_router_params(config, model_cfg: dict) -> None:
    config.router_exploration_rate = model_cfg.get("router_exploration_rate", 0.0)


def _set_deepseek_router_params(config, model_cfg: dict) -> None:
    _set_router_params(config, model_cfg)
    config.topk_scaling_factor = model_cfg.get("topk_scaling_factor", None)
    config.num_groups = model_cfg.get("num_groups", None)
    config.group_topk = model_cfg.get("group_topk", None)


def build_model(cfg: dict):
    """Build a model and its config from a raw config dict.

    Returns (model, model_config).
    """
    mtype = cfg["model"]["type"]
    mcfg = cfg["model"]
    attn_impl = mcfg.get("attn_implementation", "sdpa")

    # Reject deprecated model types with migration guidance
    if mtype in _DEPRECATED_TYPES:
        raise ValueError(
            f"Model type '{mtype}' is deprecated. "
            f"{_DEPRECATED_TYPES[mtype]}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}"
        )

    # Reject archived model types
    if mtype in _ARCHIVED_TYPES:
        raise ValueError(
            f"Model type '{mtype}' has been archived to legacy/. "
            f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}"
        )

    if mtype not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unknown model type: '{mtype}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}"
        )

    # Determine router type from config
    router_type = mcfg.get("router_type", "softmax")
    use_deepseek = router_type == "deepseek" or mcfg.get("use_deepseek_routing", False)

    if mtype == "dense":
        config = Qwen3Config(
            vocab_size=mcfg["vocab_size"],
            hidden_size=mcfg["hidden_size"],
            num_hidden_layers=mcfg["num_hidden_layers"],
            head_dim=mcfg["head_dim"],
            num_attention_heads=mcfg["num_attention_heads"],
            num_key_value_heads=mcfg["num_key_value_heads"],
            intermediate_size=mcfg["intermediate_size"],
            max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
            rope_theta=mcfg.get("rope_theta", 1_000_000.0),
            rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        )
        config._attn_implementation = attn_impl
        config.num_experts = 0
        config.num_experts_per_tok = 0
        model = Qwen3ForCausalLM(config)
        return model, config

    # Common MoE parameters
    common = dict(
        vocab_size=mcfg["vocab_size"],
        hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        head_dim=mcfg["head_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        moe_intermediate_size=mcfg["moe_intermediate_size"],
        intermediate_size=mcfg.get("intermediate_size", mcfg["moe_intermediate_size"] * 4),
        max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
        rope_theta=mcfg.get("rope_theta", 1_000_000.0),
        rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        router_aux_loss_coef=mcfg.get("router_aux_loss_coef", 0.001),
        seq_aux_loss_coef=mcfg.get("seq_aux_loss_coef", 0.0),
        norm_topk_prob=mcfg.get("norm_topk_prob", True),
        num_experts_per_tok=mcfg["num_experts_per_tok"],
        output_router_logits=True,
        attn_implementation=attn_impl,
    )

    if mtype == "standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        if use_deepseek:
            _set_deepseek_router_params(config, mcfg)
            model = DeepSeekStandardMoEModel(config)
        else:
            _set_router_params(config, mcfg)
            model = StandardMoEModel(config)

    elif mtype == "global_moe":
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        if use_deepseek:
            _set_deepseek_router_params(config, mcfg)
            model = DeepSeekGlobalMoEForCausalLM(config)
        else:
            _set_router_params(config, mcfg)
            model = GlobalMoEForCausalLM(config)

    elif mtype == "moe_everything":
        config = MoEverythingConfig(
            num_experts=mcfg["num_experts"],
            num_attn_experts=mcfg.get("num_attn_experts", 4),
            num_attn_experts_per_tok=mcfg.get("num_attn_experts_per_tok", 1),
            attn_expert_mode=mcfg.get("attn_expert_mode", "bundled"),
            branch_router_aux_loss_coef=mcfg.get("branch_router_aux_loss_coef", 0.0),
            use_deepseek_routing=use_deepseek,
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            per_layer_router=mcfg.get("per_layer_router", False),
            per_layer_mlp_router=mcfg.get("per_layer_mlp_router", False),
            per_layer_attn_router=mcfg.get("per_layer_attn_router", False),
            routed_norm=mcfg.get("routed_norm", False),
            per_layer_norm=mcfg.get("per_layer_norm", False),
            per_layer_qk_norm=mcfg.get("per_layer_qk_norm", False),
            post_norm=mcfg.get("post_norm", False),
            dynamic_depth_min=mcfg.get("dynamic_depth_min", 1.0),
            dynamic_depth_max=mcfg.get("dynamic_depth_max", 1.0),
            depthwise_attention=mcfg.get("depthwise_attention", False),
            depthwise_block_size=mcfg.get("depthwise_block_size", 0),
            per_head_compute_mode=mcfg.get("per_head_compute_mode", "auto"),
            per_head_dense_fraction_threshold=mcfg.get("per_head_dense_fraction_threshold", 0.75),
            sanity_check_mode=mcfg.get("sanity_check_mode"),
            scale_attn_by_routing_weight=mcfg.get("scale_attn_by_routing_weight", True),
            scale_branch_by_routing_weight=mcfg.get("scale_branch_by_routing_weight", True),
            router_exploration_rate=mcfg.get("router_exploration_rate", 0.0),
            branch_router_exploration_rate=mcfg.get("branch_router_exploration_rate"),
            branch_sampling=mcfg.get("branch_sampling", False),
            branch_level=mcfg.get("branch_level", "token"),
            branch_deepseek=mcfg.get("branch_deepseek", False),
            **common,
        )
        model = MoEverythingForCausalLM(config)

    if hasattr(model, "set_experts_implementation"):
        experts_impl = mcfg.get("experts_implementation", "grouped_mm")
        try:
            model.set_experts_implementation(experts_impl)
        except Exception:
            model.set_experts_implementation("eager")

    return model, config
