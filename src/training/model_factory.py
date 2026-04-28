"""Model factory: build models from config dicts.

Supported model types:
- dense: Qwen3-based dense transformer
- standard_moe: Standard MoE with softmax or DeepSeek routing (router_type config)
- global_moe: Global MoE with shared expert pool and softmax or DeepSeek routing
- moe_everything: MoE-Everything with per-head attention routing
"""

from __future__ import annotations

import os

# Re-export the canonical-block resolver so existing import sites
# (`from .model_factory import _resolve_balancing_field`) keep working.
from .balancing_fields import (  # noqa: F401
    _BALANCING_FIELDS_IN_TRAINING,
    _resolve_balancing_field,
    output_router_logits_for_method,
)

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
    """Plumb shared router knobs (softmax-family and DeepSeek both honour these)."""
    config.router_exploration_rate = model_cfg.get("router_exploration_rate", 0.0)
    # Scoring function and softmax position (softmax-family router only;
    # DeepSeek forces sigmoid + its own selection path). Softmax-position migration (
    # softmax_position): `softmax_position` is the canonical field name;
    # `router_topk_ordering` is accepted as a deprecated alias with a
    # DeprecationWarning emitted from `_resolve_softmax_position`.
    config.router_score_function = model_cfg.get("router_score_function", "softmax")
    if "softmax_position" in model_cfg:
        config.softmax_position = model_cfg["softmax_position"]
    if "router_topk_ordering" in model_cfg:
        config.router_topk_ordering = model_cfg["router_topk_ordering"]
    # Group-limited top-k (applies to both router families when set).
    config.num_groups = model_cfg.get("num_groups", None)
    config.group_topk = model_cfg.get("group_topk", None)
    # Z-loss stabiliser on raw router logits (independent of scoring function;
    # cached per router call, accumulated into aux loss by the trainer).
    config.router_z_loss_coef = model_cfg.get("router_z_loss_coef", 0.0)


def _set_deepseek_router_params(config, model_cfg: dict) -> None:
    _set_router_params(config, model_cfg)
    config.topk_scaling_factor = model_cfg.get("topk_scaling_factor", None)


def _get_branch_router_field(model_cfg: dict, key: str, default):
    """Read a `branch_router.<key>` field with a flat-schema fallback.

    Nested form (preferred):
        model:
          branch_router:
            balancing: exploration_only
            exploration_rate: 1.0
            exploration_decay: cosine
            exploration_min: 0.0
            exploration_warmup_steps: 1000

    Flat fallback (legacy yamls): `branch_balancing`,
    `branch_exploration_rate`, etc. live directly on the `model:` block.
    The nested form wins when both are present.
    """
    nested = model_cfg.get("branch_router")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    flat_key = f"branch_{key}"
    return model_cfg.get(flat_key, default)


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

    # Detach-only policy: resolve `output_router_logits` from the method.
    # For aux methods, router scores need to be gradient-bearing in the
    # model output so the aux loss term can backprop through them. For
    # non-aux methods (deepseek_bias, quantile, none), we don't return
    # them — the routing-decision state lives in router-internal buffers.
    method_for_orl = _resolve_balancing_field(cfg, "load_balancing_method", None)
    output_router_logits = output_router_logits_for_method(method_for_orl)

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
        # Per the canonical-block resolver: aux coefficients live in `training:`. Fall back to
        # `model:` with a deprecation warning for unmigrated yamls.
        router_aux_loss_coef=_resolve_balancing_field(cfg, "router_aux_loss_coef", 0.001),
        seq_aux_loss_coef=_resolve_balancing_field(cfg, "seq_aux_loss_coef", 0.0),
        norm_topk_prob=mcfg.get("norm_topk_prob", True),
        num_experts_per_tok=mcfg["num_experts_per_tok"],
        output_router_logits=output_router_logits,
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
            attn_expert_mode=mcfg.get("attn_expert_mode", "per_head_fully_independent"),
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
            branch_balancing=_get_branch_router_field(mcfg, "balancing", "none"),
            branch_exploration_rate=_get_branch_router_field(
                mcfg, "exploration_rate", 0.0
            ),
            branch_exploration_decay=_get_branch_router_field(
                mcfg, "exploration_decay", "constant"
            ),
            branch_exploration_min=_get_branch_router_field(
                mcfg, "exploration_min", 0.0
            ),
            branch_exploration_warmup_steps=_get_branch_router_field(
                mcfg, "exploration_warmup_steps", 0
            ),
            **common,
        )
        # Router-option knobs attached post-construction (the config __init__
        # does not currently enumerate them; _set_router_params is the single
        # source of truth across all MoE families). softmax_position migration:
        # `softmax_position` is the canonical field name; legacy
        # `router_topk_ordering` is accepted with a DeprecationWarning.
        config.router_score_function = mcfg.get("router_score_function", "softmax")
        if "softmax_position" in mcfg:
            config.softmax_position = mcfg["softmax_position"]
        if "router_topk_ordering" in mcfg:
            config.router_topk_ordering = mcfg["router_topk_ordering"]
        config.router_z_loss_coef = mcfg.get("router_z_loss_coef", 0.0)
        model = MoEverythingForCausalLM(config)

    if hasattr(model, "set_experts_implementation"):
        experts_impl = mcfg.get("experts_implementation", "grouped_mm")
        try:
            model.set_experts_implementation(experts_impl)
        except Exception:
            model.set_experts_implementation("eager")

    # Coefficient gating per method: stamp the resolved `load_balancing_method` onto BOTH the
    # model and its config so every caller (the trainer, ad-hoc test fixtures,
    # the post-step `update_expert_biases` walker) sees the same authoritative
    # value without needing to re-resolve from `cfg`. `normalize_balancing_config`
    # has already auto-zeroed conflicting coefficients (called from
    # `load_config`), so this is the single source of truth that downstream
    # method-driven dispatch reads.
    method = _resolve_balancing_field(cfg, "load_balancing_method", None)
    if method is not None:
        model._load_balancing_method = method
        config.load_balancing_method = method

    return model, config
