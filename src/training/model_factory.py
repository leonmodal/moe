"""Model factory: build models from config dicts.

Supported model types:
- dense: Qwen3-based dense transformer
- standard_moe: Standard MoE with softmax or DeepSeek routing (router_type config)
- global_moe: Global MoE with shared expert pool and softmax or DeepSeek routing
- moe_everything: MoE-Everything with per-head attention routing
- recurrent_standard_moe / recurrent_global_moe: 4-8-4 recurrent normal MoE blocks
- recurrent_moe_everything: 4-8-4 recurrent MoE-Everything middle bank
- hrm_recurrent_standard_moe: recurrent standard MoE split into 4-layer L/H modules
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
    RecurrentMoEConfig,
    RecurrentMoEForCausalLM,
)


# Model types that have been archived to legacy/
_ARCHIVED_TYPES = {
    "speedrun_gpt",
    "speedrun_moe_everything",
    "gpt2_dense",
}

# Supported model types
SUPPORTED_TYPES = {
    "dense",
    "standard_moe",
    "global_moe",
    "moe_everything",
    "recurrent_moe_everything",
    "recurrent_standard_moe",
    "recurrent_global_moe",
    "hrm_recurrent_standard_moe",
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
    mtype = cfg["model"]["type"]
    has_model_native_fused_ce = (
        mtype in {
            "standard_moe",
            "global_moe",
            "moe_everything",
            "recurrent_moe_everything",
            "recurrent_standard_moe",
            "recurrent_global_moe",
            "hrm_recurrent_standard_moe",
        }
        and cfg.get("model", {}).get("use_fused_linear_ce", True)
    )
    if training_cfg.get("disable_liger", False) or os.environ.get("MOE_DISABLE_LIGER", "0") == "1":
        if has_model_native_fused_ce:
            return "patching disabled; model-native fused linear CE remains active"
        return "disabled"

    if mtype in _ARCHIVED_TYPES or mtype == "gpt2_dense":
        return "disabled (unsupported model type)"

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
    except ImportError:
        return "disabled (liger_kernel not installed)"

    if mtype in {"moe_everything", "recurrent_moe_everything"}:
        apply_liger_kernel_to_qwen3_moe(
            rope=True,
            rms_norm=True,
            swiglu=False,
            fused_linear_cross_entropy=False,
            cross_entropy=False,
        )
        return "partial (rope+rms_norm patch; fused CE is model-native for moe_everything)"

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


def _stamp_per_class_router_fields(config, model_cfg: dict) -> None:
    """Stamp the per-class nested router fields
    (`model.mlp_router`, `model.attn_router`) onto `config` so the
    runtime sees per-class methods. Each field is exposed under a
    flat attribute name on `config` (`mlp_router_balancing`,
    `mlp_router_quantile_eta`, etc.) so existing callers that read
    config attributes directly do not need to traverse the nested
    structure. The branch_router is plumbed via the dedicated
    `_get_branch_router_field` helper that supports the legacy
    flat-bridge form, so it is intentionally left out of this
    walker.

    When a per-class block is absent from the yaml, no attributes
    are stamped — the runtime falls back to the top-level
    `load_balancing_method` and the legacy coefficient defaults,
    preserving behavior for unmigrated configs.
    """
    for group in ("mlp_router", "attn_router", "branch_router"):
        nested = model_cfg.get(group)
        if not isinstance(nested, dict):
            continue
        prefix = f"{group}_"
        for key, value in nested.items():
            setattr(config, prefix + key, value)
    # Branch-router: the runtime reads `config.branch_balancing`
    # (set via the dedicated flat-bridge plumbing earlier in
    # `build_model`). For per-class branch coefficients
    # (`router_aux_loss_coef`, `seq_aux_loss_coef`) we expose the
    # full nested branch_router dict on `config._mcfg_branch_router`
    # so the runtime can read knobs without parsing through the
    # legacy flat-bridge. Empty / absent block is set to {} so
    # downstream `getattr` defaults work without explicit None
    # checks.
    branch_nested = model_cfg.get("branch_router")
    config._mcfg_branch_router = (
        dict(branch_nested) if isinstance(branch_nested, dict) else {}
    )


def _get_prelude_coda_field(model_cfg: dict, key: str, default=None):
    """Read a `prelude_coda.<key>` field from the model config.

    The nested `model.prelude_coda` block carries standard-MoE-style
    settings for the prelude/coda decoder layers (num_experts,
    num_experts_per_tok, moe_intermediate_size, num_groups, group_topk,
    router_type, etc.). Returns `default` when the block or key is
    absent; absent block + `prelude_layers=0` + `coda_layers=0` is the
    "no boundary" default.
    """
    nested = model_cfg.get("prelude_coda")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return default


def _get_branch_router_field(model_cfg: dict, key: str, default):
    """Read a `branch_router.<key>` field with a flat-schema fallback.

    Nested form (preferred):
        model:
          branch_router:
            balancing: sampling_entropy
            entropy_coef: 0.01
            entropy_decay: cosine
            entropy_min: 0.0
            entropy_decay_steps: 1000

    Flat fallback (legacy yamls): `branch_balancing`,
    `branch_exploration_rate`, `branch_entropy_coef`, etc. live directly
    on the `model:` block.
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
    attn_impl = os.environ.get("MOE_EVERYTHING_FORCE_ATTN_IMPL") or mcfg.get(
        "attn_implementation", "sdpa"
    )

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
    #
    # Resolution order:
    #   1. Top-level `training.load_balancing_method` (or `model:`
    #      back-compat).
    #   2. Nested `model.mlp_router.balancing` (the nested-only
    #      runtime — when the migrator has stripped the top-level
    #      method, this is the authoritative source). Standard /
    #      Global MoE have only an MLP router class, so the MLP
    #      method is the model-level method.
    #   3. Nested `model.attn_router.balancing` (used as a fallback
    #      for moe_everything when MLP is unset but attention is).
    method_for_orl = _resolve_balancing_field(cfg, "load_balancing_method", None)
    if method_for_orl is None:
        nested_mlp = mcfg.get("mlp_router")
        if isinstance(nested_mlp, dict):
            cand = nested_mlp.get("balancing")
            if cand is not None:
                method_for_orl = cand
        if method_for_orl is None:
            nested_attn = mcfg.get("attn_router")
            if isinstance(nested_attn, dict):
                cand = nested_attn.get("balancing")
                if cand is not None:
                    method_for_orl = cand
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
        use_fused_linear_ce=mcfg.get("use_fused_linear_ce", True),
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

    elif mtype in {"moe_everything", "recurrent_moe_everything"}:
        recurrent_moe_everything = mtype == "recurrent_moe_everything"
        config = MoEverythingConfig(
            num_experts=mcfg["num_experts"],
            num_attn_experts=mcfg.get("num_attn_experts", 4),
            num_attn_experts_per_tok=mcfg.get("num_attn_experts_per_tok", 1),
            attn_expert_mode=mcfg.get("attn_expert_mode", "per_head_no_recompute"),
            attn_routing_bundle=mcfg.get("attn_routing_bundle", None),
            branch_router_aux_loss_coef=mcfg.get("branch_router_aux_loss_coef", 0.0),
            use_deepseek_routing=use_deepseek,
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            per_layer_router=mcfg.get("per_layer_router", False),
            per_layer_mlp_router=mcfg.get("per_layer_mlp_router", False),
            per_layer_attn_router=mcfg.get("per_layer_attn_router", False),
            per_layer_expert_bank=mcfg.get("per_layer_expert_bank", False),
            per_pair_v_routing=mcfg.get("per_pair_v_routing", False),
            routed_norm=mcfg.get("routed_norm", False),
            per_layer_norm=mcfg.get("per_layer_norm", False),
            per_layer_qk_norm=mcfg.get("per_layer_qk_norm", False),
            post_norm=mcfg.get("post_norm", False),
            dynamic_depth_min=mcfg.get("dynamic_depth_min", 1.0),
            dynamic_depth_max=mcfg.get("dynamic_depth_max", 1.0),
            depthwise_attention=mcfg.get("depthwise_attention", False),
            depthwise_block_size=mcfg.get("depthwise_block_size", 0),
            sanity_check_mode=mcfg.get("sanity_check_mode"),
            scale_attn_by_routing_weight=mcfg.get("scale_attn_by_routing_weight", True),
            scale_branch_by_routing_weight=mcfg.get("scale_branch_by_routing_weight", True),
            attn_router_context=mcfg.get("attn_router_context", "none"),
            attn_router_context_decay=mcfg.get("attn_router_context_decay", 0.95),
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
            branch_entropy_coef=_get_branch_router_field(
                mcfg, "entropy_coef", 0.0
            ),
            branch_entropy_decay=_get_branch_router_field(
                mcfg, "entropy_decay", "constant"
            ),
            branch_entropy_min=_get_branch_router_field(
                mcfg, "entropy_min", 0.0
            ),
            branch_entropy_decay_steps=_get_branch_router_field(
                mcfg, "entropy_decay_steps", 0
            ),
            prelude_layers=mcfg.get("prelude_layers", 0),
            coda_layers=mcfg.get("coda_layers", 0),
            boundary_num_experts=_get_prelude_coda_field(mcfg, "num_experts"),
            boundary_num_experts_per_tok=_get_prelude_coda_field(
                mcfg, "num_experts_per_tok"
            ),
            boundary_moe_intermediate_size=_get_prelude_coda_field(
                mcfg, "moe_intermediate_size"
            ),
            boundary_num_groups=_get_prelude_coda_field(mcfg, "num_groups"),
            boundary_group_topk=_get_prelude_coda_field(mcfg, "group_topk"),
            boundary_router_type=_get_prelude_coda_field(
                mcfg, "router_type", "deepseek"
            ),
            boundary_bias_update_rate=_get_prelude_coda_field(
                mcfg, "bias_update_rate", 0.001
            ),
            boundary_bias_update_zero_sum=_get_prelude_coda_field(
                mcfg, "bias_update_zero_sum", True
            ),
            boundary_norm_topk_prob=_get_prelude_coda_field(mcfg, "norm_topk_prob"),
            boundary_topk_scaling_factor=_get_prelude_coda_field(
                mcfg, "topk_scaling_factor"
            ),
            recurrent=recurrent_moe_everything or mcfg.get("recurrent", False),
            recurrent_adapter=mcfg.get("recurrent_adapter", True),
            recurrent_adapter_intermediate_size=mcfg.get("recurrent_adapter_intermediate_size"),
            recurrent_adapter_activation=mcfg.get("recurrent_adapter_activation", "gelu"),
            recurrent_adapter_bias=mcfg.get("recurrent_adapter_bias", False),
            eval_recurrence=mcfg.get("eval_recurrence", cfg.get("recurrence", {}).get("eval_recurrence", 32)),
            mean_backprop_depth=mcfg.get(
                "mean_backprop_depth",
                cfg.get("recurrence", {}).get("mean_backprop_depth", 8),
            ),
            **common,
        )
        config.model_type = mtype
        # Branch quantile knobs — only used when branch_balancing=="quantile",
        # but the model constructor reads them unconditionally from config.
        bq_target = _get_branch_router_field(mcfg, "quantile_target_q", None)
        bq_eta = _get_branch_router_field(mcfg, "quantile_eta", None)
        if bq_target is not None:
            config.branch_quantile_target_q = float(bq_target)
        if bq_eta is not None:
            config.branch_quantile_eta = float(bq_eta)
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

    elif mtype in {
        "recurrent_standard_moe",
        "recurrent_global_moe",
        "hrm_recurrent_standard_moe",
    }:
        use_global_pool = mtype == "recurrent_global_moe"
        use_hrm_loop = mtype == "hrm_recurrent_standard_moe"
        config = RecurrentMoEConfig(
            num_experts=mcfg["num_experts"],
            prelude_layers=mcfg.get("prelude_layers", 4),
            recurrent_layers=mcfg.get("recurrent_layers", 8),
            coda_layers=mcfg.get("coda_layers", 4),
            recurrent_expert_pool="global" if use_global_pool else "per_layer",
            core_attention=mcfg.get("core_attention", True),
            core_attention_layers=mcfg.get("core_attention_layers"),
            recurrent_loop="hrm" if use_hrm_loop else mcfg.get("recurrent_loop", "flat"),
            hrm_l_layers=mcfg.get("hrm_l_layers", 4),
            hrm_h_layers=mcfg.get("hrm_h_layers", 4),
            hrm_history_attention=mcfg.get("hrm_history_attention", use_hrm_loop),
            recurrent_sandwich_norm=mcfg.get("recurrent_sandwich_norm", False),
            recurrent_history_attention=mcfg.get("recurrent_history_attention", False),
            recurrent_adapter=mcfg.get("recurrent_adapter", True),
            recurrent_adapter_type=mcfg.get("recurrent_adapter_type", "mlp"),
            recurrent_adapter_intermediate_size=mcfg.get("recurrent_adapter_intermediate_size"),
            recurrent_adapter_activation=mcfg.get("recurrent_adapter_activation", "gelu"),
            recurrent_adapter_bias=mcfg.get("recurrent_adapter_bias", False),
            eval_recurrence=mcfg.get("eval_recurrence", cfg.get("recurrence", {}).get("eval_recurrence", 32)),
            mean_backprop_depth=mcfg.get(
                "mean_backprop_depth",
                cfg.get("recurrence", {}).get("mean_backprop_depth", 8),
            ),
            boundary_num_experts=_get_prelude_coda_field(mcfg, "num_experts"),
            boundary_num_experts_per_tok=_get_prelude_coda_field(
                mcfg, "num_experts_per_tok"
            ),
            boundary_moe_intermediate_size=_get_prelude_coda_field(
                mcfg, "moe_intermediate_size"
            ),
            use_deepseek_routing=use_deepseek,
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            global_router_update=mcfg.get("global_router_update", use_global_pool),
            router_exploration_rate=mcfg.get("router_exploration_rate", 0.0),
            router_z_loss_coef=mcfg.get("router_z_loss_coef", 0.0),
            **common,
        )
        config.router_score_function = mcfg.get("router_score_function", "softmax")
        if "softmax_position" in mcfg:
            config.softmax_position = mcfg["softmax_position"]
        if "router_topk_ordering" in mcfg:
            config.router_topk_ordering = mcfg["router_topk_ordering"]
        config.model_type = mtype
        model = RecurrentMoEForCausalLM(config)

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

    _stamp_per_class_router_fields(config, mcfg)

    # Per-class method override on the model: when the nested-schema
    # yaml has set `model.mlp_router.balancing` to a real method, that
    # value WINS over any top-level method for the post-step
    # bias-update dispatch. Standard / Global MoE families have only
    # an MLP router class, so stamping the MLP per-class method onto
    # `model._load_balancing_method` makes `update_expert_biases`
    # dispatch on the per-class value automatically. MoE-Everything
    # has its own per-class gating in `forward`; we still stamp here
    # so the post-step walker sees the right method when MLP is the
    # active one.
    mlp_class_method = mcfg.get("mlp_router", {}).get("balancing") if isinstance(
        mcfg.get("mlp_router"), dict,
    ) else None
    if mlp_class_method is not None and mlp_class_method != "none":
        model._load_balancing_method = mlp_class_method
        config.load_balancing_method = mlp_class_method

    # Per-owner deepseek_bias knob mirroring. When several routers
    # opt into `deepseek_bias` simultaneously (e.g. MLP at rate 1e-3
    # and branch at rate 5e-4), the trainer must dispatch the OWNER's
    # rate / warmup / zero-sum to that owner — not collapse them onto
    # one global value. Stamp `effective_<owner>_bias_*` for each
    # nested per-class block; keep the un-prefixed `effective_bias_*`
    # alias pointing at MLP first then branch for back-compat with
    # callers that still read the global field.
    owner_to_block: list[tuple[str, dict]] = []
    if isinstance(mcfg.get("mlp_router"), dict):
        owner_to_block.append(("mlp", mcfg["mlp_router"]))
    if isinstance(mcfg.get("attn_router"), dict):
        owner_to_block.append(("attn", mcfg["attn_router"]))
    if isinstance(mcfg.get("branch_router"), dict):
        owner_to_block.append(("branch", mcfg["branch_router"]))
    bias_keys = (
        "bias_update_rate", "bias_update_zero_sum",
        "bias_warmup_start", "bias_warmup_steps",
    )
    for owner, block in owner_to_block:
        if block.get("balancing") != "deepseek_bias":
            continue
        for key in bias_keys:
            if key in block:
                setattr(config, f"effective_{owner}_{key}", block[key])
                # Back-compat global alias: first writer wins so MLP
                # takes precedence over branch under mixed configs,
                # matching prior behavior.
                if not hasattr(config, f"effective_{key}"):
                    setattr(config, f"effective_{key}", block[key])

    return model, config
