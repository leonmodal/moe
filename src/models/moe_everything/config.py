"""MoE-Everything configuration."""

from src.models.configuration_qwen3_moe import Qwen3MoeConfig


class MoEverythingConfig(Qwen3MoeConfig):
    model_type = "moe_everything"

    def __init__(
        self,
        num_attn_experts: int = 4,
        num_attn_experts_per_tok: int = 1,
        attn_expert_mode: str = "per_head_no_recompute",
        attn_routing_bundle: str | None = None,
        branch_router_aux_loss_coef: float = 0.0,
        use_deepseek_routing: bool = False,
        per_layer_router: bool = False,
        per_layer_mlp_router: bool = False,
        per_layer_attn_router: bool = False,
        # When True, each depth gets its own independent attention + MLP
        # expert bank (separate weight tensors per depth) instead of the
        # default global-bank where one set of expert weights is shared
        # across all depths.
        per_layer_expert_bank: bool = False,
        # When True (and `attn_expert_mode='per_head_recompute_kv'`), the
        # V-expert is re-routed per (query position, key position) pair via
        # a new "pair-V" router that takes (hidden[q], hidden[k]) inputs.
        # K still routes per-query (same K-projection across all key
        # positions for a given query) — only V varies per pair. See
        # `attention_bank.py::_run_pair_v_routing`.
        per_pair_v_routing: bool = False,
        routed_norm: bool = False,
        per_layer_norm: bool = False,
        per_layer_qk_norm: bool = False,
        post_norm: bool = False,
        dynamic_depth_min: float = 1.0,
        dynamic_depth_max: float = 1.0,
        depthwise_attention: bool = False,
        depthwise_block_size: int = 0,
        sanity_check_mode: str | None = None,
        scale_attn_by_routing_weight: bool = True,
        scale_branch_by_routing_weight: bool = True,
        attn_router_context: str = "none",
        attn_router_context_decay: float = 0.95,
        router_exploration_rate: float = 0.0,
        branch_router_exploration_rate: float | None = None,
        # BranchRouter mode options.
        branch_sampling: bool = False,
        branch_level: str = "token",  # "token" or "seq"
        branch_deepseek: bool = False,
        # Branch-router balancing knobs. Active configs should use the
        # nested `model.branch_router` block; these flat attributes are
        # the runtime surface after model_factory expands that block.
        #   `branch_balancing` is one of:
        #     none, sampling_entropy, aux_loss, seq_aux_loss,
        #     deepseek_bias, quantile, exploration_only.
        #     `deepseek_bias` routes with two independent sigmoid scores
        #     plus a persistent ATTN/MLP bias. `sampling_entropy`
        #     samples the branch categorical during training and
        #     adds a decaying branch entropy bonus in the model
        #     forward. `exploration_only` turns
        #     the branch router into a uniform-random-with-rate
        #     exploration probe and disables every branch aux/bias path
        #     by construction.
        #   `branch_exploration_rate`: initial `p_explore` rate at
        #     step 0. Default 0.0 so the BranchRouter auto-promote
        #     rule (rate>0 + balancing="none" → balancing="exploration_only")
        #     never fires when the user has not opted into the mode;
        #     callers that opt into `branch_balancing="exploration_only"`
        #     must set this explicitly to a positive value.
        #   `branch_exploration_decay` ∈ {"constant", "linear", "cosine"}.
        #     Decay shape applied by the trainer's per-step schedule.
        #   `branch_exploration_min`: floor for the decay schedule.
        #   `branch_exploration_warmup_steps`: decay length.
        #   `branch_entropy_*`: coefficient schedule for
        #     `branch_balancing="sampling_entropy"`.
        branch_balancing: str = "none",
        branch_exploration_rate: float = 0.0,
        branch_exploration_decay: str = "constant",
        branch_exploration_min: float = 0.0,
        branch_exploration_warmup_steps: int = 0,
        branch_entropy_coef: float = 0.0,
        branch_entropy_decay: str = "constant",
        branch_entropy_min: float = 0.0,
        branch_entropy_decay_steps: int = 0,
        # Prelude / coda hybrid: prepend `prelude_layers` and append
        # `coda_layers` standard-MoE-style decoder blocks (own per-layer
        # MLP expert pool + dense GQA, no bank) around the recurrent
        # depth loop. Defaults of 0 keep the model bit-identical to a
        # pure MoE-Everything build. Boundary MLP geometry is configured
        # via the `boundary_*` fields below; boundary attention reuses
        # `hidden_size / num_attention_heads / num_key_value_heads /
        # head_dim / rms_norm_eps / rope_theta` from the main config.
        prelude_layers: int = 0,
        coda_layers: int = 0,
        boundary_num_experts: int | None = None,
        boundary_num_experts_per_tok: int | None = None,
        boundary_moe_intermediate_size: int | None = None,
        boundary_num_groups: int | None = None,
        boundary_group_topk: int | None = None,
        boundary_router_type: str = "deepseek",
        boundary_bias_update_rate: float = 0.001,
        boundary_bias_update_zero_sum: bool = True,
        boundary_norm_topk_prob: bool | None = None,
        boundary_topk_scaling_factor: float | None = None,
        recurrent: bool = False,
        recurrent_adapter: bool = True,
        recurrent_adapter_intermediate_size: int | None = None,
        recurrent_adapter_activation: str = "gelu",
        recurrent_adapter_bias: bool = False,
        eval_recurrence: int = 32,
        mean_backprop_depth: int = 8,
        use_fused_linear_ce: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attn_experts = num_attn_experts
        self.num_attn_experts_per_tok = num_attn_experts_per_tok
        if attn_routing_bundle is None:
            attn_routing_bundle = (
                "q_k_v_o" if attn_expert_mode == "per_head_no_recompute" else "qkvo"
            )
        self.attn_expert_mode = attn_expert_mode
        self.attn_routing_bundle = attn_routing_bundle
        self.branch_router_aux_loss_coef = branch_router_aux_loss_coef
        self.use_deepseek_routing = use_deepseek_routing
        self.per_layer_router = per_layer_router
        self.per_layer_mlp_router = per_layer_mlp_router
        self.per_layer_attn_router = per_layer_attn_router
        self.per_layer_expert_bank = per_layer_expert_bank
        self.per_pair_v_routing = per_pair_v_routing
        self.routed_norm = routed_norm
        self.per_layer_norm = per_layer_norm
        self.per_layer_qk_norm = per_layer_qk_norm
        self.post_norm = post_norm
        self.dynamic_depth_min = dynamic_depth_min
        self.dynamic_depth_max = dynamic_depth_max
        self.depthwise_attention = depthwise_attention
        self.depthwise_block_size = depthwise_block_size
        self.sanity_check_mode = sanity_check_mode
        self.scale_attn_by_routing_weight = scale_attn_by_routing_weight
        self.scale_branch_by_routing_weight = scale_branch_by_routing_weight
        self.attn_router_context = attn_router_context
        self.attn_router_context_decay = attn_router_context_decay
        self.router_exploration_rate = router_exploration_rate
        self.branch_router_exploration_rate = (
            router_exploration_rate
            if branch_router_exploration_rate is None
            else branch_router_exploration_rate
        )
        self.branch_sampling = branch_sampling
        self.branch_level = branch_level
        self.branch_deepseek = branch_deepseek
        self.branch_balancing = branch_balancing
        self.branch_exploration_rate = branch_exploration_rate
        self.branch_exploration_decay = branch_exploration_decay
        self.branch_exploration_min = branch_exploration_min
        self.branch_exploration_warmup_steps = branch_exploration_warmup_steps
        self.branch_entropy_coef = branch_entropy_coef
        self.branch_entropy_decay = branch_entropy_decay
        self.branch_entropy_min = branch_entropy_min
        self.branch_entropy_decay_steps = branch_entropy_decay_steps
        self.current_branch_entropy_coef = branch_entropy_coef
        # Prelude / coda hybrid knobs. Boundary MLP geometry falls back
        # to the main bank's geometry when the explicit boundary_* field
        # is None, so a config that only sets prelude_layers/coda_layers
        # still produces a buildable model.
        self.prelude_layers = int(prelude_layers)
        self.coda_layers = int(coda_layers)
        if self.prelude_layers < 0 or self.coda_layers < 0:
            raise ValueError(
                f"prelude_layers and coda_layers must be >= 0, "
                f"got prelude_layers={self.prelude_layers}, coda_layers={self.coda_layers}"
            )
        self.boundary_num_experts = (
            int(boundary_num_experts) if boundary_num_experts is not None else None
        )
        self.boundary_num_experts_per_tok = (
            int(boundary_num_experts_per_tok)
            if boundary_num_experts_per_tok is not None else None
        )
        self.boundary_moe_intermediate_size = (
            int(boundary_moe_intermediate_size)
            if boundary_moe_intermediate_size is not None else None
        )
        self.boundary_num_groups = (
            int(boundary_num_groups) if boundary_num_groups is not None else None
        )
        self.boundary_group_topk = (
            int(boundary_group_topk) if boundary_group_topk is not None else None
        )
        self.boundary_router_type = str(boundary_router_type)
        self.boundary_bias_update_rate = float(boundary_bias_update_rate)
        self.boundary_bias_update_zero_sum = bool(boundary_bias_update_zero_sum)
        self.boundary_norm_topk_prob = boundary_norm_topk_prob
        self.boundary_topk_scaling_factor = (
            float(boundary_topk_scaling_factor)
            if boundary_topk_scaling_factor is not None else None
        )
        self.recurrent = bool(recurrent)
        self.recurrent_adapter = bool(recurrent_adapter)
        self.recurrent_adapter_intermediate_size = (
            int(recurrent_adapter_intermediate_size)
            if recurrent_adapter_intermediate_size is not None
            else None
        )
        self.recurrent_adapter_activation = str(recurrent_adapter_activation)
        self.recurrent_adapter_bias = bool(recurrent_adapter_bias)
        self.eval_recurrence = int(eval_recurrence)
        self.mean_backprop_depth = int(mean_backprop_depth)
        self.recurrent_layers = int(self.num_hidden_layers)
        self.use_fused_linear_ce = use_fused_linear_ce
