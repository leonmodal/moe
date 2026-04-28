"""MoE-Everything configuration."""

from src.models.configuration_qwen3_moe import Qwen3MoeConfig


class MoEverythingConfig(Qwen3MoeConfig):
    model_type = "moe_everything"

    def __init__(
        self,
        num_attn_experts: int = 4,
        num_attn_experts_per_tok: int = 1,
        attn_expert_mode: str = "per_head_fully_independent",
        branch_router_aux_loss_coef: float = 0.0,
        use_deepseek_routing: bool = False,
        per_layer_router: bool = False,
        per_layer_mlp_router: bool = False,
        per_layer_attn_router: bool = False,
        routed_norm: bool = False,
        per_layer_norm: bool = False,
        per_layer_qk_norm: bool = False,
        post_norm: bool = False,
        dynamic_depth_min: float = 1.0,
        dynamic_depth_max: float = 1.0,
        depthwise_attention: bool = False,
        depthwise_block_size: int = 0,
        per_head_compute_mode: str = "auto",
        per_head_dense_fraction_threshold: float = 0.75,
        sanity_check_mode: str | None = None,
        scale_attn_by_routing_weight: bool = True,
        scale_branch_by_routing_weight: bool = True,
        router_exploration_rate: float = 0.0,
        branch_router_exploration_rate: float | None = None,
        # BranchRouter mode options (from speedrun extraction)
        branch_sampling: bool = False,
        branch_level: str = "token",  # "token" or "seq"
        branch_deepseek: bool = False,
        # Branch-router balancing knobs (flat-schema bridge until the
        # nested-schema migration lands). The full nested schema will
        # eventually live under `branch_router.{...}` in yaml; this
        # flat surface keeps that migration low-risk.
        #   `branch_balancing` ∈ {"none", "exploration_only"}.
        #     Default "none" preserves the existing routing
        #     behavior. "exploration_only" turns the branch router
        #     into a uniform-random-with-rate exploration probe and
        #     disables every branch aux/bias path by construction.
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
        branch_balancing: str = "none",
        branch_exploration_rate: float = 0.0,
        branch_exploration_decay: str = "constant",
        branch_exploration_min: float = 0.0,
        branch_exploration_warmup_steps: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attn_experts = num_attn_experts
        self.num_attn_experts_per_tok = num_attn_experts_per_tok
        self.attn_expert_mode = attn_expert_mode
        self.branch_router_aux_loss_coef = branch_router_aux_loss_coef
        self.use_deepseek_routing = use_deepseek_routing
        self.per_layer_router = per_layer_router
        self.per_layer_mlp_router = per_layer_mlp_router
        self.per_layer_attn_router = per_layer_attn_router
        self.routed_norm = routed_norm
        self.per_layer_norm = per_layer_norm
        self.per_layer_qk_norm = per_layer_qk_norm
        self.post_norm = post_norm
        self.dynamic_depth_min = dynamic_depth_min
        self.dynamic_depth_max = dynamic_depth_max
        self.depthwise_attention = depthwise_attention
        self.depthwise_block_size = depthwise_block_size
        self.per_head_compute_mode = per_head_compute_mode
        self.per_head_dense_fraction_threshold = per_head_dense_fraction_threshold
        self.sanity_check_mode = sanity_check_mode
        self.scale_attn_by_routing_weight = scale_attn_by_routing_weight
        self.scale_branch_by_routing_weight = scale_branch_by_routing_weight
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
