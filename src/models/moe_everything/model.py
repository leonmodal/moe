"""MoE-Everything model assembly: MoEverythingModel and MoEverythingForCausalLM.

Hierarchical per-token routing with shared attention/MLP expert banks.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

from src.models.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)
from src.models.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from src.models.routing.routers import BranchRouter, BranchRouterRecorder
from src.models.router import DeepSeekRouter, ExplorationTopKRouter, checkpoint_recompute_context
from src.models.load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from src.models.mask_compat import call_mask_function
from .config import MoEverythingConfig
from .attention_bank import AttentionExpertBank, NormExpertBank
from .mlp_bank import MlpExpertBank


def _build_boundary_config(config: MoEverythingConfig) -> Qwen3MoeConfig:
    """Build the Qwen3MoeConfig consumed by prelude/coda decoder layers.

    Boundary blocks are standard-MoE-style: dense GQA + per-layer SwiGLU
    expert pool. Attention geometry mirrors the main config; MLP geometry
    overrides default to the boundary_* fields, falling back to the bank's
    geometry when an explicit boundary_* override is None.
    """
    boundary = Qwen3MoeConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=getattr(config, "intermediate_size", config.hidden_size * 4),
        num_hidden_layers=max(int(config.prelude_layers), int(config.coda_layers), 1),
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=getattr(config, "rope_theta", 1_000_000.0),
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=getattr(config, "attention_bias", False),
        attention_dropout=getattr(config, "attention_dropout", 0.0),
        moe_intermediate_size=(
            config.boundary_moe_intermediate_size
            if config.boundary_moe_intermediate_size is not None
            else config.moe_intermediate_size
        ),
        num_experts=(
            config.boundary_num_experts
            if config.boundary_num_experts is not None
            else config.num_experts
        ),
        num_experts_per_tok=(
            config.boundary_num_experts_per_tok
            if config.boundary_num_experts_per_tok is not None
            else config.num_experts_per_tok
        ),
        norm_topk_prob=(
            bool(config.boundary_norm_topk_prob)
            if config.boundary_norm_topk_prob is not None
            else getattr(config, "norm_topk_prob", True)
        ),
    )
    boundary._attn_implementation = getattr(config, "_attn_implementation", "sdpa")
    boundary.num_groups = (
        config.boundary_num_groups
        if config.boundary_num_groups is not None
        else getattr(config, "num_groups", None)
    )
    boundary.group_topk = (
        config.boundary_group_topk
        if config.boundary_group_topk is not None
        else getattr(config, "group_topk", None)
    )
    boundary.topk_scaling_factor = (
        config.boundary_topk_scaling_factor
        if config.boundary_topk_scaling_factor is not None
        else getattr(config, "topk_scaling_factor", None)
    )
    boundary.router_exploration_rate = getattr(config, "router_exploration_rate", 0.0)
    boundary.router_z_loss_coef = getattr(config, "router_z_loss_coef", 0.0)
    boundary.router_score_function = getattr(config, "router_score_function", "softmax")
    if hasattr(config, "softmax_position"):
        boundary.softmax_position = config.softmax_position
    if hasattr(config, "router_topk_ordering"):
        boundary.router_topk_ordering = config.router_topk_ordering
    boundary.output_router_logits = False
    return boundary


def _make_boundary_layer(
    boundary_config: Qwen3MoeConfig,
    layer_idx: int,
    router_type: str,
) -> Qwen3MoeDecoderLayer:
    """Build one prelude/coda block and (optionally) swap its gate for a
    DeepSeekRouter so it integrates with the deepseek-bias balancing
    walker the same way DeepSeekStandardMoEModel does."""
    layer = Qwen3MoeDecoderLayer(boundary_config, layer_idx=layer_idx)
    if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
        raise RuntimeError(
            "Boundary layer constructed without a sparse MoE block; ensure "
            "boundary_num_experts > 0 and decoder_sparse_step == 1."
        )
    old_gate = layer.mlp.gate
    if router_type == "deepseek":
        new_gate = DeepSeekRouter(boundary_config)
    elif router_type == "softmax":
        new_gate = ExplorationTopKRouter(boundary_config)
    else:
        raise ValueError(
            f"boundary_router_type must be 'deepseek' or 'softmax', got {router_type!r}"
        )
    new_gate.weight.data.copy_(old_gate.weight.data)
    layer.mlp.gate = new_gate
    return layer


class RecurrentAdapterMLP(nn.Module):
    """Adapter MLP from concat recurrent state back to model hidden size."""

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        input_size = config.hidden_size * 2
        intermediate_size = (
            config.recurrent_adapter_intermediate_size
            if config.recurrent_adapter_intermediate_size is not None
            else getattr(config, "intermediate_size", config.hidden_size * 4)
        )
        bias = bool(getattr(config, "recurrent_adapter_bias", False))
        activation = str(getattr(config, "recurrent_adapter_activation", "gelu"))
        self.up_proj = nn.Linear(input_size, int(intermediate_size), bias=bias)
        self.down_proj = nn.Linear(int(intermediate_size), config.hidden_size, bias=bias)
        self.act_fn = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class MoEverythingModel(nn.Module):
    """Mixture-of-Everything transformer backbone.

    All components (branch router, attention bank, MLP bank) are shared
    across depths. The forward is a simple for loop — only activations
    change per depth, not weights.
    """

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_depths = config.num_hidden_layers
        self.sanity_check_mode = getattr(config, "sanity_check_mode", None)
        if self.sanity_check_mode == "alternating_global_moe" and self.num_depths % 2 != 0:
            raise ValueError("sanity_check_mode='alternating_global_moe' requires an even num_hidden_layers")

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.head_dim = head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.kv_dim = self.num_kv_heads * head_dim

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.attn_expert_mode in {"per_head_recompute_k", "per_head_recompute_kv"}:
            self.init_k_proj = None
            self.init_v_proj = None
            self.init_k_norm = None
        else:
            self.init_k_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
            self.init_v_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
            self.init_k_norm = Qwen3MoeRMSNorm(head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

        # Feature 1: per-layer vs shared branch router
        self.per_layer_router = getattr(config, "per_layer_router", False)
        branch_exploration_rate = getattr(
            config,
            "branch_router_exploration_rate",
            getattr(config, "router_exploration_rate", 0.0),
        )
        scale_branch = getattr(config, "scale_branch_by_routing_weight", True)
        use_sampling = getattr(config, "branch_sampling", False)
        use_seq_level = getattr(config, "branch_level", "token") == "seq"
        use_deepseek_style = getattr(config, "branch_deepseek", False)
        # Branch-router balancing knobs (flat-schema bridge).
        # `branch_balancing="exploration_only"` flips every BranchRouter on this
        # model into rate-driven random branch picks; the trainer's per-step
        # `apply_branch_exploration_only_rate(model, rate)` hook then pushes the
        # current `p_explore(step)` into `exploration_only_rate` each step.
        # The constructor seeds `exploration_only_rate` to
        # `branch_exploration_rate` so eval-mode forwards (and step-0 forwards
        # before the trainer hook fires) see the configured initial rate.
        branch_balancing = getattr(config, "branch_balancing", "none")
        self.branch_balancing = branch_balancing
        branch_exploration_only_rate = getattr(config, "branch_exploration_rate", 0.0)
        # Branch quantile knobs (only consumed when balancing == "quantile").
        # The factory copies these from `model.branch_router.{quantile_target_q,
        # quantile_eta}` onto the top-level config.
        branch_q_target = getattr(config, "branch_quantile_target_q", None)
        branch_q_eta = getattr(config, "branch_quantile_eta", None)
        if self.sanity_check_mode == "alternating_global_moe" or branch_balancing == "fixed_alternating":
            if self.per_layer_router:
                self.branch_routers = nn.ModuleList([BranchRouterRecorder() for _ in range(self.num_depths)])
            else:
                self.branch_router = BranchRouterRecorder()
        elif self.per_layer_router:
            self.branch_routers = nn.ModuleList([
                BranchRouter(config.hidden_size, exploration_rate=branch_exploration_rate,
                             scale_by_routing_weight=scale_branch,
                             use_sampling=use_sampling, use_seq_level=use_seq_level,
                             use_deepseek_style=use_deepseek_style,
                             exploration_only_rate=branch_exploration_only_rate,
                             balancing=branch_balancing,
                             quantile_target_q=branch_q_target,
                             quantile_eta=branch_q_eta)
                for _ in range(self.num_depths)
            ])
        else:
            self.branch_router = BranchRouter(config.hidden_size, exploration_rate=branch_exploration_rate,
                                              scale_by_routing_weight=scale_branch,
                                              use_sampling=use_sampling, use_seq_level=use_seq_level,
                                              use_deepseek_style=use_deepseek_style,
                                              exploration_only_rate=branch_exploration_only_rate,
                                              balancing=branch_balancing,
                                              quantile_target_q=branch_q_target,
                                              quantile_eta=branch_q_eta)

        # Per-layer expert bank: separate attention + MLP expert weights for
        # each "paper-layer". When False (default), one shared bank is used
        # at every depth.
        #
        # A "paper-layer" is the natural unit of one attn + one MLP step.
        # MoE-Everything counts a paper-layer as TWO routed depths (because
        # each depth runs only attn OR only mlp under hard branching), so:
        #   num_paper_layers = num_depths // 2
        # Under `fixed_alternating` (AMAMAMAM…) this gives the correct
        # mapping: depth 2k uses attn_banks[k]; depth 2k+1 uses mlp_banks[k].
        # The other modality's bank at each depth would be dead weight — we
        # don't allocate it.
        self.per_layer_expert_bank = getattr(config, "per_layer_expert_bank", False)
        if self.per_layer_expert_bank:
            num_paper_layers = max(1, self.num_depths // 2)
            self.attn_banks_per_depth = nn.ModuleList(
                [AttentionExpertBank(config) for _ in range(num_paper_layers)]
            )
            self.mlp_banks_per_depth = nn.ModuleList(
                [MlpExpertBank(config) for _ in range(num_paper_layers)]
            )
            self._num_paper_layers = num_paper_layers
            # `self.attn_bank` / `self.mlp_bank` are convenience pointers
            # to the bank used at the CURRENT depth. We register them via
            # `object.__setattr__` to bypass `nn.Module.__setattr__`'s
            # sub-module registration — the bank's parameters are already
            # owned by `attn_banks_per_depth` / `mlp_banks_per_depth`, so
            # making them sub-module attributes here would double-count
            # parameters under `.parameters()`. The depth loop rebinds them
            # before each step via `_select_bank_for_depth`.
            object.__setattr__(self, "attn_bank", self.attn_banks_per_depth[0])
            object.__setattr__(self, "mlp_bank", self.mlp_banks_per_depth[0])
        else:
            self.attn_bank = AttentionExpertBank(config)
            self.mlp_bank = MlpExpertBank(config)
            self.attn_banks_per_depth = None
            self.mlp_banks_per_depth = None
            self._num_paper_layers = None

        # Post-norm: RMSNorm on branch output before residual addition
        self.post_norm = getattr(config, "post_norm", False)
        if self.post_norm:
            per_layer = getattr(config, "per_layer_norm", False)
            if per_layer:
                self.attn_post_norms = nn.ModuleList([Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(self.num_depths)])
                self.mlp_post_norms = nn.ModuleList([Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(self.num_depths)])
            else:
                self.attn_post_norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.mlp_post_norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.recurrent = bool(getattr(config, "recurrent", False))
        if self.recurrent and bool(getattr(config, "recurrent_adapter", True)):
            self.recurrent_adapter = RecurrentAdapterMLP(config)
        else:
            self.recurrent_adapter = None

        # Feature 2: dynamic depth
        self.dynamic_depth_min = getattr(config, "dynamic_depth_min", 1.0)
        self.dynamic_depth_max = getattr(config, "dynamic_depth_max", 1.0)

        # Feature 3: depthwise attention (AttnRes)
        self.depthwise_attention = getattr(config, "depthwise_attention", False)
        self.depthwise_block_size = getattr(config, "depthwise_block_size", 0)
        if self.depthwise_attention:
            block_size = self.depthwise_block_size if self.depthwise_block_size > 0 else 1
            num_queries = (self.num_depths + block_size - 1) // block_size
            self.depth_queries = nn.Parameter(
                torch.randn(num_queries, config.hidden_size) * config.initializer_range
            )
            self.depth_norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Prelude / coda hybrid: optional standard-MoE-style decoder
        # blocks prepended / appended around the recurrent loop. Each
        # boundary layer owns its own dense GQA + per-layer MLP expert
        # pool (no sharing with the bank). Built off a separate
        # `Qwen3MoeConfig` derived from the boundary_* fields so the
        # bank's geometry is preserved.
        self.prelude_layers_count = int(getattr(config, "prelude_layers", 0))
        self.coda_layers_count = int(getattr(config, "coda_layers", 0))
        if self.prelude_layers_count > 0 or self.coda_layers_count > 0:
            self._boundary_config = _build_boundary_config(config)
            router_type = getattr(config, "boundary_router_type", "deepseek")
            self.prelude_blocks = nn.ModuleList([
                _make_boundary_layer(self._boundary_config, i, router_type)
                for i in range(self.prelude_layers_count)
            ])
            self.coda_blocks = nn.ModuleList([
                _make_boundary_layer(self._boundary_config, i, router_type)
                for i in range(self.coda_layers_count)
            ])
        else:
            self._boundary_config = None
            self.prelude_blocks = nn.ModuleList()
            self.coda_blocks = nn.ModuleList()

        self._all_mlp_router_logits = []
        self._all_mlp_selected_experts = []
        self._all_mlp_token_masks = []
        self._all_branch_probs = []
        self._all_branch_selected_experts = []
        self._all_attn_router_info = []
        self._all_attention_maps = []
        self._all_mlp_router_scopes = []
        self._last_num_steps_no_grad = 0
        self._last_num_steps_with_grad = 0

        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {"use_reentrant": False}

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _checkpoint_context_fn():
        return checkpoint_recompute_context(False), checkpoint_recompute_context(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None):
        self.gradient_checkpointing = True
        kwargs = {"use_reentrant": False, "context_fn": self._checkpoint_context_fn}
        if gradient_checkpointing_kwargs:
            kwargs.update(gradient_checkpointing_kwargs)
        self._gradient_checkpointing_kwargs = kwargs

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _resolve_num_steps(self, num_steps: torch.Tensor | tuple[int, int] | list[int] | int | None) -> tuple[int, int]:
        if not self.recurrent:
            return 0, 1
        if num_steps is None:
            if self.training:
                return 0, max(1, int(getattr(self.config, "eval_recurrence", 1)))
            return int(getattr(self.config, "eval_recurrence", 1)), 0
        if isinstance(num_steps, torch.Tensor):
            flat = num_steps.detach().view(-1).to("cpu")
            if flat.numel() == 1:
                return 0, int(flat[0].item())
            return int(flat[0].item()), int(flat[1].item())
        if isinstance(num_steps, int):
            return 0, int(num_steps)
        if len(num_steps) == 1:
            return 0, int(num_steps[0])
        return int(num_steps[0]), int(num_steps[1])

    def _inject_recurrent_input(self, hidden_states: torch.Tensor, recurrent_input: torch.Tensor) -> torch.Tensor:
        if self.recurrent_adapter is None:
            return hidden_states
        return self.recurrent_adapter(
            torch.cat(
                [hidden_states, recurrent_input.to(device=hidden_states.device)],
                dim=-1,
            )
        )

    @staticmethod
    def _manual_count_selected(
        router: nn.Module | None,
        selected_experts: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        if router is None or selected_experts is None or not hasattr(router, "local_tokens_per_expert"):
            return
        selected = selected_experts.detach()
        if token_mask is not None:
            mask = token_mask.detach().reshape(-1).bool().to(selected.device)
            if selected.shape[0] == mask.numel():
                selected = selected[mask]
        if selected.numel() == 0:
            return
        with torch.no_grad():
            counts = torch.bincount(
                selected.reshape(-1),
                minlength=int(getattr(router, "num_experts", selected.max().item() + 1)),
            ).to(device=router.local_tokens_per_expert.device, dtype=torch.float32)
            router.local_tokens_per_expert += counts

    def _first_attn_router_for(self, name: str, depth_idx: int) -> nn.Module | None:
        attr = {
            "q": "q_routers",
            "k": "k_routers",
            "v": "v_routers",
            "o": "o_routers",
            "qk": "qk_routers",
            "qkv": "qkv_routers",
            "qkvo": "qkvo_routers",
            "vo": "vo_routers",
        }.get(name)
        if attr is None or not hasattr(self.attn_bank, attr):
            return None
        routers = getattr(self.attn_bank, attr)
        if getattr(self.attn_bank, "per_layer_attn_router", False):
            routers = routers[depth_idx]
        if isinstance(routers, nn.ModuleList) and len(routers) > 0:
            return routers[0]
        return routers

    def _accumulate_no_grad_counts(self, depth_idx: int) -> None:
        if not self.training:
            return
        self._manual_count_selected(
            self.mlp_bank._select_gate(depth_idx),
            self.mlp_bank.last_selected_experts,
            self.mlp_bank.last_token_mask,
        )
        if not bool(getattr(self.config, "global_router_update", False)):
            return
        for name, info in (self.attn_bank.last_router_info or {}).items():
            if not isinstance(info, dict):
                continue
            self._manual_count_selected(
                self._first_attn_router_for(str(name), depth_idx),
                info.get("selected_experts"),
                info.get("token_mask"),
            )

    def _select_bank_for_depth(self, depth_idx: int) -> None:
        """When `per_layer_expert_bank` is True, rebind `self.attn_bank` /
        `self.mlp_bank` to the bank instance for the **paper-layer** that
        owns this depth. Under fixed_alternating (AMAMAMAM…), depths
        `2k` and `2k+1` together form paper-layer `k`: the attn bank at
        index `k` is used at depth `2k` (the attn-active depth), the mlp
        bank at index `k` is used at depth `2k+1` (the mlp-active depth).

        We always assign BOTH `attn_bank` and `mlp_bank` (to the same
        paper-layer's banks) so downstream `self.attn_bank.X` /
        `self.mlp_bank.X` lookups always resolve. The unused branch's
        compute is skipped by the `attn_mask.sum() == 0` /
        `mlp_mask.sum() == 0` short-circuits already in the depth code.

        No-op for the default global-bank case.
        """
        if self.attn_banks_per_depth is not None:
            paper_layer_idx = min(depth_idx // 2, self._num_paper_layers - 1)
            object.__setattr__(self, "attn_bank", self.attn_banks_per_depth[paper_layer_idx])
            object.__setattr__(self, "mlp_bank", self.mlp_banks_per_depth[paper_layer_idx])

    def _run_depth_loop(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        *,
        manual_count_no_grad: bool = False,
    ) -> torch.Tensor:
        B, T = hidden_states.shape[:2]
        cos, sin = position_embeddings
        use_recompute_attention = self.attn_bank.mode in {"per_head_recompute_k", "per_head_recompute_kv"}

        if self.recurrent and not use_recompute_attention:
            raise RuntimeError(
                "recurrent_moe_everything requires attn_expert_mode='per_head_recompute_k' "
                "or 'per_head_recompute_kv' so each recurrent pass recomputes attention "
                "from the current state."
            )

        if not use_recompute_attention:
            K_init = self.init_k_proj(hidden_states)
            V_init = self.init_v_proj(hidden_states)
            K_init = self.init_k_norm(K_init.view(B, T, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            V_init = V_init.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos_unsq = cos.unsqueeze(1)
            sin_unsq = sin.unsqueeze(1)
            K_init = (K_init * cos_unsq) + (self._rotate_half(K_init) * sin_unsq)
            kv_state = (K_init, V_init)

        if self.training and self.dynamic_depth_min < 1.0:
            min_d = max(1, int(self.num_depths * self.dynamic_depth_min))
            max_d = max(min_d, int(self.num_depths * self.dynamic_depth_max))
            actual_depths = torch.randint(min_d, max_d + 1, (1,)).item()
        else:
            actual_depths = self.num_depths

        if self.depthwise_attention:
            block_size = self.depthwise_block_size if self.depthwise_block_size > 0 else 1
            depth_cache = [hidden_states]

        for d in range(actual_depths):
            self._select_bank_for_depth(d)
            if self.depthwise_attention and len(depth_cache) > 1:
                is_boundary = (block_size <= 1) or (d % block_size == 0)
                if is_boundary:
                    query_idx = d // block_size if block_size > 1 else d
                    if query_idx < self.depth_queries.shape[0]:
                        V_stack = torch.stack(depth_cache, dim=0)
                        K_stack = self.depth_norm(V_stack)
                        w = self.depth_queries[query_idx]
                        logits = torch.einsum("h, l b t h -> l b t", w, K_stack)
                        alpha = F.softmax(logits, dim=0)
                        hidden_states = torch.einsum(
                            "l b t, l b t h -> b t h", alpha, V_stack
                        )

            if use_recompute_attention:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                    checkpoint_kwargs.setdefault("use_reentrant", False)
                    checkpoint_kwargs.setdefault("context_fn", self._checkpoint_context_fn)

                    def depth_step_recompute(h, _depth_idx=d):
                        return self._depth_step_recompute_attention(
                            h, position_embeddings, causal_mask, depth_idx=_depth_idx
                        )

                    hidden_states = checkpoint(depth_step_recompute, hidden_states, **checkpoint_kwargs)
                else:
                    hidden_states = self._depth_step_recompute_attention(
                        hidden_states,
                        position_embeddings,
                        causal_mask,
                        depth_idx=d,
                    )
            else:
                K_old, V_old = kv_state
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                    checkpoint_kwargs.setdefault("use_reentrant", False)
                    checkpoint_kwargs.setdefault("context_fn", self._checkpoint_context_fn)

                    def depth_step(h, k, v, _depth_idx=d):
                        return self._depth_step_no_recompute(
                            h, k, v, position_embeddings, causal_mask, depth_idx=_depth_idx
                        )

                    hidden_states, K_new, V_new = checkpoint(
                        depth_step, hidden_states, K_old, V_old, **checkpoint_kwargs
                    )
                else:
                    hidden_states, K_new, V_new = self._depth_step_no_recompute(
                        hidden_states, K_old, V_old, position_embeddings, causal_mask,
                        depth_idx=d,
                    )

            if manual_count_no_grad:
                self._accumulate_no_grad_counts(d)

            if self.per_layer_router:
                self._all_branch_probs.append(self.branch_routers[d].last_probs)
                self._all_branch_selected_experts.append(self.branch_routers[d].last_selected_experts)
            else:
                self._all_branch_probs.append(self.branch_router.last_probs)
                self._all_branch_selected_experts.append(self.branch_router.last_selected_experts)
            self._all_mlp_router_logits.append(self.mlp_bank.last_router_logits)
            self._all_mlp_selected_experts.append(self.mlp_bank.last_selected_experts)
            self._all_mlp_token_masks.append(self.mlp_bank.last_token_mask)
            self._all_mlp_router_scopes.append({
                "pool": "mlp_recurrent" if self.recurrent else "mlp",
                "block_index": self.prelude_layers_count + d if self.recurrent else d,
                "num_experts": int(getattr(self.config, "num_experts", 0) or 0),
            })
            self._all_attn_router_info.append(self.attn_bank.last_router_info)
            if getattr(self.attn_bank, "capture_attention_maps", False):
                # Detached + CPU'd already; safe to hold across depths.
                self._all_attention_maps.append(list(self.attn_bank.last_attention_maps))

            if not use_recompute_attention:
                kv_state = (K_new, V_new)

            if self.depthwise_attention:
                should_store = (
                    block_size <= 1
                    or (d + 1) % block_size == 0
                    or d == actual_depths - 1
                )
                if should_store:
                    depth_cache.append(hidden_states)

        return hidden_states

    def _fixed_alternating_branch_route(
        self,
        hidden_states: torch.Tensor,
        depth_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        choose_attn = (depth_idx % 2) == 0
        probs = hidden_states.new_zeros(*hidden_states.shape[:2], 2)
        probs[..., 0 if choose_attn else 1] = 1.0
        attn_mask = torch.full(
            (*hidden_states.shape[:2], 1),
            choose_attn,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        mlp_mask = ~attn_mask
        w_attn = attn_mask.to(hidden_states.dtype)
        w_mlp = mlp_mask.to(hidden_states.dtype)
        if self.per_layer_router:
            self.branch_routers[depth_idx].last_probs = probs
            self.branch_routers[depth_idx].last_selected_experts = (
                hidden_states.new_full((*hidden_states.shape[:2], 1), 0 if choose_attn else 1, dtype=torch.long)
            )
        else:
            self.branch_router.last_probs = probs
            self.branch_router.last_selected_experts = (
                hidden_states.new_full((*hidden_states.shape[:2], 1), 0 if choose_attn else 1, dtype=torch.long)
            )
        return w_attn, w_mlp, attn_mask, mlp_mask

    def _sanity_branch_route(
        self,
        hidden_states: torch.Tensor,
        depth_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._fixed_alternating_branch_route(hidden_states, depth_idx)

    def _depth_step_no_recompute(
        self,
        hidden_states: torch.Tensor,
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        depth_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.attn_bank.mode != "per_head_no_recompute":
            raise RuntimeError("_depth_step_no_recompute requires per_head_no_recompute attention")
        if self.sanity_check_mode == "alternating_global_moe":
            w_attn, w_mlp, attn_mask, mlp_mask = self._sanity_branch_route(hidden_states, depth_idx)
        elif self.branch_balancing == "fixed_alternating":
            w_attn, w_mlp, attn_mask, mlp_mask = self._fixed_alternating_branch_route(hidden_states, depth_idx)
        elif self.per_layer_router:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_routers[depth_idx](hidden_states)
        else:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_router(hidden_states)

        attn_mask_bool = attn_mask.bool()

        Q, K_fresh, V_fresh = self.attn_bank.project(hidden_states, position_embeddings, depth_idx=depth_idx)
        # Tokens that chose MLP keep old KV in the blend.
        attn_mask_kv = attn_mask_bool.unsqueeze(1)  # (B, 1, T, 1)
        K_blend = torch.where(attn_mask_kv, K_fresh, K_old)
        V_blend = torch.where(attn_mask_kv, V_fresh, V_old)
        attn_out = self.attn_bank.attend(Q, K_blend, V_blend, causal_mask, depth_idx=depth_idx)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)
        self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)
        if self.post_norm:
            pn = self.attn_post_norms[depth_idx] if hasattr(self, "attn_post_norms") else self.attn_post_norm
            attn_out = pn(attn_out)
        hidden_states = hidden_states + w_attn * attn_out

        # MLP branch
        mlp_out = self.mlp_bank(hidden_states, depth_idx=depth_idx, token_mask=mlp_mask.bool())
        if self.post_norm:
            pn = self.mlp_post_norms[depth_idx] if hasattr(self, "mlp_post_norms") else self.mlp_post_norm
            mlp_out = pn(mlp_out)
        hidden_states = hidden_states + w_mlp * mlp_out

        return hidden_states, K_new, V_new

    def _depth_step_recompute_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        depth_idx: int = 0,
    ) -> torch.Tensor:
        if self.sanity_check_mode == "alternating_global_moe":
            w_attn, w_mlp, attn_mask, mlp_mask = self._sanity_branch_route(hidden_states, depth_idx)
        elif self.branch_balancing == "fixed_alternating":
            w_attn, w_mlp, attn_mask, mlp_mask = self._fixed_alternating_branch_route(hidden_states, depth_idx)
        elif self.per_layer_router:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_routers[depth_idx](hidden_states)
        else:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_router(hidden_states)

        attn_mask_bool = attn_mask.bool()

        if not bool(attn_mask_bool.any().item()):
            attn_out = self.attn_bank._empty_attn_output(
                hidden_states,
                self.attn_bank.recompute_router_specs(),
            )
        elif bool(attn_mask_bool.all().item()):
            attn_out = self.attn_bank.project_and_attend_per_head_recompute(
                hidden_states,
                position_embeddings,
                causal_mask,
                depth_idx=depth_idx,
            )
            self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)
        else:
            attn_out = self.attn_bank.project_and_attend_per_head_recompute_dense_mixed(
                hidden_states,
                position_embeddings,
                attn_mask_bool,
                causal_mask,
                depth_idx=depth_idx,
            )
            self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)

        if self.post_norm:
            pn = self.attn_post_norms[depth_idx] if hasattr(self, "attn_post_norms") else self.attn_post_norm
            attn_out = pn(attn_out)

        if self.branch_balancing == "weighted_sum":
            # Parallel wiring: both branches read the SAME pre-attn
            # hidden_states. This decouples the MLP gate's input from the
            # attention output, so bf16 noise in `attn_out` (e.g. from
            # FlashAttention's atomic-add reductions) cannot flip a
            # borderline MLP top-K decision under
            # `torch.utils.checkpoint` recompute.
            mlp_out = self.mlp_bank(hidden_states, depth_idx=depth_idx, token_mask=mlp_mask.bool())
            if self.post_norm:
                pn = self.mlp_post_norms[depth_idx] if hasattr(self, "mlp_post_norms") else self.mlp_post_norm
                mlp_out = pn(mlp_out)
            hidden_states = hidden_states + w_attn * attn_out + w_mlp * mlp_out
        else:
            # Serial wiring (default for top-1 branch modes): attn
            # residual is added before the MLP gate sees `hidden_states`.
            # Safe under recompute because top-1 modes only have ONE
            # branch contribute per token, so the attn-output residual
            # never reaches the MLP gate's input for the same token.
            hidden_states = hidden_states + w_attn * attn_out
            mlp_out = self.mlp_bank(hidden_states, depth_idx=depth_idx, token_mask=mlp_mask.bool())
            if self.post_norm:
                pn = self.mlp_post_norms[depth_idx] if hasattr(self, "mlp_post_norms") else self.mlp_post_norm
                mlp_out = pn(mlp_out)
            hidden_states = hidden_states + w_mlp * mlp_out

        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        num_steps: torch.Tensor | tuple[int, int] | list[int] | int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        cache_position = torch.arange(T, device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        cos, sin = position_embeddings

        mask_function = (
            create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        )
        causal_mask = call_mask_function(
            mask_function,
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        use_recompute_attention = self.attn_bank.mode in {"per_head_recompute_k", "per_head_recompute_kv"}

        # Prelude: standard-MoE-style decoder blocks before the recurrent
        # bank. Each block does dense GQA + per-layer MLP expert pool with
        # the same causal mask and rotary embeddings as the recurrent loop.
        if len(self.prelude_blocks) > 0:
            hidden_states = self._run_boundary_blocks(
                self.prelude_blocks,
                hidden_states,
                position_embeddings,
                causal_mask,
            )

        if not use_recompute_attention:
            K_init = self.init_k_proj(hidden_states)
            V_init = self.init_v_proj(hidden_states)
            K_init = self.init_k_norm(K_init.view(B, T, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            V_init = V_init.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos_unsq = cos.unsqueeze(1)
            sin_unsq = sin.unsqueeze(1)
            K_init = (K_init * cos_unsq) + (self._rotate_half(K_init) * sin_unsq)

            kv_state = (K_init, V_init)

        self._all_mlp_router_logits = []
        self._all_mlp_selected_experts = []
        self._all_mlp_token_masks = []
        self._all_branch_probs = []
        self._all_branch_selected_experts = []
        self._all_attn_router_info = []
        self._all_attention_maps = []
        self._all_mlp_router_scopes = []

        if self.recurrent:
            recurrent_input = hidden_states
            n_no_grad, n_with_grad = self._resolve_num_steps(num_steps)
            self._last_num_steps_no_grad = n_no_grad
            self._last_num_steps_with_grad = n_with_grad

            if n_no_grad > 0:
                with torch.no_grad():
                    for _ in range(n_no_grad):
                        hidden_states = self._inject_recurrent_input(hidden_states, recurrent_input)
                        hidden_states = self._run_depth_loop(
                            hidden_states,
                            position_embeddings,
                            causal_mask,
                            manual_count_no_grad=True,
                        )
                hidden_states = hidden_states.detach()

            for _ in range(n_with_grad):
                hidden_states = self._inject_recurrent_input(hidden_states, recurrent_input)
                hidden_states = self._run_depth_loop(
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                )

            if len(self.coda_blocks) > 0:
                hidden_states = self._run_boundary_blocks(
                    self.coda_blocks,
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                )

            hidden_states = self.norm(hidden_states)
            return hidden_states

        self._last_num_steps_no_grad = 0
        self._last_num_steps_with_grad = 1

        # Dynamic depth: randomize iteration count during training
        if self.training and self.dynamic_depth_min < 1.0:
            min_d = max(1, int(self.num_depths * self.dynamic_depth_min))
            max_d = max(min_d, int(self.num_depths * self.dynamic_depth_max))
            actual_depths = torch.randint(min_d, max_d + 1, (1,)).item()
        else:
            actual_depths = self.num_depths

        # Depthwise attention (AttnRes): cache of prior depth outputs
        if self.depthwise_attention:
            block_size = self.depthwise_block_size if self.depthwise_block_size > 0 else 1
            depth_cache = [hidden_states]

        for d in range(actual_depths):
            self._select_bank_for_depth(d)
            # Depthwise attention: replace input with learned combination of prior outputs
            if self.depthwise_attention and len(depth_cache) > 1:
                is_boundary = (block_size <= 1) or (d % block_size == 0)
                if is_boundary:
                    query_idx = d // block_size if block_size > 1 else d
                    if query_idx < self.depth_queries.shape[0]:
                        V_stack = torch.stack(depth_cache, dim=0)
                        K_stack = self.depth_norm(V_stack)
                        w = self.depth_queries[query_idx]
                        logits = torch.einsum("h, l b t h -> l b t", w, K_stack)
                        alpha = F.softmax(logits, dim=0)
                        hidden_states = torch.einsum(
                            "l b t, l b t h -> b t h", alpha, V_stack
                        )

            if use_recompute_attention:
                if self.gradient_checkpointing and self.training:
                    checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                    checkpoint_kwargs.setdefault("use_reentrant", False)
                    checkpoint_kwargs.setdefault("context_fn", self._checkpoint_context_fn)

                    def depth_step_recompute(h, _depth_idx=d):
                        return self._depth_step_recompute_attention(
                            h, position_embeddings, causal_mask, depth_idx=_depth_idx
                        )

                    hidden_states = checkpoint(depth_step_recompute, hidden_states, **checkpoint_kwargs)
                else:
                    hidden_states = self._depth_step_recompute_attention(
                        hidden_states,
                        position_embeddings,
                        causal_mask,
                        depth_idx=d,
                    )
            else:
                K_old, V_old = kv_state

                if self.gradient_checkpointing and self.training:
                    checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                    checkpoint_kwargs.setdefault("use_reentrant", False)
                    checkpoint_kwargs.setdefault("context_fn", self._checkpoint_context_fn)

                    def depth_step(h, k, v, _depth_idx=d):
                        return self._depth_step_no_recompute(
                            h, k, v, position_embeddings, causal_mask, depth_idx=_depth_idx
                        )

                    hidden_states, K_new, V_new = checkpoint(
                        depth_step, hidden_states, K_old, V_old, **checkpoint_kwargs
                    )
                else:
                    hidden_states, K_new, V_new = self._depth_step_no_recompute(
                        hidden_states, K_old, V_old, position_embeddings, causal_mask,
                        depth_idx=d,
                    )

            if self.per_layer_router:
                self._all_branch_probs.append(self.branch_routers[d].last_probs)
                self._all_branch_selected_experts.append(self.branch_routers[d].last_selected_experts)
            else:
                self._all_branch_probs.append(self.branch_router.last_probs)
                self._all_branch_selected_experts.append(self.branch_router.last_selected_experts)
            self._all_mlp_router_logits.append(self.mlp_bank.last_router_logits)
            self._all_mlp_selected_experts.append(self.mlp_bank.last_selected_experts)
            self._all_mlp_token_masks.append(self.mlp_bank.last_token_mask)
            self._all_attn_router_info.append(self.attn_bank.last_router_info)
            if getattr(self.attn_bank, "capture_attention_maps", False):
                # Detached + CPU'd already; safe to hold across depths.
                self._all_attention_maps.append(list(self.attn_bank.last_attention_maps))

            if not use_recompute_attention:
                kv_state = (K_new, V_new)

            # Store depth output for depthwise attention
            if self.depthwise_attention:
                should_store = (
                    block_size <= 1
                    or (d + 1) % block_size == 0
                    or d == actual_depths - 1
                )
                if should_store:
                    depth_cache.append(hidden_states)

        # Coda: standard-MoE-style decoder blocks after the recurrent bank.
        if len(self.coda_blocks) > 0:
            hidden_states = self._run_boundary_blocks(
                self.coda_blocks,
                hidden_states,
                position_embeddings,
                causal_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _run_boundary_blocks(
        self,
        blocks: nn.ModuleList,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a stack of prelude/coda Qwen3MoeDecoderLayer blocks.

        Each block does pre-norm → dense GQA → residual → pre-norm → MoE
        MLP → residual. Gradient checkpointing wraps each block when
        enabled, mirroring the recurrent loop's behavior.
        """
        for block in blocks:
            if self.gradient_checkpointing and self.training:
                checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                checkpoint_kwargs.setdefault("use_reentrant", False)
                checkpoint_kwargs.setdefault("context_fn", self._checkpoint_context_fn)

                def run_block(h, _block=block):
                    return _block(
                        hidden_states=h,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                    )

                hidden_states = checkpoint(run_block, hidden_states, **checkpoint_kwargs)
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                )
        return hidden_states


class MoEverythingForCausalLM(Qwen3MoePreTrainedModel):
    """Causal LM head over MoEverythingModel.

    Computes CE loss plus optional MoE auxiliary losses.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MoEverythingConfig):
        super().__init__(config)
        self.config = config
        self.model = MoEverythingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vocab_size = config.vocab_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.branch_router_aux_loss_coef = config.branch_router_aux_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self._seq_aux_loss_coef = getattr(config, "seq_aux_loss_coef", 0.0)

        # Match Qwen3/Qwen3-MoE init semantics, including router weight init
        # and experts implementation dispatch through the shared config object.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def _compute_lm_ce_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        if not getattr(self.config, "use_fused_linear_ce", True):
            logits = self.lm_head(hidden_states)
            return self.loss_function(logits, labels, self.vocab_size, **kwargs)

        ignore_index = int(kwargs.get("ignore_index", -100))
        label_smoothing = float(kwargs.get("label_smoothing", 0.0) or 0.0)
        if hidden_states.is_cuda:
            try:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            except ImportError as exc:
                raise RuntimeError(
                    "MoE-Everything uses Liger fused linear CE by default on CUDA. "
                    "Install liger-kernel in the training image."
                ) from exc

            flat_hidden = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
            shifted_labels = F.pad(labels, (0, 1), value=ignore_index)
            shifted_labels = shifted_labels[..., 1:].contiguous().view(-1)
            loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
                reduction="mean",
            )
            return loss_fn(self.lm_head.weight, flat_hidden, shifted_labels)

        logits = self.lm_head(hidden_states)
        return self.loss_function(logits, labels, self.vocab_size, **kwargs)

    def get_all_balancing_owners(self):
        """Yield (owner_module, label) for every load-balancing owner.

        Until bank-level balancing state is introduced, routers across
        the MLP bank, the four attention router classes, and the
        branch router(s) each own their own `expert_bias` /
        `local_tokens_per_expert` buffers. Labels match the
        per-projection bias-rate keys consumed by the trainer
        (`bias_rate_q/k/v/o/mlp/branch`). A future bank-level
        refactor will collapse the MLP and attention routers to
        bank-level owners.
        """
        inner = self.model

        # When per-layer expert banks are active there's one MlpExpertBank
        # and one AttentionExpertBank PER depth; the walker must visit
        # every one so each depth's balancing buffers are updated. Falls
        # back to the single shared bank in the default global case.
        mlp_banks_list = getattr(inner, "mlp_banks_per_depth", None)
        attn_banks_list = getattr(inner, "attn_banks_per_depth", None)
        if mlp_banks_list is not None:
            mlp_banks_iter = list(mlp_banks_list)
        else:
            mlp_banks_iter = [getattr(inner, "mlp_bank", None)]
        if attn_banks_list is not None:
            attn_banks_iter = list(attn_banks_list)
        else:
            attn_banks_iter = [getattr(inner, "attn_bank", None)]

        # MLP bank: walk every submodule that exposes the balancing-owner
        # interface; name path is unused (label is always "mlp").
        for mlp_bank in mlp_banks_iter:
            if mlp_bank is None:
                continue
            for _name, m in mlp_bank.named_modules():
                if hasattr(m, "expert_bias") and hasattr(m, "local_tokens_per_expert"):
                    yield m, "mlp"

        # Attention bank: derive q/k/v/o label from the attribute path so the
        # trainer's per-projection bias rates land on the right routers.
        for attn_bank in attn_banks_iter:
            if attn_bank is None:
                continue
            for name, m in attn_bank.named_modules():
                if not (hasattr(m, "expert_bias") and hasattr(m, "local_tokens_per_expert")):
                    continue
                head = name.split(".", 1)[0] if name else ""
                if head.startswith(("q_", "qk_", "qkv_", "qkvo_")):
                    label = "q"
                elif head.startswith("k_"):
                    label = "k"
                elif head.startswith(("v_", "vo_")):
                    label = "v"
                elif head.startswith("o_"):
                    label = "o"
                else:
                    # Single non-q/k/v/o-split set (e.g. attn_expert_mode where
                    # one router serves all heads); fall back to "q" so the
                    # default per_proj_rates dict still resolves to a numeric
                    # rate. Specialized split-rate users can extend the map
                    # once the bank-level state rule lands a cleaner naming.
                    label = "q"
                yield m, label

        # Branch router(s): singular when per_layer_router=False, plural list
        # when per_layer_router=True. Both attribute names exist depending on
        # config; check both.
        branch_router = getattr(inner, "branch_router", None)
        if branch_router is not None and hasattr(branch_router, "expert_bias") \
                and hasattr(branch_router, "local_tokens_per_expert"):
            yield branch_router, "branch"

        branch_routers = getattr(inner, "branch_routers", None)
        if branch_routers is not None:
            for br in branch_routers:
                if br is not None and hasattr(br, "expert_bias") \
                        and hasattr(br, "local_tokens_per_expert"):
                    yield br, "branch"

        # Prelude / coda blocks: each owns its own per-layer MLP expert
        # pool and router, matching standard_moe's bias-update semantics.
        # Label them "mlp" so the trainer's per-projection bias rates
        # land on the right buckets.
        for boundary_blocks_attr in ("prelude_blocks", "coda_blocks"):
            blocks = getattr(inner, boundary_blocks_attr, None)
            if blocks is None:
                continue
            for block in blocks:
                gate = getattr(getattr(block, "mlp", None), "gate", None)
                if gate is not None and hasattr(gate, "expert_bias") \
                        and hasattr(gate, "local_tokens_per_expert"):
                    yield gate, "mlp"

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_router_logits: bool = False,
        return_logits: bool = True,
        num_steps: torch.Tensor | tuple[int, int] | list[int] | int | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            num_steps=num_steps,
        )
        logits = self.lm_head(hidden_states) if return_logits else None

        loss = None
        ce_loss = None
        aux_loss = None
        seq_aux_loss = None
        branch_aux_loss = None
        branch_entropy_loss = None
        attention_aux_loss = None

        mlp_router_logits = tuple(self.model._all_mlp_router_logits) or None
        mlp_selected_experts = tuple(self.model._all_mlp_selected_experts) or None
        mlp_token_masks = tuple(self.model._all_mlp_token_masks) or None
        branch_prob_tensors = tuple(self.model._all_branch_probs) or None
        attention_router_info = tuple(self.model._all_attn_router_info) or None

        if labels is not None:
            ce_loss = self._compute_lm_ce_loss(hidden_states, labels, **kwargs)
            loss = ce_loss

            # Method gating: gate aux / seq-aux additions by the resolved
            # `load_balancing_method`. `None` keeps the legacy coefficient-
            # driven behavior; explicit methods restrict to the method's
            # active loss term. `normalize_balancing_config` already
            # auto-zeros conflicting coefficients, but the explicit gate
            # here is belt-and-suspenders against future regressions.
            method = getattr(self, "_load_balancing_method", None)
            top_aux_active = method is None or method == "aux_loss"
            top_seq_aux_active = method is None or method == "seq_aux_loss"

            # Per-class method gating: when the nested-schema yaml sets
            # `model.mlp_router.balancing` or `model.attn_router.balancing`,
            # the per-class value WINS over the top-level method. This is
            # the runtime side of the nested-schema contract: a yaml that
            # opts MLP into aux_loss and attention into seq_aux_loss can
            # express that without the legacy top-level method having to
            # be both at once. When a per-class field is unset, fall back
            # to the top-level decision so legacy yamls keep working.
            mlp_class_method = getattr(self.config, "mlp_router_balancing", None)
            attn_class_method = getattr(self.config, "attn_router_balancing", None)
            mlp_aux_active = (
                top_aux_active if mlp_class_method is None
                else mlp_class_method == "aux_loss"
            )
            mlp_seq_aux_active = (
                top_seq_aux_active if mlp_class_method is None
                else mlp_class_method == "seq_aux_loss"
            )
            attn_aux_active = (
                top_aux_active if attn_class_method is None
                else attn_class_method == "aux_loss"
            )
            attn_seq_aux_active = (
                top_seq_aux_active if attn_class_method is None
                else attn_class_method == "seq_aux_loss"
            )

            # Per-class coefficients: when the nested-schema yaml sets
            # `model.{mlp,attn}_router.router_aux_loss_coef` or
            # `model.{mlp,attn}_router.seq_aux_loss_coef`, the per-class
            # value WINS over the top-level training coefficient. Falls
            # back to top-level when unset so legacy yamls still work.
            seq_aux_coef = getattr(self, "_seq_aux_loss_coef", 0.0)
            mlp_aux_coef = getattr(
                self.config, "mlp_router_router_aux_loss_coef",
                self.router_aux_loss_coef,
            )
            mlp_seq_aux_coef = getattr(
                self.config, "mlp_router_seq_aux_loss_coef", seq_aux_coef,
            )
            attn_aux_coef = getattr(
                self.config, "attn_router_router_aux_loss_coef",
                self.router_aux_loss_coef,
            )
            attn_seq_aux_coef = getattr(
                self.config, "attn_router_seq_aux_loss_coef", seq_aux_coef,
            )

            if mlp_aux_active and mlp_router_logits is not None:
                mlp_aux = load_balancing_loss_func(
                    mlp_router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    token_masks=mlp_token_masks,
                    selected_experts=mlp_selected_experts,
                )
                if isinstance(mlp_aux, torch.Tensor):
                    aux_loss = mlp_aux
                    loss = loss + mlp_aux_coef * mlp_aux

            if mlp_seq_aux_active and mlp_seq_aux_coef > 0 and mlp_router_logits is not None:
                seq_aux = seq_load_balancing_loss_func(
                    mlp_router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    batch_size=input_ids.shape[0],
                    selected_experts=mlp_selected_experts,
                    token_masks=mlp_token_masks,
                )
                if isinstance(seq_aux, torch.Tensor):
                    seq_aux_loss = seq_aux
                    loss = loss + mlp_seq_aux_coef * seq_aux

            # Attention expert router losses.
            # Method gating: only run when method allows aux or seq_aux contributions.
            # Skip auxiliary terms in sanity mode because routing is deterministic there.
            if (
                (attn_aux_active or attn_seq_aux_active)
                and attention_router_info is not None
                and getattr(self.config, "sanity_check_mode", None) != "alternating_global_moe"
            ):
                # Gather per-router logits and selected experts across depths
                attn_router_names = sorted({n for d in attention_router_info for n in d})
                num_attn_experts = self.config.num_attn_experts
                num_kv_experts = getattr(self.model.attn_bank, "num_kv_experts", num_attn_experts)
                num_o_experts = getattr(self.model.attn_bank, "num_o_experts", num_attn_experts)
                for rname in attn_router_names:
                    r_logits = tuple(d[rname]["router_logits"] for d in attention_router_info if rname in d)
                    r_selected = tuple(d[rname]["selected_experts"] for d in attention_router_info if rname in d)
                    r_masks = tuple(d[rname].get("token_mask") for d in attention_router_info if rname in d)
                    if not r_logits:
                        continue
                    if rname in ("k", "v"):
                        n_experts = num_kv_experts
                        n_per_tok = 1  # per-head top-1 routing: each head picks 1 expert
                    elif rname == "o":
                        n_experts = num_o_experts
                        n_per_tok = 1  # per-head top-1 routing
                    else:
                        n_experts = num_attn_experts
                        n_per_tok = 1  # per-head top-1 routing
                    if attn_aux_active:
                        attn_aux = load_balancing_loss_func(
                            r_logits,
                            n_experts,
                            n_per_tok,
                            token_masks=r_masks,
                            selected_experts=r_selected,
                        )
                        if isinstance(attn_aux, torch.Tensor):
                            attention_aux_loss = attn_aux if attention_aux_loss is None else attention_aux_loss + attn_aux
                    if attn_seq_aux_active and attn_seq_aux_coef > 0:
                        attn_seq_aux = seq_load_balancing_loss_func(
                            r_logits, n_experts, n_per_tok,
                            batch_size=input_ids.shape[0],
                            selected_experts=r_selected,
                            token_masks=r_masks,
                        )
                        if isinstance(attn_seq_aux, torch.Tensor):
                            loss = loss + attn_seq_aux_coef * attn_seq_aux

            if attn_aux_active and isinstance(attention_aux_loss, torch.Tensor):
                aux_loss = attention_aux_loss if aux_loss is None else aux_loss + attention_aux_loss
                loss = loss + attn_aux_coef * attention_aux_loss

            # Branch-router aux/seq-aux contribution: when the
            # nested-schema yaml sets `model.branch_router.balancing`
            # to `aux_loss` or `seq_aux_loss`, the branch routing's
            # per-token probability tensors flow into the same
            # load-balancing loss helpers used for MLP / attention.
            # The branch is binary (ATTN vs MLP) so num_experts == 2
            # and num_experts_per_tok == 1 (each token picks one).
            # When the per-class field is unset or `none` /
            # `exploration_only`, no branch aux contribution is
            # added (the prior contract).
            #
            # Config field: `MoEverythingConfig.branch_balancing`
            # (set from yaml `model.branch_router.balancing` by the
            # factory) — NOT `branch_router_balancing` (which is
            # never set). The factory's flat-bridge plumbing keeps
            # this name, so the runtime reads it directly.
            branch_class_method = getattr(self.config, "branch_balancing", None)
            if branch_class_method in ("aux_loss", "seq_aux_loss") and branch_prob_tensors is not None:
                # Branch probs are (B, T, 2); the load-balancing
                # helpers expect (B*T, E). Reshape per layer before
                # passing through.
                B = input_ids.shape[0]
                branch_prob_2d = tuple(
                    t.reshape(-1, 2) for t in branch_prob_tensors
                )
                branch_selected_raw = self.model._all_branch_selected_experts
                # Selected experts per layer: shape (B, T, 1) -> (B*T, 1)
                branch_selected = tuple(
                    t.reshape(-1, 1) for t in branch_selected_raw
                ) if branch_selected_raw else None
                # Per-class branch coefficients are stamped into the
                # branch_router config block by `model_factory.py`'s
                # `_get_branch_router_field` helper. The flat bridge
                # exposes them as `branch_router_aux_loss_coef` /
                # `branch_router_seq_aux_loss_coef` (NOT
                # `branch_router_router_aux_loss_coef`).
                _mcfg = getattr(self.config, "_mcfg_branch_router", {}) or {}
                branch_aux_coef = float(
                    _mcfg.get("router_aux_loss_coef")
                    if "router_aux_loss_coef" in _mcfg
                    else self.branch_router_aux_loss_coef
                )
                branch_seq_aux_coef = float(
                    _mcfg.get("seq_aux_loss_coef")
                    if "seq_aux_loss_coef" in _mcfg
                    else seq_aux_coef
                )
                if branch_class_method == "aux_loss" and branch_aux_coef > 0:
                    branch_aux_term = load_balancing_loss_func(
                        branch_prob_2d,
                        2,  # binary ATTN/MLP pool
                        1,  # one branch per token
                        selected_experts=branch_selected,
                    )
                    if isinstance(branch_aux_term, torch.Tensor):
                        branch_aux_loss = branch_aux_term
                        loss = loss + branch_aux_coef * branch_aux_term
                if branch_class_method == "seq_aux_loss" and branch_seq_aux_coef > 0:
                    branch_seq_aux_term = seq_load_balancing_loss_func(
                        branch_prob_2d,
                        2, 1,
                        batch_size=B,
                        selected_experts=branch_selected,
                    )
                    if isinstance(branch_seq_aux_term, torch.Tensor):
                        branch_aux_loss = branch_seq_aux_term
                        loss = loss + branch_seq_aux_coef * branch_seq_aux_term
            if branch_class_method == "sampling_entropy" and branch_prob_tensors is not None:
                entropy_coef = float(
                    getattr(
                        self.config,
                        "current_branch_entropy_coef",
                        getattr(self.config, "branch_entropy_coef", 0.0),
                    ) or 0.0
                )
                if entropy_coef > 0.0:
                    entropies = []
                    for probs in branch_prob_tensors:
                        p = probs.float().clamp_min(1e-20)
                        entropies.append(-(p * p.log()).sum(dim=-1).mean())
                    if entropies:
                        branch_entropy_loss = torch.stack(entropies).mean()
                        loss = loss - entropy_coef * branch_entropy_loss

        if output_router_logits:
            # Aux methods need `router_logits` to be gradient-bearing
            # in the model output so the aux/seq-aux loss term can
            # backprop through them; non-aux methods detach for
            # telemetry-only output (no autograd graph cost).
            # Unconditional detach silently breaks aux-method gradient
            # flow.
            method = getattr(self, "_load_balancing_method", None)
            mlp_grad_bearing = method is None or method in ("aux_loss", "seq_aux_loss")
            if mlp_router_logits is not None:
                if mlp_grad_bearing:
                    router_logits_out = tuple(mlp_router_logits)
                else:
                    router_logits_out = tuple(t.detach() for t in mlp_router_logits)
            else:
                router_logits_out = None
            selected_experts_out = tuple(t.detach() for t in mlp_selected_experts) if mlp_selected_experts is not None else None
            router_token_masks_out = tuple(t.detach() if t is not None else None for t in mlp_token_masks) if mlp_token_masks is not None else None
            branch_probs_out = tuple(t.detach() for t in branch_prob_tensors) if branch_prob_tensors is not None else None
            attention_router_info_out = None
            if attention_router_info is not None:
                # DETACH-ONLY: for non-aux methods, expose the
                # `router_logits_detached` side channel that
                # `attention_bank._store_router_info` always populates. The
                # gradient-bearing `router_logits` view is reserved for aux
                # methods that need it for the in-forward aux-loss
                # accumulator. Falling back to detach() if the side channel
                # is missing (older paths that may not have populated it).
                def _attn_router_logits(info):
                    if mlp_grad_bearing:
                        return info["router_logits"]
                    detached = info.get("router_logits_detached")
                    return detached if detached is not None else info["router_logits"].detach()

                attention_router_info_out = tuple(
                    {
                        name: {
                            "router_logits": _attn_router_logits(info),
                            "selected_experts": info["selected_experts"],
                            **({"token_mask": info["token_mask"]} if "token_mask" in info else {}),
                        }
                        for name, info in depth_info.items()
                    }
                    for depth_info in attention_router_info
                )
        else:
            router_logits_out = None
            selected_experts_out = None
            router_token_masks_out = None
            branch_probs_out = None
            attention_router_info_out = None

        return _MoEverythingOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            router_logits=router_logits_out,
            ce_loss=ce_loss,
            seq_aux_loss=seq_aux_loss,
            branch_aux_loss=branch_aux_loss,
            branch_entropy_loss=branch_entropy_loss,
            attention_aux_loss=attention_aux_loss,
            selected_experts=selected_experts_out,
            router_token_masks=router_token_masks_out,
            branch_probs=branch_probs_out,
            attention_router_info=attention_router_info_out,
            num_steps_no_grad=self.model._last_num_steps_no_grad,
            num_steps_with_grad=self.model._last_num_steps_with_grad,
        )


class _MoEverythingOutput(dict):
    """Dict-like output object compatible with FSDP and trainer attributes.

    FSDP traverses standard Python containers in the forward output to
    attach pre-backward hooks. A plain custom object hides the `loss`
    tensor from that traversal and leaves FSDP in `IDLE` when backward
    starts, so this mirrors the relevant HuggingFace `ModelOutput`
    behavior without adding another dependency.
    """

    def __init__(
        self,
        loss,
        logits,
        aux_loss=None,
        router_logits=None,
        ce_loss=None,
        seq_aux_loss=None,
        branch_aux_loss=None,
        branch_entropy_loss=None,
        attention_aux_loss=None,
        selected_experts=None,
        router_token_masks=None,
        branch_probs=None,
        attention_router_info=None,
        num_steps_no_grad=None,
        num_steps_with_grad=None,
    ):
        values = {
            "loss": loss,
            "logits": logits,
            "aux_loss": aux_loss,
            "router_logits": router_logits,
            "ce_loss": ce_loss,
            "seq_aux_loss": seq_aux_loss,
            "branch_aux_loss": branch_aux_loss,
            "branch_entropy_loss": branch_entropy_loss,
            "attention_aux_loss": attention_aux_loss,
            "selected_experts": selected_experts,
            "router_token_masks": router_token_masks,
            "branch_probs": branch_probs,
            "attention_router_info": attention_router_info,
            "num_steps_no_grad": num_steps_no_grad,
            "num_steps_with_grad": num_steps_with_grad,
            "past_key_values": None,
            "hidden_states": None,
            "attentions": None,
        }
        super().__init__((key, value) for key, value in values.items() if value is not None)
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if value is None:
            self.pop(key, None)
        else:
            self[key] = value
