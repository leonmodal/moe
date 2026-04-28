"""MoE-Everything model assembly: MoEverythingModel and MoEverythingForCausalLM.

Hierarchical per-token routing with shared attention/MLP expert banks.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.modeling_qwen3_moe import (
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from src.models.routing.routers import BranchRouter, BranchRouterRecorder
from src.models.router import checkpoint_recompute_context
from src.models.load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from .config import MoEverythingConfig
from .attention_bank import AttentionExpertBank, NormExpertBank
from .mlp_bank import MlpExpertBank

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
        if config.attn_expert_mode == "per_head_precompute_kv":
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
        if self.sanity_check_mode == "alternating_global_moe":
            if self.per_layer_router:
                self.branch_routers = nn.ModuleList([BranchRouterRecorder() for _ in range(self.num_depths)])
            else:
                self.branch_router = BranchRouterRecorder()
        elif self.per_layer_router:
            self.branch_routers = nn.ModuleList([
                BranchRouter(config.hidden_size, exploration_rate=branch_exploration_rate,
                             scale_by_routing_weight=scale_branch,
                             use_sampling=use_sampling, use_seq_level=use_seq_level,
                             use_deepseek_style=use_deepseek_style)
                for _ in range(self.num_depths)
            ])
        else:
            self.branch_router = BranchRouter(config.hidden_size, exploration_rate=branch_exploration_rate,
                                              scale_by_routing_weight=scale_branch,
                                              use_sampling=use_sampling, use_seq_level=use_seq_level,
                                              use_deepseek_style=use_deepseek_style)

        self.attn_bank = AttentionExpertBank(config)
        self.mlp_bank = MlpExpertBank(config)

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

        self._all_mlp_router_logits = []
        self._all_mlp_selected_experts = []
        self._all_mlp_token_masks = []
        self._all_branch_probs = []
        self._all_branch_selected_experts = []
        self._all_attn_router_info = []

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

    def _sanity_branch_route(
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

    def _depth_step(
        self,
        hidden_states: torch.Tensor,
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        depth_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sanity_check_mode == "alternating_global_moe":
            w_attn, w_mlp, attn_mask, mlp_mask = self._sanity_branch_route(hidden_states, depth_idx)
        elif self.per_layer_router:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_routers[depth_idx](hidden_states)
        else:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_router(hidden_states)

        attn_mask_bool = attn_mask.bool()
        use_sparse_attn = self.attn_bank.should_use_sparse_path(attn_mask_bool)

        # Attention branch
        if self.attn_bank.mode == "per_head_precompute_kv" and use_sparse_attn:
            attn_out, K_new, V_new = self.attn_bank.project_and_attend_per_head_precompute_kv_sparse(
                hidden_states, position_embeddings, K_old, V_old, attn_mask_bool, causal_mask, depth_idx=depth_idx
            )
        elif self.attn_bank.mode == "per_head_precompute_kv":
            attn_out, K_new, V_new = self.attn_bank.project_and_attend_per_head_precompute_kv_dense_mixed(
                hidden_states, position_embeddings, K_old, V_old, attn_mask_bool, causal_mask, depth_idx=depth_idx
            )
            self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)
        elif self.attn_bank.mode == "per_head_fully_independent" and use_sparse_attn:
            attn_out, K_new, V_new = self.attn_bank.project_and_attend_per_head_fully_independent_sparse(
                hidden_states, position_embeddings, K_old, V_old, attn_mask_bool, causal_mask, depth_idx=depth_idx
            )
        else:
            Q, K_fresh, V_fresh = self.attn_bank.project(hidden_states, position_embeddings, depth_idx=depth_idx)
            # Tokens that chose MLP keep old KV in the blend
            attn_mask_kv = attn_mask_bool.unsqueeze(1)  # (B, 1, T, 1)
            K_blend = torch.where(attn_mask_kv, K_fresh, K_old)
            V_blend = torch.where(attn_mask_kv, V_fresh, V_old)
            attn_out = self.attn_bank.attend(Q, K_blend, V_blend, causal_mask, depth_idx=depth_idx)
            K_new = torch.where(attn_mask_kv, K_fresh, K_old)
            V_new = torch.where(attn_mask_kv, V_fresh, V_old)
            if self.attn_bank.mode in ("per_head_fully_independent", "per_head_precompute_kv"):
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

    def _depth_step_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        depth_idx: int = 0,
    ) -> torch.Tensor:
        if self.sanity_check_mode == "alternating_global_moe":
            w_attn, w_mlp, attn_mask, mlp_mask = self._sanity_branch_route(hidden_states, depth_idx)
        elif self.per_layer_router:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_routers[depth_idx](hidden_states)
        else:
            w_attn, w_mlp, attn_mask, mlp_mask = self.branch_router(hidden_states)

        attn_mask_bool = attn_mask.bool()
        use_sparse_attn = self.attn_bank.should_use_sparse_path(attn_mask_bool)
        B, T, _ = hidden_states.shape
        dummy_k = hidden_states.new_zeros(B, self.num_kv_heads, T, self.head_dim)
        dummy_v = hidden_states.new_zeros(B, self.num_kv_heads, T, self.head_dim)

        if not bool(attn_mask_bool.any().item()):
            attn_out, _, _ = self.attn_bank._empty_sparse_attn_result(
                hidden_states,
                dummy_k,
                dummy_v,
                [("attn", self.attn_bank.num_experts, self.attn_bank.num_kv_heads)],
            )
        elif bool(attn_mask_bool.all().item()):
            attn_out, _, _ = self.attn_bank.project_and_attend_per_head_precompute_kv(
                hidden_states,
                position_embeddings,
                causal_mask,
                depth_idx=depth_idx,
            )
            self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)
        elif use_sparse_attn:
            attn_out, _, _ = self.attn_bank.project_and_attend_per_head_precompute_kv_sparse(
                hidden_states,
                position_embeddings,
                dummy_k,
                dummy_v,
                attn_mask_bool,
                causal_mask,
                depth_idx=depth_idx,
            )
            self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)
        else:
            attn_out, _, _ = self.attn_bank.project_and_attend_per_head_precompute_kv_dense_mixed(
                hidden_states,
                position_embeddings,
                dummy_k,
                dummy_v,
                attn_mask_bool,
                causal_mask,
                depth_idx=depth_idx,
            )
            self.attn_bank._attach_token_mask_to_last_router_info(attn_mask_bool)

        if self.post_norm:
            pn = self.attn_post_norms[depth_idx] if hasattr(self, "attn_post_norms") else self.attn_post_norm
            attn_out = pn(attn_out)
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
        causal_mask = mask_function(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        use_precompute_kv = self.attn_bank.mode == "per_head_precompute_kv"
        if not use_precompute_kv:
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

            if use_precompute_kv:
                if self.gradient_checkpointing and self.training:
                    checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                    checkpoint_kwargs.setdefault("use_reentrant", False)
                    checkpoint_kwargs.setdefault("context_fn", self._checkpoint_context_fn)

                    def depth_step_precompute(h, _depth_idx=d):
                        return self._depth_step_precompute_kv(
                            h, position_embeddings, causal_mask, depth_idx=_depth_idx
                        )

                    hidden_states = checkpoint(depth_step_precompute, hidden_states, **checkpoint_kwargs)
                else:
                    hidden_states = self._depth_step_precompute_kv(
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
                        return self._depth_step(
                            h, k, v, position_embeddings, causal_mask, depth_idx=_depth_idx
                        )

                    hidden_states, K_new, V_new = checkpoint(
                        depth_step, hidden_states, K_old, V_old, **checkpoint_kwargs
                    )
                else:
                    hidden_states, K_new, V_new = self._depth_step(
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

            if not use_precompute_kv:
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

        hidden_states = self.norm(hidden_states)
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

        # MLP bank: walk every submodule that exposes the balancing-owner
        # interface; name path is unused (label is always "mlp").
        mlp_bank = getattr(inner, "mlp_bank", None)
        if mlp_bank is not None:
            for _name, m in mlp_bank.named_modules():
                if hasattr(m, "expert_bias") and hasattr(m, "local_tokens_per_expert"):
                    yield m, "mlp"

        # Attention bank: derive q/k/v/o label from the attribute path so the
        # trainer's per-projection bias rates land on the right routers.
        attn_bank = getattr(inner, "attn_bank", None)
        if attn_bank is not None:
            for name, m in attn_bank.named_modules():
                if not (hasattr(m, "expert_bias") and hasattr(m, "local_tokens_per_expert")):
                    continue
                head = name.split(".", 1)[0] if name else ""
                if head.startswith("q_"):
                    label = "q"
                elif head.startswith("k_"):
                    label = "k"
                elif head.startswith("v_"):
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_router_logits: bool = False,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        aux_loss = None
        seq_aux_loss = None
        branch_aux_loss = None
        attention_aux_loss = None

        mlp_router_logits = tuple(self.model._all_mlp_router_logits) or None
        mlp_selected_experts = tuple(self.model._all_mlp_selected_experts) or None
        mlp_token_masks = tuple(self.model._all_mlp_token_masks) or None
        branch_prob_tensors = tuple(self.model._all_branch_probs) or None
        attention_router_info = tuple(self.model._all_attn_router_info) or None

        if labels is not None:
            ce_loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
            loss = ce_loss

            # Method gating: gate aux / seq-aux additions by the resolved
            # `load_balancing_method`. `None` keeps the legacy coefficient-
            # driven behavior; explicit methods restrict to the method's
            # active loss term. `normalize_balancing_config` already
            # auto-zeros conflicting coefficients, but the explicit gate
            # here is belt-and-suspenders against future regressions.
            method = getattr(self, "_load_balancing_method", None)
            aux_active = method is None or method == "aux_loss"
            seq_aux_active = method is None or method == "seq_aux_loss"

            if aux_active and mlp_router_logits is not None:
                mlp_aux = load_balancing_loss_func(
                    mlp_router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    token_masks=mlp_token_masks,
                    selected_experts=mlp_selected_experts,
                )
                if isinstance(mlp_aux, torch.Tensor):
                    aux_loss = mlp_aux
                    loss = loss + self.router_aux_loss_coef * mlp_aux

            seq_aux_coef = getattr(self, "_seq_aux_loss_coef", 0.0)
            if seq_aux_active and seq_aux_coef > 0 and mlp_router_logits is not None:
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
                    loss = loss + seq_aux_coef * seq_aux

            # Attention expert router losses.
            # Method gating: only run when method allows aux or seq_aux contributions.
            # Skip auxiliary terms in sanity mode because routing is deterministic there.
            if (
                (aux_active or seq_aux_active)
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
                    if aux_active:
                        attn_aux = load_balancing_loss_func(
                            r_logits,
                            n_experts,
                            n_per_tok,
                            token_masks=r_masks,
                            selected_experts=r_selected,
                        )
                        if isinstance(attn_aux, torch.Tensor):
                            attention_aux_loss = attn_aux if attention_aux_loss is None else attention_aux_loss + attn_aux
                    if seq_aux_active and seq_aux_coef > 0:
                        attn_seq_aux = seq_load_balancing_loss_func(
                            r_logits, n_experts, n_per_tok,
                            batch_size=input_ids.shape[0],
                            selected_experts=r_selected,
                            token_masks=r_masks,
                        )
                        if isinstance(attn_seq_aux, torch.Tensor):
                            loss = loss + seq_aux_coef * attn_seq_aux

            if aux_active and isinstance(attention_aux_loss, torch.Tensor):
                aux_loss = attention_aux_loss if aux_loss is None else aux_loss + attention_aux_loss
                loss = loss + self.router_aux_loss_coef * attention_aux_loss

            # Do not apply auxiliary balancing to the branch router. The
            # branch split is part of the model behavior we want to observe,
            # not something we want to regularize toward 50/50 usage.

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
            attention_aux_loss=attention_aux_loss,
            selected_experts=selected_experts_out,
            router_token_masks=router_token_masks_out,
            branch_probs=branch_probs_out,
            attention_router_info=attention_router_info_out,
        )


class _MoEverythingOutput:
    """Minimal output object compatible with train.py expectations."""

    def __init__(
        self,
        loss,
        logits,
        aux_loss=None,
        router_logits=None,
        ce_loss=None,
        seq_aux_loss=None,
        branch_aux_loss=None,
        attention_aux_loss=None,
        selected_experts=None,
        router_token_masks=None,
        branch_probs=None,
        attention_router_info=None,
    ):
        self.loss = loss
        self.logits = logits
        self.aux_loss = aux_loss
        self.router_logits = router_logits
        self.ce_loss = ce_loss
        self.seq_aux_loss = seq_aux_loss
        self.branch_aux_loss = branch_aux_loss
        self.attention_aux_loss = attention_aux_loss
        self.selected_experts = selected_experts
        self.router_token_masks = router_token_masks
        self.branch_probs = branch_probs
        self.attention_router_info = attention_router_info
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
