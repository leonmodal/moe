"""
Mixture-of-Everything (Section 4.1) — hierarchical per-token routing.

Each token at each depth chooses ATTENTION or MLP via a branch router,
then selects expert weights within the chosen branch.

Key design: ALL components are shared across depths.  The forward pass
is a simple for loop over the same branch router, attention bank, and
MLP bank — like a Universal Transformer with heterogeneous expert
branches.  Only activations change per depth, not weights.

Attention bank modes:
  - "bundled"            : (Q,K,V,O) selected as one unit — 1 router
  - "kv_paired"          : (K,V) paired + Q,O independent — 3 routers
  - "qk_paired"          : (Q,K) paired + V,O independent — 3 routers
  - "fully_independent"  : Q, K, V, O each from separate banks — 4 routers
  - "per_head_fully_independent": each Q/K/V/O head routes independently
  - "precompute_kv"      : per-expert KV tables, routed Q and O — 1 router
  - "per_head_precompute_kv": one flat router picks per-head experts; KV/O reuse that routing

MLP bank: standard top-k MoE over SwiGLU experts (reuses Qwen3MoeExperts).

Design notes:
  - Token state is (E, K, V).  K,V persist across depths and are only
    refreshed when a token takes the attention branch.
  - Hard branch routing: each token picks EITHER attention OR MLP via
    argmax. MLP compute is restricted to selected tokens, and the
    per-head attention modes also restrict attention compute to
    selected tokens. The output is scaled by the softmax probability
    of the chosen branch for gradient flow (same pattern as MoE
    expert routing).
  - KV state: tokens that chose attention get fresh KV; tokens that
    chose MLP keep their old KV from the previous depth.
  - Pre-norm is bundled with each bank: the attention bank and MLP bank
    each include their own RMSNorm, applied before computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from .router import DeepSeekRouter, checkpoint_recompute_context, is_checkpoint_recompute


# ─── Config ────────────────────────────────────────────────────────────────── #

class MoEverythingConfig(Qwen3MoeConfig):
    model_type = "moe_everything"

    def __init__(
        self,
        # Attention expert bank
        num_attn_experts: int = 4,
        num_attn_experts_per_tok: int = 1,
        attn_expert_mode: str = "bundled",
        # Branch router
        branch_router_aux_loss_coef: float = 0.0,
        # Routing style
        use_deepseek_routing: bool = False,
        # Per-layer router: one branch router per depth instead of shared
        per_layer_router: bool = False,
        # Per-layer MLP router: one MLP gate per depth instead of shared
        per_layer_mlp_router: bool = False,
        # Per-layer attention router: one set of attn expert routers per depth
        per_layer_attn_router: bool = False,
        # Routed norm: bank of RMSNorm experts with per-token routing
        routed_norm: bool = False,
        # Per-layer norm: separate attn + MLP norms per depth (standard transformer style)
        per_layer_norm: bool = False,
        # Post-norm: RMSNorm on branch output before residual addition
        post_norm: bool = False,
        # Dynamic depth: random perturbation of depth during training
        dynamic_depth_min: float = 1.0,
        dynamic_depth_max: float = 1.0,
        # Depthwise attention (AttnRes): learned weighted combination across depths
        depthwise_attention: bool = False,
        depthwise_block_size: int = 0,
        # Per-head attention execution strategy
        per_head_compute_mode: str = "auto",
        per_head_dense_fraction_threshold: float = 0.75,
        # Deterministic routing mode used for architecture sanity checks
        sanity_check_mode: str | None = None,
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
        self.post_norm = post_norm
        self.dynamic_depth_min = dynamic_depth_min
        self.dynamic_depth_max = dynamic_depth_max
        self.depthwise_attention = depthwise_attention
        self.depthwise_block_size = depthwise_block_size
        self.per_head_compute_mode = per_head_compute_mode
        self.per_head_dense_fraction_threshold = per_head_dense_fraction_threshold
        self.sanity_check_mode = sanity_check_mode


# ─── Branch router ─────────────────────────────────────────────────────────── #

class BranchRouter(nn.Module):
    """Binary router: ATTN (0) or MLP (1) per token.

    Hard routing: each token picks one branch via argmax.
    The selected branch output is scaled by its softmax probability
    for gradient flow (same pattern as MoE expert routing).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)
        self.last_probs = None

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate(hidden_states.float())
        probs = F.softmax(logits, dim=-1).to(hidden_states.dtype)
        self.last_probs = probs
        # Hard selection: 0 = attn, 1 = mlp
        choice = probs.argmax(dim=-1)              # (...,)
        attn_mask = (choice == 0).unsqueeze(-1)     # (..., 1)
        mlp_mask = (choice == 1).unsqueeze(-1)      # (..., 1)
        # Weight = probability of selected branch (differentiable)
        w_attn = probs[..., 0:1] * attn_mask       # (..., 1)
        w_mlp = probs[..., 1:2] * mlp_mask         # (..., 1)
        return w_attn, w_mlp, attn_mask, mlp_mask


# ─── Routed norm bank ─────────────────────────────────────────────────────── #

class NormExpertBank(nn.Module):
    """Bank of RMSNorm experts with per-token top-1 routing.

    Each expert is a learned per-element scale (like RMSNorm weight).
    A router selects which scale to apply per token, with the output
    weighted by the routing probability for gradient flow.
    """

    def __init__(self, num_experts: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.eps = eps
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.weight = nn.Parameter(torch.ones(num_experts, hidden_size))
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)

        # Top-1 routing
        logits = self.router(flat.float())
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)                           # (N,)
        router_weight = probs.gather(1, idx.unsqueeze(-1))   # (N, 1)

        # Track token counts (skip checkpoint recompute)
        if torch.is_grad_enabled() and not is_checkpoint_recompute():
            with torch.no_grad():
                counts = torch.bincount(idx, minlength=self.num_experts).float()
                self.local_tokens_per_expert += counts

        # RMSNorm with selected expert's weight
        x_float = flat.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)

        selected_weight = self.weight[idx]                    # (N, H)
        out = (selected_weight * x_normed).to(hidden_states.dtype)
        out = out * router_weight.to(out.dtype)

        return out.reshape(orig_shape)


# ─── Attention expert bank ─────────────────────────────────────────────────── #

class AttentionExpertBank(nn.Module):
    """Bank of attention weight sets with per-token expert routing.

    Includes pre-norm (RMSNorm applied before routing/projection).
    All weights are shared across depths.
    """

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.config = config
        self.mode = config.attn_expert_mode
        self.num_experts = config.num_attn_experts
        self.top_k = config.num_attn_experts_per_tok
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.eps = config.rms_norm_eps
        self.use_deepseek_routing = getattr(config, "use_deepseek_routing", False)
        self.per_layer_attn_router = getattr(config, "per_layer_attn_router", False)
        self.routed_norm = getattr(config, "routed_norm", False)
        self.per_layer_norm = getattr(config, "per_layer_norm", False)
        self.num_depths = config.num_hidden_layers
        self.per_head_compute_mode = getattr(config, "per_head_compute_mode", "auto")
        self.per_head_dense_fraction_threshold = getattr(
            config, "per_head_dense_fraction_threshold", 0.75
        )
        self.sanity_check_mode = getattr(config, "sanity_check_mode", None)
        self.last_router_info = {}
        self._last_routing = None

        _MODES = {
            "bundled",
            "kv_paired",
            "qk_paired",
            "fully_independent",
            "per_head_fully_independent",
            "precompute_kv",
            "per_head_precompute_kv",
        }
        if self.mode not in _MODES:
            raise ValueError(f"Unknown attn_expert_mode: {self.mode}, must be one of {_MODES}")
        if self.per_head_compute_mode not in {"auto", "sparse", "dense"}:
            raise ValueError(
                f"Unknown per_head_compute_mode: {self.per_head_compute_mode}, must be one of "
                "{'auto', 'sparse', 'dense'}"
            )
        if self.sanity_check_mode is not None and self.mode != "per_head_precompute_kv":
            raise ValueError(
                "sanity_check_mode is only supported with attn_expert_mode='per_head_precompute_kv'"
            )
        getattr(self, f"_init_{self.mode}")()

    def _make_router(self, input_dim):
        """Create a router for this bank — DeepSeek sigmoid or plain Linear."""
        if self.use_deepseek_routing:
            from types import SimpleNamespace

            cfg = SimpleNamespace(
                hidden_size=input_dim,
                num_local_experts=self.num_experts,
                num_experts=self.num_experts,
                num_experts_per_tok=self.top_k,
                norm_topk_prob=getattr(self.config, "norm_topk_prob", True),
                topk_scaling_factor=getattr(self.config, "topk_scaling_factor", None),
                num_groups=getattr(self.config, "num_groups", None),
                group_topk=getattr(self.config, "group_topk", None),
            )
            return DeepSeekRouter(cfg)
        return nn.Linear(input_dim, self.num_experts, bias=False)

    def _make_flat_bank_router(self, input_dim, top_k, num_experts=None):
        """Create a router for flat bank modes — selects top_k experts from the pool."""
        if num_experts is None:
            num_experts = self.num_experts
        if self.use_deepseek_routing:
            from types import SimpleNamespace

            num_groups = getattr(self.config, "num_groups", None)
            group_topk = getattr(self.config, "group_topk", None)
            # Disable group-limited routing if it can't provide enough candidates
            if num_groups and group_topk:
                experts_per_group = num_experts // num_groups
                max_candidates = group_topk * experts_per_group
                if max_candidates < top_k:
                    num_groups = None
                    group_topk = None
            cfg = SimpleNamespace(
                hidden_size=input_dim,
                num_local_experts=num_experts,
                num_experts=num_experts,
                num_experts_per_tok=top_k,
                norm_topk_prob=getattr(self.config, "norm_topk_prob", True),
                topk_scaling_factor=getattr(self.config, "topk_scaling_factor", None),
                num_groups=num_groups,
                group_topk=group_topk,
            )
            return DeepSeekRouter(cfg)
        return nn.Linear(input_dim, num_experts, bias=False)

    def _route_flat(self, router, x, top_k):
        """Route with explicit top_k for flat bank."""
        if isinstance(router, DeepSeekRouter):
            router_probs, weights, idx = router(x)
            return idx, weights, router_probs

        logits = router(x.float())
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-20)
        return top_idx, top_vals.to(x.dtype), probs

    def _project_flat_head(self, flat, weight_bank, expert_idx, expert_weights, norm_weights=None):
        """Project one head position from flat bank (1D expert routing)."""
        N = flat.shape[0]
        out = flat.new_zeros(N, weight_bank.shape[2])
        for e in expert_idx.unique():
            mask = expert_idx == e
            proj = flat[mask] @ weight_bank[e]
            if norm_weights is not None:
                proj_f = proj.float()
                var = proj_f.pow(2).mean(-1, keepdim=True)
                proj_normed = proj_f * torch.rsqrt(var + self.eps)
                proj = (norm_weights[e] * proj_normed).to(flat.dtype)
            out[mask] = (proj * expert_weights[mask].unsqueeze(-1)).to(out.dtype)
        return out

    def _apply_single_head_norm(self, proj, norm_weight):
        proj_f = proj.float()
        var = proj_f.pow(2).mean(-1, keepdim=True)
        proj_normed = proj_f * torch.rsqrt(var + self.eps)
        return (norm_weight * proj_normed).to(proj.dtype)

    def _project_shared_inputs_grouped(self, flat, weight_bank, idx, weights, norm_weights=None):
        """Project shared token inputs for all head slots, grouped by expert."""
        N, num_slots = idx.shape
        H_out = weight_bank.shape[2]
        pair_expert = idx.reshape(-1)
        pair_weight = weights.reshape(-1)
        token_idx = (
            torch.arange(N, device=flat.device)
            .unsqueeze(1)
            .expand(N, num_slots)
            .reshape(-1)
        )
        pair_out = flat.new_zeros(pair_expert.shape[0], H_out)
        if pair_expert.numel() == 0:
            return pair_out.view(N, num_slots, H_out)

        sort_order = torch.argsort(pair_expert)
        sorted_expert = pair_expert[sort_order]
        sorted_token_idx = token_idx[sort_order]
        sorted_weight = pair_weight[sort_order]
        unique_experts, counts = torch.unique_consecutive(sorted_expert, return_counts=True)

        start = 0
        for expert, count in zip(unique_experts.tolist(), counts.tolist()):
            end = start + count
            proj = flat.index_select(0, sorted_token_idx[start:end]) @ weight_bank[expert]
            if norm_weights is not None:
                proj = self._apply_single_head_norm(proj, norm_weights[expert])
            proj = proj * sorted_weight[start:end].unsqueeze(-1)
            pair_out[sort_order[start:end]] = proj.to(pair_out.dtype)
            start = end

        return pair_out.view(N, num_slots, H_out)

    def _project_pair_inputs_grouped(
        self,
        inputs,
        weight_bank,
        idx,
        weights,
        norm_weights=None,
        reduce_tokens: bool = False,
    ):
        """Project per-head inputs grouped by expert, optionally summing back to tokens."""
        N, num_slots, head_dim = inputs.shape
        H_out = weight_bank.shape[2]
        pair_inputs = inputs.reshape(-1, head_dim)
        pair_expert = idx.reshape(-1)
        pair_weight = weights.reshape(-1)
        if pair_expert.numel() == 0:
            if reduce_tokens:
                return inputs.new_zeros(N, H_out)
            return inputs.new_zeros(N, num_slots, H_out)

        token_idx = (
            torch.arange(N, device=inputs.device)
            .unsqueeze(1)
            .expand(N, num_slots)
            .reshape(-1)
        )
        sort_order = torch.argsort(pair_expert)
        sorted_expert = pair_expert[sort_order]
        sorted_inputs = pair_inputs[sort_order]
        sorted_weight = pair_weight[sort_order]
        sorted_token_idx = token_idx[sort_order]
        unique_experts, counts = torch.unique_consecutive(sorted_expert, return_counts=True)

        if reduce_tokens:
            token_out = inputs.new_zeros(N, H_out)
        else:
            pair_out = inputs.new_zeros(pair_inputs.shape[0], H_out)

        start = 0
        for expert, count in zip(unique_experts.tolist(), counts.tolist()):
            end = start + count
            proj = sorted_inputs[start:end] @ weight_bank[expert]
            if norm_weights is not None:
                proj = self._apply_single_head_norm(proj, norm_weights[expert])
            proj = proj * sorted_weight[start:end].unsqueeze(-1)
            proj = proj.to(inputs.dtype)
            if reduce_tokens:
                token_out.index_add_(0, sorted_token_idx[start:end], proj)
            else:
                pair_out[sort_order[start:end]] = proj
            start = end

        if reduce_tokens:
            return token_out
        return pair_out.view(N, num_slots, H_out)

    def _project_heads_batched(self, flat, weight_bank, idx, weights, norm_weights=None):
        """Project all heads using grouped expert matmuls across all head slots.

        Args:
            flat: (N, H) input tokens
            weight_bank: (E, H_in, H_out) expert weight bank
            idx: (N, num_heads) expert indices per head
            weights: (N, num_heads) routing weights per head
            norm_weights: (E, H_out) optional per-expert RMSNorm weights

        Returns:
            (N, num_heads, H_out)
        """
        return self._project_shared_inputs_grouped(flat, weight_bank, idx, weights, norm_weights)

    def _init_bundled(self):
        """(Q,K,V,O) all selected as one unit.  1 norm, 1 router."""
        self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
        else:
            self.router = self._make_router(self.hidden_size)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_kv_paired(self):
        """(K,V) paired — memory coherence.  Q and O separate.  2 norms, 3 routers."""
        self.kv_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.q_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.kv_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.q_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.o_routers = nn.ModuleList(
                [self._make_router(self.q_dim) for _ in range(self.num_depths)]
            )
        else:
            self.kv_router = self._make_router(self.hidden_size)
            self.q_router = self._make_router(self.hidden_size)
            self.o_router = self._make_router(self.q_dim)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_qk_paired(self):
        """(Q,K) paired — dot-product compatibility.  V and O separate.  2 norms, 3 routers."""
        self.qk_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.v_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.qk_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.v_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.o_routers = nn.ModuleList(
                [self._make_router(self.q_dim) for _ in range(self.num_depths)]
            )
        else:
            self.qk_router = self._make_router(self.hidden_size)
            self.v_router = self._make_router(self.hidden_size)
            self.o_router = self._make_router(self.q_dim)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_fully_independent(self):
        """Q, K, V, O each from a separate bank.  3 norms, 4 routers."""
        self.q_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.k_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.v_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.q_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.k_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.v_routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
            self.o_routers = nn.ModuleList(
                [self._make_router(self.q_dim) for _ in range(self.num_depths)]
            )
        else:
            self.q_router = self._make_router(self.hidden_size)
            self.k_router = self._make_router(self.hidden_size)
            self.v_router = self._make_router(self.hidden_size)
            self.o_router = self._make_router(self.q_dim)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_per_head_fully_independent(self):
        """Flat-bank Q/K/V/O with GQA: Q,O pools = E, K,V pools = E_kv.

        Q router picks top-num_heads from E, K/V pick top-num_kv_heads from E_kv,
        O router picks top-num_heads from E.  GQA via repeat_kv in attend().
        """
        if self.routed_norm:
            self.attn_pre_norm = NormExpertBank(self.num_depths, self.hidden_size, eps=self.eps)
        elif self.per_layer_norm:
            self.attn_pre_norms = nn.ModuleList([Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps) for _ in range(self.num_depths)])
        else:
            self.q_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
            self.k_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
            self.v_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        E_kv = E * self.num_kv_heads // self.num_heads
        self.num_kv_experts = E_kv
        if self.per_layer_attn_router:
            self.q_routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_heads) for _ in range(self.num_depths)])
            self.k_routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv) for _ in range(self.num_depths)])
            self.v_routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv) for _ in range(self.num_depths)])
            self.o_routers = nn.ModuleList([self._make_flat_bank_router(self.q_dim, self.num_heads) for _ in range(self.num_depths)])
        else:
            self.q_router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
            self.k_router = self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv)
            self.v_router = self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv)
            self.o_router = self._make_flat_bank_router(self.q_dim, self.num_heads)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.k_proj = nn.Parameter(torch.empty(E_kv, self.hidden_size, self.head_dim))
        self.v_proj = nn.Parameter(torch.empty(E_kv, self.hidden_size, self.head_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.head_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E_kv, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_precompute_kv(self):
        """Per-expert KV tables, routed Q and O.  1 norm, 1 router."""
        self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.routers = nn.ModuleList(
                [self._make_router(self.hidden_size) for _ in range(self.num_depths)]
            )
        else:
            self.router = self._make_router(self.hidden_size)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_per_head_precompute_kv(self):
        """Flat-bank precompute KV: 1 router, corresponding QKVO.  1 norm.

        Single router picks top-num_heads from E experts.
        Each selected expert provides Q, K, V, and O for that head.
        """
        if self.routed_norm:
            self.norm = NormExpertBank(self.num_depths, self.hidden_size, eps=self.eps)
        elif self.per_layer_norm:
            self.norms = nn.ModuleList([Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps) for _ in range(self.num_depths)])
        else:
            self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_heads) for _ in range(self.num_depths)])
        else:
            self.router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.head_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _select_router(self, name: str, depth_idx: int | None = None):
        plural_name = f"{name}s"
        if self.per_layer_attn_router and depth_idx is not None and hasattr(self, plural_name):
            return getattr(self, plural_name)[depth_idx]
        return getattr(self, name)

    def _maybe_build_sanity_attention_routing(
        self,
        num_tokens: int,
        num_slots: int,
        depth_idx: int | None,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        token_mask: torch.Tensor | None = None,
    ):
        if self.sanity_check_mode != "alternating_global_moe":
            return None
        logical_layer = 0 if depth_idx is None else depth_idx // 2
        base = (logical_layer * self.num_heads) % self.num_experts
        slot_idx = (base + torch.arange(num_slots, device=device)) % self.num_experts
        idx = slot_idx.unsqueeze(0).expand(num_tokens, -1).contiguous()
        weights = torch.ones(num_tokens, num_slots, device=device, dtype=dtype)
        probs = torch.zeros(num_tokens, self.num_experts, device=device, dtype=dtype)
        probs.scatter_(1, idx, 1.0 / max(1, num_slots))
        if token_mask is not None:
            self._store_router_info("attn", probs, idx, token_mask=token_mask)
        else:
            self._store_router_info("attn", probs, idx)
        return idx, weights, probs

    def should_use_sparse_path(self, token_mask: torch.Tensor) -> bool:
        flat_mask = token_mask.reshape(-1).bool()
        if not flat_mask.any():
            return True
        if flat_mask.all():
            return False
        if self.mode not in {"per_head_fully_independent", "per_head_precompute_kv"}:
            return False
        if self.per_head_compute_mode == "dense":
            return False
        if self.per_head_compute_mode == "sparse":
            return True
        attn_fraction = flat_mask.float().mean().item()
        return attn_fraction < self.per_head_dense_fraction_threshold

    def _init_params(self, params):
        std = self.config.initializer_range
        for p in params:
            nn.init.normal_(p, mean=0.0, std=std)

    def _apply_expert_head_norm(self, x, norm_weights, expert_idx, num_heads):
        """Per-expert RMSNorm on head-organized vectors."""
        N = x.shape[0]
        orig_dtype = x.dtype
        x = x.view(N, num_heads, self.head_dim).float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        w = norm_weights[expert_idx]
        x = w.unsqueeze(1) * x
        return x.to(orig_dtype).view(N, num_heads * self.head_dim)

    def _store_router_info(
        self,
        name: str,
        router_probs: torch.Tensor,
        expert_idx: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        if token_mask is not None:
            flat_mask = token_mask.reshape(-1).bool().to(router_probs.device)
            dense_probs = router_probs.new_zeros(flat_mask.numel(), router_probs.shape[-1])
            dense_probs[flat_mask] = router_probs

            if expert_idx.ndim == 1:
                dense_idx = expert_idx.new_zeros(flat_mask.numel())
            else:
                dense_idx = expert_idx.new_zeros((flat_mask.numel(), expert_idx.shape[-1]))
            dense_idx[flat_mask] = expert_idx

            self.last_router_info[name] = {
                "router_logits": dense_probs.detach(),
                "selected_experts": dense_idx.detach(),
                "token_mask": flat_mask.detach(),
            }
            return

        self.last_router_info[name] = {
            "router_logits": router_probs.detach(),
            "selected_experts": expert_idx.detach(),
        }

    def _zero_dummy(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dummy = torch.zeros((), device=device, dtype=dtype)
        for param in self.parameters():
            dummy = dummy + param.reshape(-1)[0].to(dtype) * 0.0
        return dummy

    def _route(self, router, x):
        """Compute routing and return (expert_idx, expert_weights, router_probs)."""
        if isinstance(router, DeepSeekRouter):
            router_probs, weights, idx = router(x)
            if self.top_k == 1:
                idx = idx.squeeze(-1)
            return idx, weights, router_probs

        logits = router(x.float())
        probs = F.softmax(logits, dim=-1)
        if self.top_k == 1:
            idx = probs.argmax(dim=-1)
            weights = probs.gather(1, idx.unsqueeze(-1)).to(x.dtype)
            return idx, weights, probs

        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-20)
        return top_idx, top_vals.to(x.dtype), probs

    def _apply_projection(
        self,
        x,
        weight_bank,
        expert_idx,
        expert_weights,
        head_norm_weights=None,
        num_heads_for_norm=None,
    ):
        """Project x using given expert routing."""
        N = x.shape[0]
        out = x.new_zeros(N, weight_bank.shape[2])

        if self.top_k == 1:
            for e in expert_idx.unique():
                mask = expert_idx == e
                proj = x[mask] @ weight_bank[e]
                if head_norm_weights is not None:
                    proj = self._apply_expert_head_norm(
                        proj, head_norm_weights, expert_idx[mask], num_heads_for_norm
                    )
                out[mask] = (proj * expert_weights[mask]).to(out.dtype)
        else:
            for k in range(self.top_k):
                idx_k = expert_idx[:, k]
                w_k = expert_weights[:, k : k + 1]
                for e in idx_k.unique():
                    mask = idx_k == e
                    proj = x[mask] @ weight_bank[e]
                    if head_norm_weights is not None:
                        proj = self._apply_expert_head_norm(
                            proj, head_norm_weights, idx_k[mask], num_heads_for_norm
                        )
                    out[mask] = out[mask] + (w_k[mask] * proj).to(out.dtype)
        return out

    def _route_and_project(
        self,
        router,
        x,
        weight_bank,
        head_norm_weights=None,
        num_heads_for_norm=None,
        router_name: str | None = None,
    ):
        """Route + project in one call."""
        idx, weights, router_probs = self._route(router, x)
        if router_name is not None:
            self._store_router_info(router_name, router_probs, idx)
        return self._apply_projection(
            x, weight_bank, idx, weights, head_norm_weights, num_heads_for_norm
        )

    def project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-routing-group norm, then compute Q, K_fresh, V_fresh via expert routing."""
        B, T, H = hidden_states.shape
        self.last_router_info = {}

        if self.mode == "bundled":
            flat = self.norm(hidden_states).reshape(B * T, H)
            idx, w, router_probs = self._route(self._select_router("router", depth_idx), flat)
            self._last_routing = (idx, w)
            self._store_router_info("attn", router_probs, idx)
            Q = self._apply_projection(flat, self.q_proj, idx, w, self.q_norm_weight, self.num_heads)
            K = self._apply_projection(flat, self.k_proj, idx, w, self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(flat, self.v_proj, idx, w)

        elif self.mode == "kv_paired":
            kv_flat = self.kv_norm(hidden_states).reshape(B * T, H)
            q_flat = self.q_norm(hidden_states).reshape(B * T, H)
            kv_idx, kv_w, kv_router_probs = self._route(self._select_router("kv_router", depth_idx), kv_flat)
            self._store_router_info("kv", kv_router_probs, kv_idx)
            K = self._apply_projection(kv_flat, self.k_proj, kv_idx, kv_w, self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(kv_flat, self.v_proj, kv_idx, kv_w)
            Q = self._route_and_project(
                self._select_router("q_router", depth_idx),
                q_flat,
                self.q_proj,
                self.q_norm_weight,
                self.num_heads,
                router_name="q",
            )

        elif self.mode == "qk_paired":
            qk_flat = self.qk_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_norm(hidden_states).reshape(B * T, H)
            qk_idx, qk_w, qk_router_probs = self._route(self._select_router("qk_router", depth_idx), qk_flat)
            self._store_router_info("qk", qk_router_probs, qk_idx)
            Q = self._apply_projection(qk_flat, self.q_proj, qk_idx, qk_w, self.q_norm_weight, self.num_heads)
            K = self._apply_projection(qk_flat, self.k_proj, qk_idx, qk_w, self.k_norm_weight, self.num_kv_heads)
            V = self._route_and_project(
                self._select_router("v_router", depth_idx),
                v_flat,
                self.v_proj,
                router_name="v",
            )

        elif self.mode == "fully_independent":
            q_flat = self.q_pre_norm(hidden_states).reshape(B * T, H)
            k_flat = self.k_pre_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_pre_norm(hidden_states).reshape(B * T, H)
            Q = self._route_and_project(
                self._select_router("q_router", depth_idx),
                q_flat,
                self.q_proj,
                self.q_norm_weight,
                self.num_heads,
                router_name="q",
            )
            K = self._route_and_project(
                self._select_router("k_router", depth_idx),
                k_flat,
                self.k_proj,
                self.k_norm_weight,
                self.num_kv_heads,
                router_name="k",
            )
            V = self._route_and_project(
                self._select_router("v_router", depth_idx),
                v_flat,
                self.v_proj,
                router_name="v",
            )

        elif self.mode == "per_head_fully_independent":
            if self.routed_norm:
                normed = self.attn_pre_norm(hidden_states).reshape(B * T, H)
                q_flat = k_flat = v_flat = normed
            elif self.per_layer_norm and depth_idx is not None:
                normed = self.attn_pre_norms[depth_idx](hidden_states).reshape(B * T, H)
                q_flat = k_flat = v_flat = normed
            else:
                q_flat = self.q_pre_norm(hidden_states).reshape(B * T, H)
                k_flat = self.k_pre_norm(hidden_states).reshape(B * T, H)
                v_flat = self.v_pre_norm(hidden_states).reshape(B * T, H)

            q_router = self._select_router("q_router", depth_idx)
            k_router = self._select_router("k_router", depth_idx)
            v_router = self._select_router("v_router", depth_idx)

            q_idx, q_w, q_probs = self._route_flat(q_router, q_flat, self.num_heads)
            k_idx, k_w, k_probs = self._route_flat(k_router, k_flat, self.num_kv_heads)
            v_idx, v_w, v_probs = self._route_flat(v_router, v_flat, self.num_kv_heads)
            self._store_router_info("q", q_probs, q_idx)
            self._store_router_info("k", k_probs, k_idx)
            self._store_router_info("v", v_probs, v_idx)

            # Batched projections: (N, num_heads, head_dim)
            Q_heads = self._project_heads_batched(q_flat, self.q_proj, q_idx, q_w, self.q_norm_weight)
            K_heads = self._project_heads_batched(k_flat, self.k_proj, k_idx, k_w, self.k_norm_weight)
            V_heads = self._project_heads_batched(v_flat, self.v_proj, v_idx, v_w)

            Q = Q_heads.reshape(B * T, self.num_heads * self.head_dim)
            K = K_heads.reshape(B * T, self.num_kv_heads * self.head_dim)
            V = V_heads.reshape(B * T, self.num_kv_heads * self.head_dim)

        elif self.mode == "precompute_kv":
            raise RuntimeError("precompute_kv should use project_and_attend_precompute_kv()")
        elif self.mode == "per_head_precompute_kv":
            raise RuntimeError("per_head_precompute_kv should use project_and_attend_per_head_precompute_kv()")

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        return Q, K, V

    def attend(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        """Run attention and O projection."""
        B = Q.shape[0]
        T = Q.shape[2]

        K_expanded = repeat_kv(K, self.num_kv_groups)
        V_expanded = repeat_kv(V, self.num_kv_groups)
        if Q.is_cuda:
            attn_output = F.scaled_dot_product_attention(
                Q,
                K_expanded,
                V_expanded,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
            )
        else:
            attn_weights = torch.matmul(Q, K_expanded.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_output = torch.matmul(attn_weights, V_expanded)

        if self.mode == "per_head_fully_independent":
            N = B * T
            attn_heads = attn_output.transpose(1, 2).reshape(N, self.num_heads, self.head_dim)
            attn_flat = attn_heads.reshape(N, self.q_dim)
            o_router = self._select_router("o_router", depth_idx)
            o_idx, o_w, o_probs = self._route_flat(o_router, attn_flat, self.num_heads)
            self._store_router_info("o", o_probs, o_idx)
            o_out = self._project_pair_inputs_grouped(
                attn_heads,
                self.o_proj,
                o_idx,
                o_w,
                reduce_tokens=True,
            )
            return o_out.view(B, T, self.hidden_size)

        attn_output = attn_output.transpose(1, 2).reshape(B * T, self.q_dim)

        if self.mode in ("bundled", "precompute_kv"):
            idx, w = self._last_routing
            attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        elif self.mode in ("kv_paired", "qk_paired", "fully_independent"):
            attn_output = self._route_and_project(
                self._select_router("o_router", depth_idx),
                attn_output,
                self.o_proj,
                router_name="o",
            )

        return attn_output.view(B, T, self.hidden_size)

    def _select_attn_routers(self, depth_idx: int | None = None):
        if self.mode != "per_head_fully_independent":
            return None
        return (
            self._select_router("q_router", depth_idx),
            self._select_router("k_router", depth_idx),
            self._select_router("v_router", depth_idx),
            self._select_router("o_router", depth_idx),
        )

    def _empty_sparse_attn_result(
        self,
        hidden_states: torch.Tensor,
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        router_specs: list[tuple[str, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = hidden_states.shape
        flat_mask = torch.zeros(B * T, dtype=torch.bool, device=hidden_states.device)
        for name, num_experts, top_k in router_specs:
            empty_probs = hidden_states.new_zeros((0, num_experts))
            empty_idx = torch.zeros((0, top_k), device=hidden_states.device, dtype=torch.long)
            self._store_router_info(name, empty_probs, empty_idx, token_mask=flat_mask)
        dummy = self._zero_dummy(hidden_states.device, hidden_states.dtype)
        attn_out = hidden_states.new_zeros(B, T, self.hidden_size) + dummy
        return attn_out, K_old, V_old

    def project_and_attend_per_head_fully_independent_sparse(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H = hidden_states.shape
        self.last_router_info = {}
        flat_mask = token_mask.reshape(-1).bool()

        if not flat_mask.any():
            return self._empty_sparse_attn_result(
                hidden_states,
                K_old,
                V_old,
                [
                    ("q", self.num_experts, self.num_heads),
                    ("k", self.num_kv_experts, self.num_kv_heads),
                    ("v", self.num_kv_experts, self.num_kv_heads),
                    ("o", self.num_experts, self.num_heads),
                ],
            )

        if flat_mask.all():
            Q, K_fresh, V_fresh = self.project(hidden_states, position_embeddings, depth_idx=depth_idx)
            attn_out = self.attend(Q, K_fresh, V_fresh, attention_mask, depth_idx=depth_idx)
            return attn_out, K_fresh, V_fresh

        flat_hidden = hidden_states.reshape(B * T, H)
        hidden_selected = flat_hidden[flat_mask]

        if self.routed_norm:
            normed = self.attn_pre_norm(hidden_selected)
            q_flat = k_flat = v_flat = normed
        elif self.per_layer_norm and depth_idx is not None:
            normed = self.attn_pre_norms[depth_idx](hidden_selected)
            q_flat = k_flat = v_flat = normed
        else:
            q_flat = self.q_pre_norm(hidden_selected)
            k_flat = self.k_pre_norm(hidden_selected)
            v_flat = self.v_pre_norm(hidden_selected)

        q_router, k_router, v_router, o_router = self._select_attn_routers(depth_idx)
        q_idx, q_w, q_probs = self._route_flat(q_router, q_flat, self.num_heads)
        k_idx, k_w, k_probs = self._route_flat(k_router, k_flat, self.num_kv_heads)
        v_idx, v_w, v_probs = self._route_flat(v_router, v_flat, self.num_kv_heads)
        self._store_router_info("q", q_probs, q_idx, token_mask=flat_mask)
        self._store_router_info("k", k_probs, k_idx, token_mask=flat_mask)
        self._store_router_info("v", v_probs, v_idx, token_mask=flat_mask)

        Q_sel = self._project_heads_batched(q_flat, self.q_proj, q_idx, q_w, self.q_norm_weight)
        K_sel = self._project_heads_batched(k_flat, self.k_proj, k_idx, k_w, self.k_norm_weight)
        V_sel = self._project_heads_batched(v_flat, self.v_proj, v_idx, v_w)

        Q_flat = hidden_states.new_zeros(B * T, self.num_heads, self.head_dim)
        K_flat = hidden_states.new_zeros(B * T, self.num_kv_heads, self.head_dim)
        V_flat = hidden_states.new_zeros(B * T, self.num_kv_heads, self.head_dim)
        Q_flat[flat_mask] = Q_sel
        K_flat[flat_mask] = K_sel
        V_flat[flat_mask] = V_sel

        Q = Q_flat.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K_fresh = K_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        Q, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        attn_mask_kv = token_mask.unsqueeze(1)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)

        K_expanded = repeat_kv(K_new, self.num_kv_groups)
        V_expanded = repeat_kv(V_new, self.num_kv_groups)
        attn_heads = hidden_states.new_zeros(B, self.num_heads, T, self.head_dim)
        token_mask_2d = token_mask.squeeze(-1).bool()

        for b in range(B):
            pos = token_mask_2d[b].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            Q_b = Q[b : b + 1, :, pos, :]
            if Q_b.is_cuda:
                attn_b = F.scaled_dot_product_attention(
                    Q_b,
                    K_expanded[b : b + 1],
                    V_expanded[b : b + 1],
                    attn_mask=attention_mask[:, :, pos, :] if attention_mask is not None else None,
                    dropout_p=0.0,
                    scale=self.scaling,
                )
            else:
                scores = torch.matmul(Q_b, K_expanded[b : b + 1].transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    scores = scores + attention_mask[:, :, pos, :]
                scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                attn_b = torch.matmul(scores, V_expanded[b : b + 1])
            attn_heads[b, :, pos, :] = attn_b.squeeze(0).to(attn_heads.dtype)

        attn_selected = attn_heads.transpose(1, 2).reshape(B * T, self.num_heads, self.head_dim)[flat_mask]
        attn_flat = attn_selected.reshape(attn_selected.shape[0], self.q_dim)
        o_idx, o_w, o_probs = self._route_flat(o_router, attn_flat, self.num_heads)
        self._store_router_info("o", o_probs, o_idx, token_mask=flat_mask)

        o_selected = self._project_pair_inputs_grouped(
            attn_selected,
            self.o_proj,
            o_idx,
            o_w,
            reduce_tokens=True,
        )

        attn_out = hidden_states.new_zeros(B * T, self.hidden_size)
        attn_out[flat_mask] = o_selected
        return attn_out.view(B, T, self.hidden_size), K_new, V_new

    def _build_per_head_precompute_kv_tables(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        depth_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, H = hidden_states.shape
        N = B * T
        if self.per_layer_norm and depth_idx is not None:
            normed = self.norms[depth_idx](hidden_states)
        else:
            normed = self.norm(hidden_states)
        flat = normed.reshape(N, H)

        routed = self._maybe_build_sanity_attention_routing(
            N, self.num_heads, depth_idx, hidden_states.device, dtype=flat.dtype
        )
        if routed is None:
            router = self._select_router("router", depth_idx)
            idx, w, probs = self._route_flat(router, flat, self.num_heads)
            self._store_router_info("attn", probs, idx)
        else:
            idx, w, probs = routed

        Q_heads = self._project_heads_batched(flat, self.q_proj, idx, w, self.q_norm_weight)
        kv_ranks = torch.arange(self.num_kv_heads, device=flat.device) * self.num_kv_groups
        kv_idx = idx.index_select(1, kv_ranks)
        kv_weight = w.index_select(1, kv_ranks)
        ones = torch.ones_like(kv_weight)
        K_heads = self._project_heads_batched(flat, self.k_proj, kv_idx, ones, self.k_norm_weight)
        V_heads = self._project_heads_batched(flat, self.v_proj, kv_idx, ones)

        Q = Q_heads.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K_fresh = K_heads.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_heads.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        Q, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        return {
            "flat": flat,
            "idx": idx,
            "weights": w,
            "kv_weight": kv_weight,
            "Q": Q,
            "K_fresh": K_fresh,
            "V_fresh": V_fresh,
        }

    def project_and_attend_per_head_precompute_kv_dense_mixed(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = hidden_states.shape
        self.last_router_info = {}
        tables = self._build_per_head_precompute_kv_tables(
            hidden_states, position_embeddings, depth_idx=depth_idx
        )
        idx = tables["idx"]
        w = tables["weights"]
        kv_weight = tables["kv_weight"]
        Q = tables["Q"]
        K_fresh = tables["K_fresh"]
        V_fresh = tables["V_fresh"]

        attn_mask_kv = token_mask.unsqueeze(1)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)

        K_expanded = repeat_kv(K_new, self.num_kv_groups)
        V_expanded = repeat_kv(V_new, self.num_kv_groups)
        if Q.is_cuda:
            attn_output = F.scaled_dot_product_attention(
                Q,
                K_expanded,
                V_expanded,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
            )
        else:
            scores = torch.matmul(Q, K_expanded.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_output = torch.matmul(scores, V_expanded)

        kv_weight_expanded = (
            kv_weight.view(B, T, self.num_kv_heads)
            .transpose(1, 2)
            .repeat_interleave(self.num_kv_groups, dim=1)
            .unsqueeze(-1)
        )
        attn_output = attn_output * kv_weight_expanded.to(attn_output.dtype)
        attn_heads = attn_output.transpose(1, 2).reshape(B * T, self.num_heads, self.head_dim)
        o_out = self._project_pair_inputs_grouped(
            attn_heads,
            self.o_proj,
            idx,
            w,
            reduce_tokens=True,
        )
        return o_out.view(B, T, self.hidden_size), K_new, V_new

    def project_and_attend_per_head_precompute_kv_sparse(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H = hidden_states.shape
        self.last_router_info = {}
        flat_mask = token_mask.reshape(-1).bool()

        if not flat_mask.any():
            return self._empty_sparse_attn_result(
                hidden_states,
                K_old,
                V_old,
                [("attn", self.num_experts, self.num_heads)],
            )

        if flat_mask.all():
            return self.project_and_attend_per_head_precompute_kv(
                hidden_states,
                position_embeddings,
                attention_mask,
                depth_idx=depth_idx,
            )

        flat_hidden = hidden_states.reshape(B * T, H)
        hidden_selected = flat_hidden[flat_mask]

        if self.per_layer_norm and depth_idx is not None:
            normed = self.norms[depth_idx](hidden_selected)
        else:
            normed = self.norm(hidden_selected)

        routed = self._maybe_build_sanity_attention_routing(
            normed.shape[0],
            self.num_heads,
            depth_idx,
            hidden_states.device,
            dtype=normed.dtype,
            token_mask=flat_mask,
        )
        if routed is None:
            router = self._select_router("router", depth_idx)
            idx, w, probs = self._route_flat(router, normed, self.num_heads)
            self._store_router_info("attn", probs, idx, token_mask=flat_mask)
        else:
            idx, w, probs = routed

        Q_sel = self._project_heads_batched(normed, self.q_proj, idx, w, self.q_norm_weight)
        Q_flat = hidden_states.new_zeros(B * T, self.num_heads, self.head_dim)
        Q_flat[flat_mask] = Q_sel
        Q = Q_flat.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        kv_ranks = torch.arange(self.num_kv_heads, device=normed.device) * self.num_kv_groups
        kv_idx = idx.index_select(1, kv_ranks)
        kv_weight_parts = [kv_weight for kv_weight in w.index_select(1, kv_ranks).unbind(dim=1)]
        ones = torch.ones_like(kv_idx, dtype=normed.dtype)
        K_sel = self._project_heads_batched(normed, self.k_proj, kv_idx, ones, self.k_norm_weight)
        V_sel = self._project_heads_batched(normed, self.v_proj, kv_idx, ones)

        K_flat = hidden_states.new_zeros(B * T, self.num_kv_heads, self.head_dim)
        V_flat = hidden_states.new_zeros(B * T, self.num_kv_heads, self.head_dim)
        K_flat[flat_mask] = K_sel
        V_flat[flat_mask] = V_sel

        K_fresh = K_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        Q, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        attn_mask_kv = token_mask.unsqueeze(1)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)

        attn_heads = hidden_states.new_zeros(B, self.num_heads, T, self.head_dim)
        token_mask_2d = token_mask.squeeze(-1).bool()
        selected_counts = token_mask_2d.sum(dim=1)
        selected_ends = selected_counts.cumsum(dim=0)

        for b in range(B):
            pos = token_mask_2d[b].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            start = 0 if b == 0 else int(selected_ends[b - 1].item())
            end = int(selected_ends[b].item())
            for g in range(self.num_kv_heads):
                q_start = g * self.num_kv_groups
                q_end = q_start + self.num_kv_groups
                Q_b = Q[b : b + 1, q_start:q_end, pos, :]
                K_b = K_new[b : b + 1, g : g + 1].expand(-1, self.num_kv_groups, -1, -1)
                V_b = V_new[b : b + 1, g : g + 1].expand(-1, self.num_kv_groups, -1, -1)
                if Q_b.is_cuda:
                    attn_b = F.scaled_dot_product_attention(
                        Q_b,
                        K_b,
                        V_b,
                        attn_mask=attention_mask[:, :, pos, :] if attention_mask is not None else None,
                        dropout_p=0.0,
                        scale=self.scaling,
                    )
                else:
                    scores = torch.matmul(Q_b, K_b.transpose(2, 3)) * self.scaling
                    if attention_mask is not None:
                        scores = scores + attention_mask[:, :, pos, :]
                    scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                    attn_b = torch.matmul(scores, V_b)
                kv_weight = kv_weight_parts[g][start:end]
                attn_b = attn_b * kv_weight.view(1, 1, -1, 1)
                attn_heads[b, q_start:q_end, pos, :] = attn_b.squeeze(0).to(attn_heads.dtype)

        o_selected = self._project_pair_inputs_grouped(
            attn_heads.transpose(1, 2).reshape(B * T, self.num_heads, self.head_dim)[flat_mask],
            self.o_proj,
            idx,
            w,
            reduce_tokens=True,
        )

        attn_out = hidden_states.new_zeros(B * T, self.hidden_size)
        attn_out[flat_mask] = o_selected
        return attn_out.view(B, T, self.hidden_size), K_new, V_new

    def project_and_attend_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Section 4.1.2: apply norm, route first, compute per-expert KV tables, attend."""
        B, T, H = hidden_states.shape
        self.last_router_info = {}
        normed = self.norm(hidden_states)
        N = B * T
        flat = normed.reshape(N, H)

        idx, w, router_probs = self._route(self._select_router("router", depth_idx), flat)
        self._last_routing = (idx, w)
        self._store_router_info("attn", router_probs, idx)

        if self.top_k == 1:
            token_expert = idx
        else:
            token_expert = idx[:, 0]

        active_experts = token_expert.unique()
        Q = self._apply_projection(flat, self.q_proj, idx, w, self.q_norm_weight, self.num_heads)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        K_per_token = flat.new_zeros(N, self.kv_dim)
        V_per_token = flat.new_zeros(N, self.kv_dim)

        for e in active_experts:
            mask_e = token_expert == e
            mask_2d = mask_e.view(B, T)

            K_e = flat @ self.k_proj[e]
            V_e = flat @ self.v_proj[e]

            K_e_normed = K_e.view(N, self.num_kv_heads, self.head_dim).float()
            var_k = K_e_normed.pow(2).mean(-1, keepdim=True)
            K_e_normed = K_e_normed * torch.rsqrt(var_k + self.eps)
            K_e_normed = (self.k_norm_weight[e].unsqueeze(0) * K_e_normed).to(flat.dtype)

            K_e_heads = K_e_normed.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            V_e_heads = V_e.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            Q_rope, K_e_rope = apply_rotary_pos_emb(Q, K_e_heads, cos, sin)
            K_e_exp = repeat_kv(K_e_rope, self.num_kv_groups)
            V_e_exp = repeat_kv(V_e_heads, self.num_kv_groups)

            if Q_rope.is_cuda:
                attn_e = F.scaled_dot_product_attention(
                    Q_rope,
                    K_e_exp,
                    V_e_exp,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    scale=self.scaling,
                )
            else:
                scores = torch.matmul(Q_rope, K_e_exp.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    scores = scores + attention_mask
                scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                attn_e = torch.matmul(scores, V_e_exp)

            mask_head = mask_2d.unsqueeze(1).unsqueeze(-1)
            attn_output = attn_output + attn_e * mask_head

            K_per_token[mask_e] = K_e[mask_e].to(K_per_token.dtype)
            V_per_token[mask_e] = V_e[mask_e].to(V_per_token.dtype)

        attn_output = attn_output.transpose(1, 2).reshape(N, self.q_dim)
        attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        attn_output = attn_output.view(B, T, H)

        K_normed = K_per_token.view(N, self.num_kv_heads, self.head_dim).float()
        var_k = K_normed.pow(2).mean(-1, keepdim=True)
        K_normed = K_normed * torch.rsqrt(var_k + self.eps)
        kw = self.k_norm_weight[token_expert]
        K_normed = (kw.unsqueeze(1) * K_normed).to(flat.dtype)

        K_fresh = K_normed.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_per_token.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        _, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        return attn_output, K_fresh, V_fresh

    def project_and_attend_per_head_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat-bank precompute KV: 1 router, corresponding QKVO, with GQA."""
        self.last_router_info = {}
        tables = self._build_per_head_precompute_kv_tables(
            hidden_states, position_embeddings, depth_idx=depth_idx
        )
        idx = tables["idx"]
        w = tables["weights"]
        kv_weight = tables["kv_weight"]
        Q = tables["Q"]
        K_fresh = tables["K_fresh"]
        V_fresh = tables["V_fresh"]
        B, _, T, _ = Q.shape

        K_expanded = repeat_kv(K_fresh, self.num_kv_groups)
        V_expanded = repeat_kv(V_fresh, self.num_kv_groups)
        if Q.is_cuda:
            attn_output = F.scaled_dot_product_attention(
                Q,
                K_expanded,
                V_expanded,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
            )
        else:
            scores = torch.matmul(Q, K_expanded.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_output = torch.matmul(scores, V_expanded)

        kv_weight_expanded = (
            kv_weight.view(B, T, self.num_kv_heads)
            .transpose(1, 2)
            .repeat_interleave(self.num_kv_groups, dim=1)
            .unsqueeze(-1)
        )
        attn_output = attn_output * kv_weight_expanded.to(attn_output.dtype)
        o_out = self._project_pair_inputs_grouped(
            attn_output.transpose(1, 2).reshape(B * T, self.num_heads, self.head_dim),
            self.o_proj,
            idx,
            w,
            reduce_tokens=True,
        )

        return o_out.view(B, T, self.hidden_size), K_fresh, V_fresh


# ─── MLP expert bank ──────────────────────────────────────────────────────── #

class MlpExpertBank(nn.Module):
    """MLP expert bank with pre-norm. Shared across all depths."""

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.per_layer_norm = getattr(config, "per_layer_norm", False)
        self.sanity_check_mode = getattr(config, "sanity_check_mode", None)
        self.logical_layer_gates = self.sanity_check_mode == "alternating_global_moe"
        self.per_layer_gate = (
            getattr(config, "per_layer_mlp_router", False)
            or self.sanity_check_mode == "alternating_global_moe"
        )
        if getattr(config, "routed_norm", False):
            self.norm = NormExpertBank(config.num_hidden_layers, config.hidden_size, eps=config.rms_norm_eps)
        elif self.per_layer_norm:
            self.norms = nn.ModuleList([Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(config.num_hidden_layers)])
        else:
            self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.per_layer_gate:
            num_gate_layers = config.num_hidden_layers // 2 if self.logical_layer_gates else config.num_hidden_layers
            self.gates = nn.ModuleList([self._make_gate(config) for _ in range(num_gate_layers)])
        else:
            self.gate = self._make_gate(config)
        self.experts = Qwen3MoeExperts(config)
        self.last_router_logits = None
        self.last_selected_experts = None
        self.last_token_mask = None

    def _make_gate(self, config: MoEverythingConfig):
        if getattr(config, "use_deepseek_routing", False):
            return DeepSeekRouter(config)
        return Qwen3MoeTopKRouter(config)

    def _select_gate(self, depth_idx: int | None):
        if not self.per_layer_gate:
            return self.gate
        if depth_idx is None:
            raise ValueError("per_layer_mlp_router requires depth_idx during MLP routing")
        gate_depth_idx = depth_idx // 2 if self.logical_layer_gates else depth_idx
        return self.gates[gate_depth_idx]

    def _zero_dummy(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dummy = torch.zeros((), device=device, dtype=dtype)
        for param in self.parameters():
            dummy = dummy + param.reshape(-1)[0].to(dtype) * 0.0
        return dummy

    def forward(
        self,
        hidden_states: torch.Tensor,
        depth_idx: int | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H = hidden_states.shape
        total_tokens = B * T
        gate = self._select_gate(depth_idx)
        if token_mask is None:
            self.last_token_mask = None
        else:
            flat_mask = token_mask.reshape(-1).bool()
            self.last_token_mask = flat_mask.detach()
            if not flat_mask.any():
                self.last_router_logits = hidden_states.new_zeros(total_tokens, gate.num_experts)
                self.last_selected_experts = torch.zeros(
                    total_tokens,
                    gate.top_k,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                dummy = self._zero_dummy(hidden_states.device, hidden_states.dtype)
                return hidden_states.new_zeros(B, T, H) + dummy
            if flat_mask.all():
                token_mask = None

        if self.per_layer_norm and depth_idx is not None:
            normed = self.norms[depth_idx](hidden_states)
        else:
            normed = self.norm(hidden_states)
        flat = normed.view(-1, H)

        if token_mask is None:
            router_logits, routing_weights, selected_experts = gate(flat)
            out = self.experts(flat, selected_experts, routing_weights)
            self.last_router_logits = router_logits
            self.last_selected_experts = selected_experts
            return out.view(B, T, H)

        flat_mask = token_mask.reshape(-1).bool()
        flat_selected = flat[flat_mask]
        router_logits, routing_weights, selected_experts = gate(flat_selected)
        out_selected = self.experts(flat_selected, selected_experts, routing_weights)

        out = flat.new_zeros(total_tokens, H)
        out[flat_mask] = out_selected

        dense_logits = router_logits.new_zeros(total_tokens, router_logits.shape[-1])
        dense_logits[flat_mask] = router_logits
        dense_selected = torch.zeros(
            total_tokens,
            selected_experts.shape[-1],
            device=selected_experts.device,
            dtype=selected_experts.dtype,
        )
        dense_selected[flat_mask] = selected_experts

        self.last_router_logits = dense_logits
        self.last_selected_experts = dense_selected
        return out.view(B, T, H)


# ─── Full model ────────────────────────────────────────────────────────────── #

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
        self.init_k_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
        self.init_v_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
        self.init_k_norm = Qwen3MoeRMSNorm(head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

        # Feature 1: per-layer vs shared branch router
        self.per_layer_router = getattr(config, "per_layer_router", False)
        if self.per_layer_router:
            self.branch_routers = nn.ModuleList([
                BranchRouter(config.hidden_size) for _ in range(self.num_depths)
            ])
        else:
            self.branch_router = BranchRouter(config.hidden_size)

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
        else:
            self.branch_router.last_probs = probs
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
            flat_attn_mask = attn_mask_bool.reshape(-1).detach()
            for info in self.attn_bank.last_router_info.values():
                info["token_mask"] = flat_attn_mask
        elif self.attn_bank.mode == "per_head_fully_independent" and use_sparse_attn:
            attn_out, K_new, V_new = self.attn_bank.project_and_attend_per_head_fully_independent_sparse(
                hidden_states, position_embeddings, K_old, V_old, attn_mask_bool, causal_mask, depth_idx=depth_idx
            )
        elif self.attn_bank.mode == "precompute_kv":
            attn_out, K_fresh, V_fresh = self.attn_bank.project_and_attend_precompute_kv(
                hidden_states, position_embeddings, causal_mask, depth_idx=depth_idx
            )
            attn_mask_kv = attn_mask_bool.unsqueeze(1)
            K_new = torch.where(attn_mask_kv, K_fresh, K_old)
            V_new = torch.where(attn_mask_kv, V_fresh, V_old)
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
                flat_attn_mask = attn_mask_bool.reshape(-1).detach()
                for info in self.attn_bank.last_router_info.values():
                    info["token_mask"] = flat_attn_mask
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        cos, sin = position_embeddings

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

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
            else:
                self._all_branch_probs.append(self.branch_router.last_probs)
            self._all_mlp_router_logits.append(self.mlp_bank.last_router_logits)
            self._all_mlp_selected_experts.append(self.mlp_bank.last_selected_experts)
            self._all_mlp_token_masks.append(self.mlp_bank.last_token_mask)
            self._all_attn_router_info.append(self.attn_bank.last_router_info)

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

        mlp_router_logits = tuple(self.model._all_mlp_router_logits) or None
        mlp_selected_experts = tuple(self.model._all_mlp_selected_experts) or None
        mlp_token_masks = tuple(self.model._all_mlp_token_masks) or None
        branch_prob_tensors = tuple(self.model._all_branch_probs) or None
        attention_router_info = tuple(self.model._all_attn_router_info) or None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss

            if mlp_router_logits is not None:
                mlp_aux = load_balancing_loss_func(
                    mlp_router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    token_masks=mlp_token_masks,
                )
                if isinstance(mlp_aux, torch.Tensor):
                    aux_loss = mlp_aux
                    loss = loss + self.router_aux_loss_coef * mlp_aux

            seq_aux_coef = getattr(self, "_seq_aux_loss_coef", 0.0)
            if seq_aux_coef > 0 and mlp_router_logits is not None:
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

            # Attention expert seq aux loss (same coef as MLP).
            # Skip this in sanity mode because attention routing is deterministic
            # and would only add a constant term to the loss.
            if (
                seq_aux_coef > 0
                and attention_router_info is not None
                and getattr(self.config, "sanity_check_mode", None) != "alternating_global_moe"
            ):
                # Gather per-router logits and selected experts across depths
                attn_router_names = sorted({n for d in attention_router_info for n in d})
                num_attn_experts = self.config.num_attn_experts
                num_kv_experts = num_attn_experts * self.config.num_key_value_heads // self.config.num_attention_heads
                for rname in attn_router_names:
                    r_logits = tuple(d[rname]["router_logits"] for d in attention_router_info if rname in d)
                    r_selected = tuple(d[rname]["selected_experts"] for d in attention_router_info if rname in d)
                    r_masks = tuple(d[rname].get("token_mask") for d in attention_router_info if rname in d)
                    if not r_logits:
                        continue
                    n_experts = num_kv_experts if rname in ("k", "v") else num_attn_experts
                    n_per_tok = self.config.num_key_value_heads if rname in ("k", "v") else self.config.num_attention_heads
                    attn_seq_aux = seq_load_balancing_loss_func(
                        r_logits, n_experts, n_per_tok,
                        batch_size=input_ids.shape[0],
                        selected_experts=r_selected,
                        token_masks=r_masks,
                    )
                    if isinstance(attn_seq_aux, torch.Tensor):
                        loss = loss + seq_aux_coef * attn_seq_aux

            # Branch probs are tracked for logging only. We do not force
            # balance between attention and MLP; the model is free to learn
            # any branch ratio per depth.

        if output_router_logits:
            router_logits_out = tuple(t.detach() for t in mlp_router_logits) if mlp_router_logits is not None else None
            selected_experts_out = tuple(t.detach() for t in mlp_selected_experts) if mlp_selected_experts is not None else None
            router_token_masks_out = tuple(t.detach() if t is not None else None for t in mlp_token_masks) if mlp_token_masks is not None else None
            branch_probs_out = tuple(t.detach() for t in branch_prob_tensors) if branch_prob_tensors is not None else None
            attention_router_info_out = None
            if attention_router_info is not None:
                attention_router_info_out = tuple(
                    {
                        name: {
                            "router_logits": info["router_logits"],
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
        self.selected_experts = selected_experts
        self.router_token_masks = router_token_masks
        self.branch_probs = branch_probs
        self.attention_router_info = attention_router_info
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
