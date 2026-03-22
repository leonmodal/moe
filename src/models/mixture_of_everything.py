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
  - "precompute_kv"      : per-expert KV tables, routed Q and O — 1 router

MLP bank: standard top-k MoE over SwiGLU experts (reuses Qwen3MoeExperts).

Design notes:
  - Token state is (E, K, V).  K,V persist across depths and are only
    refreshed when a token takes the attention branch.
  - Training uses soft branch routing (both branches computed, weighted
    by probability) so gradients flow through the router.  KV state is
    soft-interpolated: K_out = p_attn * K_fresh + p_mlp * K_old.
  - Pre-norm is bundled with each bank: the attention bank and MLP bank
    each include their own RMSNorm, applied before computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .load_balancing import load_balancing_loss_func
from .router import DeepSeekRouter


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
        branch_router_aux_loss_coef: float = 0.01,
        # Routing style
        use_deepseek_routing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attn_experts = num_attn_experts
        self.num_attn_experts_per_tok = num_attn_experts_per_tok
        self.attn_expert_mode = attn_expert_mode
        self.branch_router_aux_loss_coef = branch_router_aux_loss_coef
        self.use_deepseek_routing = use_deepseek_routing


# ─── Branch router ─────────────────────────────────────────────────────────── #

class BranchRouter(nn.Module):
    """Binary router: ATTN (0) or MLP (1) per token.

    Returns soft probabilities in [0, 1] for each branch.
    During training both branches are computed and weighted.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate(hidden_states.float())
        probs = F.softmax(logits, dim=-1).to(hidden_states.dtype)
        return probs  # [..., 0] = attn, [..., 1] = mlp


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
        self.use_deepseek_routing = getattr(config, 'use_deepseek_routing', False)

        # Build projection banks based on mode (each _init_* creates its own norms)
        _MODES = {"bundled", "kv_paired", "qk_paired", "fully_independent", "precompute_kv"}
        if self.mode not in _MODES:
            raise ValueError(f"Unknown attn_expert_mode: {self.mode}, must be one of {_MODES}")
        getattr(self, f"_init_{self.mode}")()

    # ── Router factory ──

    def _make_router(self, input_dim):
        """Create a router for this bank — DeepSeek sigmoid or plain Linear."""
        if self.use_deepseek_routing:
            from types import SimpleNamespace
            cfg = SimpleNamespace(
                hidden_size=input_dim,
                num_local_experts=self.num_experts,
                num_experts=self.num_experts,  # alias used by HF attribute_map
                num_experts_per_tok=self.top_k,
                norm_topk_prob=True,
                topk_scaling_factor=None,
                num_groups=None,
                group_topk=None,
            )
            return DeepSeekRouter(cfg)
        else:
            return nn.Linear(input_dim, self.num_experts, bias=False)

    # ── Initialization helpers ──

    def _init_bundled(self):
        """(Q,K,V,O) all selected as one unit.  1 norm, 1 router."""
        self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
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
        self.kv_router = self._make_router(self.hidden_size)
        self.q_router = self._make_router(self.hidden_size)
        self.o_router = self._make_router(self.q_dim)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        # q_norm follows Q router, k_norm follows KV router
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_qk_paired(self):
        """(Q,K) paired — dot-product compatibility.  V and O separate.  2 norms, 3 routers."""
        self.qk_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.v_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
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

    def _init_precompute_kv(self):
        """Per-expert KV tables, routed Q and O.  1 norm, 1 router."""
        self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        self.router = self._make_router(self.hidden_size)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_params(self, params):
        std = self.config.initializer_range
        for p in params:
            nn.init.normal_(p, mean=0.0, std=std)

    # ── Expert dispatch helpers ──

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

    def _route(self, router, x):
        """Compute routing: returns (expert_idx, expert_weights).

        Handles both plain Linear (softmax) and DeepSeekRouter (sigmoid + bias).
        """
        if isinstance(router, DeepSeekRouter):
            # DeepSeekRouter.forward returns (scores, weights, indices)
            # where weights are already normalized and indices are [T, K]
            _, weights, idx = router(x)
            if self.top_k == 1:
                return idx.squeeze(-1), weights
            return idx, weights
        else:
            logits = router(x.float())
            probs = F.softmax(logits, dim=-1)
            if self.top_k == 1:
                idx = probs.argmax(dim=-1)
                weights = probs.gather(1, idx.unsqueeze(-1))
                return idx, weights
            else:
                top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
                top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-20)
                return top_idx, top_vals

    def _apply_projection(self, x, weight_bank, expert_idx, expert_weights,
                          head_norm_weights=None, num_heads_for_norm=None):
        """Project x using given expert routing.

        Memory-efficient: loops over active experts with regular matmul
        instead of gathering a [N, H_in, H_out] tensor.
        """
        N = x.shape[0]
        out = x.new_zeros(N, weight_bank.shape[2])

        if self.top_k == 1:
            for e in expert_idx.unique():
                mask = (expert_idx == e)
                proj = x[mask] @ weight_bank[e]
                if head_norm_weights is not None:
                    proj = self._apply_expert_head_norm(
                        proj, head_norm_weights,
                        expert_idx[mask], num_heads_for_norm)
                out[mask] = proj * expert_weights[mask]
        else:
            for k in range(self.top_k):
                idx_k = expert_idx[:, k]
                w_k = expert_weights[:, k:k+1]
                for e in idx_k.unique():
                    mask = (idx_k == e)
                    proj = x[mask] @ weight_bank[e]
                    if head_norm_weights is not None:
                        proj = self._apply_expert_head_norm(
                            proj, head_norm_weights,
                            idx_k[mask], num_heads_for_norm)
                    out[mask] = out[mask] + w_k[mask] * proj
        return out

    def _route_and_project(self, router, x, weight_bank,
                           head_norm_weights=None, num_heads_for_norm=None):
        """Route + project in one call."""
        idx, weights = self._route(router, x)
        return self._apply_projection(x, weight_bank, idx, weights,
                                      head_norm_weights, num_heads_for_norm)

    # ── Forward: project + attend ──

    def project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-routing-group norm, then compute Q, K_fresh, V_fresh via expert routing.

        Each routing group gets its own pre-norm (mirrors standard transformer
        where the norm is part of the sublayer, not floating above it).

        Returns:
            Q: [B, n_heads, T, head_dim] with RoPE applied
            K_fresh: [B, n_kv_heads, T, head_dim] with RoPE applied
            V_fresh: [B, n_kv_heads, T, head_dim]
        """
        B, T, H = hidden_states.shape

        if self.mode == "bundled":
            flat = self.norm(hidden_states).reshape(B * T, H)
            idx, w = self._route(self.router, flat)
            self._last_routing = (idx, w)
            Q = self._apply_projection(flat, self.q_proj, idx, w,
                                       self.q_norm_weight, self.num_heads)
            K = self._apply_projection(flat, self.k_proj, idx, w,
                                       self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(flat, self.v_proj, idx, w)

        elif self.mode == "kv_paired":
            kv_flat = self.kv_norm(hidden_states).reshape(B * T, H)
            q_flat = self.q_norm(hidden_states).reshape(B * T, H)
            kv_idx, kv_w = self._route(self.kv_router, kv_flat)
            K = self._apply_projection(kv_flat, self.k_proj, kv_idx, kv_w,
                                       self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(kv_flat, self.v_proj, kv_idx, kv_w)
            Q = self._route_and_project(self.q_router, q_flat, self.q_proj,
                                        self.q_norm_weight, self.num_heads)

        elif self.mode == "qk_paired":
            qk_flat = self.qk_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_norm(hidden_states).reshape(B * T, H)
            qk_idx, qk_w = self._route(self.qk_router, qk_flat)
            Q = self._apply_projection(qk_flat, self.q_proj, qk_idx, qk_w,
                                       self.q_norm_weight, self.num_heads)
            K = self._apply_projection(qk_flat, self.k_proj, qk_idx, qk_w,
                                       self.k_norm_weight, self.num_kv_heads)
            V = self._route_and_project(self.v_router, v_flat, self.v_proj)

        elif self.mode == "fully_independent":
            q_flat = self.q_pre_norm(hidden_states).reshape(B * T, H)
            k_flat = self.k_pre_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_pre_norm(hidden_states).reshape(B * T, H)
            Q = self._route_and_project(self.q_router, q_flat, self.q_proj,
                                        self.q_norm_weight, self.num_heads)
            K = self._route_and_project(self.k_router, k_flat, self.k_proj,
                                        self.k_norm_weight, self.num_kv_heads)
            V = self._route_and_project(self.v_router, v_flat, self.v_proj)

        elif self.mode == "precompute_kv":
            raise RuntimeError("precompute_kv should use project_and_attend_precompute_kv()")

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
    ) -> torch.Tensor:
        """Run attention and O projection.

        For bundled/precompute_kv, O reuses routing saved by project().
        For kv_paired/qk_paired/fully_independent, O has its own router.
        """
        B = Q.shape[0]
        T = Q.shape[2]

        K_expanded = repeat_kv(K, self.num_kv_groups)
        V_expanded = repeat_kv(V, self.num_kv_groups)

        attn_weights = torch.matmul(Q, K_expanded.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn_output = torch.matmul(attn_weights, V_expanded)

        attn_output = attn_output.transpose(1, 2).reshape(B * T, self.q_dim)

        if self.mode in ("bundled", "precompute_kv"):
            idx, w = self._last_routing
            attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        elif self.mode in ("kv_paired", "qk_paired", "fully_independent"):
            attn_output = self._route_and_project(self.o_router, attn_output, self.o_proj)

        return attn_output.view(B, T, self.hidden_size)

    def project_and_attend_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Section 4.1.2: apply norm, route first, compute per-expert KV tables, attend.

        Returns:
            attn_out: [B, T, H]
            K_fresh: [B, n_kv_heads, T, head_dim]
            V_fresh: [B, n_kv_heads, T, head_dim]
        """
        B, T, H = hidden_states.shape
        normed = self.norm(hidden_states)
        N = B * T
        flat = normed.reshape(N, H)

        # 1. Route
        idx, w = self._route(self.router, flat)
        self._last_routing = (idx, w)

        if self.top_k == 1:
            token_expert = idx
        else:
            token_expert = idx[:, 0]

        active_experts = token_expert.unique()

        # 2. Q for each token
        Q = self._apply_projection(flat, self.q_proj, idx, w,
                                   self.q_norm_weight, self.num_heads)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        # 3. Per-expert KV tables + attention
        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        K_per_token = flat.new_zeros(N, self.kv_dim)
        V_per_token = flat.new_zeros(N, self.kv_dim)

        for e in active_experts:
            mask_e = (token_expert == e)
            mask_2d = mask_e.view(B, T)

            K_e = flat @ self.k_proj[e]
            V_e = flat @ self.v_proj[e]

            # Per-expert k_norm
            K_e_normed = K_e.view(N, self.num_kv_heads, self.head_dim).float()
            var_k = K_e_normed.pow(2).mean(-1, keepdim=True)
            K_e_normed = K_e_normed * torch.rsqrt(var_k + self.eps)
            K_e_normed = (self.k_norm_weight[e].unsqueeze(0) * K_e_normed).to(flat.dtype)

            K_e_heads = K_e_normed.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            V_e_heads = V_e.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            _, K_e_rope = apply_rotary_pos_emb(Q, K_e_heads, cos, sin)
            K_e_exp = repeat_kv(K_e_rope, self.num_kv_groups)
            V_e_exp = repeat_kv(V_e_heads, self.num_kv_groups)

            scores = torch.matmul(Q, K_e_exp.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_e = torch.matmul(scores, V_e_exp)

            mask_head = mask_2d.unsqueeze(1).unsqueeze(-1)
            attn_output = attn_output + attn_e * mask_head

            K_per_token[mask_e] = K_e[mask_e]
            V_per_token[mask_e] = V_e[mask_e]

        # 4. O projection
        attn_output = attn_output.transpose(1, 2).reshape(N, self.q_dim)
        attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        attn_output = attn_output.view(B, T, H)

        # 5. Per-token K,V for state update
        K_normed = K_per_token.view(N, self.num_kv_heads, self.head_dim).float()
        var_k = K_normed.pow(2).mean(-1, keepdim=True)
        K_normed = K_normed * torch.rsqrt(var_k + self.eps)
        kw = self.k_norm_weight[token_expert]
        K_normed = (kw.unsqueeze(1) * K_normed).to(flat.dtype)

        K_fresh = K_normed.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_per_token.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        _, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        return attn_output, K_fresh, V_fresh


# ─── MLP expert bank ──────────────────────────────────────────────────────── #

class MlpExpertBank(nn.Module):
    """MLP expert bank with pre-norm. Shared across all depths."""

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, 'use_deepseek_routing', False):
            self.gate = DeepSeekRouter(config)
        else:
            self.gate = Qwen3MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(config)
        self.last_router_logits = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        normed = self.norm(hidden_states)
        flat = normed.view(-1, H)
        router_logits, routing_weights, selected_experts = self.gate(flat)
        out = self.experts(flat, selected_experts, routing_weights)
        self.last_router_logits = router_logits
        return out.view(B, T, H)


# ─── Full model ────────────────────────────────────────────────────────────── #

class MoEverythingModel(nn.Module):
    """Mixture-of-Everything transformer backbone.

    All components (branch router, attention bank, MLP bank) are shared
    across depths.  The forward is a simple for loop — only activations
    change per depth, not weights.
    """

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_depths = config.num_hidden_layers

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * head_dim

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Initial KV projections (depth 0)
        self.init_k_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
        self.init_v_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
        self.init_k_norm = Qwen3MoeRMSNorm(head_dim, eps=config.rms_norm_eps)

        # RoPE
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

        # Shared components — one instance each, reused at every depth
        self.branch_router = BranchRouter(config.hidden_size)
        self.attn_bank = AttentionExpertBank(config)
        self.mlp_bank = MlpExpertBank(config)

        # Final norm
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Populated during forward for loss computation
        self._all_mlp_router_logits = []
        self._all_branch_probs = []

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        # ── Embed ──
        hidden_states = self.embed_tokens(input_ids)

        # ── Position embeddings ──
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        cos, sin = position_embeddings

        # ── Causal mask ──
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        # ── Initialize KV state (depth 0) ──
        K_init = self.init_k_proj(hidden_states)
        V_init = self.init_v_proj(hidden_states)
        K_init = self.init_k_norm(
            K_init.view(B, T, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        V_init = V_init.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to initial K
        cos_unsq = cos.unsqueeze(1)
        sin_unsq = sin.unsqueeze(1)
        K_init = (K_init * cos_unsq) + (self._rotate_half(K_init) * sin_unsq)

        kv_state = (K_init, V_init)

        # ── Reset loss accumulators ──
        self._all_mlp_router_logits = []
        self._all_branch_probs = []

        # ── Depth loop — all weights shared ──
        for depth in range(self.num_depths):
            # Branch routing
            branch_probs = self.branch_router(hidden_states)
            p_attn = branch_probs[..., 0:1]  # [B, T, 1]
            p_mlp = branch_probs[..., 1:2]   # [B, T, 1]
            self._all_branch_probs.append(branch_probs)

            K_old, V_old = kv_state
            p_attn_kv = p_attn.unsqueeze(1)  # [B, 1, T, 1]
            p_mlp_kv = p_mlp.unsqueeze(1)

            if self.attn_bank.mode == "precompute_kv":
                attn_out, K_fresh, V_fresh = self.attn_bank.project_and_attend_precompute_kv(
                    hidden_states, position_embeddings, causal_mask
                )
            else:
                Q, K_fresh, V_fresh = self.attn_bank.project(hidden_states, position_embeddings)
                # Blend KV: attn tokens refresh, mlp tokens keep old
                K_blend = p_attn_kv * K_fresh + p_mlp_kv * K_old
                V_blend = p_attn_kv * V_fresh + p_mlp_kv * V_old
                attn_out = self.attn_bank.attend(Q, K_blend, V_blend, causal_mask)

            # MLP branch (bank applies its own norm)
            mlp_out = self.mlp_bank(hidden_states)
            self._all_mlp_router_logits.append(self.mlp_bank.last_router_logits)

            # Combine via soft routing
            hidden_states = hidden_states + p_attn * attn_out + p_mlp * mlp_out

            # Update KV state
            kv_state = (
                p_attn_kv * K_fresh + p_mlp_kv * K_old,
                p_attn_kv * V_fresh + p_mlp_kv * V_old,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# ─── Causal LM wrapper ────────────────────────────────────────────────────── #

class MoEverythingForCausalLM(nn.Module):
    """Causal LM head over MoEverythingModel.

    Computes CE loss + MLP load-balancing aux loss + branch entropy loss.
    """

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.config = config
        self.model = MoEverythingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.vocab_size = config.vocab_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.branch_router_aux_loss_coef = config.branch_router_aux_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, Qwen3MoeExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        aux_loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # MLP expert load-balancing loss
            mlp_router_logits = self.model._all_mlp_router_logits
            if mlp_router_logits:
                mlp_aux = load_balancing_loss_func(
                    tuple(mlp_router_logits),
                    self.num_experts,
                    self.num_experts_per_tok,
                )
                if isinstance(mlp_aux, torch.Tensor):
                    aux_loss = mlp_aux
                    loss = loss + self.router_aux_loss_coef * mlp_aux

            # Branch balance loss: encourage ~50/50 split
            branch_probs = self.model._all_branch_probs
            if branch_probs:
                cat_probs = torch.cat([p.reshape(-1, 2) for p in branch_probs], dim=0)
                mean_probs = cat_probs.mean(dim=0)
                branch_aux = 2.0 * (mean_probs ** 2).sum()
                loss = loss + self.branch_router_aux_loss_coef * branch_aux

        return _MoEverythingOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
        )


class _MoEverythingOutput:
    """Minimal output object compatible with train.py expectations."""

    def __init__(self, loss, logits, aux_loss=None, router_logits=None):
        self.loss = loss
        self.logits = logits
        self.aux_loss = aux_loss
        self.router_logits = router_logits
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
