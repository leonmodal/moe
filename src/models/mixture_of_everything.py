"""
Mixture-of-Everything (Section 4.1) — hierarchical per-token routing.

Each token at each depth chooses ATTENTION or MLP via a branch router,
then selects expert weights within the chosen branch.

Attention bank modes:
  - "bundled"            : (Q,K,V,O) selected as one unit
  - "qo_kv_paired"      : (Q,O) bank + (K,V) bank, independently routed
  - "fully_independent"  : Q, K, V, O each from separate banks
  - "precompute_kv"      : shared KV projections, routed Q and O (Section 4.1.2)

MLP bank: standard top-k MoE over SwiGLU experts (reuses Qwen3MoeExperts).

Design notes:
  - Token state is (E, K, V) as described in the doc.  K,V persist across
    layers and are only refreshed when a token takes the attention branch.
  - Training uses soft branch routing (both branches computed, weighted by
    probability) so gradients flow through the router.  KV state is
    soft-interpolated: K_out = p_attn * K_fresh + p_mlp * K_old.
  - Pre-norm is shared: a single RMSNorm feeds whichever branch is selected.
    Its parameters are tied to the layer (not duplicated per branch).
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from .load_balancing import load_balancing_loss_func


# ─── Config ────────────────────────────────────────────────────────────────── #

class MoEverythingConfig(Qwen3MoeConfig):
    model_type = "moe_everything"

    def __init__(
        self,
        # Attention expert bank
        num_attn_experts: int = 4,
        num_attn_experts_per_tok: int = 1,
        attn_expert_mode: str = "bundled",  # bundled | qo_kv_paired | fully_independent | precompute_kv
        # MLP experts (num_experts / num_experts_per_tok from parent)
        # Branch router
        branch_router_aux_loss_coef: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attn_experts = num_attn_experts
        self.num_attn_experts_per_tok = num_attn_experts_per_tok
        self.attn_expert_mode = attn_expert_mode
        self.branch_router_aux_loss_coef = branch_router_aux_loss_coef


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
        """
        Args:
            hidden_states: [B, T, H] or [N, H]
        Returns:
            branch_probs: same leading dims + [2], softmax probabilities
        """
        logits = self.gate(hidden_states.float())
        probs = F.softmax(logits, dim=-1).to(hidden_states.dtype)
        return probs  # [..., 0] = attn, [..., 1] = mlp


# ─── Attention expert bank ─────────────────────────────────────────────────── #

class AttentionExpertBank(nn.Module):
    """Bank of attention weight sets with per-token expert routing.

    Supports four modes for how Q, K, V, O projections are banked.
    All modes share the same forward() signature.
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

        # Build projection banks based on mode
        # Each mode creates per-expert q/k norm weights tied to the projections.
        _MODES = {"bundled", "qo_kv_paired", "qk_paired", "fully_independent", "precompute_kv"}
        if self.mode not in _MODES:
            raise ValueError(f"Unknown attn_expert_mode: {self.mode}, must be one of {_MODES}")
        getattr(self, f"_init_{self.mode}")()

    # ── Initialization helpers ──

    def _init_bundled(self):
        """(Q,K,V,O) all selected as one unit.  Single router."""
        E = self.num_experts
        self.router = nn.Linear(self.hidden_size, E, bias=False)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        # Per-expert Q/K norms — tied to projection weights
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_qo_kv_paired(self):
        """(Q,O) paired bank + (K,V) paired bank."""
        E = self.num_experts
        self.qo_router = nn.Linear(self.hidden_size, E, bias=False)
        self.kv_router = nn.Linear(self.hidden_size, E, bias=False)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        # Per-expert norms: q_norm tied to QO bank, k_norm tied to KV bank
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.o_proj, self.k_proj, self.v_proj])

    def _init_qk_paired(self):
        """(Q,K) paired — dot-product compatibility.  V and O separate."""
        E = self.num_experts
        self.qk_router = nn.Linear(self.hidden_size, E, bias=False)
        self.v_router = nn.Linear(self.hidden_size, E, bias=False)
        self.o_router = nn.Linear(self.q_dim, E, bias=False)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        # Per-expert norms: q and k both tied to QK bank
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_fully_independent(self):
        """Q, K, V, O each from a separate bank with its own router."""
        E = self.num_experts
        self.q_router = nn.Linear(self.hidden_size, E, bias=False)
        self.k_router = nn.Linear(self.hidden_size, E, bias=False)
        self.v_router = nn.Linear(self.hidden_size, E, bias=False)
        self.o_router = nn.Linear(self.q_dim, E, bias=False)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        # Per-expert norms: q_norm tied to Q bank, k_norm tied to K bank
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_precompute_kv(self):
        """Full KV table per expert (Section 4.1.2).

        Route first → find active experts → compute K,V for all tokens
        using only the active experts.  Each token's Q attends over its
        expert's complete KV table, so cross-token subspace mismatch is
        eliminated.  Q,O are bundled (same expert selection).
        """
        E = self.num_experts
        self.router = nn.Linear(self.hidden_size, E, bias=False)
        # All experts have Q, K, V, O (same as bundled)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.kv_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.q_dim, self.hidden_size))
        # Per-expert Q/K norms
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_params(self, params):
        std = self.config.initializer_range
        for p in params:
            nn.init.normal_(p, mean=0.0, std=std)

    # ── Expert dispatch helpers ──

    def _apply_expert_head_norm(self, x, norm_weights, expert_idx, num_heads):
        """Per-expert RMSNorm on head-organized vectors.

        Args:
            x: [N, num_heads * head_dim]
            norm_weights: [E, head_dim] — per-expert learnable scale
            expert_idx: [N] — which expert each token selected
            num_heads: int
        Returns:
            [N, num_heads * head_dim]
        """
        N = x.shape[0]
        orig_dtype = x.dtype
        x = x.view(N, num_heads, self.head_dim).float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        w = norm_weights[expert_idx]          # [N, head_dim]
        x = w.unsqueeze(1) * x               # [N, 1, hd] * [N, nh, hd]
        return x.to(orig_dtype).view(N, num_heads * self.head_dim)

    def _route(self, router, x):
        """Compute routing: returns (expert_idx, expert_weights).

        For top-1: idx [N], weights [N, 1]
        For top-k: idx [N, K], weights [N, K]
        """
        logits = router(x.float())
        probs = F.softmax(logits, dim=-1)

        if self.top_k == 1:
            idx = probs.argmax(dim=-1)                          # [N]
            weights = probs.gather(1, idx.unsqueeze(-1))        # [N, 1]
            return idx, weights
        else:
            top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
            top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-20)
            return top_idx, top_vals                            # [N, K] each

    def _apply_projection(self, x, weight_bank, expert_idx, expert_weights,
                          head_norm_weights=None, num_heads_for_norm=None):
        """Project x using given expert routing (no router call).

        Memory-efficient: loops over active experts with regular matmul
        instead of gathering a [N, H_in, H_out] tensor.

        Args:
            x: [N, H_in]
            weight_bank: [E, H_in, H_out]
            expert_idx: [N] (top-1) or [N, K] (top-k)
            expert_weights: [N, 1] (top-1) or [N, K] (top-k)
            head_norm_weights: Optional [E, head_dim]
            num_heads_for_norm: int
        """
        N = x.shape[0]
        out = x.new_zeros(N, weight_bank.shape[2])

        if self.top_k == 1:
            for e in expert_idx.unique():
                mask = (expert_idx == e)
                proj = x[mask] @ weight_bank[e]                 # [n_e, H_out]
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
        """Route + project in one call (convenience wrapper)."""
        idx, weights = self._route(router, x)
        return self._apply_projection(x, weight_bank, idx, weights,
                                      head_norm_weights, num_heads_for_norm)

    # ── Forward: split into project + attend for KV blending ──

    def project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K_fresh, V_fresh via expert routing.

        Bundled projections share a single routing decision.
        Saves routing info in self._last_*_routing for attend() to reuse.

        Returns:
            Q: [B, n_heads, T, head_dim] with RoPE applied
            K_fresh: [B, n_kv_heads, T, head_dim] with RoPE applied
            V_fresh: [B, n_kv_heads, T, head_dim]
        """
        B, T, H = hidden_states.shape
        flat = hidden_states.reshape(B * T, H)

        if self.mode == "bundled":
            # Route once — Q,K,V,O all use same expert
            idx, w = self._route(self.router, flat)
            self._last_routing = (idx, w)
            Q = self._apply_projection(flat, self.q_proj, idx, w,
                                       self.q_norm_weight, self.num_heads)
            K = self._apply_projection(flat, self.k_proj, idx, w,
                                       self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(flat, self.v_proj, idx, w)

        elif self.mode == "qo_kv_paired":
            # QO bank: route once, save for O
            qo_idx, qo_w = self._route(self.qo_router, flat)
            self._last_routing = (qo_idx, qo_w)
            Q = self._apply_projection(flat, self.q_proj, qo_idx, qo_w,
                                       self.q_norm_weight, self.num_heads)
            # KV bank: route once, K and V share
            kv_idx, kv_w = self._route(self.kv_router, flat)
            K = self._apply_projection(flat, self.k_proj, kv_idx, kv_w,
                                       self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(flat, self.v_proj, kv_idx, kv_w)

        elif self.mode == "qk_paired":
            # QK bank: route once, Q and K share (dot-product compatibility)
            qk_idx, qk_w = self._route(self.qk_router, flat)
            Q = self._apply_projection(flat, self.q_proj, qk_idx, qk_w,
                                       self.q_norm_weight, self.num_heads)
            K = self._apply_projection(flat, self.k_proj, qk_idx, qk_w,
                                       self.k_norm_weight, self.num_kv_heads)
            # V and O are independently routed
            V = self._route_and_project(self.v_router, flat, self.v_proj)

        elif self.mode == "fully_independent":
            Q = self._route_and_project(self.q_router, flat, self.q_proj,
                                        self.q_norm_weight, self.num_heads)
            K = self._route_and_project(self.k_router, flat, self.k_proj,
                                        self.k_norm_weight, self.num_kv_heads)
            V = self._route_and_project(self.v_router, flat, self.v_proj)

        elif self.mode == "precompute_kv":
            # Route first, then compute per-expert KV tables.
            # Handled entirely in project_and_attend_precompute_kv().
            raise RuntimeError("precompute_kv should use project_and_attend_precompute_kv()")

        # Reshape to [B, n_heads, T, head_dim] — norms already applied
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
        hidden_states_flat: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention and O projection.

        For bundled/paired modes, O reuses the expert selection saved by project().

        Args:
            Q: [B, n_heads, T, head_dim]
            K: [B, n_kv_heads, T, head_dim] — already blended with cache
            V: [B, n_kv_heads, T, head_dim] — already blended with cache
            hidden_states_flat: [N, H] — unused (kept for API consistency)
            attention_mask: [B, 1, T, T]

        Returns:
            attn_out: [B, T, H]
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

        # O projection — bundled modes reuse saved routing from project()
        if self.mode in ("bundled", "qo_kv_paired", "precompute_kv"):
            idx, w = self._last_routing
            attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        elif self.mode == "qk_paired":
            attn_output = self._route_and_project(self.o_router, attn_output, self.o_proj)
        elif self.mode == "fully_independent":
            attn_output = self._route_and_project(self.o_router, attn_output, self.o_proj)

        return attn_output.view(B, T, self.hidden_size)

    def project_and_attend_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Section 4.1.2: route first, compute per-expert KV tables, attend.

        1. Route to decide each token's expert (Q,O bundled).
        2. Find active experts.
        3. For each active expert: compute K,V for ALL tokens using that expert.
        4. Each token's Q attends over its expert's full KV table.
        5. O projection reuses same expert selection.

        Returns:
            attn_out: [B, T, H]
            K_fresh: [B, n_kv_heads, T, head_dim] — per-token K (from selected expert)
            V_fresh: [B, n_kv_heads, T, head_dim] — per-token V (from selected expert)
        """
        B, T, H = hidden_states.shape
        N = B * T
        flat = hidden_states.reshape(N, H)

        # 1. Route first — decide each token's expert
        idx, w = self._route(self.router, flat)
        self._last_routing = (idx, w)

        # For top-1: idx is [N], w is [N, 1]
        # For top-k: idx is [N, K], w is [N, K] — we handle top-1 path for clarity
        if self.top_k == 1:
            token_expert = idx                  # [N]
        else:
            token_expert = idx[:, 0]            # primary expert for KV table selection

        # 2. Find active experts
        active_experts = token_expert.unique()

        # 3. Compute Q for each token using its expert
        Q = self._apply_projection(flat, self.q_proj, idx, w,
                                   self.q_norm_weight, self.num_heads)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        # 4. For each active expert: compute K,V for ALL tokens, run full
        #    attention, then keep only results for tokens assigned to that expert.
        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        K_per_token = flat.new_zeros(N, self.kv_dim)
        V_per_token = flat.new_zeros(N, self.kv_dim)

        for e in active_experts:
            mask_e = (token_expert == e)                        # [N]
            mask_2d = mask_e.view(B, T)                         # [B, T]

            # K,V for ALL tokens using expert e's projections
            K_e = flat @ self.k_proj[e]                         # [N, kv_dim]
            V_e = flat @ self.v_proj[e]                         # [N, kv_dim]

            # Per-expert k_norm
            K_e_normed = K_e.view(N, self.num_kv_heads, self.head_dim).float()
            var_k = K_e_normed.pow(2).mean(-1, keepdim=True)
            K_e_normed = K_e_normed * torch.rsqrt(var_k + self.eps)
            K_e_normed = (self.k_norm_weight[e].unsqueeze(0) * K_e_normed).to(flat.dtype)

            # Reshape, RoPE, GQA expand
            K_e_heads = K_e_normed.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            V_e_heads = V_e.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            _, K_e_rope = apply_rotary_pos_emb(Q, K_e_heads, cos, sin)
            K_e_exp = repeat_kv(K_e_rope, self.num_kv_groups)  # [B, n_heads, T, hd]
            V_e_exp = repeat_kv(V_e_heads, self.num_kv_groups)

            # Full attention: all Q against expert e's KV table
            scores = torch.matmul(Q, K_e_exp.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_e = torch.matmul(scores, V_e_exp)             # [B, n_heads, T, hd]

            # Keep only results for tokens assigned to this expert
            mask_head = mask_2d.unsqueeze(1).unsqueeze(-1)      # [B, 1, T, 1]
            attn_output = attn_output + attn_e * mask_head

            # Save per-token K,V from selected expert (for KV state)
            K_per_token[mask_e] = K_e[mask_e]
            V_per_token[mask_e] = V_e[mask_e]

        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).reshape(N, self.q_dim)

        # 5. O projection — reuse same routing
        attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        attn_output = attn_output.view(B, T, H)

        # 6. Per-token K,V for state update (from each token's selected expert)
        #    Apply per-expert k_norm + RoPE
        K_normed = K_per_token.view(N, self.num_kv_heads, self.head_dim).float()
        var_k = K_normed.pow(2).mean(-1, keepdim=True)
        K_normed = K_normed * torch.rsqrt(var_k + self.eps)
        kw = self.k_norm_weight[token_expert]                   # [N, head_dim]
        K_normed = (kw.unsqueeze(1) * K_normed).to(flat.dtype)

        K_fresh = K_normed.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_per_token.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        _, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        return attn_output, K_fresh, V_fresh


# ─── MoE-Everything decoder layer ─────────────────────────────────────────── #

class MoEverythingLayer(nn.Module):
    """Single decoder layer with branch routing between attention and MLP.

    Uses a single pre-norm (tied to the layer, not per-branch).
    Token state is (E, K, V); K,V persist and are only refreshed on ATTN.
    """

    def __init__(self, config: MoEverythingConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Shared pre-norm
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Branch router
        self.branch_router = BranchRouter(config.hidden_size)

        # Attention expert bank
        self.attn_bank = AttentionExpertBank(config)

        # MLP expert bank (reuses standard MoE block)
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )
        self.mlp_bank = Qwen3MoeSparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_state: tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, H]
            kv_state: (K, V) each [B, n_kv_heads, T, head_dim]
            position_embeddings: (cos, sin)
            attention_mask: [B, 1, T, T] causal mask

        Returns:
            hidden_states: [B, T, H] updated
            kv_state: (K, V) updated
        """
        B, T, H = hidden_states.shape

        # Pre-norm (shared)
        normed = self.norm(hidden_states)

        # Branch routing: [B, T, 2]
        branch_probs = self.branch_router(normed)
        p_attn = branch_probs[..., 0:1]  # [B, T, 1]
        p_mlp = branch_probs[..., 1:2]   # [B, T, 1]

        if self.attn_bank.mode == "precompute_kv":
            # precompute_kv does its own per-expert attention internally
            attn_out, K_fresh, V_fresh = self.attn_bank.project_and_attend_precompute_kv(
                normed, position_embeddings, attention_mask
            )
        else:
            # ── Attention branch: project Q, K_fresh, V_fresh ──
            Q, K_fresh, V_fresh = self.attn_bank.project(normed, position_embeddings)

            # ── Blend KV: soft "attn tokens refresh, mlp tokens keep cache" ──
            K_old, V_old = kv_state
            p_attn_kv = p_attn.unsqueeze(1)  # [B, 1, T, 1]
            p_mlp_kv = p_mlp.unsqueeze(1)
            K_blend = p_attn_kv * K_fresh + p_mlp_kv * K_old
            V_blend = p_attn_kv * V_fresh + p_mlp_kv * V_old

            # ── Attention using blended KV (cache flows into scores) ──
            flat = normed.reshape(B * T, H)
            attn_out = self.attn_bank.attend(Q, K_blend, V_blend, flat, attention_mask)

        # ── MLP branch ──
        mlp_out = self.mlp_bank(normed)  # [B, T, H]

        # ── Combine via soft routing ──
        hidden_states = hidden_states + p_attn * attn_out + p_mlp * mlp_out

        # ── KV state update ──
        K_old, V_old = kv_state
        p_attn_kv = p_attn.unsqueeze(1)
        p_mlp_kv = p_mlp.unsqueeze(1)
        K_out = p_attn_kv * K_fresh + p_mlp_kv * K_old
        V_out = p_attn_kv * V_fresh + p_mlp_kv * V_old

        return hidden_states, (K_out, V_out)


# ─── Full model ────────────────────────────────────────────────────────────── #

class MoEverythingModel(nn.Module):
    """Mixture-of-Everything transformer backbone.

    Token state (E, K, V) is initialized from the embedding and flows
    through all layers.  K,V persist and are refreshed only when the
    branch router selects attention.
    """

    def __init__(self, config: MoEverythingConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

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

        # Decoder layers
        self.layers = nn.ModuleList([
            MoEverythingLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        # ── Embed ──
        E = self.embed_tokens(input_ids)  # [B, T, H]

        # ── Position embeddings ──
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(E, position_ids=position_ids)
        cos, sin = position_embeddings

        # ── Causal mask ──
        # Standard causal: [1, 1, T, T]
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=E.device, dtype=E.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        # ── Initialize KV state (depth 0) ──
        K_init = self.init_k_proj(E)  # [B, T, kv_dim]
        V_init = self.init_v_proj(E)  # [B, T, kv_dim]
        K_init = self.init_k_norm(
            K_init.view(B, T, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        V_init = V_init.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to initial K
        # cos/sin: [B, T, head_dim] — need to unsqueeze for [B, n_heads, T, head_dim]
        cos_unsq = cos.unsqueeze(1)  # [B, 1, T, head_dim]
        sin_unsq = sin.unsqueeze(1)
        K_init = (K_init * cos_unsq) + (self._rotate_half(K_init) * sin_unsq)

        kv_state = (K_init, V_init)

        # ── Decoder layers ──
        hidden_states = E
        for layer in self.layers:
            hidden_states, kv_state = layer(
                hidden_states, kv_state, position_embeddings, causal_mask
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

        # Tie embeddings if configured
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
            mlp_router_logits = self._collect_mlp_router_logits()
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
            branch_aux = self._branch_balance_loss()
            if branch_aux is not None:
                loss = loss + self.branch_router_aux_loss_coef * branch_aux

        # Return a simple namespace-like object compatible with train.py
        return _MoEverythingOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
        )

    def _collect_mlp_router_logits(self) -> list[torch.Tensor]:
        """Collect router logits from MLP expert banks."""
        logits = []
        for layer in self.model.layers:
            gate = layer.mlp_bank.gate
            if hasattr(gate, '_last_logits'):
                logits.append(gate._last_logits)
        return logits

    def _branch_balance_loss(self) -> torch.Tensor | None:
        """Entropy-based loss to prevent branch collapse."""
        all_probs = []
        for layer in self.model.layers:
            router = layer.branch_router
            if hasattr(router, '_last_probs'):
                all_probs.append(router._last_probs)
        if not all_probs:
            return None
        # Mean branch probability across all tokens and layers
        # Encourage balanced usage by penalizing deviation from uniform
        cat_probs = torch.cat([p.reshape(-1, 2) for p in all_probs], dim=0)  # [N_total, 2]
        mean_probs = cat_probs.mean(dim=0)  # [2]
        # Loss = 2 * sum(mean_i^2) — minimized at uniform (0.5, 0.5) = 0.5
        return 2.0 * (mean_probs ** 2).sum()


class _MoEverythingOutput:
    """Minimal output object compatible with train.py expectations."""

    def __init__(self, loss, logits, aux_loss=None, router_logits=None):
        self.loss = loss
        self.logits = logits
        self.aux_loss = aux_loss
        self.router_logits = router_logits
        # For compatibility
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
