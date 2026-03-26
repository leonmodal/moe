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
  - "per_head_precompute_kv": Q, KV, O heads route independently (K,V share router) — per-head precompute KV

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
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from .router import DeepSeekRouter, checkpoint_recompute_context


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
        # Per-layer router: one branch router per depth instead of shared
        per_layer_router: bool = False,
        # Dynamic depth: random perturbation of depth during training
        dynamic_depth_min: float = 1.0,
        dynamic_depth_max: float = 1.0,
        # Depthwise attention (AttnRes): learned weighted combination across depths
        depthwise_attention: bool = False,
        depthwise_block_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attn_experts = num_attn_experts
        self.num_attn_experts_per_tok = num_attn_experts_per_tok
        self.attn_expert_mode = attn_expert_mode
        self.branch_router_aux_loss_coef = branch_router_aux_loss_coef
        self.use_deepseek_routing = use_deepseek_routing
        self.per_layer_router = per_layer_router
        self.dynamic_depth_min = dynamic_depth_min
        self.dynamic_depth_max = dynamic_depth_max
        self.depthwise_attention = depthwise_attention
        self.depthwise_block_size = depthwise_block_size


# ─── Branch router ─────────────────────────────────────────────────────────── #

class BranchRouter(nn.Module):
    """Binary router: ATTN (0) or MLP (1) per token.

    Returns soft probabilities in [0, 1] for each branch.
    During training both branches are computed and weighted.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)
        self.last_probs = None

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate(hidden_states.float())
        probs = F.softmax(logits, dim=-1).to(hidden_states.dtype)
        self.last_probs = probs
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
        # Flat-bank fully_independent drops GQA: KV heads = Q heads
        if self.mode == "per_head_fully_independent":
            self.num_kv_heads = self.num_heads
            self.num_kv_groups = 1
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.eps = config.rms_norm_eps
        self.use_deepseek_routing = getattr(config, "use_deepseek_routing", False)
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

    def _make_flat_bank_router(self, input_dim, top_k):
        """Create a router for flat bank modes — selects top_k experts from the pool."""
        if self.use_deepseek_routing:
            from types import SimpleNamespace

            num_groups = getattr(self.config, "num_groups", None)
            group_topk = getattr(self.config, "group_topk", None)
            # Disable group-limited routing if it can't provide enough candidates
            if num_groups and group_topk:
                experts_per_group = self.num_experts // num_groups
                max_candidates = group_topk * experts_per_group
                if max_candidates < top_k:
                    num_groups = None
                    group_topk = None
            cfg = SimpleNamespace(
                hidden_size=input_dim,
                num_local_experts=self.num_experts,
                num_experts=self.num_experts,
                num_experts_per_tok=top_k,
                norm_topk_prob=getattr(self.config, "norm_topk_prob", True),
                topk_scaling_factor=getattr(self.config, "topk_scaling_factor", None),
                num_groups=num_groups,
                group_topk=group_topk,
            )
            return DeepSeekRouter(cfg)
        return nn.Linear(input_dim, self.num_experts, bias=False)

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

    def _init_per_head_fully_independent(self):
        """Flat-bank Q/K/V/O: 4 routers select from shared pools.  3 norms.

        Q router picks top-num_heads, K/V each pick top-num_heads (no GQA),
        O router picks top-num_heads.  K and V route independently.
        """
        self.q_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.k_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        self.v_pre_norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        self.q_router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
        self.k_router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
        self.v_router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
        self.o_router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.head_dim, self.hidden_size))
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

    def _init_per_head_precompute_kv(self):
        """Flat-bank precompute KV: 1 router, corresponding QKVO.  1 norm.

        Single router picks top-num_heads from E experts.
        Each selected expert provides Q, K, V, and O for that head.
        """
        self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        self.router = self._make_flat_bank_router(self.hidden_size, self.num_heads)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
        self.o_proj = nn.Parameter(torch.empty(E, self.head_dim, self.hidden_size))
        self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

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

    def _store_router_info(self, name: str, router_probs: torch.Tensor, expert_idx: torch.Tensor) -> None:
        self.last_router_info[name] = {
            "router_logits": router_probs.detach(),
            "selected_experts": expert_idx.detach(),
        }

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-routing-group norm, then compute Q, K_fresh, V_fresh via expert routing."""
        B, T, H = hidden_states.shape
        self.last_router_info = {}

        if self.mode == "bundled":
            flat = self.norm(hidden_states).reshape(B * T, H)
            idx, w, router_probs = self._route(self.router, flat)
            self._last_routing = (idx, w)
            self._store_router_info("attn", router_probs, idx)
            Q = self._apply_projection(flat, self.q_proj, idx, w, self.q_norm_weight, self.num_heads)
            K = self._apply_projection(flat, self.k_proj, idx, w, self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(flat, self.v_proj, idx, w)

        elif self.mode == "kv_paired":
            kv_flat = self.kv_norm(hidden_states).reshape(B * T, H)
            q_flat = self.q_norm(hidden_states).reshape(B * T, H)
            kv_idx, kv_w, kv_router_probs = self._route(self.kv_router, kv_flat)
            self._store_router_info("kv", kv_router_probs, kv_idx)
            K = self._apply_projection(kv_flat, self.k_proj, kv_idx, kv_w, self.k_norm_weight, self.num_kv_heads)
            V = self._apply_projection(kv_flat, self.v_proj, kv_idx, kv_w)
            Q = self._route_and_project(
                self.q_router, q_flat, self.q_proj, self.q_norm_weight, self.num_heads, router_name="q"
            )

        elif self.mode == "qk_paired":
            qk_flat = self.qk_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_norm(hidden_states).reshape(B * T, H)
            qk_idx, qk_w, qk_router_probs = self._route(self.qk_router, qk_flat)
            self._store_router_info("qk", qk_router_probs, qk_idx)
            Q = self._apply_projection(qk_flat, self.q_proj, qk_idx, qk_w, self.q_norm_weight, self.num_heads)
            K = self._apply_projection(qk_flat, self.k_proj, qk_idx, qk_w, self.k_norm_weight, self.num_kv_heads)
            V = self._route_and_project(self.v_router, v_flat, self.v_proj, router_name="v")

        elif self.mode == "fully_independent":
            q_flat = self.q_pre_norm(hidden_states).reshape(B * T, H)
            k_flat = self.k_pre_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_pre_norm(hidden_states).reshape(B * T, H)
            Q = self._route_and_project(
                self.q_router, q_flat, self.q_proj, self.q_norm_weight, self.num_heads, router_name="q"
            )
            K = self._route_and_project(
                self.k_router, k_flat, self.k_proj, self.k_norm_weight, self.num_kv_heads, router_name="k"
            )
            V = self._route_and_project(self.v_router, v_flat, self.v_proj, router_name="v")

        elif self.mode == "per_head_fully_independent":
            q_flat = self.q_pre_norm(hidden_states).reshape(B * T, H)
            k_flat = self.k_pre_norm(hidden_states).reshape(B * T, H)
            v_flat = self.v_pre_norm(hidden_states).reshape(B * T, H)

            q_idx, q_w, q_probs = self._route_flat(self.q_router, q_flat, self.num_heads)
            k_idx, k_w, k_probs = self._route_flat(self.k_router, k_flat, self.num_heads)
            v_idx, v_w, v_probs = self._route_flat(self.v_router, v_flat, self.num_heads)
            o_idx, o_w, o_probs = self._route_flat(self.o_router, q_flat, self.num_heads)
            self._store_router_info("q", q_probs, q_idx)
            self._store_router_info("k", k_probs, k_idx)
            self._store_router_info("v", v_probs, v_idx)
            self._store_router_info("o", o_probs, o_idx)
            self._flat_o_idx = o_idx
            self._flat_o_w = o_w

            q_parts, k_parts, v_parts = [], [], []
            for h in range(self.num_heads):
                q_parts.append(self._project_flat_head(
                    q_flat, self.q_proj, q_idx[:, h], q_w[:, h], self.q_norm_weight))
                k_parts.append(self._project_flat_head(
                    k_flat, self.k_proj, k_idx[:, h], k_w[:, h], self.k_norm_weight))
                v_parts.append(self._project_flat_head(
                    v_flat, self.v_proj, v_idx[:, h], v_w[:, h]))

            Q = torch.cat(q_parts, dim=-1)
            K = torch.cat(k_parts, dim=-1)
            V = torch.cat(v_parts, dim=-1)

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
    ) -> torch.Tensor:
        """Run attention and O projection."""
        B = Q.shape[0]
        T = Q.shape[2]

        K_expanded = repeat_kv(K, self.num_kv_groups)
        V_expanded = repeat_kv(V, self.num_kv_groups)

        attn_weights = torch.matmul(Q, K_expanded.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn_output = torch.matmul(attn_weights, V_expanded)

        if self.mode == "per_head_fully_independent":
            head_outputs = []
            for h in range(self.num_heads):
                attn_h = attn_output[:, h, :, :].reshape(B * T, self.head_dim)
                head_outputs.append(self._project_flat_head(
                    attn_h, self.o_proj, self._flat_o_idx[:, h], self._flat_o_w[:, h]))
            return torch.stack(head_outputs, dim=0).sum(dim=0).view(B, T, self.hidden_size)

        attn_output = attn_output.transpose(1, 2).reshape(B * T, self.q_dim)

        if self.mode in ("bundled", "precompute_kv"):
            idx, w = self._last_routing
            attn_output = self._apply_projection(attn_output, self.o_proj, idx, w)
        elif self.mode in ("kv_paired", "qk_paired", "fully_independent"):
            attn_output = self._route_and_project(self.o_router, attn_output, self.o_proj, router_name="o")

        return attn_output.view(B, T, self.hidden_size)

    def project_and_attend_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Section 4.1.2: apply norm, route first, compute per-expert KV tables, attend."""
        B, T, H = hidden_states.shape
        self.last_router_info = {}
        normed = self.norm(hidden_states)
        N = B * T
        flat = normed.reshape(N, H)

        idx, w, router_probs = self._route(self.router, flat)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat-bank precompute KV: 1 router, corresponding QKVO, with GQA."""
        B, T, H = hidden_states.shape
        self.last_router_info = {}
        N = B * T
        normed = self.norm(hidden_states)
        flat = normed.reshape(N, H)
        cos, sin = position_embeddings

        # Single routing decision — picks num_heads experts
        idx, w, probs = self._route_flat(self.router, flat, self.num_heads)  # (N, num_heads)
        self._store_router_info("attn", probs, idx)

        # Q projection: all num_heads positions
        q_parts = []
        for h in range(self.num_heads):
            q_h = self._project_flat_head(flat, self.q_proj, idx[:, h], w[:, h], self.q_norm_weight)
            q_parts.append(q_h.view(B, T, self.head_dim).unsqueeze(1))
        Q = torch.cat(q_parts, dim=1)  # (B, num_heads, T, head_dim)

        # KV precompute + attention with GQA grouping
        # KV for group g comes from the expert at rank g * num_kv_groups
        head_outputs = []
        K_parts = []
        V_parts = []
        kv_head_expert_idx = []

        for g in range(self.num_kv_heads):
            kv_rank = g * self.num_kv_groups
            kv_expert = idx[:, kv_rank]   # (N,) — KV expert for this group
            kv_weight = w[:, kv_rank]     # (N,)
            kv_head_expert_idx.append(kv_expert)
            active_experts = kv_expert.unique()

            q_start = g * self.num_kv_groups
            q_end = q_start + self.num_kv_groups
            Q_g = Q[:, q_start:q_end, :, :]  # (B, num_kv_groups, T, head_dim)

            head_attn = flat.new_zeros(B, self.num_kv_groups, T, self.head_dim)
            head_K = flat.new_zeros(B, T, self.head_dim)
            head_V = flat.new_zeros(B, T, self.head_dim)

            for e in active_experts:
                mask_e = kv_expert == e
                mask_2d = mask_e.view(B, T)

                K_he = flat @ self.k_proj[e]
                V_he = flat @ self.v_proj[e]

                K_he_f = K_he.float()
                var_k = K_he_f.pow(2).mean(-1, keepdim=True)
                K_he_normed = (K_he_f * torch.rsqrt(var_k + self.eps))
                K_he_normed = (self.k_norm_weight[e] * K_he_normed).to(flat.dtype)

                K_he_heads = K_he_normed.view(B, T, 1, self.head_dim).transpose(1, 2)
                V_he_heads = V_he.view(B, T, 1, self.head_dim).transpose(1, 2)

                _, K_he_rope = apply_rotary_pos_emb(Q_g, K_he_heads, cos, sin)

                K_he_exp = K_he_rope.expand(-1, self.num_kv_groups, -1, -1)
                V_he_exp = V_he_heads.expand(-1, self.num_kv_groups, -1, -1)

                scores = torch.matmul(Q_g, K_he_exp.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    scores = scores + attention_mask
                scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                attn_he = torch.matmul(scores, V_he_exp)

                mask_head = mask_2d.unsqueeze(1).unsqueeze(-1)
                head_attn = head_attn + attn_he * mask_head

                head_K = head_K + K_he.view(B, T, self.head_dim) * mask_2d.unsqueeze(-1)
                head_V = head_V + V_he.view(B, T, self.head_dim) * mask_2d.unsqueeze(-1)

            routing_scale = kv_weight.view(B, T).unsqueeze(1).unsqueeze(-1)
            head_attn = head_attn * routing_scale

            head_outputs.append(head_attn)
            K_parts.append(head_K.unsqueeze(1))
            V_parts.append(head_V.unsqueeze(1))

        attn_output = torch.cat(head_outputs, dim=1)  # (B, num_heads, T, head_dim)
        K_per_token = torch.cat(K_parts, dim=1)  # (B, num_kv_heads, T, head_dim)
        V_per_token = torch.cat(V_parts, dim=1)

        # O projection: same expert as Q for each head
        o_projected_heads = []
        for h in range(self.num_heads):
            attn_h = attn_output[:, h, :, :].reshape(N, self.head_dim)
            o_projected_heads.append(
                self._project_flat_head(attn_h, self.o_proj, idx[:, h], w[:, h]))
        attn_output = torch.stack(o_projected_heads, dim=0).sum(dim=0).view(B, T, H)

        # K_fresh with QK norm
        for h in range(self.num_kv_heads):
            K_h = K_per_token[:, h, :, :].reshape(N, self.head_dim).float()
            var_k = K_h.pow(2).mean(-1, keepdim=True)
            K_h_normed = K_h * torch.rsqrt(var_k + self.eps)
            kw = self.k_norm_weight[kv_head_expert_idx[h]]
            K_per_token[:, h, :, :] = (kw * K_h_normed).to(flat.dtype).view(B, T, self.head_dim)

        K_fresh = K_per_token
        V_fresh = V_per_token
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
        self.last_selected_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        normed = self.norm(hidden_states)
        flat = normed.view(-1, H)
        router_logits, routing_weights, selected_experts = self.gate(flat)
        out = self.experts(flat, selected_experts, routing_weights)
        self.last_router_logits = router_logits
        self.last_selected_experts = selected_experts
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

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.head_dim = head_dim
        attn_mode = getattr(config, "attn_expert_mode", "bundled")
        if attn_mode == "per_head_fully_independent":
            self.num_kv_heads = config.num_attention_heads  # no GQA for flat-bank
        else:
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

    def _depth_step(
        self,
        hidden_states: torch.Tensor,
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        depth_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.per_layer_router:
            branch_probs = self.branch_routers[depth_idx](hidden_states)
        else:
            branch_probs = self.branch_router(hidden_states)
        p_attn = branch_probs[..., 0:1]
        p_mlp = branch_probs[..., 1:2]

        p_attn_kv = p_attn.unsqueeze(1)
        p_mlp_kv = p_mlp.unsqueeze(1)

        if self.attn_bank.mode == "per_head_precompute_kv":
            attn_out, K_fresh, V_fresh = self.attn_bank.project_and_attend_per_head_precompute_kv(
                hidden_states, position_embeddings, causal_mask
            )
        elif self.attn_bank.mode == "precompute_kv":
            attn_out, K_fresh, V_fresh = self.attn_bank.project_and_attend_precompute_kv(
                hidden_states, position_embeddings, causal_mask
            )
        else:
            Q, K_fresh, V_fresh = self.attn_bank.project(hidden_states, position_embeddings)
            K_blend = p_attn_kv * K_fresh + p_mlp_kv * K_old
            V_blend = p_attn_kv * V_fresh + p_mlp_kv * V_old
            attn_out = self.attn_bank.attend(Q, K_blend, V_blend, causal_mask)

        mlp_out = self.mlp_bank(hidden_states)
        hidden_states = hidden_states + p_attn * attn_out + p_mlp * mlp_out

        K_new = p_attn_kv * K_fresh + p_mlp_kv * K_old
        V_new = p_attn_kv * V_fresh + p_mlp_kv * V_old
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


class MoEverythingForCausalLM(nn.Module):
    """Causal LM head over MoEverythingModel.

    Computes CE loss + MLP load-balancing aux loss + branch entropy loss.
    """

    supports_gradient_checkpointing = True

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
        self._seq_aux_loss_coef = getattr(config, "seq_aux_loss_coef", 0.0)

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
                )
                if isinstance(seq_aux, torch.Tensor):
                    seq_aux_loss = seq_aux
                    loss = loss + seq_aux_coef * seq_aux

            if branch_prob_tensors:
                cat_probs = torch.cat([p.reshape(-1, 2) for p in branch_prob_tensors], dim=0)
                mean_probs = cat_probs.mean(dim=0)
                branch_aux_loss = 2.0 * (mean_probs**2).sum()
                loss = loss + self.branch_router_aux_loss_coef * branch_aux_loss

        if output_router_logits:
            router_logits_out = tuple(t.detach() for t in mlp_router_logits) if mlp_router_logits is not None else None
            selected_experts_out = tuple(t.detach() for t in mlp_selected_experts) if mlp_selected_experts is not None else None
            branch_probs_out = tuple(t.detach() for t in branch_prob_tensors) if branch_prob_tensors is not None else None
            attention_router_info_out = None
            if attention_router_info is not None:
                attention_router_info_out = tuple(
                    {
                        name: {
                            "router_logits": info["router_logits"],
                            "selected_experts": info["selected_experts"],
                        }
                        for name, info in depth_info.items()
                    }
                    for depth_info in attention_router_info
                )
        else:
            router_logits_out = None
            selected_experts_out = None
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
        self.branch_probs = branch_probs
        self.attention_router_info = attention_router_info
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
