"""Attention expert bank: NormExpertBank and AttentionExpertBank.

Per-head routed attention with shared expert weight banks across depths.
Modes: per_head_fully_independent, per_head_precompute_kv.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fp32_routing import fp32_index_add, fp32_index_put, fp32_index_select
from src.models.triton_grouped_gemm import triton_grouped_gemm
from src.models.modeling_qwen3_moe import (
    Qwen3MoeRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from src.models.router import (
    DeepSeekRouter,
    ExplorationTopKRouter,
    checkpoint_recompute_context,
    is_checkpoint_recompute,
)
from src.models.routing.routers import _straight_through_ones

DEFAULT_PER_HEAD_DENSE_FRACTION_THRESHOLD = 0.75
AUTO_PER_HEAD_SPARSE_THRESHOLDS = {
    "per_head_fully_independent": 0.75,
    "per_head_precompute_kv": 0.75,
}

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
        with torch.autocast(device_type=flat.device.type, enabled=False):
            logits = self.router(flat.float())
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
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


# _straight_through_ones is now in src/models/routing/routers.py and imported at the top


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
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.num_key_value_groups = self.num_kv_groups
        self.is_causal = True
        self.q_heads_per_kv = self.num_kv_groups
        self.q_group_dim = self.q_heads_per_kv * self.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.eps = config.rms_norm_eps
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.sliding_window = getattr(config, "sliding_window", None)
        self.use_deepseek_routing = getattr(config, "use_deepseek_routing", False)
        self.per_layer_attn_router = getattr(config, "per_layer_attn_router", False)
        self.routed_norm = getattr(config, "routed_norm", False)
        self.per_layer_norm = getattr(config, "per_layer_norm", False)
        self.per_layer_qk_norm = getattr(config, "per_layer_qk_norm", False)
        self.num_depths = config.num_hidden_layers
        self.per_head_compute_mode = getattr(config, "per_head_compute_mode", "auto")
        self.per_head_dense_fraction_threshold = getattr(
            config, "per_head_dense_fraction_threshold", 0.75
        )
        self.sanity_check_mode = getattr(config, "sanity_check_mode", None)
        self.scale_attn_by_routing_weight = getattr(config, "scale_attn_by_routing_weight", False)
        self.last_router_info = {}

        _MODES = {
            "per_head_fully_independent",
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

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            modules = object.__getattribute__(self, "_modules")
            if name == "q_proj" and "logical_q_proj" in modules:
                return self._logical_q_proj_bank()
            if name == "k_proj" and "logical_k_proj" in modules:
                return self._logical_k_proj_bank()
            if name == "v_proj" and "logical_v_proj" in modules:
                return self._logical_v_proj_bank()
            if name == "o_proj" and "logical_o_proj" in modules:
                return self._logical_o_proj_bank()
            raise exc

    def _logical_q_proj_bank(self) -> torch.Tensor:
        layers = super().__getattr__("logical_q_proj")
        per_layer = [
            weight.view(self.num_kv_heads, self.q_group_dim, self.hidden_size).permute(0, 2, 1)
            for weight in layers
        ]
        return torch.stack(per_layer, dim=0).reshape(self.num_experts, self.hidden_size, self.q_group_dim)

    def _logical_k_proj_bank(self) -> torch.Tensor:
        layers = super().__getattr__("logical_k_proj")
        per_layer = [
            weight.view(self.num_kv_heads, self.head_dim, self.hidden_size).permute(0, 2, 1)
            for weight in layers
        ]
        return torch.stack(per_layer, dim=0).reshape(self.num_experts, self.hidden_size, self.head_dim)

    def _logical_v_proj_bank(self) -> torch.Tensor:
        layers = super().__getattr__("logical_v_proj")
        per_layer = [
            weight.view(self.num_kv_heads, self.head_dim, self.hidden_size).permute(0, 2, 1)
            for weight in layers
        ]
        return torch.stack(per_layer, dim=0).reshape(self.num_experts, self.hidden_size, self.head_dim)

    def _logical_o_proj_bank(self) -> torch.Tensor:
        layers = super().__getattr__("logical_o_proj")
        per_layer = [
            weight.view(self.hidden_size, self.num_kv_heads, self.q_group_dim).permute(1, 2, 0)
            for weight in layers
        ]
        return torch.stack(per_layer, dim=0).reshape(self.num_experts, self.q_group_dim, self.hidden_size)

    def _make_flat_bank_router(self, input_dim, top_k, num_experts=None):
        """Create a router for flat bank modes — selects top_k experts from the pool."""
        if num_experts is None:
            num_experts = self.num_experts
        from types import SimpleNamespace

        base_cfg = SimpleNamespace(
            hidden_size=input_dim,
            num_local_experts=num_experts,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            norm_topk_prob=getattr(self.config, "norm_topk_prob", True),
            router_exploration_rate=getattr(self.config, "router_exploration_rate", 0.0),
        )
        if self.use_deepseek_routing:
            num_groups = getattr(self.config, "num_groups", None)
            group_topk = getattr(self.config, "group_topk", None)
            # Disable group-limited routing if it can't provide enough candidates
            if num_groups and group_topk:
                experts_per_group = num_experts // num_groups
                max_candidates = group_topk * experts_per_group
                if max_candidates < top_k:
                    num_groups = None
                    group_topk = None
            base_cfg.topk_scaling_factor = getattr(self.config, "topk_scaling_factor", None)
            base_cfg.num_groups = num_groups
            base_cfg.group_topk = group_topk
            return DeepSeekRouter(base_cfg)
        return ExplorationTopKRouter(base_cfg)

    def _route_flat(self, router, x, top_k):
        """Route with explicit top_k for flat bank."""
        if isinstance(router, (DeepSeekRouter, ExplorationTopKRouter)):
            router_probs, weights, idx = router(x)
            return idx, weights, router_probs

        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = router(x.float())
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
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
                proj = self._apply_single_head_norm(proj, norm_weights[e])
            out[mask] = (proj * expert_weights[mask].unsqueeze(-1)).to(out.dtype)
        return out

    def _apply_single_head_norm(self, proj, norm_weight):
        input_dtype = proj.dtype
        proj_f = proj.float()
        var = proj_f.pow(2).mean(-1, keepdim=True)
        proj_normed = proj_f * torch.rsqrt(var + self.eps)
        # Match Qwen3MoeRMSNorm: round the normalized activations back to the
        # projection dtype before multiplying by the learned weight.
        return norm_weight * proj_normed.to(input_dtype)

    def _run_grouped_expert_matmul(
        self,
        sorted_inputs: torch.Tensor,
        sorted_expert: torch.Tensor,
        unique_experts: torch.Tensor,
        counts: torch.Tensor,
        weight_bank: torch.Tensor,
        norm_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            os.environ.get("MOE_EVERYTHING_DISABLE_GROUPED_MM", "0") == "0"
            and os.environ.get("MOE_EVERYTHING_DISABLE_ATTN_GROUPED_MM", "0") == "0"
            and sorted_inputs.is_cuda
            and sorted_inputs.dtype in (torch.bfloat16, torch.float16)
        ):
            proj = triton_grouped_gemm(sorted_inputs.contiguous(), weight_bank.to(sorted_inputs.dtype).contiguous(), unique_experts, counts)
        else:
            proj = torch.empty(
                sorted_inputs.shape[0],
                weight_bank.shape[2],
                device=sorted_inputs.device,
                dtype=sorted_inputs.dtype,
            )
            start = 0
            for expert, count in zip(unique_experts.tolist(), counts.tolist()):
                end = start + count
                if count > 0:
                    proj[start:end] = sorted_inputs[start:end] @ weight_bank[expert]
                start = end

        if norm_weights is not None:
            input_dtype = proj.dtype
            proj_f = proj.float()
            variance = proj_f.pow(2).mean(-1, keepdim=True)
            proj_normed = proj_f * torch.rsqrt(variance + self.eps)
            proj = fp32_index_select(norm_weights, 0, sorted_expert) * proj_normed.to(input_dtype)

        return proj

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

        sorted_inputs = fp32_index_select(flat, 0, sorted_token_idx)
        proj = self._run_grouped_expert_matmul(
            sorted_inputs,
            sorted_expert,
            unique_experts,
            counts,
            weight_bank,
            norm_weights=norm_weights,
        )
        proj = proj * sorted_weight.unsqueeze(-1).to(proj.dtype)
        pair_out = fp32_index_put(pair_out, sort_order, proj.to(pair_out.dtype))

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
        sorted_inputs = fp32_index_select(pair_inputs, 0, sort_order)
        sorted_weight = pair_weight[sort_order]
        sorted_token_idx = token_idx[sort_order]
        unique_experts, counts = torch.unique_consecutive(sorted_expert, return_counts=True)

        if reduce_tokens:
            # When we sum per-slot O projections back into token outputs, low-precision
            # accumulation can drift noticeably from the monolithic dense O projection
            # used by the baseline models. Keep the reduction in fp32, then cast back.
            accum_dtype = (
                torch.float32
                if inputs.dtype in (torch.bfloat16, torch.float16)
                else inputs.dtype
            )
            token_out = torch.zeros(N, H_out, device=inputs.device, dtype=accum_dtype)
        else:
            pair_out = inputs.new_zeros(pair_inputs.shape[0], H_out)

        proj = self._run_grouped_expert_matmul(
            sorted_inputs,
            sorted_expert,
            unique_experts,
            counts,
            weight_bank,
            norm_weights=norm_weights,
        )
        proj_output_dtype = proj.dtype
        if reduce_tokens:
            proj = proj.to(token_out.dtype)
            proj = proj * sorted_weight.unsqueeze(-1).to(token_out.dtype)
            token_out = fp32_index_add(token_out, 0, sorted_token_idx, proj)
        else:
            proj = proj * sorted_weight.unsqueeze(-1).to(proj.dtype)
            proj = proj.to(inputs.dtype)
            pair_out = fp32_index_put(pair_out, sort_order, proj)

        if reduce_tokens:
            return token_out.to(proj_output_dtype)
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

    def _project_grouped_query_heads_batched(self, flat, weight_bank, idx, weights, norm_weights=None):
        """Project grouped-query outputs for per_head_precompute_kv GQA routing."""
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

        sorted_inputs = fp32_index_select(flat, 0, sorted_token_idx)
        proj = self._run_grouped_expert_matmul(
            sorted_inputs,
            sorted_expert,
            unique_experts,
            counts,
            weight_bank,
            norm_weights=None,
        )

        if norm_weights is not None:
            input_dtype = proj.dtype
            proj_f = proj.view(-1, self.q_heads_per_kv, self.head_dim).float()
            variance = proj_f.pow(2).mean(-1, keepdim=True)
            proj_normed = proj_f * torch.rsqrt(variance + self.eps)
            gathered = fp32_index_select(norm_weights, 0, sorted_expert).unsqueeze(1)
            proj = (gathered * proj_normed.to(input_dtype)).view(-1, H_out)

        proj = proj * sorted_weight.unsqueeze(-1).to(proj.dtype)
        pair_out = fp32_index_put(pair_out, sort_order, proj.to(pair_out.dtype))
        return pair_out.view(N, num_slots, H_out)

    def _init_per_head_fully_independent(self):
        """Flat-bank Q/K/V/O with GQA: Q bundled per KV group, K/V/O per head.

        Q router picks top-num_kv_heads from E (each expert produces q_group_dim),
        K/V pick top-num_kv_heads from E,
        O router picks top-num_heads from E (each expert takes head_dim).
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
        # Q routes per KV-group, K/V route per KV-head — same pool size.
        # O routes per head from a larger pool (E_o = E * heads_per_group)
        # so that O total params match the standard per-layer O projection.
        E_kv = E
        E_o = E * self.q_heads_per_kv
        self.num_kv_experts = E_kv
        self.num_o_experts = E_o
        if self.per_layer_attn_router:
            self.q_routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_kv_heads) for _ in range(self.num_depths)])
            self.k_routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv) for _ in range(self.num_depths)])
            self.v_routers = nn.ModuleList([self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv) for _ in range(self.num_depths)])
            self.o_routers = nn.ModuleList([self._make_flat_bank_router(self.q_dim, self.num_heads, num_experts=E_o) for _ in range(self.num_depths)])
        else:
            self.q_router = self._make_flat_bank_router(self.hidden_size, self.num_kv_heads)
            self.k_router = self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv)
            self.v_router = self._make_flat_bank_router(self.hidden_size, self.num_kv_heads, num_experts=E_kv)
            self.o_router = self._make_flat_bank_router(self.q_dim, self.num_heads, num_experts=E_o)
        self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_group_dim))
        self.k_proj = nn.Parameter(torch.empty(E_kv, self.hidden_size, self.head_dim))
        self.v_proj = nn.Parameter(torch.empty(E_kv, self.hidden_size, self.head_dim))
        self.o_proj = nn.Parameter(torch.empty(E_o, self.head_dim, self.hidden_size))
        if self.per_layer_qk_norm:
            self.layer_q_norm_weight = nn.Parameter(torch.ones(self.num_depths, self.head_dim))
            self.layer_k_norm_weight = nn.Parameter(torch.ones(self.num_depths, self.head_dim))
        else:
            self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
            self.k_norm_weight = nn.Parameter(torch.ones(E_kv, self.head_dim))
        self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _init_per_head_precompute_kv(self):
        """Flat-bank precompute KV with GQA-group routing.

        A single router picks top-num_kv_heads experts from E. Each selected
        expert provides one KV head plus the grouped-query slice that attends
        against it.
        """
        if self.routed_norm:
            self.norm = NormExpertBank(self.num_depths, self.hidden_size, eps=self.eps)
        elif self.per_layer_norm:
            self.norms = nn.ModuleList([Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps) for _ in range(self.num_depths)])
        else:
            self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        if self.per_layer_attn_router:
            self.routers = nn.ModuleList(
                [self._make_flat_bank_router(self.hidden_size, self.num_kv_heads) for _ in range(self.num_depths)]
            )
        else:
            self.router = self._make_flat_bank_router(self.hidden_size, self.num_kv_heads)
        if self.sanity_check_mode == "alternating_global_moe":
            num_logical_layers = max(1, self.num_depths // 2)
            self.logical_q_proj = nn.ParameterList(
                [nn.Parameter(torch.empty(self.q_dim, self.hidden_size)) for _ in range(num_logical_layers)]
            )
            self.logical_k_proj = nn.ParameterList(
                [nn.Parameter(torch.empty(self.kv_dim, self.hidden_size)) for _ in range(num_logical_layers)]
            )
            self.logical_v_proj = nn.ParameterList(
                [nn.Parameter(torch.empty(self.kv_dim, self.hidden_size)) for _ in range(num_logical_layers)]
            )
            self.logical_o_proj = nn.ParameterList(
                [nn.Parameter(torch.empty(self.hidden_size, self.q_dim)) for _ in range(num_logical_layers)]
            )
            self.logical_q_norm_weight = nn.Parameter(torch.ones(num_logical_layers, self.head_dim))
            self.logical_k_norm_weight = nn.Parameter(torch.ones(num_logical_layers, self.head_dim))
            expert_to_layer = torch.arange(E, dtype=torch.long) // max(1, self.num_kv_heads)
            expert_to_layer.clamp_(max=num_logical_layers - 1)
            self.register_buffer(
                "logical_qk_norm_layer_index",
                expert_to_layer,
                persistent=False,
            )
        elif self.per_layer_qk_norm:
            self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_group_dim))
            self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
            self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
            self.o_proj = nn.Parameter(torch.empty(E, self.q_group_dim, self.hidden_size))
            self.layer_q_norm_weight = nn.Parameter(torch.ones(self.num_depths, self.head_dim))
            self.layer_k_norm_weight = nn.Parameter(torch.ones(self.num_depths, self.head_dim))
        else:
            self.q_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.q_group_dim))
            self.k_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
            self.v_proj = nn.Parameter(torch.empty(E, self.hidden_size, self.head_dim))
            self.o_proj = nn.Parameter(torch.empty(E, self.q_group_dim, self.hidden_size))
            self.q_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
            self.k_norm_weight = nn.Parameter(torch.ones(E, self.head_dim))
        if self.sanity_check_mode == "alternating_global_moe":
            self._init_params(
                [
                    *list(self.logical_q_proj),
                    *list(self.logical_k_proj),
                    *list(self.logical_v_proj),
                    *list(self.logical_o_proj),
                ]
            )
        else:
            self._init_params([self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def _select_router(self, name: str, depth_idx: int | None = None):
        plural_name = f"{name}s"
        if self.per_layer_attn_router and depth_idx is not None and hasattr(self, plural_name):
            return getattr(self, plural_name)[depth_idx]
        return getattr(self, name)

    def _get_q_norm_weight_bank(self, depth_idx: int | None, num_experts: int) -> torch.Tensor:
        if hasattr(self, "logical_q_norm_weight"):
            return self.logical_q_norm_weight.index_select(0, self.logical_qk_norm_layer_index)
        if hasattr(self, "layer_q_norm_weight"):
            if depth_idx is None:
                raise ValueError("per_layer_qk_norm requires depth_idx")
            return self.layer_q_norm_weight[depth_idx].unsqueeze(0).expand(num_experts, -1)
        return self.q_norm_weight

    def _get_k_norm_weight_bank(self, depth_idx: int | None, num_experts: int) -> torch.Tensor:
        if hasattr(self, "logical_k_norm_weight"):
            return self.logical_k_norm_weight.index_select(0, self.logical_qk_norm_layer_index)
        if hasattr(self, "layer_k_norm_weight"):
            if depth_idx is None:
                raise ValueError("per_layer_qk_norm requires depth_idx")
            return self.layer_k_norm_weight[depth_idx].unsqueeze(0).expand(num_experts, -1)
        return self.k_norm_weight

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
        base = (logical_layer * num_slots) % self.num_experts
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
        threshold = self.per_head_dense_fraction_threshold
        if threshold == DEFAULT_PER_HEAD_DENSE_FRACTION_THRESHOLD:
            threshold = AUTO_PER_HEAD_SPARSE_THRESHOLDS.get(self.mode, threshold)
        return attn_fraction < threshold

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

    def _attach_token_mask_to_last_router_info(self, token_mask: torch.Tensor) -> None:
        flat_mask = token_mask.reshape(-1).bool().detach()
        for info in self.last_router_info.values():
            info["token_mask"] = flat_mask
            if info["router_logits"].shape[0] != flat_mask.numel():
                continue
            info["router_logits"] = info["router_logits"].clone()
            info["router_logits"][~flat_mask] = 0
            info["selected_experts"] = info["selected_experts"].clone()
            info["selected_experts"][~flat_mask] = 0

    def _zero_dummy(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dummy = torch.zeros((), device=device, dtype=dtype)
        for param in self.parameters():
            dummy = dummy + param.reshape(-1)[0].to(dtype) * 0.0
        return dummy

    def project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-routing-group norm, then compute Q, K_fresh, V_fresh via expert routing."""
        B, T, H = hidden_states.shape
        self.last_router_info = {}

        if self.mode == "per_head_fully_independent":
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

            q_idx, q_w, q_probs = self._route_flat(q_router, q_flat, self.num_kv_heads)
            k_idx, k_w, k_probs = self._route_flat(k_router, k_flat, self.num_kv_heads)
            v_idx, v_w, v_probs = self._route_flat(v_router, v_flat, self.num_kv_heads)
            self._store_router_info("q", q_probs, q_idx)
            self._store_router_info("k", k_probs, k_idx)
            self._store_router_info("v", v_probs, v_idx)

            q_norm_weight = self._get_q_norm_weight_bank(depth_idx, self.num_experts)
            k_norm_weight = self._get_k_norm_weight_bank(depth_idx, self.num_kv_experts)
            pq_w = q_w if self.scale_attn_by_routing_weight else _straight_through_ones(q_w)
            pk_w = k_w if self.scale_attn_by_routing_weight else _straight_through_ones(k_w)
            pv_w = v_w if self.scale_attn_by_routing_weight else _straight_through_ones(v_w)
            # Q: bundled per GQA group — (N, num_kv_heads, q_group_dim)
            Q_groups = self._project_grouped_query_heads_batched(q_flat, self.q_proj, q_idx, pq_w, q_norm_weight)
            # Reshape from (N, num_kv_heads, q_group_dim) to (N, num_heads, head_dim)
            Q = Q_groups.reshape(B * T, self.num_heads, self.head_dim).reshape(B * T, self.q_dim)
            K_heads = self._project_heads_batched(k_flat, self.k_proj, k_idx, pk_w, k_norm_weight)
            V_heads = self._project_heads_batched(v_flat, self.v_proj, v_idx, pv_w)

            K = K_heads.reshape(B * T, self.num_kv_heads * self.head_dim)
            V = V_heads.reshape(B * T, self.num_kv_heads * self.head_dim)

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

        attn_output = self._run_attention(Q, K, V, attention_mask)

        assert self.mode == "per_head_fully_independent"
        N = B * T
        attn_heads = attn_output.transpose(1, 2).reshape(N, self.num_heads, self.head_dim)
        attn_flat = attn_heads.reshape(N, self.q_dim)
        o_router = self._select_router("o_router", depth_idx)
        o_idx, o_w, o_probs = self._route_flat(o_router, attn_flat, self.num_heads)
        self._store_router_info("o", o_probs, o_idx)
        po_w = o_w if self.scale_attn_by_routing_weight else _straight_through_ones(o_w)
        o_out = self._project_pair_inputs_grouped(
            attn_heads,
            self.o_proj,
            o_idx,
            po_w,
            reduce_tokens=True,
        )
        return o_out.view(B, T, self.hidden_size)

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
                    ("q", self.num_experts, self.num_kv_heads),
                    ("k", self.num_kv_experts, self.num_kv_heads),
                    ("v", self.num_kv_experts, self.num_kv_heads),
                    ("o", self.num_o_experts, self.num_heads),
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
        q_idx, q_w, q_probs = self._route_flat(q_router, q_flat, self.num_kv_heads)
        k_idx, k_w, k_probs = self._route_flat(k_router, k_flat, self.num_kv_heads)
        v_idx, v_w, v_probs = self._route_flat(v_router, v_flat, self.num_kv_heads)
        self._store_router_info("q", q_probs, q_idx, token_mask=flat_mask)
        self._store_router_info("k", k_probs, k_idx, token_mask=flat_mask)
        self._store_router_info("v", v_probs, v_idx, token_mask=flat_mask)

        q_norm_weight = self._get_q_norm_weight_bank(depth_idx, self.num_experts)
        k_norm_weight = self._get_k_norm_weight_bank(depth_idx, self.num_kv_experts)
        pq_w = q_w if self.scale_attn_by_routing_weight else _straight_through_ones(q_w)
        pk_w = k_w if self.scale_attn_by_routing_weight else _straight_through_ones(k_w)
        pv_w = v_w if self.scale_attn_by_routing_weight else _straight_through_ones(v_w)
        # Q: bundled per GQA group — (N_sel, num_kv_heads, q_group_dim)
        Q_groups = self._project_grouped_query_heads_batched(q_flat, self.q_proj, q_idx, pq_w, q_norm_weight)
        # Reshape from (N_sel, num_kv_heads, q_group_dim) to (N_sel, num_heads, head_dim)
        Q_sel = Q_groups.reshape(-1, self.num_heads, self.head_dim)
        K_sel = self._project_heads_batched(k_flat, self.k_proj, k_idx, pk_w, k_norm_weight)
        V_sel = self._project_heads_batched(v_flat, self.v_proj, v_idx, pv_w)

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

        attn_heads = hidden_states.new_zeros(B, self.num_heads, T, self.head_dim)
        token_mask_2d = token_mask.squeeze(-1).bool()

        for b in range(B):
            pos = token_mask_2d[b].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            Q_b = Q[b : b + 1, :, pos, :]
            attn_b = self._run_attention(
                Q_b,
                K_new[b : b + 1],
                V_new[b : b + 1],
                attention_mask[:, :, pos, :] if attention_mask is not None else None,
                query_positions=None if attention_mask is not None else pos,
            )
            attn_heads[b, :, pos, :] = attn_b.squeeze(0).to(attn_heads.dtype)

        attn_selected = attn_heads.transpose(1, 2).reshape(B * T, self.num_heads, self.head_dim)[flat_mask]
        attn_flat = attn_selected.reshape(attn_selected.shape[0], self.q_dim)
        o_idx, o_w, o_probs = self._route_flat(o_router, attn_flat, self.num_heads)
        self._store_router_info("o", o_probs, o_idx, token_mask=flat_mask)
        po_w = o_w if self.scale_attn_by_routing_weight else _straight_through_ones(o_w)

        o_selected = self._project_pair_inputs_grouped(
            attn_selected,
            self.o_proj,
            o_idx,
            po_w,
            reduce_tokens=True,
        )

        attn_out = hidden_states.new_zeros(B * T, self.hidden_size)
        attn_out[flat_mask] = o_selected.to(attn_out.dtype)
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
            N, self.num_kv_heads, depth_idx, hidden_states.device, dtype=flat.dtype
        )
        if routed is None:
            router = self._select_router("router", depth_idx)
            idx, w, probs = self._route_flat(router, flat, self.num_kv_heads)
            self._store_router_info("attn", probs, idx)
        else:
            idx, w, probs = routed

        q_norm_weight = self._get_q_norm_weight_bank(depth_idx, self.num_experts)
        k_norm_weight = self._get_k_norm_weight_bank(depth_idx, self.num_experts)
        q_w = w if self.scale_attn_by_routing_weight else _straight_through_ones(w)
        kv_w = w if self.scale_attn_by_routing_weight else _straight_through_ones(w)
        Q_groups = self._project_grouped_query_heads_batched(flat, self.q_proj, idx, q_w, q_norm_weight)
        K_heads = self._project_heads_batched(flat, self.k_proj, idx, kv_w, k_norm_weight)
        V_heads = self._project_heads_batched(flat, self.v_proj, idx, kv_w)

        Q = (
            Q_groups.view(B, T, self.num_kv_heads, self.q_heads_per_kv, self.head_dim)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K_fresh = K_heads.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_fresh = V_heads.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        Q, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        return {
            "flat": flat,
            "idx": idx,
            "weights": w,
            "kv_weight": w,
            "Q": Q,
            "K_fresh": K_fresh,
            "V_fresh": V_fresh,
        }

    def _project_sanity_logical_o(
        self,
        attn_output: torch.Tensor,
        depth_idx: int | None,
    ) -> torch.Tensor | None:
        if self.sanity_check_mode != "alternating_global_moe" or depth_idx is None:
            return None
        logical_layer = depth_idx // 2
        logical_o_proj = super().__getattr__("logical_o_proj")[logical_layer]
        return F.linear(attn_output, logical_o_proj)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb_k_only(
        self,
        k: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (k * cos) + (self._rotate_half(k) * sin)

    def _run_per_head_precompute_kv_expert_tables(
        self,
        tables: dict[str, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        query_token_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        """Run attention against expert-specific KV tables for routed query groups."""
        flat = tables["flat"]
        idx = tables["idx"]
        Q = tables["Q"]
        B, _, T, _ = Q.shape
        N = B * T

        if query_token_mask is None:
            query_group_mask = torch.ones((B, self.num_kv_heads, T), device=flat.device, dtype=torch.bool)
        else:
            query_group_mask = query_token_mask.squeeze(-1).bool().unsqueeze(1).expand(-1, self.num_kv_heads, -1)

        if not query_group_mask.any():
            return flat.new_zeros(B, self.num_heads, T, self.head_dim)

        active_experts = idx.view(B, T, self.num_kv_heads).permute(0, 2, 1)[query_group_mask].unique()
        slot_experts = idx.view(B, T, self.num_kv_heads).permute(0, 2, 1)
        k_norm_weight = self._get_k_norm_weight_bank(depth_idx, self.num_experts)

        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        for expert in active_experts.tolist():
            K_e = flat @ self.k_proj[expert]
            V_e = flat @ self.v_proj[expert]

            K_e = self._apply_single_head_norm(K_e, k_norm_weight[expert])
            K_e_heads = K_e.view(B, T, 1, self.head_dim).transpose(1, 2)
            V_e_heads = V_e.view(B, T, 1, self.head_dim).transpose(1, 2)
            K_e_heads = self._apply_rotary_pos_emb_k_only(K_e_heads, position_embeddings)
            # Dense per-expert attention is intentionally kept here. It does
            # redundant work on query groups routed to other experts, but on
            # GPU it is markedly faster than launching many ragged attention
            # kernels over the routed subsets.
            attn_e = self._run_attention(
                Q,
                K_e_heads.expand(-1, self.num_kv_heads, -1, -1),
                V_e_heads.expand(-1, self.num_kv_heads, -1, -1),
                attention_mask,
            )

            group_mask = (slot_experts == expert) & query_group_mask
            head_mask = group_mask.unsqueeze(-1).repeat_interleave(self.num_kv_groups, dim=1)
            attn_output = attn_output + attn_e * head_mask.to(attn_e.dtype)

        return attn_output

    def _apply_logical_head_norm(
        self,
        proj: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = proj.dtype
        proj_f = proj.float()
        variance = proj_f.pow(2).mean(-1, keepdim=True)
        proj_normed = proj_f * torch.rsqrt(variance + self.eps)
        return norm_weight * proj_normed.to(input_dtype)

    def _project_and_attend_sanity_logical_dense(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        depth_idx: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.sanity_check_mode != "alternating_global_moe" or depth_idx is None:
            return None
        if not bool(token_mask.bool().all().item()):
            return None

        B, T, H = hidden_states.shape
        logical_layer = depth_idx // 2
        if self.per_layer_norm and depth_idx is not None:
            normed = self.norms[depth_idx](hidden_states)
        else:
            normed = self.norm(hidden_states)

        self.last_router_info = {}
        self._maybe_build_sanity_attention_routing(
            B * T,
            self.num_kv_heads,
            depth_idx,
            hidden_states.device,
            dtype=normed.dtype,
        )

        logical_q_proj = super().__getattr__("logical_q_proj")[logical_layer]
        logical_k_proj = super().__getattr__("logical_k_proj")[logical_layer]
        logical_v_proj = super().__getattr__("logical_v_proj")[logical_layer]
        q_norm_weight = super().__getattr__("logical_q_norm_weight")[logical_layer]
        k_norm_weight = super().__getattr__("logical_k_norm_weight")[logical_layer]

        Q = F.linear(normed, logical_q_proj).view(B, T, self.num_heads, self.head_dim)
        K_fresh = F.linear(normed, logical_k_proj).view(B, T, self.num_kv_heads, self.head_dim)
        V_fresh = F.linear(normed, logical_v_proj).view(B, T, self.num_kv_heads, self.head_dim)

        Q = self._apply_logical_head_norm(Q, q_norm_weight).transpose(1, 2)
        K_fresh = self._apply_logical_head_norm(K_fresh, k_norm_weight).transpose(1, 2)
        V_fresh = V_fresh.transpose(1, 2)

        cos, sin = position_embeddings
        Q, K_fresh = apply_rotary_pos_emb(Q, K_fresh, cos, sin)

        attn_mask_kv = token_mask.unsqueeze(1)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)
        attn_output = self._run_attention(
            Q,
            K_new,
            V_new,
            attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.q_dim)
        o_out = self._project_sanity_logical_o(attn_output, depth_idx)
        return o_out, K_new, V_new

    def _run_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        query_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None and query_positions is not None:
            attention_mask = self._build_query_position_mask(
                query_positions,
                key_length=K.shape[-2],
                device=Q.device,
                dtype=Q.dtype,
            )
        return self._run_attention_backend(Q, K, V, attention_mask)

    def _build_query_position_mask(
        self,
        query_positions: torch.Tensor,
        *,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key_positions = torch.arange(key_length, device=device)
        future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        mask = torch.zeros(
            (query_positions.shape[0], key_length),
            device=device,
            dtype=dtype,
        )
        mask.masked_fill_(future_mask, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def _run_attention_backend(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, _ = attention_interface(
            self,
            Q,
            K,
            V,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )
        return attn_output.transpose(1, 2).contiguous()

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
        flat_mask = token_mask.reshape(-1).bool()
        if not flat_mask.any():
            return self._empty_sparse_attn_result(
                hidden_states,
                K_old,
                V_old,
                [("attn", self.num_experts, self.num_kv_heads)],
            )
        sanity_dense = self._project_and_attend_sanity_logical_dense(
            hidden_states,
            position_embeddings,
            K_old,
            V_old,
            token_mask,
            attention_mask,
            depth_idx,
        )
        if sanity_dense is not None:
            return sanity_dense
        self.last_router_info = {}
        tables = self._build_per_head_precompute_kv_tables(
            hidden_states, position_embeddings, depth_idx=depth_idx
        )
        idx = tables["idx"]
        w = tables["weights"]
        K_fresh = tables["K_fresh"]
        V_fresh = tables["V_fresh"]

        attn_mask_kv = token_mask.unsqueeze(1)
        K_new = torch.where(attn_mask_kv, K_fresh, K_old)
        V_new = torch.where(attn_mask_kv, V_fresh, V_old)
        self._attach_token_mask_to_last_router_info(token_mask)
        attn_output = self._run_per_head_precompute_kv_expert_tables(
            tables,
            position_embeddings,
            attention_mask=attention_mask,
            query_token_mask=token_mask,
            depth_idx=depth_idx,
        )

        if self.scale_attn_by_routing_weight:
            kv_weight = hidden_states.new_zeros(B * T, self.num_kv_heads)
            kv_weight[flat_mask] = w[flat_mask]
            kv_weight_expanded = (
                kv_weight.view(B, T, self.num_kv_heads)
                .transpose(1, 2)
                .repeat_interleave(self.num_kv_groups, dim=1)
                .unsqueeze(-1)
            )
            attn_output = attn_output * kv_weight_expanded.to(attn_output.dtype)
        attn_dense = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.q_dim)
        logical_o = self._project_sanity_logical_o(attn_dense, depth_idx)
        if logical_o is not None:
            return logical_o, K_new, V_new
        attn_groups = attn_dense.reshape(B * T, self.num_kv_heads, self.q_group_dim)
        o_w = w[flat_mask] if self.scale_attn_by_routing_weight else _straight_through_ones(w[flat_mask])
        o_selected = self._project_pair_inputs_grouped(
            attn_groups[flat_mask],
            self.o_proj,
            idx[flat_mask],
            o_w,
            reduce_tokens=True,
        )
        o_out = hidden_states.new_zeros(B * T, self.hidden_size)
        o_out[flat_mask] = o_selected.to(o_out.dtype)
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
        flat_mask = token_mask.reshape(-1).bool()
        if not flat_mask.any():
            return self._empty_sparse_attn_result(
                hidden_states,
                K_old,
                V_old,
                [("attn", self.num_experts, self.num_kv_heads)],
            )
        return self.project_and_attend_per_head_precompute_kv_dense_mixed(
            hidden_states,
            position_embeddings,
            K_old,
            V_old,
            token_mask,
            attention_mask,
            depth_idx,
        )

    def project_and_attend_per_head_precompute_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat-bank precompute KV with one routed expert per KV group."""
        self.last_router_info = {}
        tables = self._build_per_head_precompute_kv_tables(
            hidden_states, position_embeddings, depth_idx=depth_idx
        )
        idx = tables["idx"]
        w = tables["weights"]
        kv_weight = tables["kv_weight"]
        K_fresh = tables["K_fresh"]
        V_fresh = tables["V_fresh"]
        B, _, T, _ = tables["Q"].shape

        attn_output = self._run_per_head_precompute_kv_expert_tables(
            tables,
            position_embeddings,
            attention_mask=attention_mask,
            depth_idx=depth_idx,
        )

        if self.scale_attn_by_routing_weight:
            kv_weight_expanded = (
                kv_weight.view(B, T, self.num_kv_heads)
                .transpose(1, 2)
                .repeat_interleave(self.num_kv_groups, dim=1)
                .unsqueeze(-1)
            )
            attn_output = attn_output * kv_weight_expanded.to(attn_output.dtype)
        attn_dense = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.q_dim)
        logical_o = self._project_sanity_logical_o(attn_dense, depth_idx)
        if logical_o is not None:
            return logical_o, K_fresh, V_fresh
        attn_flat = attn_dense.reshape(B * T, self.q_dim)
        o_w = w if self.scale_attn_by_routing_weight else _straight_through_ones(w)
        o_out = self._project_pair_inputs_grouped(
            attn_flat.view(B * T, self.num_kv_heads, self.q_group_dim),
            self.o_proj,
            idx,
            o_w,
            reduce_tokens=True,
        )

        return o_out.view(B, T, self.hidden_size), K_fresh, V_fresh


# ─── MLP expert bank ──────────────────────────────────────────────────────── #

