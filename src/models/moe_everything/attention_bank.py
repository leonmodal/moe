"""Attention expert bank: NormExpertBank and AttentionExpertBank.

Per-head routed attention with shared expert weight banks across depths.
Modes: per_head_no_recompute, per_head_recompute_k, per_head_recompute_kv.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.models.fp32_routing import fp32_index_add, fp32_index_put, fp32_index_select
try:
    from src.models.triton_grouped_gemm import triton_grouped_gemm
except ModuleNotFoundError:  # triton is CUDA-only; fall back when unavailable.
    triton_grouped_gemm = None
from src.models.modeling_qwen3_moe import (
    Qwen3MoeRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from src.models.router import is_checkpoint_recompute
from src.models.routing.routers import _straight_through_ones
from src.models.routing.helpers import make_top1_router
from .config import MoEverythingConfig

ATTN_EXPERT_MODES = {
    "per_head_no_recompute",
    "per_head_recompute_k",
    "per_head_recompute_kv",
}
ATTN_ROUTING_BUNDLES = {
    "q_k_v_o",
    "qk_v_o",
    "qk_vo",
    "qkv_o",
    "qkvo",
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
        self.routing_bundle = getattr(
            config,
            "attn_routing_bundle",
            "q_k_v_o" if self.mode == "per_head_no_recompute" else "qkvo",
        )
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
        self.sanity_check_mode = getattr(config, "sanity_check_mode", None)
        self.scale_attn_by_routing_weight = getattr(config, "scale_attn_by_routing_weight", False)
        self.attn_router_context = getattr(config, "attn_router_context", "none")
        self.attn_router_context_decay = float(getattr(config, "attn_router_context_decay", 0.95))
        if self.attn_router_context not in {"none", "ema_qk_v"}:
            raise ValueError(
                "attn_router_context must be one of {'none', 'ema_qk_v'}, "
                f"got {self.attn_router_context!r}"
            )
        if not (0.0 <= self.attn_router_context_decay < 1.0):
            raise ValueError(
                "attn_router_context_decay must be in [0, 1), "
                f"got {self.attn_router_context_decay}"
            )
        self.attn_router_input_dim = (
            self.hidden_size * 2
            if self.attn_router_context == "ema_qk_v"
            else self.hidden_size
        )
        self.last_router_info = {}
        # Attention-map capture for synthetic-task eval. When
        # `capture_attention_maps` is True, `_run_per_head_recompute_expert_tables`
        # populates `last_attention_maps` with a list of per-(k_expert, v_expert)
        # entries containing the softmax attention weights and the routing mask
        # that determines which tokens used that pair. Disabled by default to
        # keep the production forward path on `F.scaled_dot_product_attention`.
        self.capture_attention_maps = False
        self.last_attention_maps: list[dict] = []

        # Per-(q, k) V-routing: when enabled, the V expert at attended
        # position k for query q is selected by a learned router taking
        # (hidden[q], hidden[k]) as input. K still routes per-query.
        # Compose the per-pair score from two separable linear maps so
        # we never materialize a [B, T, T, num_experts] tensor:
        #   score(q, k, e) = W_q · hidden[q] [e] + W_k · hidden[k] [e]
        self.per_pair_v_routing = getattr(config, "per_pair_v_routing", False)
        if self.per_pair_v_routing:
            if self.mode != "per_head_recompute_kv":
                raise ValueError(
                    "per_pair_v_routing requires attn_expert_mode='per_head_recompute_kv'"
                )
            num_v_experts = self.num_experts
            if self.per_layer_attn_router:
                self.pair_v_router_q = nn.ModuleList(
                    [nn.Linear(self.hidden_size, num_v_experts, bias=False) for _ in range(self.num_depths)]
                )
                self.pair_v_router_k = nn.ModuleList(
                    [nn.Linear(self.hidden_size, num_v_experts, bias=False) for _ in range(self.num_depths)]
                )
            else:
                self.pair_v_router_q = nn.Linear(self.hidden_size, num_v_experts, bias=False)
                self.pair_v_router_k = nn.Linear(self.hidden_size, num_v_experts, bias=False)

        if self.mode not in ATTN_EXPERT_MODES:
            raise ValueError(
                f"Unknown attn_expert_mode: {self.mode}, must be one of {ATTN_EXPERT_MODES}"
            )
        if self.routing_bundle not in ATTN_ROUTING_BUNDLES:
            raise ValueError(
                f"Unknown attn_routing_bundle: {self.routing_bundle}, "
                f"must be one of {ATTN_ROUTING_BUNDLES}"
            )
        if self.mode == "per_head_no_recompute" and self.routing_bundle != "q_k_v_o":
            raise ValueError(
                "per_head_no_recompute currently supports only "
                "attn_routing_bundle='q_k_v_o'."
            )
        if self.mode in {"per_head_recompute_k", "per_head_recompute_kv"}:
            if not self.routing_bundle.startswith("qk"):
                raise ValueError(
                    f"{self.mode} requires Q/K to share a route; got "
                    f"attn_routing_bundle={self.routing_bundle!r}."
                )
        if self.sanity_check_mode is not None and self.mode != "per_head_recompute_kv":
            raise ValueError(
                "sanity_check_mode is only supported with attn_expert_mode='per_head_recompute_kv'"
            )
        getattr(self, f"_init_{self.mode}")()

    def _causal_prefix_ema(self, x: torch.Tensor) -> torch.Tensor:
        """Return EMA of previous tokens only, preserving causal routing."""
        B, T, H = x.shape
        state = x.new_zeros(B, H)
        states = []
        decay = self.attn_router_context_decay
        update = 1.0 - decay
        for t in range(T):
            states.append(state)
            state = state * decay + x[:, t, :] * update
        return torch.stack(states, dim=1)

    def _attn_router_inputs(self, normed_states: torch.Tensor) -> torch.Tensor:
        if self.attn_router_context == "none":
            return normed_states.reshape(-1, self.hidden_size)
        prefix_ema = self._causal_prefix_ema(normed_states)
        return torch.cat((normed_states, prefix_ema), dim=-1).reshape(
            -1,
            self.hidden_size * 2,
        )

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

    def _route_per_head(self, routers, x, depth_idx=None):
        """Per-head top-1 routing: H routers each pick 1 expert.

        Returns (idx, weights, probs_list) where:
        - idx: (N, H) expert indices (one per head)
        - weights: (N, H) routing weights
        - probs_list: list of H (N, E) prob tensors for aux loss
        """
        if isinstance(routers, nn.ModuleList) and len(routers) > 0 and isinstance(routers[0], nn.ModuleList):
            # Per-depth routers: select the depth-specific set
            routers = routers[depth_idx] if depth_idx is not None else routers[0]

        H = len(routers)
        all_idx = []
        all_weights = []
        all_probs = []
        for h in range(H):
            probs_h, weights_h, idx_h = routers[h](x)  # top-1: weights (N,1), idx (N,1)
            all_idx.append(idx_h.squeeze(-1))      # (N,)
            all_weights.append(weights_h.squeeze(-1))  # (N,)
            all_probs.append(probs_h)              # (N, E)
        idx = torch.stack(all_idx, dim=1)          # (N, H)
        weights = torch.stack(all_weights, dim=1)  # (N, H)
        return idx, weights, all_probs

    def _route_per_slot_inputs(self, routers, x_slots, depth_idx=None):
        """Per-slot top-1 routing where each slot has its own input vector.

        Args:
            routers: H routers, optionally nested by depth.
            x_slots: (N, H, D) slot-specific inputs.

        Returns:
            idx: (N, H)
            weights: (N, H)
            probs_list: list of H tensors with shape (N, E)
        """
        if isinstance(routers, nn.ModuleList) and len(routers) > 0 and isinstance(routers[0], nn.ModuleList):
            routers = routers[depth_idx] if depth_idx is not None else routers[0]

        num_slots = len(routers)
        if x_slots.shape[1] != num_slots:
            raise ValueError(
                f"router/input slot mismatch: {num_slots} routers for "
                f"{x_slots.shape[1]} input slots"
            )
        all_idx = []
        all_weights = []
        all_probs = []
        for h in range(num_slots):
            probs_h, weights_h, idx_h = routers[h](x_slots[:, h, :])
            all_idx.append(idx_h.squeeze(-1))
            all_weights.append(weights_h.squeeze(-1))
            all_probs.append(probs_h)
        idx = torch.stack(all_idx, dim=1)
        weights = torch.stack(all_weights, dim=1)
        return idx, weights, all_probs

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
            triton_grouped_gemm is not None
            and os.environ.get("MOE_EVERYTHING_DISABLE_GROUPED_MM", "0") == "0"
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
        """Project grouped-query outputs for recompute GQA routing."""
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

    def _init_per_head_no_recompute(self):
        """Per-head top-1 routing: H separate routers per projection, each doing top-1.

        Design:
        - num_kv_heads Q-routers (each picks 1 expert producing q_group_dim)
        - num_kv_heads K-routers (each picks 1 expert producing head_dim)
        - num_kv_heads V-routers (each picks 1 expert producing head_dim)
        - num_heads O-routers (each picks 1 expert producing hidden_size from head_dim)
        Total: 3*num_kv_heads + num_heads routers, each doing top-1.

        This is the no-recompute, fully separate routing bundle:
        Q, K, V, and O each choose their own experts.
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
        E_kv = E
        E_o = E * self.q_heads_per_kv
        self.num_kv_experts = E_kv
        self.num_o_experts = E_o

        # H separate top-1 routers per projection.
        H_kv = self.num_kv_heads
        H = self.num_heads
        if self.per_layer_attn_router:
            # Per-depth routers: num_depths sets, each with H routers per projection
            self.q_routers = nn.ModuleList([
                nn.ModuleList([make_top1_router(self.attn_router_input_dim, E, self.config) for _ in range(H_kv)])
                for _ in range(self.num_depths)
            ])
            self.k_routers = nn.ModuleList([
                nn.ModuleList([make_top1_router(self.attn_router_input_dim, E_kv, self.config) for _ in range(H_kv)])
                for _ in range(self.num_depths)
            ])
            self.v_routers = nn.ModuleList([
                nn.ModuleList([make_top1_router(self.attn_router_input_dim, E_kv, self.config) for _ in range(H_kv)])
                for _ in range(self.num_depths)
            ])
            self.o_routers = nn.ModuleList([
                nn.ModuleList([make_top1_router(self.q_dim, E_o, self.config) for _ in range(H)])
                for _ in range(self.num_depths)
            ])
        else:
            # Shared routers across depths: H routers per projection
            self.q_routers = nn.ModuleList([make_top1_router(self.attn_router_input_dim, E, self.config) for _ in range(H_kv)])
            self.k_routers = nn.ModuleList([make_top1_router(self.attn_router_input_dim, E_kv, self.config) for _ in range(H_kv)])
            self.v_routers = nn.ModuleList([make_top1_router(self.attn_router_input_dim, E_kv, self.config) for _ in range(H_kv)])
            self.o_routers = nn.ModuleList([make_top1_router(self.q_dim, E_o, self.config) for _ in range(H)])

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

    def _init_per_head_recompute_k(self):
        self._init_per_head_recompute()

    def _init_per_head_recompute_kv(self):
        self._init_per_head_recompute()

    def _init_per_head_recompute(self):
        """Per-KV-group top-1 routing for recompute attention modes.

        Q/K always share a route. Depending on `attn_routing_bundle`, V and O
        either share that route, share a separate V/O route, or route
        separately. O is token-local; only K or K/V are sequence-side
        recompute tables.
        """
        if self.routed_norm:
            self.norm = NormExpertBank(self.num_depths, self.hidden_size, eps=self.eps)
        elif self.per_layer_norm:
            self.norms = nn.ModuleList([Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps) for _ in range(self.num_depths)])
        else:
            self.norm = Qwen3MoeRMSNorm(self.hidden_size, eps=self.eps)
        E = self.num_experts
        H_kv = self.num_kv_heads
        self.num_kv_experts = E
        self.num_o_experts = E

        def make_hidden_routers():
            return nn.ModuleList([make_top1_router(self.attn_router_input_dim, E, self.config) for _ in range(H_kv)])

        def make_group_routers():
            return nn.ModuleList([make_top1_router(self.q_group_dim, E, self.config) for _ in range(H_kv)])

        if self.per_layer_attn_router:
            self.qk_routers = nn.ModuleList([make_hidden_routers() for _ in range(self.num_depths)])
            if self.routing_bundle in {"qk_v_o", "qk_vo"}:
                self.v_routers = nn.ModuleList([make_hidden_routers() for _ in range(self.num_depths)])
            if self.routing_bundle in {"qk_v_o", "qkv_o"}:
                self.o_routers = nn.ModuleList([make_group_routers() for _ in range(self.num_depths)])
        else:
            self.qk_routers = make_hidden_routers()
            if self.routing_bundle in {"qk_v_o", "qk_vo"}:
                self.v_routers = make_hidden_routers()
            if self.routing_bundle in {"qk_v_o", "qkv_o"}:
                self.o_routers = make_group_routers()
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
        name: str = "qk",
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
            self._store_router_info(name, probs, idx, token_mask=token_mask)
        else:
            self._store_router_info(name, probs, idx)
        return idx, weights, probs

    def _init_params(self, params):
        std = self.config.initializer_range
        for p in params:
            nn.init.normal_(p, mean=0.0, std=std)

    def _store_router_info(
        self,
        name: str,
        router_probs,
        expert_idx: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        # Handle per-head probs list: concatenate into (N*H, E) for aux loss compatibility
        num_head_repeats = 1
        if isinstance(router_probs, list):
            num_head_repeats = len(router_probs)
            router_probs = torch.cat(router_probs, dim=0)  # (N*H, E)
            # Also reshape idx to match: (N, H) -> (N*H, 1)
            if expert_idx.ndim == 2:
                expert_idx = expert_idx.reshape(-1, 1)

        if token_mask is not None:
            flat_mask = token_mask.reshape(-1).bool().to(router_probs.device)
            # Expand mask to match per-head flattened probs: (N,) -> (N*H,)
            if num_head_repeats > 1:
                flat_mask = flat_mask.repeat(num_head_repeats)
            # Functional index_copy keeps gradients flowing from dense_probs
            # back to router_probs and onward to the attention router weight.
            mask_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
            dense_probs = torch.zeros(
                flat_mask.numel(),
                router_probs.shape[-1],
                dtype=router_probs.dtype,
                device=router_probs.device,
            )
            dense_probs = dense_probs.index_copy(0, mask_idx, router_probs)

            # selected_experts is not gradient-bearing.
            if expert_idx.ndim == 1:
                dense_idx = expert_idx.new_zeros(flat_mask.numel())
            else:
                dense_idx = expert_idx.new_zeros((flat_mask.numel(), expert_idx.shape[-1]))
            dense_idx[flat_mask] = expert_idx

            # Store an additional `router_logits_detached` view for any
            # consumer that explicitly needs the no-grad form (telemetry
            # plots, etc.).
            self.last_router_info[name] = {
                "router_logits": dense_probs,
                "router_logits_detached": dense_probs.detach(),
                "selected_experts": dense_idx.detach(),
                "token_mask": flat_mask.detach(),
            }
            return

        # Gradient-preservation rule: same gradient-preserving rule for the non-token-masked path.
        self.last_router_info[name] = {
            "router_logits": router_probs,
            "router_logits_detached": router_probs.detach(),
            "selected_experts": expert_idx.detach(),
        }

    def _attach_token_mask_to_last_router_info(self, token_mask: torch.Tensor) -> None:
        flat_mask = token_mask.reshape(-1).bool().detach()
        for info in self.last_router_info.values():
            # Expand mask if router_logits has more rows (per-head flattened)
            probs_rows = info["router_logits"].shape[0]
            mask_rows = flat_mask.numel()
            if probs_rows > mask_rows and probs_rows % mask_rows == 0:
                expanded_mask = flat_mask.repeat(probs_rows // mask_rows)
            else:
                expanded_mask = flat_mask
            info["token_mask"] = expanded_mask
            if info["router_logits"].shape[0] != expanded_mask.numel():
                continue
            info["router_logits"] = info["router_logits"].clone()
            info["router_logits"][~expanded_mask] = 0
            info["selected_experts"] = info["selected_experts"].clone()
            info["selected_experts"][~expanded_mask] = 0

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

        if self.mode == "per_head_no_recompute":
            if self.routed_norm:
                normed = self.attn_pre_norm(hidden_states)
                q_normed = k_normed = v_normed = normed
            elif self.per_layer_norm and depth_idx is not None:
                normed = self.attn_pre_norms[depth_idx](hidden_states)
                q_normed = k_normed = v_normed = normed
            else:
                q_normed = self.q_pre_norm(hidden_states)
                k_normed = self.k_pre_norm(hidden_states)
                v_normed = self.v_pre_norm(hidden_states)

            q_flat = q_normed.reshape(B * T, H)
            k_flat = k_normed.reshape(B * T, H)
            v_flat = v_normed.reshape(B * T, H)
            q_route_flat = self._attn_router_inputs(q_normed)
            k_route_flat = self._attn_router_inputs(k_normed)
            v_route_flat = self._attn_router_inputs(v_normed)

            q_idx, q_w, q_probs = self._route_per_head(self.q_routers, q_route_flat, depth_idx)
            k_idx, k_w, k_probs = self._route_per_head(self.k_routers, k_route_flat, depth_idx)
            v_idx, v_w, v_probs = self._route_per_head(self.v_routers, v_route_flat, depth_idx)
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

        elif self.mode in {"per_head_recompute_k", "per_head_recompute_kv"}:
            raise RuntimeError("recompute attention modes should use project_and_attend_per_head_recompute()")

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

        assert self.mode == "per_head_no_recompute"
        N = B * T
        attn_heads = attn_output.transpose(1, 2).reshape(N, self.num_heads, self.head_dim)
        attn_flat = attn_heads.reshape(N, self.q_dim)
        o_idx, o_w, o_probs = self._route_per_head(self.o_routers, attn_flat, depth_idx)
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

    def _empty_attn_output(
        self,
        hidden_states: torch.Tensor,
        router_specs: list[tuple[str, int, int]],
    ) -> torch.Tensor:
        self.last_router_info = {}
        B, T, _ = hidden_states.shape
        flat_mask = torch.zeros(B * T, dtype=torch.bool, device=hidden_states.device)
        for name, num_experts, top_k in router_specs:
            empty_probs = hidden_states.new_zeros((0, num_experts))
            empty_idx = torch.zeros((0, top_k), device=hidden_states.device, dtype=torch.long)
            self._store_router_info(name, empty_probs, empty_idx, token_mask=flat_mask)
        dummy = self._zero_dummy(hidden_states.device, hidden_states.dtype)
        attn_out = hidden_states.new_zeros(B, T, self.hidden_size) + dummy
        return attn_out

    def recompute_router_specs(self) -> list[tuple[str, int, int]]:
        specs = [("qk", self.num_experts, self.num_kv_heads)]
        if self.routing_bundle in {"qk_v_o", "qk_vo"}:
            specs.append(("v", self.num_experts, self.num_kv_heads))
        if self.routing_bundle in {"qk_v_o", "qkv_o"}:
            specs.append(("o", self.num_o_experts, self.num_kv_heads))
        return specs

    def _build_per_head_recompute_tables(
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
        router_flat = self._attn_router_inputs(normed)

        routed = self._maybe_build_sanity_attention_routing(
            N, self.num_kv_heads, depth_idx, hidden_states.device, dtype=flat.dtype
        )
        if routed is None:
            qk_idx, qk_w, qk_probs = self._route_per_head(self.qk_routers, router_flat, depth_idx)
            self._store_router_info("qk", qk_probs, qk_idx)
        else:
            qk_idx, qk_w, qk_probs = routed

        if self.routing_bundle in {"qk_v_o", "qk_vo"}:
            v_idx, v_w, v_probs = self._route_per_head(self.v_routers, router_flat, depth_idx)
            self._store_router_info("v", v_probs, v_idx)
        else:
            v_idx, v_w = qk_idx, qk_w

        q_norm_weight = self._get_q_norm_weight_bank(depth_idx, self.num_experts)
        q_w_eff = qk_w if self.scale_attn_by_routing_weight else _straight_through_ones(qk_w)
        Q_groups = self._project_grouped_query_heads_batched(flat, self.q_proj, qk_idx, q_w_eff, q_norm_weight)
        V_fresh = None
        if self.mode == "per_head_recompute_k":
            v_w_eff = v_w if self.scale_attn_by_routing_weight else _straight_through_ones(v_w)
            V_heads = self._project_heads_batched(flat, self.v_proj, v_idx, v_w_eff)
            V_fresh = V_heads.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = (
            Q_groups.view(B, T, self.num_kv_heads, self.q_heads_per_kv, self.head_dim)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        Q = self._apply_rotary_pos_emb_q_only(Q, position_embeddings)

        tables = {
            "flat": flat,
            "qk_idx": qk_idx,
            "qk_weights": qk_w,
            "v_idx": v_idx,
            "v_weights": v_w,
            "Q": Q,
        }
        if V_fresh is not None:
            tables["V_fresh"] = V_fresh
        return tables

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

    def _apply_rotary_pos_emb_q_only(
        self,
        q: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (q * cos) + (self._rotate_half(q) * sin)

    def _run_per_head_recompute_expert_tables(
        self,
        tables: dict[str, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        query_token_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        """Run attention against expert-specific K or K/V tables.

        `per_head_recompute_k` recomputes K by Q/K expert and uses one
        token-routed V table. `per_head_recompute_kv` recomputes V by the
        query group's V route as well, so the active table key is either
        `(k_expert,)` or `(k_expert, v_expert)`.
        """
        flat = tables["flat"]
        qk_idx = tables["qk_idx"]
        v_idx = tables["v_idx"]
        Q = tables["Q"]
        B, _, T, _ = Q.shape

        if query_token_mask is None:
            query_group_mask = torch.ones((B, self.num_kv_heads, T), device=flat.device, dtype=torch.bool)
        else:
            query_group_mask = query_token_mask.squeeze(-1).bool().unsqueeze(1).expand(-1, self.num_kv_heads, -1)

        if not query_group_mask.any():
            return flat.new_zeros(B, self.num_heads, T, self.head_dim)

        slot_k_experts = qk_idx.view(B, T, self.num_kv_heads).permute(0, 2, 1)
        slot_v_experts = v_idx.view(B, T, self.num_kv_heads).permute(0, 2, 1)
        k_norm_weight = self._get_k_norm_weight_bank(depth_idx, self.num_experts)

        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        if self.capture_attention_maps:
            # Reset on the first per-depth call. The model loop iterates
            # depths sequentially and harvests `last_attention_maps` after
            # each, so resetting here gives one fresh list per depth.
            self.last_attention_maps = []
        if self.per_pair_v_routing:
            return self._run_pair_v_routing(
                flat=flat, Q=Q, slot_k_experts=slot_k_experts,
                query_group_mask=query_group_mask,
                k_norm_weight=k_norm_weight,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                depth_idx=depth_idx,
            )
        if self.mode == "per_head_recompute_k":
            V_fresh = tables.get("V_fresh")
            if V_fresh is None:
                raise RuntimeError("per_head_recompute_k requires a token-routed V table")
            active_k_experts = slot_k_experts[query_group_mask].unique()
            for k_expert in active_k_experts.tolist():
                K_e = flat @ self.k_proj[k_expert]
                K_e = self._apply_single_head_norm(K_e, k_norm_weight[k_expert])
                K_e_heads = K_e.view(B, T, 1, self.head_dim).transpose(1, 2)
                K_e_heads = self._apply_rotary_pos_emb_k_only(K_e_heads, position_embeddings)
                group_mask = (slot_k_experts == k_expert) & query_group_mask
                attn_e = self._run_attention(
                    Q,
                    K_e_heads.expand(-1, self.num_kv_heads, -1, -1),
                    V_fresh,
                    attention_mask,
                )
                head_mask = group_mask.unsqueeze(-1).repeat_interleave(self.num_kv_groups, dim=1)
                attn_e.mul_(head_mask.to(attn_e.dtype))
                attn_output.add_(attn_e)
            return attn_output

        pair_source = torch.stack((slot_k_experts, slot_v_experts), dim=-1)
        active_pairs = pair_source[query_group_mask].unique(dim=0)
        k_cache: dict[int, torch.Tensor] = {}
        v_cache: dict[int, torch.Tensor] = {}
        for k_expert, v_expert in active_pairs.tolist():
            if k_expert not in k_cache:
                K_e = flat @ self.k_proj[k_expert]
                K_e = self._apply_single_head_norm(K_e, k_norm_weight[k_expert])
                K_e_heads = K_e.view(B, T, 1, self.head_dim).transpose(1, 2)
                k_cache[k_expert] = self._apply_rotary_pos_emb_k_only(K_e_heads, position_embeddings)
            if v_expert not in v_cache:
                V_e = flat @ self.v_proj[v_expert]
                v_cache[v_expert] = V_e.view(B, T, 1, self.head_dim).transpose(1, 2)

            group_mask = (
                (slot_k_experts == k_expert)
                & (slot_v_experts == v_expert)
                & query_group_mask
            )
            attn_e = self._run_attention_checkpointed(
                Q,
                k_cache[k_expert].expand(-1, self.num_kv_heads, -1, -1),
                v_cache[v_expert].expand(-1, self.num_kv_heads, -1, -1),
                attention_mask,
            )
            head_mask = group_mask.unsqueeze(-1).repeat_interleave(self.num_kv_groups, dim=1)
            attn_e.mul_(head_mask.to(attn_e.dtype))
            attn_output.add_(attn_e)

            if self.capture_attention_maps:
                # Compute the softmax(QK^T) attention weights explicitly for
                # this (k_expert, v_expert) pair. The production path uses
                # SDPA which doesn't return weights; we recompute here only
                # under the capture flag (eval-only, low frequency). Under
                # GQA Q has `num_heads` rows and K has `num_kv_heads` rows;
                # expand K up to `num_heads` via repeat_interleave so the
                # per-Q-head map is comparable to the routing-aware aggregate.
                K_full = k_cache[k_expert].expand(-1, self.num_kv_heads, -1, -1)
                K_per_qhead = K_full.repeat_interleave(self.num_kv_groups, dim=1)
                qk_scores = torch.matmul(Q.float(), K_per_qhead.transpose(-1, -2).float())
                qk_scores = qk_scores * self.scaling
                # ALWAYS apply a causal mask — SDPA handles `is_causal=True`
                # automatically when attention_mask is None (which is the
                # default in this codebase), but our explicit Q@K^T softmax
                # has no such fallback. Use the float's most-negative finite
                # value (NOT -inf — `0 * -inf = NaN` in IEEE 754, but
                # `0 * finite_negative = 0`).
                neg_inf_finite = torch.finfo(qk_scores.dtype).min
                q_idx = torch.arange(T, device=qk_scores.device)
                causal_local = (q_idx[None, :] > q_idx[:, None]).float() * neg_inf_finite
                qk_scores = qk_scores + causal_local.unsqueeze(0).unsqueeze(0)
                if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                    qk_scores = qk_scores + attention_mask[:, :, :T, :T].float()
                attn_weights = F.softmax(qk_scores, dim=-1).detach().cpu()
                self.last_attention_maps.append(
                    {
                        "k_expert": int(k_expert),
                        "v_expert": int(v_expert),
                        "weights": attn_weights,  # [B, num_heads, T, T]
                        "group_mask": group_mask.detach().cpu(),  # [B, num_kv_heads, T]
                    }
                )

        return attn_output

    def _run_pair_v_routing(
        self,
        *,
        flat: torch.Tensor,
        Q: torch.Tensor,
        slot_k_experts: torch.Tensor,
        query_group_mask: torch.Tensor,
        k_norm_weight: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        depth_idx: int | None,
    ) -> torch.Tensor:
        """Per-(query, key) V-routing attention.

        K still routes per-query (one K-expert per query, applied uniformly
        across all key positions for that query). V is re-routed per pair:
        the V-expert at key position k for query position q is decided by a
        router taking (hidden[q], hidden[k]) as input.

        Output formula:
            attn[q, k]   = softmax( Q[q] · K_{e_q}[k] / √d + causal_mask )
            v_expert(q, k) = argmax_e (W_q · hidden[q])[e] + (W_k · hidden[k])[e]
            V[q, k]      = hidden[k] @ v_proj[v_expert(q, k)]
            out[q]       = Σ_k attn[q, k] · V[q, k]

        Implementation strategy: pre-compute V projections per expert (only
        for experts that actually fire), pre-compute the separable router
        scores, then for each k-expert group iterate (q, k) once and sum
        per-V-expert masked contributions.
        """
        B, _, T, _ = Q.shape

        # Separable pair-V router scores.
        v_router_q = (
            self.pair_v_router_q[depth_idx]
            if isinstance(self.pair_v_router_q, nn.ModuleList)
            else self.pair_v_router_q
        )
        v_router_k = (
            self.pair_v_router_k[depth_idx]
            if isinstance(self.pair_v_router_k, nn.ModuleList)
            else self.pair_v_router_k
        )
        # `flat` is [B*T, hidden_size]; reshape router outputs to [B, T, E].
        with torch.autocast(device_type=flat.device.type, enabled=False):
            v_score_q = v_router_q(flat.float()).view(B, T, -1)  # [B, T, num_v_experts]
            v_score_k = v_router_k(flat.float()).view(B, T, -1)  # [B, T, num_v_experts]
        # score[b, q, k, e] = v_score_q[b, q, e] + v_score_k[b, k, e]
        # argmax over e → v_assign[b, q, k]
        scores_qk = v_score_q.unsqueeze(2) + v_score_k.unsqueeze(1)  # [B, T_q, T_k, E]
        v_assign = scores_qk.argmax(dim=-1)  # [B, T_q, T_k]

        # Active V experts across the batch (we only materialize V_e for
        # experts that actually fire somewhere — cheap when num_v_experts
        # is small and the assignment concentrates).
        active_v_experts = v_assign.unique().tolist()
        V_cache: dict[int, torch.Tensor] = {}
        for v_e in active_v_experts:
            V_e = (flat @ self.v_proj[v_e]).view(B, T, self.head_dim)  # [B, T, head_dim]
            V_cache[v_e] = V_e

        # Compute softmax attention weights explicitly per k-expert group
        # so we can do per-pair V combination outside SDPA.
        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        active_k_experts = slot_k_experts[query_group_mask].unique().tolist()

        # Build the additive causal mask. SDPA (the production path) handles
        # `is_causal=True` automatically when attention_mask is None, but our
        # explicit Q@K^T softmax has no such fallback — we must apply the
        # causal mask ourselves. Construct a [T_q, T_k] tensor with 0 on the
        # lower triangle and a large negative number on the strict upper
        # triangle, then OR it with any caller-supplied mask.
        scaling = self.scaling
        neg_inf = torch.finfo(Q.dtype).min
        q_idx = torch.arange(T, device=Q.device)
        causal_local = (q_idx[None, :] > q_idx[:, None]).to(Q.dtype) * neg_inf
        causal_local = causal_local.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            external = attention_mask[:, :, :T, :T].to(Q.dtype)
            causal_add = external + causal_local
        else:
            causal_add = causal_local

        for k_expert in active_k_experts:
            K_e = flat @ self.k_proj[k_expert]
            K_e = self._apply_single_head_norm(K_e, k_norm_weight[k_expert])
            K_e_heads = K_e.view(B, T, 1, self.head_dim).transpose(1, 2)
            K_e_heads = self._apply_rotary_pos_emb_k_only(K_e_heads, position_embeddings)
            # Expand K from num_kv_heads (=1 in this projection table) to num_heads
            # so QK^T comes out [B, num_heads, T, T].
            K_full = K_e_heads.expand(-1, self.num_kv_heads, -1, -1)
            K_per_qhead = K_full.repeat_interleave(self.num_kv_groups, dim=1)

            qk = torch.matmul(Q, K_per_qhead.transpose(-1, -2)) * scaling
            qk = qk + causal_add  # Always applied (we built it above unconditionally).
            attn_w = F.softmax(qk, dim=-1, dtype=torch.float32)  # [B, H, T_q, T_k]
            # Restrict to queries that routed to THIS k_expert.
            group_mask = (slot_k_experts == k_expert) & query_group_mask
            head_mask = (
                group_mask.unsqueeze(-1)
                .repeat_interleave(self.num_kv_groups, dim=1)
                .squeeze(-1)
                .to(attn_w.dtype)
            )  # [B, num_heads, T_q]
            attn_w = attn_w * head_mask.unsqueeze(-1)

            if self.capture_attention_maps:
                self.last_attention_maps.append(
                    {
                        "k_expert": int(k_expert),
                        "v_expert": -1,  # pair-routed: not a single v_expert
                        "weights": attn_w.detach().cpu(),
                        "group_mask": group_mask.detach().cpu(),
                    }
                )

            # Sum V contributions per V-expert: out[q, d] += Σ_k (attn_w[q,k] * mask_e[q,k]) * V_e[k,d]
            # v_assign is [B, T_q, T_k]; broadcast across heads.
            for v_e in active_v_experts:
                mask_e = (v_assign == v_e).to(attn_w.dtype).unsqueeze(1)  # [B, 1, T_q, T_k]
                masked_attn = attn_w * mask_e  # [B, H, T_q, T_k]
                V_e_full = V_cache[v_e].view(B, T, 1, self.head_dim).transpose(1, 2)
                V_per_qhead = V_e_full.expand(-1, self.num_kv_heads, -1, -1)
                V_per_qhead = V_per_qhead.repeat_interleave(self.num_kv_groups, dim=1)
                # [B, H, T_q, head_dim] = einsum("bhqk, bhkd -> bhqd", masked_attn, V)
                out_e = torch.matmul(masked_attn.to(V_per_qhead.dtype), V_per_qhead)
                attn_output = attn_output + out_e

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
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        depth_idx: int | None,
    ) -> torch.Tensor | None:
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

        attn_output = self._run_attention(
            Q,
            K_fresh,
            V_fresh,
            attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.q_dim)
        o_out = self._project_sanity_logical_o(attn_output, depth_idx)
        return o_out

    def _run_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run_attention_backend(Q, K, V, attention_mask)

    def _run_attention_checkpointed(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Recompute-KV already runs inside the model's per-depth activation
        # checkpoint when gradient checkpointing is enabled. A second nested
        # checkpoint around every expert-table SDPA call has been unstable on
        # H200/B200 runs (late XID 43 illegal-address / illegal-instruction
        # failures). Keep the default path conservative: materialize expanded
        # K/V views before attention, then let the outer depth checkpoint handle
        # recomputation. The old inner checkpoint can be re-enabled only for
        # targeted experiments.
        if K.stride(1) == 0:
            K = K.contiguous()
        if V.stride(1) == 0:
            V = V.contiguous()
        if os.environ.get("MOE_EVERYTHING_RECOMPUTE_KV_INNER_CKPT", "0") != "1":
            return self._run_attention(Q, K, V, attention_mask)

        if not (self.training and torch.is_grad_enabled()):
            return self._run_attention(Q, K, V, attention_mask)
        if not (Q.requires_grad or K.requires_grad or V.requires_grad):
            return self._run_attention(Q, K, V, attention_mask)

        def attention_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # H200 SDPA can fault on zero-stride expanded K/V views in this
            # recompute-KV path. Materialize inside the checkpointed region so
            # checkpoint saves the cheap expanded views, while forward/backward
            # kernels see ordinary contiguous tensors.
            if k.stride(1) == 0:
                k = k.contiguous()
            if v.stride(1) == 0:
                v = v.contiguous()
            return self._run_attention(q, k, v, attention_mask)

        return activation_checkpoint(
            attention_fn,
            Q,
            K,
            V,
            use_reentrant=False,
            preserve_rng_state=self.attention_dropout > 0,
        )

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

    def _project_recompute_o(
        self,
        attn_dense: torch.Tensor,
        tables: dict[str, torch.Tensor],
        *,
        flat_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        B, T, _ = attn_dense.shape
        attn_groups = attn_dense.reshape(B * T, self.num_kv_heads, self.q_group_dim)
        qk_idx = tables["qk_idx"]
        qk_w = tables["qk_weights"]
        v_idx = tables["v_idx"]
        v_w = tables["v_weights"]

        if self.routing_bundle == "qkvo":
            o_idx, o_w = qk_idx, qk_w
        elif self.routing_bundle == "qk_vo":
            o_idx, o_w = v_idx, v_w
        else:
            route_inputs = attn_groups if flat_mask is None else attn_groups[flat_mask]
            o_idx, o_w, o_probs = self._route_per_slot_inputs(self.o_routers, route_inputs, depth_idx)
            if flat_mask is not None:
                self._store_router_info("o", o_probs, o_idx, token_mask=flat_mask)
            else:
                self._store_router_info("o", o_probs, o_idx)

        if flat_mask is not None:
            selected_inputs = attn_groups[flat_mask]
            selected_idx = o_idx if o_idx.shape[0] == selected_inputs.shape[0] else o_idx[flat_mask]
            selected_w = o_w if o_w.shape[0] == selected_inputs.shape[0] else o_w[flat_mask]
            o_w_eff = selected_w if self.scale_attn_by_routing_weight else _straight_through_ones(selected_w)
            o_selected = self._project_pair_inputs_grouped(
                selected_inputs,
                self.o_proj,
                selected_idx,
                o_w_eff,
                reduce_tokens=True,
            )
            o_out = attn_dense.new_zeros(B * T, self.hidden_size)
            o_out[flat_mask] = o_selected.to(o_out.dtype)
            return o_out.view(B, T, self.hidden_size)

        o_w_eff = o_w if self.scale_attn_by_routing_weight else _straight_through_ones(o_w)
        o_out = self._project_pair_inputs_grouped(
            attn_groups,
            self.o_proj,
            o_idx,
            o_w_eff,
            reduce_tokens=True,
        )
        return o_out.view(B, T, self.hidden_size)

    def project_and_attend_per_head_recompute_dense_mixed(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        flat_mask = token_mask.reshape(-1).bool()
        if not flat_mask.any():
            return self._empty_attn_output(
                hidden_states,
                self.recompute_router_specs(),
            )
        sanity_dense = self._project_and_attend_sanity_logical_dense(
            hidden_states,
            position_embeddings,
            token_mask,
            attention_mask,
            depth_idx,
        )
        if sanity_dense is not None:
            return sanity_dense
        self.last_router_info = {}
        tables = self._build_per_head_recompute_tables(
            hidden_states, position_embeddings, depth_idx=depth_idx
        )
        qk_w = tables["qk_weights"]

        self._attach_token_mask_to_last_router_info(token_mask)
        attn_output = self._run_per_head_recompute_expert_tables(
            tables,
            position_embeddings,
            attention_mask=attention_mask,
            query_token_mask=token_mask,
            depth_idx=depth_idx,
        )

        if self.scale_attn_by_routing_weight:
            kv_weight = hidden_states.new_zeros(B * T, self.num_kv_heads)
            kv_weight[flat_mask] = qk_w[flat_mask]
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
            return logical_o
        o_out = self._project_recompute_o(attn_dense, tables, flat_mask=flat_mask, depth_idx=depth_idx)
        return o_out

    def project_and_attend_per_head_recompute(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        """Flat-bank recompute attention with one routed expert per KV group."""
        self.last_router_info = {}
        tables = self._build_per_head_recompute_tables(
            hidden_states, position_embeddings, depth_idx=depth_idx
        )
        qk_w = tables["qk_weights"]
        B, _, T, _ = tables["Q"].shape

        attn_output = self._run_per_head_recompute_expert_tables(
            tables,
            position_embeddings,
            attention_mask=attention_mask,
            depth_idx=depth_idx,
        )

        if self.scale_attn_by_routing_weight:
            kv_weight_expanded = (
                qk_w.view(B, T, self.num_kv_heads)
                .transpose(1, 2)
                .repeat_interleave(self.num_kv_groups, dim=1)
                .unsqueeze(-1)
            )
            attn_output = attn_output * kv_weight_expanded.to(attn_output.dtype)
        attn_dense = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.q_dim)
        logical_o = self._project_sanity_logical_o(attn_dense, depth_idx)
        if logical_o is not None:
            return logical_o
        o_out = self._project_recompute_o(attn_dense, tables, depth_idx=depth_idx)

        return o_out.view(B, T, self.hidden_size)


# ─── MLP expert bank ──────────────────────────────────────────────────────── #
