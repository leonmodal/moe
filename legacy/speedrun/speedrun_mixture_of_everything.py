from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from .mixture_of_everything import (
    AttentionExpertBank,
    MlpExpertBank,
    MoEverythingConfig,
    MoEverythingForCausalLM,
    MoEverythingModel,
    apply_rotary_pos_emb,
)


class FunctionalRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=self.eps)


def _norm_eps(module: nn.Module, default: float = 1e-6) -> float:
    return float(
        getattr(module, "variance_epsilon", getattr(module, "eps", default))
    )


def _replace_norm_modules(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, Qwen3MoeRMSNorm):
            setattr(module, name, FunctionalRMSNorm(child.weight.shape[0], eps=_norm_eps(child)))
        else:
            _replace_norm_modules(child)


class SpeedrunMoEverythingConfig(MoEverythingConfig):
    model_type = "speedrun_moe_everything"

    def __init__(self, *args, **kwargs):
        if kwargs.get("routed_norm", False):
            raise ValueError("speedrun_moe_everything does not support learned routed_norm")
        super().__init__(*args, **kwargs)


class SpeedrunAttentionExpertBank(AttentionExpertBank):
    def __init__(self, config: MoEverythingConfig):
        if getattr(config, "routed_norm", False):
            raise ValueError("speedrun_moe_everything does not support learned routed_norm")
        super().__init__(config)
        _replace_norm_modules(self)
        for attr in (
            "q_norm_weight",
            "k_norm_weight",
            "layer_q_norm_weight",
            "layer_k_norm_weight",
            "logical_q_norm_weight",
            "logical_k_norm_weight",
            "logical_qk_norm_layer_index",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_q_norm_weight_bank(self, depth_idx: int | None, num_experts: int):
        return None

    def _get_k_norm_weight_bank(self, depth_idx: int | None, num_experts: int):
        return None

    def _apply_single_head_norm(self, proj, norm_weight=None):
        return F.rms_norm(proj, (proj.shape[-1],), eps=self.eps)

    def _apply_logical_head_norm(self, proj, norm_weight=None):
        return F.rms_norm(proj, (proj.shape[-1],), eps=self.eps)

    def _project_and_attend_sanity_logical_dense(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        K_old: torch.Tensor,
        V_old: torch.Tensor,
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        depth_idx: int | None,
    ):
        if self.sanity_check_mode != "alternating_global_moe" or depth_idx is None:
            return None
        if not bool(token_mask.bool().all().item()):
            return None

        B, T, _ = hidden_states.shape
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

        Q = F.linear(normed, logical_q_proj).view(B, T, self.num_heads, self.head_dim)
        K_fresh = F.linear(normed, logical_k_proj).view(B, T, self.num_kv_heads, self.head_dim)
        V_fresh = F.linear(normed, logical_v_proj).view(B, T, self.num_kv_heads, self.head_dim)

        Q = self._apply_logical_head_norm(Q).transpose(1, 2)
        K_fresh = self._apply_logical_head_norm(K_fresh).transpose(1, 2)
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

    def _run_per_head_precompute_kv_expert_tables(
        self,
        tables: dict[str, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        query_token_mask: torch.Tensor | None = None,
        depth_idx: int | None = None,
    ) -> torch.Tensor:
        flat = tables["flat"]
        idx = tables["idx"]
        Q = tables["Q"]
        B, _, T, _ = Q.shape

        if query_token_mask is None:
            query_group_mask = torch.ones((B, self.num_kv_heads, T), device=flat.device, dtype=torch.bool)
        else:
            query_group_mask = query_token_mask.squeeze(-1).bool().unsqueeze(1).expand(-1, self.num_kv_heads, -1)

        if not query_group_mask.any():
            return flat.new_zeros(B, self.num_heads, T, self.head_dim)

        active_experts = idx.view(B, T, self.num_kv_heads).permute(0, 2, 1)[query_group_mask].unique()
        slot_experts = idx.view(B, T, self.num_kv_heads).permute(0, 2, 1)

        attn_output = flat.new_zeros(B, self.num_heads, T, self.head_dim)
        for expert in active_experts.tolist():
            K_e = flat @ self.k_proj[expert]
            V_e = flat @ self.v_proj[expert]

            K_e = self._apply_single_head_norm(K_e, None)
            K_e_heads = K_e.view(B, T, 1, self.head_dim).transpose(1, 2)
            V_e_heads = V_e.view(B, T, 1, self.head_dim).transpose(1, 2)
            K_e_heads = self._apply_rotary_pos_emb_k_only(K_e_heads, position_embeddings)
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


class SpeedrunMlpExpertBank(MlpExpertBank):
    def __init__(self, config: MoEverythingConfig):
        if getattr(config, "routed_norm", False):
            raise ValueError("speedrun_moe_everything does not support learned routed_norm")
        super().__init__(config)
        _replace_norm_modules(self)


class SpeedrunMoEverythingModel(MoEverythingModel):
    def __init__(self, config: MoEverythingConfig):
        if getattr(config, "routed_norm", False):
            raise ValueError("speedrun_moe_everything does not support learned routed_norm")
        super().__init__(config)
        self.attn_bank = SpeedrunAttentionExpertBank(config)
        self.mlp_bank = SpeedrunMlpExpertBank(config)
        if self.init_k_norm is not None:
            self.init_k_norm = FunctionalRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.norm = FunctionalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if hasattr(self, "attn_post_norm"):
            self.attn_post_norm = FunctionalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if hasattr(self, "mlp_post_norm"):
            self.mlp_post_norm = FunctionalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if hasattr(self, "attn_post_norms"):
            self.attn_post_norms = nn.ModuleList(
                [FunctionalRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(self.num_depths)]
            )
        if hasattr(self, "mlp_post_norms"):
            self.mlp_post_norms = nn.ModuleList(
                [FunctionalRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(self.num_depths)]
            )
        if hasattr(self, "depth_norm"):
            self.depth_norm = FunctionalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class SpeedrunMoEverythingForCausalLM(MoEverythingForCausalLM):
    def __init__(self, config: MoEverythingConfig):
        super().__init__(config)
        self.model = SpeedrunMoEverythingModel(config)
        self.post_init()
