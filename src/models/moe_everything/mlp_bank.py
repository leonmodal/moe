"""MLP expert bank: routed MLP experts using Qwen3MoeExperts."""

import torch
import torch.nn as nn

from src.models.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeRMSNorm,
)
from src.models.router import (
    DeepSeekRouter,
    ExplorationTopKRouter,
)
from src.models.load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func

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
        return ExplorationTopKRouter(config)

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
                # When every token routes through the MLP, use the same
                # unmasked aux-loss path as the baseline/global models.
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

