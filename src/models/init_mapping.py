from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParamPair:
    name: str
    left: torch.nn.Parameter
    right: torch.nn.Parameter
    track_grad_and_opt: bool = True


def copy_global_to_alternating_sanity(global_model, sanity_model) -> list[ParamPair]:
    """Map a global-MoE checkpoint init into alternating-global sanity weights."""

    global_inner = global_model.model
    sanity_inner = sanity_model.model

    sanity_mode = getattr(getattr(sanity_model, "config", None), "sanity_check_mode", None)
    if sanity_mode != "alternating_global_moe":
        raise ValueError("copy_global_to_alternating_sanity requires alternating_global_moe sanity mode")

    logical_layers = getattr(global_model.config, "num_hidden_layers", None)
    sanity_depths = getattr(sanity_model.config, "num_hidden_layers", None)
    if logical_layers is None or sanity_depths is None or sanity_depths != 2 * logical_layers:
        raise ValueError("sanity model depth must be exactly 2x the global model depth")

    head_dim = global_model.config.head_dim
    num_heads = global_model.config.num_attention_heads
    num_kv_heads = global_model.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    pairs: list[ParamPair] = []

    def add(
        name: str,
        left: torch.nn.Parameter,
        right: torch.nn.Parameter,
        *,
        track_grad_and_opt: bool = True,
    ) -> None:
        pairs.append(
            ParamPair(
                name=name,
                left=left,
                right=right,
                track_grad_and_opt=track_grad_and_opt,
            )
        )

    with torch.no_grad():
        sanity_inner.embed_tokens.weight.copy_(global_inner.embed_tokens.weight)
        sanity_inner.norm.weight.copy_(global_inner.norm.weight)
        sanity_model.lm_head.weight.copy_(global_model.lm_head.weight)
        sanity_inner.mlp_bank.experts.gate_up_proj.copy_(global_inner.global_experts.gate_up_proj)
        sanity_inner.mlp_bank.experts.down_proj.copy_(global_inner.global_experts.down_proj)

        add("embed", global_inner.embed_tokens.weight, sanity_inner.embed_tokens.weight)
        add("final_norm", global_inner.norm.weight, sanity_inner.norm.weight)
        add("lm_head", global_model.lm_head.weight, sanity_model.lm_head.weight)
        add("mlp_gate_up", global_inner.global_experts.gate_up_proj, sanity_inner.mlp_bank.experts.gate_up_proj)
        add("mlp_down", global_inner.global_experts.down_proj, sanity_inner.mlp_bank.experts.down_proj)

        for layer_idx, layer in enumerate(global_inner.layers):
            attn_depth = 2 * layer_idx
            mlp_depth = attn_depth + 1
            sanity_inner.attn_bank.norms[attn_depth].weight.copy_(layer.input_layernorm.weight)
            sanity_inner.mlp_bank.norms[mlp_depth].weight.copy_(layer.post_attention_layernorm.weight)

            add(
                f"layer{layer_idx}.attn_norm",
                layer.input_layernorm.weight,
                sanity_inner.attn_bank.norms[attn_depth].weight,
            )
            add(
                f"layer{layer_idx}.mlp_norm",
                layer.post_attention_layernorm.weight,
                sanity_inner.mlp_bank.norms[mlp_depth].weight,
            )

            if hasattr(sanity_inner.mlp_bank, "gates"):
                sanity_inner.mlp_bank.gates[layer_idx].weight.copy_(layer.mlp.gate.weight)
                add(
                    f"layer{layer_idx}.mlp_gate_weight",
                    layer.mlp.gate.weight,
                    sanity_inner.mlp_bank.gates[layer_idx].weight,
                )
                if hasattr(layer.mlp.gate, "expert_bias"):
                    sanity_inner.mlp_bank.gates[layer_idx].expert_bias.copy_(layer.mlp.gate.expert_bias)
                    add(
                        f"layer{layer_idx}.mlp_gate_bias",
                        layer.mlp.gate.expert_bias,
                        sanity_inner.mlp_bank.gates[layer_idx].expert_bias,
                    )

            q_proj = layer.self_attn.q_proj.weight
            k_proj = layer.self_attn.k_proj.weight
            v_proj = layer.self_attn.v_proj.weight
            o_proj = layer.self_attn.o_proj.weight
            q_norm = layer.self_attn.q_norm.weight
            k_norm = layer.self_attn.k_norm.weight

            if hasattr(sanity_inner.attn_bank, "logical_q_proj"):
                sanity_inner.attn_bank.logical_q_proj[layer_idx].copy_(q_proj)
                sanity_inner.attn_bank.logical_k_proj[layer_idx].copy_(k_proj)
                sanity_inner.attn_bank.logical_v_proj[layer_idx].copy_(v_proj)
                sanity_inner.attn_bank.logical_o_proj[layer_idx].copy_(o_proj)
                add(
                    f"layer{layer_idx}.q_proj",
                    q_proj,
                    sanity_inner.attn_bank.logical_q_proj[layer_idx],
                )
                add(
                    f"layer{layer_idx}.k_proj",
                    k_proj,
                    sanity_inner.attn_bank.logical_k_proj[layer_idx],
                )
                add(
                    f"layer{layer_idx}.v_proj",
                    v_proj,
                    sanity_inner.attn_bank.logical_v_proj[layer_idx],
                )
                add(
                    f"layer{layer_idx}.o_proj",
                    o_proj,
                    sanity_inner.attn_bank.logical_o_proj[layer_idx],
                )

            if hasattr(sanity_inner.attn_bank, "logical_q_norm_weight"):
                sanity_inner.attn_bank.logical_q_norm_weight[layer_idx].copy_(q_norm)
                sanity_inner.attn_bank.logical_k_norm_weight[layer_idx].copy_(k_norm)
                add(
                    f"layer{layer_idx}.attn_q_norm",
                    q_norm,
                    sanity_inner.attn_bank.logical_q_norm_weight[layer_idx],
                    track_grad_and_opt=False,
                )
                add(
                    f"layer{layer_idx}.attn_k_norm",
                    k_norm,
                    sanity_inner.attn_bank.logical_k_norm_weight[layer_idx],
                    track_grad_and_opt=False,
                )

            if not hasattr(sanity_inner.attn_bank, "logical_q_proj"):
                for kv_head_idx in range(num_kv_heads):
                    expert_idx = layer_idx * num_kv_heads + kv_head_idx
                    q_start = kv_head_idx * num_kv_groups * head_dim
                    q_end = q_start + num_kv_groups * head_dim
                    kv_start = kv_head_idx * head_dim
                    kv_end = kv_start + head_dim

                    sanity_inner.attn_bank.q_proj[expert_idx].copy_(q_proj[q_start:q_end].t())
                    sanity_inner.attn_bank.k_proj[expert_idx].copy_(k_proj[kv_start:kv_end].t())
                    sanity_inner.attn_bank.v_proj[expert_idx].copy_(v_proj[kv_start:kv_end].t())
                    sanity_inner.attn_bank.o_proj[expert_idx].copy_(o_proj[:, q_start:q_end].t())

                    add(
                        f"layer{layer_idx}.kv{kv_head_idx}.q_proj",
                        q_proj[q_start:q_end].t(),
                        sanity_inner.attn_bank.q_proj[expert_idx],
                        track_grad_and_opt=False,
                    )
                    add(
                        f"layer{layer_idx}.kv{kv_head_idx}.k_proj",
                        k_proj[kv_start:kv_end].t(),
                        sanity_inner.attn_bank.k_proj[expert_idx],
                        track_grad_and_opt=False,
                    )
                    add(
                        f"layer{layer_idx}.kv{kv_head_idx}.v_proj",
                        v_proj[kv_start:kv_end].t(),
                        sanity_inner.attn_bank.v_proj[expert_idx],
                        track_grad_and_opt=False,
                    )
                    add(
                        f"layer{layer_idx}.kv{kv_head_idx}.o_proj",
                        o_proj[:, q_start:q_end].t(),
                        sanity_inner.attn_bank.o_proj[expert_idx],
                        track_grad_and_opt=False,
                    )

                    if hasattr(sanity_inner.attn_bank, "q_norm_weight"):
                        sanity_inner.attn_bank.q_norm_weight[expert_idx].copy_(q_norm)
                        sanity_inner.attn_bank.k_norm_weight[expert_idx].copy_(k_norm)
                        add(
                            f"layer{layer_idx}.kv{kv_head_idx}.q_norm",
                            q_norm,
                            sanity_inner.attn_bank.q_norm_weight[expert_idx],
                            track_grad_and_opt=False,
                        )
                        add(
                            f"layer{layer_idx}.kv{kv_head_idx}.k_norm",
                            k_norm,
                            sanity_inner.attn_bank.k_norm_weight[expert_idx],
                            track_grad_and_opt=False,
                        )

    return pairs
