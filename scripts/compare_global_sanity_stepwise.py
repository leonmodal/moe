"""
Stepwise equivalence check for:

  - DeepSeek global MoE
  - per-head precompute_kv alternating_global_moe sanity mode
  - standard MoE (reference only)

The global and sanity models are initialized to the same weights with an
explicit structural copy. We then run fixed synthetic batches step-by-step and
report loss deltas, selected-expert mismatches, parameter diffs, and optimizer
state diffs.

Usage:
  uv run python scripts/compare_global_sanity_stepwise.py
  uv run python scripts/compare_global_sanity_stepwise.py --steps 20 --seq-len 128
  uv run python scripts/compare_global_sanity_stepwise.py --amp-bf16
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch

torch.backends.cuda.preferred_blas_library("cublaslt")

from train import (
    bias_alpha_schedule,
    build_model,
    get_bias_update_router_groups,
    load_config,
    update_expert_biases,
)


GLOBAL_CFG = "configs/scaling/debug8_xs_deepseek_global.yaml"
SANITY_CFG = "configs/scaling/moe_everything_per_head_precompute_kv_sanity_8gpu_smoke_nowandb.yaml"
STANDARD_CFG = "configs/scaling/debug8_xs_deepseek_standard.yaml"


class FakeAccelerator:
    num_processes = 1

    def unwrap_model(self, model):
        return model


@dataclass
class ParamPair:
    name: str
    left: torch.nn.Parameter
    right: torch.nn.Parameter
    track_grad_and_opt: bool = True


def _build_cfg(path: str, *, batch_size: int, seq_len: int) -> dict:
    cfg = load_config(path)
    cfg["training"]["batch_size"] = batch_size
    cfg["data"]["seq_len"] = seq_len
    return cfg


def _copy_global_to_sanity(global_model, sanity_model) -> list[ParamPair]:
    global_inner = global_model.model
    sanity_inner = sanity_model.model
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
            sanity_inner.mlp_bank.gates[layer_idx].weight.copy_(layer.mlp.gate.weight)
            sanity_inner.mlp_bank.gates[layer_idx].expert_bias.copy_(layer.mlp.gate.expert_bias)

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
            add(
                f"layer{layer_idx}.mlp_gate_weight",
                layer.mlp.gate.weight,
                sanity_inner.mlp_bank.gates[layer_idx].weight,
            )
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


def _compare_param_pairs(pairs: list[ParamPair]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for pair in pairs:
        diff = (pair.left.detach().float() - pair.right.detach().float()).abs().max().item()
        if diff > worst_diff:
            worst_name = pair.name
            worst_diff = diff
    return worst_name, worst_diff


def _compare_grad_pairs(pairs: list[ParamPair]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for pair in pairs:
        if not pair.track_grad_and_opt:
            continue
        if pair.left.grad is None and pair.right.grad is None:
            continue
        if pair.left.grad is None or pair.right.grad is None:
            return pair.name, float("inf")
        diff = (pair.left.grad.detach().float() - pair.right.grad.detach().float()).abs().max().item()
        if diff > worst_diff:
            worst_name = pair.name
            worst_diff = diff
    return worst_name, worst_diff


def _compare_optimizer_states(
    pairs: list[ParamPair],
    left_opt: torch.optim.Optimizer,
    right_opt: torch.optim.Optimizer,
) -> tuple[str, str, float]:
    worst_name = ""
    worst_key = ""
    worst_diff = 0.0
    for pair in pairs:
        if not pair.track_grad_and_opt:
            continue
        left_state = left_opt.state.get(pair.left, {})
        right_state = right_opt.state.get(pair.right, {})
        for key in ("step", "exp_avg", "exp_avg_sq"):
            left_value = left_state.get(key)
            right_value = right_state.get(key)
            if left_value is None and right_value is None:
                continue
            if left_value is None or right_value is None:
                return pair.name, key, float("inf")
            if torch.is_tensor(left_value):
                diff = (left_value.detach().float() - right_value.detach().float()).abs().max().item()
            else:
                diff = abs(float(left_value) - float(right_value))
            if diff > worst_diff:
                worst_name = pair.name
                worst_key = key
                worst_diff = diff
    return worst_name, worst_key, worst_diff


def _selected_expert_mismatches(global_model, sanity_model) -> list[int]:
    mismatches = []
    for layer_idx, layer in enumerate(global_model.model.layers):
        global_idx = layer.mlp.gate._last_top_k_idx
        sanity_idx = sanity_model.model.mlp_bank.gates[layer_idx]._last_top_k_idx
        mismatches.append(int(global_idx.ne(sanity_idx).sum().item()))
    return mismatches


def _step_model(
    model,
    optimizer,
    input_ids: torch.Tensor,
    *,
    amp_bf16: bool,
) -> tuple[float, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_bf16):
        output = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
        loss = output.loss
    loss.backward()
    optimizer.step()
    return float(loss.item()), output.logits.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default=GLOBAL_CFG)
    parser.add_argument("--sanity-config", default=SANITY_CFG)
    parser.add_argument("--standard-config", default=STANDARD_CFG)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=1234)
    parser.add_argument("--amp-bf16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = "cuda"
    accelerator = FakeAccelerator()

    global_cfg = _build_cfg(args.global_config, batch_size=args.batch_size, seq_len=args.seq_len)
    sanity_cfg = _build_cfg(args.sanity_config, batch_size=args.batch_size, seq_len=args.seq_len)
    standard_cfg = _build_cfg(args.standard_config, batch_size=args.batch_size, seq_len=args.seq_len)

    torch.manual_seed(args.seed)
    global_model, _ = build_model(copy.deepcopy(global_cfg))
    torch.manual_seed(args.seed)
    sanity_model, _ = build_model(copy.deepcopy(sanity_cfg))
    torch.manual_seed(args.seed)
    standard_model, _ = build_model(copy.deepcopy(standard_cfg))

    global_model = global_model.to(device).train()
    sanity_model = sanity_model.to(device).train()
    standard_model = standard_model.to(device).train()

    pairs = _copy_global_to_sanity(global_model, sanity_model)

    global_opt = torch.optim.AdamW(
        global_model.parameters(),
        lr=global_cfg["training"]["learning_rate"],
        weight_decay=global_cfg["training"]["weight_decay"],
        betas=(global_cfg["training"]["beta1"], global_cfg["training"]["beta2"]),
    )
    sanity_opt = torch.optim.AdamW(
        sanity_model.parameters(),
        lr=sanity_cfg["training"]["learning_rate"],
        weight_decay=sanity_cfg["training"]["weight_decay"],
        betas=(sanity_cfg["training"]["beta1"], sanity_cfg["training"]["beta2"]),
    )
    standard_opt = torch.optim.AdamW(
        standard_model.parameters(),
        lr=standard_cfg["training"]["learning_rate"],
        weight_decay=standard_cfg["training"]["weight_decay"],
        betas=(standard_cfg["training"]["beta1"], standard_cfg["training"]["beta2"]),
    )

    init_param_name, init_param_diff = _compare_param_pairs(pairs)
    print("initial_global_vs_sanity_param_diff")
    print(f"  worst={init_param_name or 'none'} max_abs={init_param_diff:.8g}")
    print()
    print("step  global_loss  sanity_loss  std_loss  |g-s|  logits_max  grad_max  param_max  opt_max  mismatched_selected")

    for step in range(args.steps):
        torch.manual_seed(args.data_seed + step)
        input_ids = torch.randint(
            0,
            global_cfg["model"]["vocab_size"],
            (args.batch_size, args.seq_len),
            device=device,
        )

        global_loss, global_logits = _step_model(
            global_model, global_opt, input_ids, amp_bf16=args.amp_bf16
        )
        sanity_loss, sanity_logits = _step_model(
            sanity_model, sanity_opt, input_ids, amp_bf16=args.amp_bf16
        )
        standard_loss, _ = _step_model(
            standard_model, standard_opt, input_ids, amp_bf16=args.amp_bf16
        )

        alpha = bias_alpha_schedule(step)
        update_expert_biases(global_model, global_cfg["model"]["bias_update_rate"], accelerator, is_global=True, alpha=alpha)
        update_expert_biases(
            sanity_model,
            sanity_cfg["model"]["bias_update_rate"],
            accelerator,
            is_global=True,
            alpha=alpha,
            router_groups=get_bias_update_router_groups(sanity_model, mlp_only=True),
        )
        update_expert_biases(
            standard_model,
            standard_cfg["model"]["bias_update_rate"],
            accelerator,
            is_global=False,
            alpha=0.0,
        )

        grad_name, grad_diff = _compare_grad_pairs(pairs)
        param_name, param_diff = _compare_param_pairs(pairs)
        opt_name, opt_key, opt_diff = _compare_optimizer_states(pairs, global_opt, sanity_opt)
        mismatches = _selected_expert_mismatches(global_model, sanity_model)
        logits_diff = (global_logits.float() - sanity_logits.float()).abs().max().item()
        selected_total = sum(mismatches)

        print(
            f"{step:>4d}  "
            f"{global_loss:>11.6f}  {sanity_loss:>11.6f}  {standard_loss:>8.6f}  "
            f"{abs(global_loss - sanity_loss):>7.6f}  {logits_diff:>10.6f}  "
            f"{grad_diff:>8.3g}  {param_diff:>9.3g}  {opt_diff:>7.3g}  {selected_total:>6d}"
        )

        if grad_name:
            print(f"      worst_grad={grad_name}")
        if param_name:
            print(f"      worst_param={param_name}")
        if opt_name:
            print(f"      worst_opt={opt_name}:{opt_key}")
        print(f"      mismatches_by_layer={mismatches}")

    print()
    print("notes")
    print("  global vs sanity is an exact mapped-init comparison.")
    print("  standard is a reference curve only; it is not structurally init-matched to the global pool.")


if __name__ == "__main__":
    main()
