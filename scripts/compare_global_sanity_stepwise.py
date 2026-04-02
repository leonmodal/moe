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

import torch

torch.backends.cuda.preferred_blas_library("cublaslt")

from train import (
    bias_alpha_schedule,
    build_model,
    get_bias_update_router_groups,
    load_config,
    update_expert_biases,
)
from src.models.init_mapping import ParamPair, copy_global_to_alternating_sanity


GLOBAL_CFG = "configs/scaling/debug8_xs_deepseek_global.yaml"
SANITY_CFG = "configs/scaling/moe_everything_per_head_precompute_kv_sanity_8gpu_smoke_nowandb.yaml"
STANDARD_CFG = "configs/scaling/debug8_xs_deepseek_standard.yaml"


class FakeAccelerator:
    num_processes = 1

    def unwrap_model(self, model):
        return model


def _build_cfg(path: str, *, batch_size: int, seq_len: int) -> dict:
    cfg = load_config(path)
    cfg["training"]["batch_size"] = batch_size
    cfg["data"]["seq_len"] = seq_len
    return cfg


def _copy_global_to_sanity(global_model, sanity_model) -> list[ParamPair]:
    return copy_global_to_alternating_sanity(global_model, sanity_model)


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

        global_alpha = bias_alpha_schedule(step) if global_cfg["model"].get("bias_interpolation", False) else 0.0
        sanity_alpha = bias_alpha_schedule(step) if sanity_cfg["model"].get("bias_interpolation", False) else 0.0
        update_expert_biases(
            global_model,
            global_cfg["model"]["bias_update_rate"],
            accelerator,
            is_global=True,
            alpha=global_alpha,
        )
        update_expert_biases(
            sanity_model,
            sanity_cfg["model"]["bias_update_rate"],
            accelerator,
            is_global=True,
            alpha=sanity_alpha,
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
