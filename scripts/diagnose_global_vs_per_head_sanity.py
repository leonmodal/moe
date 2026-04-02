#!/usr/bin/env python3
"""
Step-by-step parity check for DeepSeek Global MoE vs alternating-global per-head sanity.

This isolates the model path itself:
  - same copied weights
  - same batches
  - same optimizer hyperparameters
  - per-step loss / CE / logits / grad / param / optimizer-state diffs

Examples:
  uv run python scripts/diagnose_global_vs_per_head_sanity.py --steps 10
  uv run python scripts/diagnose_global_vs_per_head_sanity.py --data-mode parquet --steps 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models import (
    DeepSeekGlobalMoEForCausalLM,
    GlobalMoEConfig,
    MoEverythingConfig,
    MoEverythingForCausalLM,
)


torch.backends.cuda.preferred_blas_library("cublaslt")


TensorSelector = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TensorMap:
    name: str
    global_param: torch.nn.Parameter
    sanity_param: torch.nn.Parameter
    global_select: TensorSelector
    sanity_select: TensorSelector


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _row_selector(row: int) -> TensorSelector:
    return lambda tensor, row=row: tensor[row]


def _q_proj_selector(start: int, end: int) -> TensorSelector:
    return lambda tensor, start=start, end=end: tensor[start:end].transpose(0, 1)


def _kv_proj_selector(start: int, end: int) -> TensorSelector:
    return lambda tensor, start=start, end=end: tensor[start:end].transpose(0, 1)


def _o_proj_selector(start: int, end: int) -> TensorSelector:
    return lambda tensor, start=start, end=end: tensor[:, start:end].transpose(0, 1)


def copy_global_weights_into_sanity_moe(global_model, sanity_model) -> None:
    global_inner = global_model.model
    sanity_inner = sanity_model.model
    head_dim = global_model.config.head_dim
    num_heads = global_model.config.num_attention_heads
    num_kv_heads = global_model.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads

    with torch.no_grad():
        sanity_inner.embed_tokens.weight.copy_(global_inner.embed_tokens.weight)
        sanity_inner.norm.weight.copy_(global_inner.norm.weight)
        sanity_model.lm_head.weight.copy_(global_model.lm_head.weight)
        sanity_inner.mlp_bank.experts.gate_up_proj.copy_(global_inner.global_experts.gate_up_proj)
        sanity_inner.mlp_bank.experts.down_proj.copy_(global_inner.global_experts.down_proj)

        for layer_idx, layer in enumerate(global_inner.layers):
            attn_depth = 2 * layer_idx
            mlp_depth = attn_depth + 1
            sanity_inner.attn_bank.norms[attn_depth].weight.copy_(layer.input_layernorm.weight)
            sanity_inner.mlp_bank.norms[mlp_depth].weight.copy_(layer.post_attention_layernorm.weight)
            sanity_inner.mlp_bank.gates[layer_idx].weight.copy_(layer.mlp.gate.weight)
            if hasattr(layer.mlp.gate, "expert_bias"):
                sanity_inner.mlp_bank.gates[layer_idx].expert_bias.copy_(layer.mlp.gate.expert_bias)

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

            sanity_inner.attn_bank.logical_q_norm_weight[layer_idx].copy_(q_norm)
            sanity_inner.attn_bank.logical_k_norm_weight[layer_idx].copy_(k_norm)

            if not hasattr(sanity_inner.attn_bank, "logical_q_proj"):
                for kv_head_idx in range(num_kv_heads):
                    expert_idx = layer_idx * num_kv_heads + kv_head_idx
                    q_start = kv_head_idx * num_kv_groups * head_dim
                    q_end = q_start + num_kv_groups * head_dim
                    kv_start = kv_head_idx * head_dim
                    kv_end = kv_start + head_dim

                    sanity_inner.attn_bank.q_proj[expert_idx].copy_(q_proj[q_start:q_end].transpose(0, 1))
                    sanity_inner.attn_bank.k_proj[expert_idx].copy_(k_proj[kv_start:kv_end].transpose(0, 1))
                    sanity_inner.attn_bank.v_proj[expert_idx].copy_(v_proj[kv_start:kv_end].transpose(0, 1))
                    sanity_inner.attn_bank.o_proj[expert_idx].copy_(o_proj[:, q_start:q_end].transpose(0, 1))


def build_models(args, device: torch.device, dtype: torch.dtype):
    global_cfg = GlobalMoEConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.logical_layers,
        head_dim=args.head_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        output_router_logits=True,
        norm_topk_prob=True,
        router_aux_loss_coef=args.router_aux_loss_coef,
        tie_word_embeddings=args.tie_word_embeddings,
    )
    global_cfg.topk_scaling_factor = args.topk_scaling_factor
    global_cfg.num_groups = args.num_groups
    global_cfg.group_topk = args.group_topk

    sanity_cfg = MoEverythingConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=2 * args.logical_layers,
        head_dim=args.head_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        output_router_logits=True,
        norm_topk_prob=True,
        router_aux_loss_coef=args.router_aux_loss_coef,
        tie_word_embeddings=args.tie_word_embeddings,
        num_attn_experts=args.num_attn_experts,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_precompute_kv",
        per_head_compute_mode=args.per_head_compute_mode,
        use_deepseek_routing=True,
        topk_scaling_factor=args.topk_scaling_factor,
        num_groups=args.num_groups,
        group_topk=args.group_topk,
        seq_aux_loss_coef=args.seq_aux_loss_coef,
        per_layer_mlp_router=True,
        per_layer_norm=True,
        sanity_check_mode="alternating_global_moe",
    )

    torch.manual_seed(args.seed)
    global_model = DeepSeekGlobalMoEForCausalLM(global_cfg).to(device=device, dtype=dtype)
    torch.manual_seed(args.seed + 1)
    sanity_model = MoEverythingForCausalLM(sanity_cfg).to(device=device, dtype=dtype)
    copy_global_weights_into_sanity_moe(global_model, sanity_model)

    if args.seq_aux_loss_coef > 0:
        global_model._seq_aux_loss_coef = args.seq_aux_loss_coef
        sanity_model._seq_aux_loss_coef = args.seq_aux_loss_coef

    return global_model, sanity_model


def build_tensor_maps(global_model, sanity_model) -> list[TensorMap]:
    maps: list[TensorMap] = [
        TensorMap("embed_tokens", global_model.model.embed_tokens.weight, sanity_model.model.embed_tokens.weight, _identity, _identity),
        TensorMap("final_norm", global_model.model.norm.weight, sanity_model.model.norm.weight, _identity, _identity),
        TensorMap("lm_head", global_model.lm_head.weight, sanity_model.lm_head.weight, _identity, _identity),
        TensorMap(
            "mlp_experts.gate_up_proj",
            global_model.model.global_experts.gate_up_proj,
            sanity_model.model.mlp_bank.experts.gate_up_proj,
            _identity,
            _identity,
        ),
        TensorMap(
            "mlp_experts.down_proj",
            global_model.model.global_experts.down_proj,
            sanity_model.model.mlp_bank.experts.down_proj,
            _identity,
            _identity,
        ),
    ]

    head_dim = global_model.config.head_dim
    num_kv_heads = global_model.config.num_key_value_heads
    num_kv_groups = global_model.config.num_attention_heads // num_kv_heads

    for layer_idx, layer in enumerate(global_model.model.layers):
        attn_depth = 2 * layer_idx
        mlp_depth = attn_depth + 1
        maps.extend(
            [
                TensorMap(
                    f"layer_{layer_idx}.input_norm",
                    layer.input_layernorm.weight,
                    sanity_model.model.attn_bank.norms[attn_depth].weight,
                    _identity,
                    _identity,
                ),
                TensorMap(
                    f"layer_{layer_idx}.post_attn_norm",
                    layer.post_attention_layernorm.weight,
                    sanity_model.model.mlp_bank.norms[mlp_depth].weight,
                    _identity,
                    _identity,
                ),
                TensorMap(
                    f"layer_{layer_idx}.mlp_gate.weight",
                    layer.mlp.gate.weight,
                    sanity_model.model.mlp_bank.gates[layer_idx].weight,
                    _identity,
                    _identity,
                ),
                TensorMap(
                    f"layer_{layer_idx}.q_norm",
                    layer.self_attn.q_norm.weight,
                    sanity_model.model.attn_bank.logical_q_norm_weight,
                    _identity,
                    _row_selector(layer_idx),
                ),
                TensorMap(
                    f"layer_{layer_idx}.k_norm",
                    layer.self_attn.k_norm.weight,
                    sanity_model.model.attn_bank.logical_k_norm_weight,
                    _identity,
                    _row_selector(layer_idx),
                ),
            ]
        )
        if hasattr(layer.mlp.gate, "expert_bias"):
            maps.append(
                TensorMap(
                    f"layer_{layer_idx}.mlp_gate.expert_bias",
                    layer.mlp.gate.expert_bias,
                    sanity_model.model.mlp_bank.gates[layer_idx].expert_bias,
                    _identity,
                    _identity,
                )
            )

        if hasattr(sanity_model.model.attn_bank, "logical_q_proj"):
            maps.extend(
                [
                    TensorMap(
                        f"layer_{layer_idx}.q_proj",
                        layer.self_attn.q_proj.weight,
                        sanity_model.model.attn_bank.logical_q_proj[layer_idx],
                        _identity,
                        _identity,
                    ),
                    TensorMap(
                        f"layer_{layer_idx}.k_proj",
                        layer.self_attn.k_proj.weight,
                        sanity_model.model.attn_bank.logical_k_proj[layer_idx],
                        _identity,
                        _identity,
                    ),
                    TensorMap(
                        f"layer_{layer_idx}.v_proj",
                        layer.self_attn.v_proj.weight,
                        sanity_model.model.attn_bank.logical_v_proj[layer_idx],
                        _identity,
                        _identity,
                    ),
                    TensorMap(
                        f"layer_{layer_idx}.o_proj",
                        layer.self_attn.o_proj.weight,
                        sanity_model.model.attn_bank.logical_o_proj[layer_idx],
                        _identity,
                        _identity,
                    ),
                ]
            )
        else:
            for kv_head_idx in range(num_kv_heads):
                expert_idx = layer_idx * num_kv_heads + kv_head_idx
                q_start = kv_head_idx * num_kv_groups * head_dim
                q_end = q_start + num_kv_groups * head_dim
                kv_start = kv_head_idx * head_dim
                kv_end = kv_start + head_dim

                maps.extend(
                    [
                        TensorMap(
                            f"layer_{layer_idx}.q_proj.expert_{expert_idx}",
                            layer.self_attn.q_proj.weight,
                            sanity_model.model.attn_bank.q_proj,
                            _q_proj_selector(q_start, q_end),
                            _row_selector(expert_idx),
                        ),
                        TensorMap(
                            f"layer_{layer_idx}.k_proj.expert_{expert_idx}",
                            layer.self_attn.k_proj.weight,
                            sanity_model.model.attn_bank.k_proj,
                            _kv_proj_selector(kv_start, kv_end),
                            _row_selector(expert_idx),
                        ),
                        TensorMap(
                            f"layer_{layer_idx}.v_proj.expert_{expert_idx}",
                            layer.self_attn.v_proj.weight,
                            sanity_model.model.attn_bank.v_proj,
                            _kv_proj_selector(kv_start, kv_end),
                            _row_selector(expert_idx),
                        ),
                        TensorMap(
                            f"layer_{layer_idx}.o_proj.expert_{expert_idx}",
                            layer.self_attn.o_proj.weight,
                            sanity_model.model.attn_bank.o_proj,
                            _o_proj_selector(q_start, q_end),
                            _row_selector(expert_idx),
                        ),
                    ]
                )

    return maps


def compare_mapped_tensors(
    maps: list[TensorMap],
    *,
    tensor_getter: Callable[[torch.nn.Parameter], torch.Tensor | None],
) -> dict:
    worst_name = None
    worst_max = 0.0
    worst_mean = 0.0

    for tensor_map in maps:
        global_tensor = tensor_getter(tensor_map.global_param)
        sanity_tensor = tensor_getter(tensor_map.sanity_param)
        if global_tensor is None or sanity_tensor is None:
            continue

        global_view = tensor_map.global_select(global_tensor).float()
        sanity_view = tensor_map.sanity_select(sanity_tensor).float()
        diff = (global_view - sanity_view).abs()
        diff_max = diff.max().item()
        if diff_max > worst_max:
            worst_name = tensor_map.name
            worst_max = diff_max
            worst_mean = diff.mean().item()

    return {
        "name": worst_name,
        "max_abs": worst_max,
        "mean_abs": worst_mean,
    }


def compare_optimizer_states(maps: list[TensorMap], global_opt, sanity_opt) -> dict:
    result = {}
    for key in ("exp_avg", "exp_avg_sq"):
        worst_name = None
        worst_max = 0.0
        worst_mean = 0.0
        for tensor_map in maps:
            global_state = global_opt.state.get(tensor_map.global_param, {}).get(key)
            sanity_state = sanity_opt.state.get(tensor_map.sanity_param, {}).get(key)
            if global_state is None or sanity_state is None:
                continue
            global_view = tensor_map.global_select(global_state).float()
            sanity_view = tensor_map.sanity_select(sanity_state).float()
            diff = (global_view - sanity_view).abs()
            diff_max = diff.max().item()
            if diff_max > worst_max:
                worst_name = tensor_map.name
                worst_max = diff_max
                worst_mean = diff.mean().item()
        result[key] = {
            "name": worst_name,
            "max_abs": worst_max,
            "mean_abs": worst_mean,
        }
    return result


def optimizer_for(model, args):
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )


def make_synthetic_iterator(args, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1234)
    while True:
        ids = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len),
            generator=generator,
            dtype=torch.long,
        )
        yield {
            "input_ids": ids.to(device),
            "labels": ids.to(device),
        }


def make_parquet_iterator(args, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StatefulParquetDataset(
        DataConfig(
            data_dir=args.data_dir,
            text_column=args.text_column,
            seq_len=args.seq_len,
            tokenizer_name=args.tokenizer_name,
            num_workers=0,
        ),
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    while True:
        for batch in dataloader:
            yield {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "labels": batch["labels"].to(device, non_blocking=True),
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp32", "mp-bf16"), default="bf16")
    parser.add_argument("--data-mode", choices=("synthetic", "parquet"), default="synthetic")
    parser.add_argument("--data-dir", default="./data/parquet")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--logical-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-key-value-heads", type=int, default=8)
    parser.add_argument("--experts-per-layer", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--num-attn-experts", type=int, default=None)
    parser.add_argument("--num-experts-per-tok", type=int, default=4)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
    parser.add_argument("--tie-word-embeddings", action="store_true", default=True)
    parser.add_argument("--per-head-compute-mode", choices=("dense", "auto", "sparse"), default="dense")
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.0)
    parser.add_argument("--seq-aux-loss-coef", type=float, default=0.0)
    parser.add_argument("--topk-scaling-factor", type=float, default=2.5)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--group-topk", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    if args.num_experts is None:
        args.num_experts = args.logical_layers * args.experts_per_layer
    if args.num_attn_experts is None:
        args.num_attn_experts = args.logical_layers * args.num_key_value_heads

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this diagnostic.")

    device = torch.device("cuda")
    param_dtype = torch.float32 if args.dtype == "mp-bf16" else (
        torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    use_mixed_precision = args.dtype == "mp-bf16"

    def forward_context():
        if use_mixed_precision:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    global_model, sanity_model = build_models(args, device, param_dtype)
    global_model.train()
    sanity_model.train()

    global_opt = optimizer_for(global_model, args)
    sanity_opt = optimizer_for(sanity_model, args)
    tensor_maps = build_tensor_maps(global_model, sanity_model)

    if args.data_mode == "parquet":
        batch_iter = make_parquet_iterator(args, device)
    else:
        batch_iter = make_synthetic_iterator(args, device)

    for step in range(args.steps):
        batch = next(batch_iter)

        global_opt.zero_grad(set_to_none=True)
        sanity_opt.zero_grad(set_to_none=True)

        with forward_context():
            global_out = global_model(**batch, output_router_logits=True)
            sanity_out = sanity_model(**batch, output_router_logits=True)

        global_loss = global_out.loss
        sanity_loss = sanity_out.loss
        global_ce = getattr(global_out, "ce_loss", global_loss)
        sanity_ce = getattr(sanity_out, "ce_loss", sanity_loss)
        logit_diff = (global_out.logits.float() - sanity_out.logits.float()).abs()

        global_loss.backward()
        sanity_loss.backward()

        grad_stats = compare_mapped_tensors(tensor_maps, tensor_getter=lambda param: param.grad)

        global_opt.step()
        sanity_opt.step()

        param_stats = compare_mapped_tensors(tensor_maps, tensor_getter=lambda param: param.detach())
        opt_state_stats = compare_optimizer_states(tensor_maps, global_opt, sanity_opt)

        row = {
            "step": step,
            "loss_global": float(global_loss.detach()),
            "loss_sanity": float(sanity_loss.detach()),
            "loss_abs_diff": float((global_loss.detach() - sanity_loss.detach()).abs()),
            "ce_global": float(global_ce.detach()),
            "ce_sanity": float(sanity_ce.detach()),
            "ce_abs_diff": float((global_ce.detach() - sanity_ce.detach()).abs()),
            "logits_mean_abs_diff": logit_diff.mean().item(),
            "logits_max_abs_diff": logit_diff.max().item(),
            "grad_worst": grad_stats,
            "param_worst": param_stats,
            "optimizer_worst": opt_state_stats,
        }
        print(json.dumps(row, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
