"""
ZeRO-1 stepwise equivalence check for:

  - DeepSeek global MoE
  - per-head precompute_kv alternating_global_moe sanity mode

This mirrors the DDP parity harness but wraps each model in DeepSpeed ZeRO-1
instead of DistributedDataParallel so we can test whether optimizer-state
partitioning materially changes the matched drift behavior.
"""

from __future__ import annotations

import argparse
import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

torch.backends.cuda.preferred_blas_library("cublaslt")

from scripts.compare_global_sanity_stepwise import (  # noqa: E402
    GLOBAL_CFG,
    SANITY_CFG,
    ParamPair,
    _build_cfg,
    _compare_grad_pairs,
    _compare_param_pairs,
    _copy_global_to_sanity,
    _selected_expert_mismatches,
)
from src.data.parquet_dataset import DataConfig, StatefulParquetDataset  # noqa: E402
from train import (  # noqa: E402
    bias_alpha_schedule,
    build_model,
    get_bias_update_router_groups,
    update_expert_biases,
)


@dataclass
class _FakeAccelerator:
    num_processes: int


@dataclass
class StepResult:
    loss: float
    ce_loss: float
    logits: torch.Tensor | None


def _dist_info() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    return rank, world_size, local_rank, device


def _make_synthetic_iterator(
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> Iterator[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rank)
    while True:
        ids = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            generator=generator,
            dtype=torch.long,
        )
        batch = ids.to(device, non_blocking=True)
        yield {"input_ids": batch, "labels": batch}


def _make_parquet_iterator(
    *,
    data_dir: str,
    text_column: str,
    tokenizer_name: str,
    batch_size: int,
    seq_len: int,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
) -> Iterator[dict[str, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StatefulParquetDataset(
        DataConfig(
            data_dir=data_dir,
            text_column=text_column,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            num_workers=0,
        ),
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    while True:
        for batch in dataloader:
            yield {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "labels": batch["labels"].to(device, non_blocking=True),
            }


def _forward_backward_model(
    engine,
    batch: dict[str, torch.Tensor],
    *,
    amp_bf16: bool,
    capture_logits: bool,
) -> StepResult:
    engine.zero_grad()
    # DeepSpeed controls bf16/fp16 from its own config; nesting torch.autocast
    # around the engine only adds warnings and does not change the engine path.
    with nullcontext():
        output = engine(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            output_router_logits=True,
        )
        loss = output.loss
    with torch.no_grad():
        shift_logits = output.logits[..., :-1, :].float().contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    engine.backward(loss)
    return StepResult(
        loss=float(loss.item()),
        ce_loss=float(ce_loss.item()),
        logits=output.logits.detach() if capture_logits else None,
    )


def _reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _reduce_sum(value: int, device: torch.device) -> int:
    tensor = torch.tensor([value], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _zero1_config(*, batch_size: int, amp_bf16: bool) -> dict:
    return {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": amp_bf16},
        "fp16": {"enabled": False},
        "steps_per_print": 10_000_000,
        "wall_clock_breakdown": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default=GLOBAL_CFG)
    parser.add_argument("--sanity-config", default=SANITY_CFG)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=1234)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--amp-bf16", action="store_true")
    parser.add_argument("--data-mode", choices=("synthetic", "parquet"), default="parquet")
    parser.add_argument("--data-dir", default="./data/parquet")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override optimizer learning rate for both models.",
    )
    parser.add_argument(
        "--bias-update-rate",
        type=float,
        default=None,
        help="Override expert-bias update rate for both models. Use 0 to disable bias updates.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("config", "off", "on"),
        default="off",
        help="Apply checkpointing to both models for this diagnostic.",
    )
    parser.add_argument(
        "--capture-logits",
        choices=("on", "off"),
        default="on",
        help="Keep full logits and report max absolute logits diff. Disable for very large full-load runs.",
    )
    args = parser.parse_args()

    rank, world_size, _local_rank, device = _dist_info()
    is_main = rank == 0

    global_cfg = _build_cfg(args.global_config, batch_size=args.batch_size, seq_len=args.seq_len)
    sanity_cfg = _build_cfg(args.sanity_config, batch_size=args.batch_size, seq_len=args.seq_len)

    if args.gradient_checkpointing != "config":
        enabled = args.gradient_checkpointing == "on"
        global_cfg["training"]["gradient_checkpointing"] = enabled
        sanity_cfg["training"]["gradient_checkpointing"] = enabled
    if args.learning_rate is not None:
        global_cfg["training"]["learning_rate"] = args.learning_rate
        sanity_cfg["training"]["learning_rate"] = args.learning_rate
    if args.bias_update_rate is not None:
        global_cfg["model"]["bias_update_rate"] = args.bias_update_rate
        sanity_cfg["model"]["bias_update_rate"] = args.bias_update_rate

    torch.manual_seed(args.seed)
    global_model, _ = build_model(copy.deepcopy(global_cfg))
    torch.manual_seed(args.seed)
    sanity_model, _ = build_model(copy.deepcopy(sanity_cfg))

    global_model = global_model.to(device).train()
    sanity_model = sanity_model.to(device).train()
    pairs: list[ParamPair] = _copy_global_to_sanity(global_model, sanity_model)

    if global_cfg["training"].get("gradient_checkpointing", False):
        global_model.gradient_checkpointing_enable()
    if sanity_cfg["training"].get("gradient_checkpointing", False):
        sanity_model.gradient_checkpointing_enable()

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

    ds_config = _zero1_config(batch_size=args.batch_size, amp_bf16=args.amp_bf16)
    global_engine, _, _, _ = deepspeed.initialize(
        model=global_model,
        optimizer=global_opt,
        model_parameters=global_model.parameters(),
        config=ds_config,
        dist_init_required=False,
    )
    sanity_engine, _, _, _ = deepspeed.initialize(
        model=sanity_model,
        optimizer=sanity_opt,
        model_parameters=sanity_model.parameters(),
        config=ds_config,
        dist_init_required=False,
    )

    if args.data_mode == "synthetic":
        iterator = _make_synthetic_iterator(
            vocab_size=global_cfg["model"]["vocab_size"],
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            rank=rank,
            seed=args.data_seed,
            device=device,
        )
    else:
        iterator = _make_parquet_iterator(
            data_dir=args.data_dir,
            text_column=args.text_column,
            tokenizer_name=args.tokenizer_name,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            rank=rank,
            world_size=world_size,
            seed=args.data_seed,
            device=device,
        )

    accelerator = _FakeAccelerator(num_processes=world_size)

    if is_main:
        print(
            f"zero1_world_size={world_size} data_mode={args.data_mode} "
            f"amp_bf16={args.amp_bf16} "
            f"learning_rate={global_cfg['training']['learning_rate']} "
            f"bias_update_rate={global_cfg['model'].get('bias_update_rate', 0.0)} "
            f"gradient_checkpointing=(global={global_cfg['training'].get('gradient_checkpointing', False)}, "
            f"sanity={sanity_cfg['training'].get('gradient_checkpointing', False)})"
        )
        print(
            "step  global_ce  sanity_ce  |ce|  global_loss  sanity_loss  |loss|  "
            "logits_max  grad_max  param_max  mismatched_selected"
        )

    max_ce_diff = 0.0
    max_ce_step = -1
    max_loss_diff = 0.0
    max_loss_step = -1

    for step in range(args.steps):
        batch = next(iterator)

        global_result = _forward_backward_model(
            global_engine,
            batch,
            amp_bf16=args.amp_bf16,
            capture_logits=args.capture_logits == "on",
        )
        sanity_result = _forward_backward_model(
            sanity_engine,
            batch,
            amp_bf16=args.amp_bf16,
            capture_logits=args.capture_logits == "on",
        )

        logits_diff = 0.0
        if global_result.logits is not None and sanity_result.logits is not None:
            logits_diff = (global_result.logits.float() - sanity_result.logits.float()).abs().max().item()
        mismatches = _selected_expert_mismatches(global_engine.module, sanity_engine.module)
        selected_total = sum(mismatches)

        mean_global_loss = _reduce_mean(global_result.loss, device)
        mean_sanity_loss = _reduce_mean(sanity_result.loss, device)
        mean_global_ce = _reduce_mean(global_result.ce_loss, device)
        mean_sanity_ce = _reduce_mean(sanity_result.ce_loss, device)
        max_logits_diff = _reduce_max(logits_diff, device)
        total_selected = _reduce_sum(selected_total, device)

        grad_name, grad_diff = _compare_grad_pairs(pairs)

        global_engine.step()
        sanity_engine.step()

        if global_cfg["model"].get("bias_update_rate", 0.0) > 0:
            global_alpha = (
                bias_alpha_schedule(
                    step,
                    warmup_steps=global_cfg["model"].get("bias_interpolation_warmup_steps", 5000),
                )
                if global_cfg["model"].get("bias_interpolation", False)
                else 0.0
            )
            sanity_alpha = (
                bias_alpha_schedule(
                    step,
                    warmup_steps=sanity_cfg["model"].get("bias_interpolation_warmup_steps", 5000),
                )
                if sanity_cfg["model"].get("bias_interpolation", False)
                else 0.0
            )
            update_expert_biases(
                global_engine.module,
                global_cfg["model"]["bias_update_rate"],
                accelerator,
                is_global=True,
                alpha=global_alpha,
            )
            update_expert_biases(
                sanity_engine.module,
                sanity_cfg["model"]["bias_update_rate"],
                accelerator,
                is_global=True,
                alpha=sanity_alpha,
                router_groups=get_bias_update_router_groups(sanity_engine.module, mlp_only=True),
            )

        dist.barrier()

        if is_main:
            ce_diff = abs(mean_global_ce - mean_sanity_ce)
            loss_diff = abs(mean_global_loss - mean_sanity_loss)
            if ce_diff > max_ce_diff:
                max_ce_diff = ce_diff
                max_ce_step = step
            if loss_diff > max_loss_diff:
                max_loss_diff = loss_diff
                max_loss_step = step
            param_name, param_diff = _compare_param_pairs(pairs)

            should_report = (
                step == 0
                or step == args.steps - 1
                or (args.report_every > 0 and step % args.report_every == 0)
            )
            if should_report:
                print(
                    f"{step:>4d}  "
                    f"{mean_global_ce:>9.6f}  {mean_sanity_ce:>9.6f}  {ce_diff:>7.6f}  "
                    f"{mean_global_loss:>11.6f}  {mean_sanity_loss:>11.6f}  {loss_diff:>7.6f}  "
                    f"{max_logits_diff:>10.6f}  {grad_diff:>8.3g}  {param_diff:>9.3g}  {total_selected:>6d}"
                )
                if grad_name:
                    print(f"      worst_grad={grad_name}")
                if param_name:
                    print(f"      worst_param={param_name}")
                print(f"      mismatches_by_layer(local_rank0)={mismatches}")

    if is_main:
        print(
            f"summary max_ce_diff={max_ce_diff:.6f}@step{max_ce_step} "
            f"max_loss_diff={max_loss_diff:.6f}@step{max_loss_step}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
