"""
Pretraining script: Standard MoE vs Global MoE.

Usage:
  ./scripts/train.sh configs/scaling/xs_standard.yaml
  ./scripts/train.sh configs/scaling/xs_global.yaml
  ./scripts/train.sh configs/scaling/l_standard.yaml --resume outputs/l_standard_moe/checkpoint-5000
"""
import argparse
import json
import math
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env from cwd or any parent directory
load_dotenv(Path(__file__).parent / ".env")  # also check script directory

import torch
import yaml

# cuBLAS on Blackwell (B200, sm_100) has a bug where bf16 Linear(bias=False)
# fails with CUBLAS_STATUS_INVALID_VALUE via the default GEMM_DEFAULT_TENSOR_OP
# path. cuBLASlt uses different algorithm selection and handles this correctly.
torch.backends.cuda.preferred_blas_library("cublaslt")
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
apply_liger_kernel_to_qwen3_moe()  # monkey-patches RMSNorm, SwiGLU, RoPE, CrossEntropy before model init

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM
from src.utils.training import (
    TrainingConfig,
    build_lr_scheduler,
    build_optimizer,
    count_parameters,
    get_grad_norm,
)
from src.utils.routing_stats import compute_routing_stats


# --------------------------------------------------------------------------- #
#  Config helpers                                                              #
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    mtype = cfg["model"]["type"]
    mcfg = cfg["model"]

    # Shared Qwen3MoEConfig fields (field names match the dataclass exactly)
    common = dict(
        vocab_size=mcfg["vocab_size"],
        hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        head_dim=mcfg["head_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        moe_intermediate_size=mcfg["moe_intermediate_size"],
        intermediate_size=mcfg.get("intermediate_size", mcfg["moe_intermediate_size"] * 4),
        max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
        rope_theta=mcfg.get("rope_theta", 1_000_000.0),
        rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        router_aux_loss_coef=mcfg.get("router_aux_loss_coef", 0.001),
        norm_topk_prob=mcfg.get("norm_topk_prob", True),
        num_experts_per_tok=mcfg["num_experts_per_tok"],
        output_router_logits=True,
    )

    if mtype == "standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        return StandardMoEModel(config), config
    elif mtype == "global_moe":
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        return GlobalMoEForCausalLM(config), config
    else:
        raise ValueError(f"Unknown model type: {mtype}")


# --------------------------------------------------------------------------- #
#  Checkpointing                                                               #
# --------------------------------------------------------------------------- #

def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: str,
    dataset_state: dict | None = None,
    wandb_run_id: str | None = None,
) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        meta = {"step": step}
        if dataset_state:
            meta["dataset_state"] = dataset_state
        if wandb_run_id:
            meta["wandb_run_id"] = wandb_run_id
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
    accelerator.print(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint(
    accelerator: Accelerator,
    resume_from: str,
) -> tuple[int, dict | None]:
    accelerator.load_state(resume_from)
    meta_path = os.path.join(resume_from, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("step", 0), meta.get("dataset_state")
    return 0, None


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Scan output_dir for checkpoint-N directories and return path of latest."""
    if not os.path.isdir(output_dir):
        return None
    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    for entry in os.listdir(output_dir):
        m = pattern.match(entry)
        if m:
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path):
                checkpoints.append((int(m.group(1)), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def cleanup_checkpoints(output_dir: str, max_keep: int) -> None:
    """Keep only the most recent max_keep checkpoints, delete the rest."""
    if max_keep <= 0:
        return
    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    for entry in os.listdir(output_dir):
        m = pattern.match(entry)
        if m:
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path):
                checkpoints.append((int(m.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_keep:
        _, path = checkpoints.pop(0)
        shutil.rmtree(path)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Path to checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto_resume", action="store_true",
                        help="Auto-find and resume from latest checkpoint in output_dir")
    parser.add_argument("--data_dir", default=None,
                        help="Override data.data_dir from config")
    parser.add_argument("--output_dir", default=None,
                        help="Override training.output_dir from config")
    parser.add_argument("--max_checkpoints", type=int, default=0,
                        help="Max checkpoints to keep (0 = unlimited)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg_dict = cfg["training"]
    dcfg_dict = cfg.get("data", {})

    # --- CLI overrides -------------------------------------------------------
    if args.data_dir:
        dcfg_dict["data_dir"] = args.data_dir
    if args.output_dir:
        tcfg_dict["output_dir"] = args.output_dir

    resume_from = args.resume or cfg.get("checkpoint", {}).get("resume_from")

    # --- Accelerator --------------------------------------------------------
    train_cfg = TrainingConfig(
        learning_rate=tcfg_dict["learning_rate"],
        weight_decay=tcfg_dict["weight_decay"],
        beta1=tcfg_dict.get("beta1", 0.9),
        beta2=tcfg_dict.get("beta2", 0.95),
        max_grad_norm=tcfg_dict["max_grad_norm"],
        lr_scheduler=tcfg_dict["lr_scheduler"],
        warmup_steps=tcfg_dict["warmup_steps"],
        max_steps=tcfg_dict["max_steps"],
        min_lr_ratio=tcfg_dict["min_lr_ratio"],
        batch_size=tcfg_dict["batch_size"],
        gradient_accumulation=tcfg_dict["gradient_accumulation"],
        mixed_precision=tcfg_dict["mixed_precision"],
        gradient_checkpointing=tcfg_dict.get("gradient_checkpointing", False),
        log_every=tcfg_dict["log_every"],
        save_every=tcfg_dict["save_every"],
        output_dir=tcfg_dict["output_dir"],
        wandb_project=tcfg_dict.get("wandb_project"),
        wandb_run_name=tcfg_dict.get("wandb_run_name"),
    )

    # --- Auto-resume: find latest checkpoint --------------------------------
    if args.auto_resume and not resume_from:
        resume_from = find_latest_checkpoint(train_cfg.output_dir)

    # Read wandb run ID from checkpoint meta (for WandB resume)
    wandb_run_id = None
    if resume_from:
        meta_path = os.path.join(resume_from, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            wandb_run_id = meta.get("wandb_run_id")

    # --- Accelerator --------------------------------------------------------
    # Set CUDA device early — required for NCCL init in multi-node torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    log_with = "wandb" if train_cfg.wandb_project else None
    accelerator = Accelerator(
        mixed_precision=train_cfg.mixed_precision,
        gradient_accumulation_steps=train_cfg.gradient_accumulation,
        log_with=log_with,
        project_dir=train_cfg.output_dir,
    )
    set_seed(args.seed + accelerator.process_index)

    # --- Multi-node diagnostics (every rank prints) --------------------------
    print(
        f"[rank {accelerator.process_index}] "
        f"host={socket.gethostname()} "
        f"local_rank={accelerator.local_process_index} "
        f"num_processes={accelerator.num_processes} "
        f"device={accelerator.device}",
        flush=True,
    )
    if accelerator.is_main_process:
        accelerator.print(f"=== Accelerator state ===")
        accelerator.print(accelerator.state)

    if resume_from:
        accelerator.print(f"Will resume from: {resume_from}")

    # Accelerate counts scheduler steps per device, not per optimizer update.
    # Scale warmup/max steps so configs stay in real optimizer steps.
    train_cfg.warmup_steps = train_cfg.warmup_steps * accelerator.num_processes
    train_cfg.max_steps = train_cfg.max_steps * accelerator.num_processes

    if log_with and accelerator.is_main_process:
        tracker_kwargs = {"wandb": {"name": train_cfg.wandb_run_name}}
        if wandb_run_id:
            tracker_kwargs["wandb"]["id"] = wandb_run_id
            tracker_kwargs["wandb"]["resume"] = "must"
        accelerator.init_trackers(
            project_name=train_cfg.wandb_project,
            config={**cfg["model"], **tcfg_dict},
            init_kwargs=tracker_kwargs,
        )

    # Capture wandb run ID for checkpoint saving (new runs)
    if log_with and accelerator.is_main_process and not wandb_run_id:
        try:
            import wandb
            if wandb.run:
                wandb_run_id = wandb.run.id
        except Exception:
            pass

    # --- Model --------------------------------------------------------------
    model, model_cfg = build_model(cfg)

    params = count_parameters(model)
    expert_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "gate_up_proj" in n or "down_proj" in n
    )
    is_global = cfg["model"]["type"] == "global_moe"

    accelerator.print(
        f"\n{'='*60}\n"
        f"  Model     : {cfg['model']['type']}\n"
        f"  Params    : {params['total']/1e9:.3f}B total  |  {expert_params/1e9:.3f}B expert\n"
        f"  Dist type : {accelerator.distributed_type}\n"
        f"  Precision : {train_cfg.mixed_precision}\n"
        f"  GPUs      : {accelerator.num_processes}\n"
        f"{'='*60}\n"
    )

    # --- Dataset ------------------------------------------------------------
    data_cfg = DataConfig(
        data_dir=dcfg_dict["data_dir"],
        text_column=dcfg_dict.get("text_column", "text"),
        seq_len=dcfg_dict.get("seq_len", 2048),
        tokenizer_name=dcfg_dict.get("tokenizer_name", "gpt2"),
        num_workers=dcfg_dict.get("num_workers", 4),
    )
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StatefulParquetDataset(
        config=data_cfg,
        tokenizer=tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.gradient_accumulation,
        pin_memory=True,
    )

    # --- Optimizer & Scheduler ----------------------------------------------
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_lr_scheduler(optimizer, train_cfg)

    # --- Accelerate prepare (wraps model in DDP/FSDP) -----------------------
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # --- Resume -------------------------------------------------------------
    global_step = 0
    dataset_state = None
    if resume_from:
        global_step, dataset_state = load_checkpoint(accelerator, resume_from)
        accelerator.print(f"Resumed from step {global_step}")
        if dataset_state:
            dataset.set_state(dataset_state)

    # Advance scheduler to match resumed step (no-op if global_step==0)
    for _ in range(global_step):
        scheduler.step()

    # --- Training loop ------------------------------------------------------
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    model.train()

    data_iter = iter(dataloader)
    t0 = time.perf_counter()
    tokens_seen = 0
    routing_log_every = tcfg_dict.get("routing_log_every", 50)

    accelerator.print(f"Starting training from step {global_step}")

    while global_step < train_cfg.max_steps:
        # Fetch batch (loop dataset if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"]   # (B, T)
        labels = batch["labels"]         # (B, T)

        with accelerator.accumulate(model):
            # Qwen3MoeForCausalLM returns MoeCausalLMOutputWithPast
            output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
            loss = output.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = get_grad_norm(accelerator.unwrap_model(model))
                accelerator.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        # Only count steps at actual optimizer updates
        if accelerator.sync_gradients:
            scheduler.step()
            global_step += 1
            tokens_seen += input_ids.numel() * accelerator.num_processes

            # ── Per-step logging ────────────────────────────────────────
            if global_step % train_cfg.log_every == 0 and accelerator.is_main_process:
                elapsed = time.perf_counter() - t0
                tok_per_sec = tokens_seen / elapsed
                lr = scheduler.get_last_lr()[0]
                aux   = output.aux_loss.item() if output.aux_loss is not None else 0.0
                total = loss.item()
                ce    = total - accelerator.unwrap_model(model).router_aux_loss_coef * aux
                log_dict = {
                    "train/ce_loss":        ce,
                    "train/aux_loss":       aux,
                    "train/grad_norm":      grad_norm,
                    "train/lr":             lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_seen_B":  tokens_seen / 1e9,
                }
                accelerator.print(
                    f"step {global_step:6d}  "
                    f"loss={total:.4f}  ce={ce:.4f}  aux={aux:.4f}  "
                    f"lr={lr:.2e}  "
                    f"tok/s={tok_per_sec/1e3:.1f}k  |g|={grad_norm:.3f}"
                )
                if log_with:
                    accelerator.log(log_dict, step=global_step)

            # ── Routing stats (per-layer + cross-layer for Global MoE) ──
            if global_step % routing_log_every == 0 and accelerator.is_main_process:
                if output.router_logits is not None:
                    rstats = compute_routing_stats(
                        output.router_logits,
                        num_experts_per_tok=model_cfg.num_experts_per_tok,
                        is_global=is_global,
                    )
                    # Separate histogram data from scalar metrics
                    hist_data   = {k: v for k, v in rstats.items() if k.startswith("_hist/")}
                    scalar_stats = {k: v for k, v in rstats.items() if not k.startswith("_hist/")}

                    if log_with:
                        accelerator.log(scalar_stats, step=global_step)
                        # Log histograms via wandb directly
                        try:
                            import wandb
                            for key, values in hist_data.items():
                                name = key.replace("_hist/", "hist/")
                                wandb.log({name: wandb.Histogram(values)}, step=global_step)
                        except Exception:
                            pass

            # Checkpoint
            if global_step % train_cfg.save_every == 0:
                ds_state = dataset.get_state()
                save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    global_step, train_cfg.output_dir, ds_state,
                    wandb_run_id=wandb_run_id,
                )
                if args.max_checkpoints > 0 and accelerator.is_main_process:
                    cleanup_checkpoints(train_cfg.output_dir, args.max_checkpoints)

    # Final checkpoint
    save_checkpoint(
        accelerator, model, optimizer, scheduler,
        global_step, train_cfg.output_dir,
        dataset.get_state(),
        wandb_run_id=wandb_run_id,
    )
    if args.max_checkpoints > 0 and accelerator.is_main_process:
        cleanup_checkpoints(train_cfg.output_dir, args.max_checkpoints)
    accelerator.end_training()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
