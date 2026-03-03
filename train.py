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
from src.models import (
    Qwen3MoeConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    StandardMoEModel,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    DeepSeekGlobalMoEForCausalLM,
)
from src.models.router import DeepSeekRouter
from src.models.load_balancing import seq_load_balancing_loss_func
from src.utils.training import (
    TrainingConfig,
    build_lr_scheduler,
    build_optimizer,
    count_parameters,
    get_grad_norm,
)
from src.utils.routing_stats import compute_routing_stats, accumulate_expert_counts
from src.utils.routing_plots import plot_routing_snapshot


# --------------------------------------------------------------------------- #
#  Expert bias update (DeepSeek V3 aux-loss-free routing)                      #
# --------------------------------------------------------------------------- #

def bias_alpha_schedule(step: int, warmup_steps: int = 5000) -> float:
    """Cosine decay from 1 → 0 over warmup_steps, then stays at 0.

    Returns alpha ∈ [0, 1] that interpolates between per-layer (alpha=1)
    and global (alpha=0) bias updates.
    """
    if step >= warmup_steps:
        return 0.0
    progress = step / max(1, warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def update_expert_biases(
    model, update_rate: float, accelerator,
    is_global: bool = False, alpha: float = 0.0,
) -> dict:
    """
    Walk all DeepSeekRouter modules, all-reduce token counts across DDP ranks,
    then update expert_bias: bias += sign(avg - tokens) * rate.

    For global MoE (is_global=True): blends per-layer and global bias deltas
    using alpha ∈ [0, 1]:
      delta = alpha * per_layer_delta + (1 - alpha) * global_delta
    alpha=0 → purely global (all routers get same correction from pooled load).
    alpha=1 → purely per-layer (each router corrects from its own counts).

    For standard MoE (is_global=False): each layer's router is updated
    independently based on its own token counts (original DeepSeek V3 behavior).
    alpha is ignored.

    Returns dict of bias stats for logging (empty if no DeepSeekRouters found).
    """
    stats = {}
    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    if not routers:
        return stats

    # Step 1: All-reduce each router's token counts across DDP ranks
    per_router_counts = []
    for router in routers:
        counts = router.local_tokens_per_expert.clone()
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        per_router_counts.append(counts)
        router.local_tokens_per_expert.zero_()

    if is_global:
        # Global delta: sum across all layers → total load on shared experts
        global_counts = torch.stack(per_router_counts).sum(dim=0)  # [E]
        global_avg = global_counts.mean()
        global_delta = torch.sign(global_avg - global_counts) * update_rate

        # Blend per-layer and global deltas for each router
        for router, counts in zip(routers, per_router_counts):
            if alpha > 0:
                layer_avg = counts.mean()
                layer_delta = torch.sign(layer_avg - counts) * update_rate
                router.expert_bias += alpha * layer_delta + (1 - alpha) * global_delta
            else:
                router.expert_bias += global_delta
    else:
        # Step 2 (standard): Each router updates independently
        for router, counts in zip(routers, per_router_counts):
            avg = counts.mean()
            router.expert_bias += torch.sign(avg - counts) * update_rate

    # Aggregate bias stats across all routers
    all_bias = torch.cat([r.expert_bias for r in routers])
    stats["routing/expert_bias_mean"] = all_bias.mean().item()
    stats["routing/expert_bias_std"] = all_bias.std().item()
    stats["routing/expert_bias_min"] = all_bias.min().item()
    stats["routing/expert_bias_max"] = all_bias.max().item()
    return stats


# --------------------------------------------------------------------------- #
#  Config helpers                                                              #
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    mtype = cfg["model"]["type"]
    mcfg = cfg["model"]

    # --- Dense (non-MoE) early return ---
    if mtype == "dense":
        config = Qwen3Config(
            vocab_size=mcfg["vocab_size"],
            hidden_size=mcfg["hidden_size"],
            num_hidden_layers=mcfg["num_hidden_layers"],
            head_dim=mcfg["head_dim"],
            num_attention_heads=mcfg["num_attention_heads"],
            num_key_value_heads=mcfg["num_key_value_heads"],
            intermediate_size=mcfg["intermediate_size"],
            max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
            rope_theta=mcfg.get("rope_theta", 1_000_000.0),
            rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        )
        # Dense model has no MoE fields — set dummies for compatibility
        config.num_experts = 0
        config.num_experts_per_tok = 0
        model = Qwen3ForCausalLM(config)
        return model, config

    # --- MoE models: shared Qwen3MoEConfig fields ---
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

    def _set_deepseek_router_params(config, mcfg):
        """Attach DeepSeek V3 router params that Qwen3MoeConfig doesn't have natively."""
        config.topk_scaling_factor = mcfg.get("topk_scaling_factor", None)
        config.num_groups = mcfg.get("num_groups", None)
        config.group_topk = mcfg.get("group_topk", None)

    if mtype == "standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        model = StandardMoEModel(config)
    elif mtype == "deepseek_standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        _set_deepseek_router_params(config, mcfg)
        model = DeepSeekStandardMoEModel(config)
    elif mtype == "global_moe":
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        model = GlobalMoEForCausalLM(config)
    elif mtype == "deepseek_global_moe":
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        _set_deepseek_router_params(config, mcfg)
        model = DeepSeekGlobalMoEForCausalLM(config)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # Use transformers v5 grouped_mm expert backend (requires PyTorch 2.9+)
    # Falls back to batched_mm if grouped_mm is unavailable
    experts_impl = mcfg.get("experts_implementation", "grouped_mm")
    try:
        model.set_experts_implementation(experts_impl)
    except Exception:
        model.set_experts_implementation("eager")

    return model, config


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
    real_max_steps = train_cfg.max_steps  # unscaled, for our own schedules
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
    is_dense = cfg["model"]["type"] == "dense"
    is_global = cfg["model"]["type"] in ("global_moe", "deepseek_global_moe")
    bias_update_rate = cfg["model"].get("bias_update_rate", 0.0)
    bias_interpolation = cfg["model"].get("bias_interpolation", False)
    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)

    # Attach seq_aux_loss_coef to model (read by forward methods)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

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
        num_workers=0,       # must be 0 for IterableDataset state tracking
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

    # Note: scheduler state is already restored by accelerator.load_state().
    # Do NOT manually advance — that would double-advance the LR schedule.

    # --- Training loop ------------------------------------------------------
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    model.train()

    data_iter = iter(dataloader)
    t0 = time.perf_counter()
    tokens_seen = 0
    routing_log_every = tcfg_dict.get("routing_log_every", 50)
    expert_count_accum = None  # accumulated per-layer expert token counts

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
            output = model(input_ids=input_ids, labels=labels, **({} if is_dense else {"output_router_logits": True}))
            loss = output.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = get_grad_norm(accelerator.unwrap_model(model))
                accelerator.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        # Only count steps at actual optimizer updates
        if accelerator.sync_gradients:
            # Update expert biases (no-op if rate=0 or no DeepSeekRouters)
            if bias_update_rate > 0:
                alpha = bias_alpha_schedule(global_step) if (is_global and bias_interpolation) else 0.0
                bias_stats = update_expert_biases(
                    accelerator.unwrap_model(model), bias_update_rate, accelerator,
                    is_global=is_global, alpha=alpha,
                )
                bias_stats["routing/bias_alpha"] = alpha
            else:
                bias_stats = {}

            scheduler.step()
            global_step += 1
            tokens_seen += input_ids.numel() * accelerator.num_processes

            # ── Per-step logging ────────────────────────────────────────
            if global_step % train_cfg.log_every == 0 and accelerator.is_main_process:
                elapsed = time.perf_counter() - t0
                tok_per_sec = tokens_seen / elapsed
                lr = scheduler.get_last_lr()[0]
                aux   = getattr(output, "aux_loss", None)
                aux   = aux.item() if aux is not None else 0.0
                total = loss.item()
                aux_coef = getattr(accelerator.unwrap_model(model), "router_aux_loss_coef", 0.0)
                ce    = total - aux_coef * aux

                # Compute seq_aux_loss for logging
                seq_aux = 0.0
                if seq_aux_loss_coef > 0 and getattr(output, "router_logits", None) is not None:
                    seq_aux_val = seq_load_balancing_loss_func(
                        output.router_logits,
                        model_cfg.num_experts,
                        model_cfg.num_experts_per_tok,
                        batch_size=input_ids.shape[0],
                    )
                    seq_aux = seq_aux_val.item() if isinstance(seq_aux_val, torch.Tensor) else seq_aux_val
                    ce -= seq_aux_loss_coef * seq_aux

                log_dict = {
                    "train/ce_loss":        ce,
                    "train/aux_loss":       aux,
                    "train/seq_aux_loss":   seq_aux,
                    "train/grad_norm":      grad_norm,
                    "train/lr":             lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_seen_B":  tokens_seen / 1e9,
                }
                accelerator.print(
                    f"step {global_step:6d}  "
                    f"loss={total:.4f}  ce={ce:.4f}  aux={aux:.4f}  seq_aux={seq_aux:.4f}  "
                    f"lr={lr:.2e}  "
                    f"tok/s={tok_per_sec/1e3:.1f}k  |g|={grad_norm:.3f}"
                )
                if bias_stats:
                    log_dict.update(bias_stats)
                if log_with:
                    accelerator.log(log_dict, step=global_step)

            # ── Accumulate expert token counts every step (all ranks) ──
            if getattr(output, "router_logits", None) is not None:
                expert_count_accum = accumulate_expert_counts(
                    output.router_logits,
                    num_experts_per_tok=model_cfg.num_experts_per_tok,
                    accumulator=expert_count_accum,
                )

            # ── Routing stats (per-layer + cross-layer for Global MoE) ──
            if global_step % routing_log_every == 0:
                # All-reduce accumulated counts across ranks
                if expert_count_accum is not None and accelerator.num_processes > 1:
                    for k in expert_count_accum:
                        torch.distributed.all_reduce(
                            expert_count_accum[k], op=torch.distributed.ReduceOp.SUM,
                        )

                if accelerator.is_main_process and getattr(output, "router_logits", None) is not None:
                    rstats = compute_routing_stats(
                        output.router_logits,
                        num_experts_per_tok=model_cfg.num_experts_per_tok,
                        is_global=is_global,
                    )
                    # Only log lightweight scalar summaries to wandb
                    scalar_stats = {k: v for k, v in rstats.items() if not k.startswith("_hist/")}
                    if log_with:
                        accelerator.log(scalar_stats, step=global_step)

                    # Save detailed per-expert data as JSON (to output_dir on Modal volume)
                    routing_dir = os.path.join(train_cfg.output_dir, "routing_logs")
                    os.makedirs(routing_dir, exist_ok=True)
                    snapshot = {"step": global_step, "layers": {}, "global_pool": None}

                    for layer_idx in sorted(expert_count_accum or {}):
                        counts = expert_count_accum[layer_idx]
                        total = counts.sum().item()
                        fracs = (counts / total).cpu().tolist() if total > 0 else [0.0] * counts.shape[0]
                        snapshot["layers"][layer_idx] = {
                            "token_counts": counts.cpu().tolist(),
                            "token_fracs": fracs,
                        }

                    if is_global and expert_count_accum and len(expert_count_accum) > 1:
                        all_counts = torch.stack([expert_count_accum[i] for i in sorted(expert_count_accum)])
                        pool_counts = all_counts.sum(dim=0)  # [E]
                        pool_total = pool_counts.sum().item()
                        pool_fracs = (pool_counts / pool_total).cpu().tolist() if pool_total > 0 else []

                        # Per-expert: how many layers use it (> 0 tokens)
                        layer_usage = (all_counts > 0).float().sum(dim=0).cpu().tolist()  # [E]

                        snapshot["global_pool"] = {
                            "token_counts": pool_counts.cpu().tolist(),
                            "token_fracs": pool_fracs,
                            "layer_usage_count": layer_usage,
                            "num_layers": all_counts.shape[0],
                        }

                    json_path = os.path.join(routing_dir, f"step_{global_step:08d}.json")
                    with open(json_path, "w") as f:
                        json.dump(snapshot, f)

                    plot_routing_snapshot(snapshot, routing_dir, global_step)

                # Reset accumulator on all ranks
                expert_count_accum = None

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
