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
from dataclasses import replace
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def enable_liger_kernels() -> None:
    """Enable Liger monkey-patches only for real training entrypoints.

    Import-time patching leaks into unrelated CPU-only tests that import train.py,
    causing Triton kernels to run on CPU tensors.
    """
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

    apply_liger_kernel_to_qwen3_moe()

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
    MoEverythingConfig,
    MoEverythingForCausalLM,
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
from src.utils.routing_stats import (
    accumulate_expert_counts,
    accumulate_router_margins,
    compute_routing_stats_from_counts,
    router_margin_accumulator_to_stats,
)
from src.utils.routing_plots import plot_routing_snapshot
from src.models.mixture_of_everything import NormExpertBank


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


def get_selected_experts_for_seq_aux(model) -> tuple[torch.Tensor, ...] | None:
    """Return the last biased top-k assignments from DeepSeek routers, if present."""
    try:
        inner_model = getattr(model, "model", None)
        layers = getattr(inner_model, "layers", None)
        if layers is not None:
            selected = []
            for layer in layers:
                gate = getattr(getattr(layer, "mlp", None), "gate", None)
                idx = getattr(gate, "_last_top_k_idx", None)
                if idx is None:
                    return None
                selected.append(idx)
            return tuple(selected) if selected else None

        selected = getattr(inner_model, "_all_mlp_selected_experts", None)
        if selected:
            return tuple(selected)
        return None
    except Exception:
        return None


def get_output_selected_experts(output, model) -> tuple[torch.Tensor, ...] | None:
    selected = getattr(output, "selected_experts", None)
    if selected:
        return tuple(selected)
    return get_selected_experts_for_seq_aux(model)


def reduce_scalar(accelerator: Accelerator, value: float, reduction: str = "mean") -> float:
    tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        if reduction == "mean":
            tensor /= accelerator.num_processes
    return tensor.item()


def counts_accumulator_to_snapshot(
    accumulator: dict[int, torch.Tensor] | None,
    is_global: bool = False,
) -> dict:
    snapshot = {"layers": {}, "global_pool": None}
    if not accumulator:
        return snapshot

    for layer_idx in sorted(accumulator):
        counts = accumulator[layer_idx]
        total = counts.sum().item()
        fracs = (counts / total).cpu().tolist() if total > 0 else [0.0] * counts.shape[0]
        snapshot["layers"][int(layer_idx)] = {
            "token_counts": counts.cpu().tolist(),
            "token_fracs": fracs,
        }

    if is_global and len(accumulator) > 1:
        all_counts = torch.stack([accumulator[i] for i in sorted(accumulator)])
        pool_counts = all_counts.sum(dim=0)
        pool_total = pool_counts.sum().item()
        pool_fracs = (pool_counts / pool_total).cpu().tolist() if pool_total > 0 else []
        layer_usage = (all_counts > 0).float().sum(dim=0).cpu().tolist()
        snapshot["global_pool"] = {
            "token_counts": pool_counts.cpu().tolist(),
            "token_fracs": pool_fracs,
            "layer_usage_count": layer_usage,
            "num_layers": all_counts.shape[0],
        }

    return snapshot


def accumulate_branch_probs(branch_probs, accumulator=None):
    if not branch_probs:
        return accumulator
    if accumulator is None:
        accumulator = {}

    for depth_idx, probs in enumerate(branch_probs):
        probs = probs.detach().float()
        entry = accumulator.get(depth_idx)
        if entry is None:
            entry = {
                "sum": torch.zeros(2, device=probs.device, dtype=torch.float32),
                "count": torch.zeros((), device=probs.device, dtype=torch.float32),
            }
            accumulator[depth_idx] = entry
        entry["sum"] += probs.reshape(-1, 2).sum(dim=0)
        entry["count"] += probs.shape[0] * probs.shape[1]

    return accumulator


def branch_accumulator_to_stats(accumulator) -> dict:
    if not accumulator:
        return {}

    stats = {}
    total_sum = None
    total_count = 0.0
    for depth_idx in sorted(accumulator):
        entry = accumulator[depth_idx]
        probs = entry["sum"] / entry["count"].clamp(min=1.0)
        attn_frac = probs[0].item()
        mlp_frac = probs[1].item()
        ratio = attn_frac / max(mlp_frac, 1e-12)
        stats[f"routing/branch_layer_{depth_idx:02d}_attn_frac"] = attn_frac
        stats[f"routing/branch_layer_{depth_idx:02d}_mlp_frac"] = mlp_frac
        stats[f"routing/branch_layer_{depth_idx:02d}_attn_to_mlp_ratio"] = ratio
        total_sum = entry["sum"].clone() if total_sum is None else total_sum + entry["sum"]
        total_count += entry["count"].item()

    total_probs = total_sum / max(total_count, 1.0)
    stats["routing/branch_total_attn_frac"] = total_probs[0].item()
    stats["routing/branch_total_mlp_frac"] = total_probs[1].item()
    stats["routing/branch_total_attn_to_mlp_ratio"] = total_probs[0].item() / max(total_probs[1].item(), 1e-12)
    return stats


def branch_accumulator_to_snapshot(accumulator) -> dict | None:
    if not accumulator:
        return None

    snapshot = {"layers": {}, "total": {}}
    total_sum = None
    total_count = 0.0
    for depth_idx in sorted(accumulator):
        entry = accumulator[depth_idx]
        probs = entry["sum"] / entry["count"].clamp(min=1.0)
        attn_frac = probs[0].item()
        mlp_frac = probs[1].item()
        snapshot["layers"][int(depth_idx)] = {
            "attn_frac": attn_frac,
            "mlp_frac": mlp_frac,
            "attn_to_mlp_ratio": attn_frac / max(mlp_frac, 1e-12),
        }
        total_sum = entry["sum"].clone() if total_sum is None else total_sum + entry["sum"]
        total_count += entry["count"].item()

    total_probs = total_sum / max(total_count, 1.0)
    snapshot["total"] = {
        "attn_frac": total_probs[0].item(),
        "mlp_frac": total_probs[1].item(),
        "attn_to_mlp_ratio": total_probs[0].item() / max(total_probs[1].item(), 1e-12),
    }
    return snapshot


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
    elif mtype == "moe_everything":
        config = MoEverythingConfig(
            num_experts=mcfg["num_experts"],
            num_attn_experts=mcfg.get("num_attn_experts", 4),
            num_attn_experts_per_tok=mcfg.get("num_attn_experts_per_tok", 1),
            attn_expert_mode=mcfg.get("attn_expert_mode", "bundled"),
            branch_router_aux_loss_coef=mcfg.get("branch_router_aux_loss_coef", 0.01),
            use_deepseek_routing=mcfg.get("use_deepseek_routing", False),
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            seq_aux_loss_coef=mcfg.get("seq_aux_loss_coef", 0.0),
            per_layer_router=mcfg.get("per_layer_router", False),
            per_layer_attn_router=mcfg.get("per_layer_attn_router", False),
            routed_norm=mcfg.get("routed_norm", False),
            per_layer_norm=mcfg.get("per_layer_norm", False),
            post_norm=mcfg.get("post_norm", False),
            dynamic_depth_min=mcfg.get("dynamic_depth_min", 1.0),
            dynamic_depth_max=mcfg.get("dynamic_depth_max", 1.0),
            depthwise_attention=mcfg.get("depthwise_attention", False),
            depthwise_block_size=mcfg.get("depthwise_block_size", 0),
            **common,
        )
        model = MoEverythingForCausalLM(config)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # Use transformers v5 grouped_mm expert backend (requires PyTorch 2.9+)
    # Falls back to batched_mm if grouped_mm is unavailable
    if hasattr(model, "set_experts_implementation"):
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
    tokens_seen: float = 0.0,
) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        meta = {"step": step, "tokens_seen": tokens_seen}
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
) -> tuple[int, dict | None, float]:
    accelerator.load_state(resume_from)
    meta_path = os.path.join(resume_from, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("step", 0), meta.get("dataset_state"), meta.get("tokens_seen", 0.0)
    return 0, None, 0.0


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
    enable_liger_kernels()

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
    ddp_kwargs = []
    if (
        cfg["model"]["type"] == "moe_everything"
        and cfg["model"].get("attn_expert_mode") in (
            "precompute_kv",
            "per_head_precompute_kv",
        )
    ):
        ddp_kwargs.append(DistributedDataParallelKwargs(static_graph=True))
    accelerator = Accelerator(
        mixed_precision=train_cfg.mixed_precision,
        gradient_accumulation_steps=train_cfg.gradient_accumulation,
        log_with=log_with,
        project_dir=train_cfg.output_dir,
        kwargs_handlers=ddp_kwargs,
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

    # Training config stays in real optimizer-step units for loop control, logging,
    # checkpoint cadence, and WandB config. Accelerate's prepared scheduler, however,
    # advances once per process when split_batches=False, so only the scheduler needs
    # world-size-scaled warmup/total steps.
    scheduler_cfg = replace(
        train_cfg,
        warmup_steps=train_cfg.warmup_steps * accelerator.num_processes,
        max_steps=train_cfg.max_steps * accelerator.num_processes,
    )

    if log_with and accelerator.is_main_process:
        tracker_kwargs = {"wandb": {"name": train_cfg.wandb_run_name}}
        if wandb_run_id:
            tracker_kwargs["wandb"]["id"] = wandb_run_id
            # "allow" instead of "must": if the run logged steps beyond
            # this checkpoint (e.g. a later run crashed), wandb won't
            # reject the earlier steps — it starts a new run instead.
            tracker_kwargs["wandb"]["resume"] = "allow"
        accelerator.init_trackers(
            project_name=train_cfg.wandb_project,
            config={**cfg["model"], **tcfg_dict},
            init_kwargs=tracker_kwargs,
        )

    # Capture wandb run ID for checkpoint saving (new runs)
    if log_with and accelerator.is_main_process:
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
    is_moe_everything = cfg["model"]["type"] == "moe_everything"
    shared_mlp_pool = is_global or is_moe_everything
    bias_update_rate = cfg["model"].get("bias_update_rate", 0.0)
    bias_interpolation = cfg["model"].get("bias_interpolation", False)
    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)

    # Attach seq_aux_loss_coef to model (read by forward methods)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    if train_cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            accelerator.print("Gradient checkpointing enabled.")
        else:
            accelerator.print("Gradient checkpointing requested, but this model does not expose gradient_checkpointing_enable().")

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
    if accelerator.local_process_index == 0:
        print(
            f"[rank {accelerator.process_index}] Loading tokenizer {data_cfg.tokenizer_name}",
            flush=True,
        )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if accelerator.local_process_index == 0:
        print(f"[rank {accelerator.process_index}] Tokenizer ready", flush=True)
        print(
            f"[rank {accelerator.process_index}] Building dataset from {data_cfg.data_dir}",
            flush=True,
        )

    dataset = StatefulParquetDataset(
        config=data_cfg,
        tokenizer=tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    if accelerator.local_process_index == 0:
        print(
            f"[rank {accelerator.process_index}] Dataset ready: {len(dataset.files)} shard files",
            flush=True,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=0,       # must be 0 for IterableDataset state tracking
        pin_memory=True,
    )

    # --- Optimizer & Scheduler ----------------------------------------------
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg)

    # --- Accelerate prepare (wraps model in DDP/FSDP) -----------------------
    accelerator.print("Calling accelerator.prepare()")
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    accelerator.print("accelerator.prepare() complete")

    # --- Resume -------------------------------------------------------------
    global_step = 0
    dataset_state = None
    tokens_seen = 0.0
    if resume_from:
        global_step, dataset_state, tokens_seen = load_checkpoint(accelerator, resume_from)
        accelerator.print(f"Resumed from step {global_step}, tokens_seen={tokens_seen/1e9:.3f}B")
        if dataset_state:
            dataset.set_state(dataset_state)

    # Note: scheduler state is already restored by accelerator.load_state().
    # Do NOT manually advance — that would double-advance the LR schedule.

    # --- Training loop ------------------------------------------------------
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    model.train()

    data_iter = iter(dataloader)
    t0 = time.perf_counter()
    tokens_this_session = 0.0
    routing_log_every = tcfg_dict.get("routing_log_every", 50)
    expert_count_accum = None
    expert_margin_accum = None
    attention_expert_count_accum = {}
    attention_router_margin_accum = {}
    branch_prob_accum = None

    loss_window_sum = 0.0
    ce_window_sum = 0.0
    aux_window_sum = 0.0
    seq_aux_window_sum = 0.0
    branch_aux_window_sum = 0.0
    microbatches_in_step = 0
    local_tokens_in_step = 0

    accelerator.print(f"Starting training from step {global_step}")

    while global_step < train_cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        with accelerator.accumulate(model):
            output = model(
                input_ids=input_ids,
                labels=labels,
                **({} if is_dense else {"output_router_logits": True}),
            )
            loss = output.loss
            raw_model = accelerator.unwrap_model(model)

            aux = getattr(output, "aux_loss", None)
            aux_value = aux.detach().float().item() if isinstance(aux, torch.Tensor) else float(aux or 0.0)
            total_value = loss.detach().float().item()

            ce_tensor = getattr(output, "ce_loss", None)
            if isinstance(ce_tensor, torch.Tensor):
                ce_value = ce_tensor.detach().float().item()
            elif ce_tensor is not None:
                ce_value = float(ce_tensor)
            else:
                ce_value = total_value - getattr(raw_model, "router_aux_loss_coef", 0.0) * aux_value

            seq_aux = getattr(output, "seq_aux_loss", None)
            if seq_aux is None and seq_aux_loss_coef > 0 and getattr(output, "router_logits", None) is not None:
                selected_for_seq_aux = get_output_selected_experts(output, raw_model)
                seq_aux = seq_load_balancing_loss_func(
                    output.router_logits,
                    model_cfg.num_experts,
                    model_cfg.num_experts_per_tok,
                    batch_size=input_ids.shape[0],
                    selected_experts=selected_for_seq_aux,
                )
            if isinstance(seq_aux, torch.Tensor):
                seq_aux_value = seq_aux.detach().float().item()
            elif seq_aux is not None:
                seq_aux_value = float(seq_aux)
            else:
                seq_aux_value = 0.0

            branch_aux = getattr(output, "branch_aux_loss", None)
            if isinstance(branch_aux, torch.Tensor):
                branch_aux_value = branch_aux.detach().float().item()
            elif branch_aux is not None:
                branch_aux_value = float(branch_aux)
            else:
                branch_aux_value = 0.0

            if ce_tensor is None:
                ce_value -= seq_aux_loss_coef * seq_aux_value
                ce_value -= getattr(raw_model, "branch_router_aux_loss_coef", 0.0) * branch_aux_value

            loss_window_sum += total_value
            ce_window_sum += ce_value
            aux_window_sum += aux_value
            seq_aux_window_sum += seq_aux_value
            branch_aux_window_sum += branch_aux_value
            microbatches_in_step += 1
            local_tokens_in_step += input_ids.numel()

            selected_experts = get_output_selected_experts(output, raw_model)
            if getattr(output, "router_logits", None) is not None:
                expert_count_accum = accumulate_expert_counts(
                    output.router_logits,
                    num_experts_per_tok=model_cfg.num_experts_per_tok,
                    accumulator=expert_count_accum,
                    selected_experts=selected_experts,
                )
                expert_margin_accum = accumulate_router_margins(
                    output.router_logits,
                    num_experts_per_tok=model_cfg.num_experts_per_tok,
                    accumulator=expert_margin_accum,
                )

            attention_router_info = getattr(output, "attention_router_info", None)
            if attention_router_info is not None:
                router_names = sorted({name for depth_info in attention_router_info for name in depth_info})
                for router_name in router_names:
                    router_logits = []
                    router_selected = []
                    for depth_info in attention_router_info:
                        info = depth_info.get(router_name)
                        if info is None:
                            continue
                        router_logits.append(info["router_logits"])
                        router_selected.append(info["selected_experts"])
                    attention_expert_count_accum[router_name] = accumulate_expert_counts(
                        router_logits,
                        num_experts_per_tok=model_cfg.num_attn_experts_per_tok,
                        accumulator=attention_expert_count_accum.get(router_name),
                        selected_experts=router_selected,
                    )
                    attention_router_margin_accum[router_name] = accumulate_router_margins(
                        router_logits,
                        num_experts_per_tok=model_cfg.num_attn_experts_per_tok,
                        accumulator=attention_router_margin_accum.get(router_name),
                    )

            branch_prob_accum = accumulate_branch_probs(getattr(output, "branch_probs", None), branch_prob_accum)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = get_grad_norm(accelerator.unwrap_model(model))
                accelerator.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            if bias_update_rate > 0:
                alpha = bias_alpha_schedule(global_step) if (is_global and bias_interpolation) else 0.0
                bias_stats = update_expert_biases(
                    accelerator.unwrap_model(model),
                    bias_update_rate,
                    accelerator,
                    is_global=is_global,
                    alpha=alpha,
                )
                bias_stats["routing/bias_alpha"] = alpha
            else:
                bias_stats = {}

            scheduler.step()
            global_step += 1

            tokens_this_step = reduce_scalar(accelerator, float(local_tokens_in_step), reduction="sum")
            tokens_seen += tokens_this_step
            tokens_this_session += tokens_this_step

            avg_total = reduce_scalar(accelerator, loss_window_sum / max(1, microbatches_in_step))
            avg_ce = reduce_scalar(accelerator, ce_window_sum / max(1, microbatches_in_step))
            avg_aux = reduce_scalar(accelerator, aux_window_sum / max(1, microbatches_in_step))
            avg_seq_aux = reduce_scalar(accelerator, seq_aux_window_sum / max(1, microbatches_in_step))
            avg_branch_aux = reduce_scalar(accelerator, branch_aux_window_sum / max(1, microbatches_in_step))
            avg_grad_norm = reduce_scalar(accelerator, grad_norm)

            if global_step % train_cfg.log_every == 0 and accelerator.is_main_process:
                elapsed = time.perf_counter() - t0
                tok_per_sec = tokens_this_session / max(elapsed, 1e-6)
                lr = scheduler.get_last_lr()[0]
                log_dict = {
                    "train/loss": avg_total,
                    "train/ce_loss": avg_ce,
                    "train/aux_loss": avg_aux,
                    "train/seq_aux_loss": avg_seq_aux,
                    "train/branch_aux_loss": avg_branch_aux,
                    "train/grad_norm": avg_grad_norm,
                    "train/lr": lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_seen_B": tokens_seen / 1e9,
                }
                accelerator.print(
                    f"step {global_step:6d}  "
                    f"loss={avg_total:.4f}  ce={avg_ce:.4f}  aux={avg_aux:.4f}  "
                    f"seq_aux={avg_seq_aux:.4f}  branch_aux={avg_branch_aux:.4f}  "
                    f"lr={lr:.2e}  tok/s={tok_per_sec/1e3:.1f}k  |g|={avg_grad_norm:.3f}"
                )
                if bias_stats:
                    log_dict.update(bias_stats)
                if log_with:
                    accelerator.log(log_dict, step=global_step)

            if global_step % routing_log_every == 0:
                if expert_count_accum is not None and accelerator.num_processes > 1:
                    for layer_idx in expert_count_accum:
                        torch.distributed.all_reduce(expert_count_accum[layer_idx], op=torch.distributed.ReduceOp.SUM)

                if expert_margin_accum and accelerator.num_processes > 1:
                    for entry in expert_margin_accum.values():
                        torch.distributed.all_reduce(entry["sum"], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(entry["count"], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(entry["min"], op=torch.distributed.ReduceOp.MIN)

                if attention_expert_count_accum and accelerator.num_processes > 1:
                    for router_accum in attention_expert_count_accum.values():
                        for layer_idx in router_accum:
                            torch.distributed.all_reduce(router_accum[layer_idx], op=torch.distributed.ReduceOp.SUM)

                if attention_router_margin_accum and accelerator.num_processes > 1:
                    for router_accum in attention_router_margin_accum.values():
                        for entry in router_accum.values():
                            torch.distributed.all_reduce(entry["sum"], op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(entry["count"], op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(entry["min"], op=torch.distributed.ReduceOp.MIN)

                if branch_prob_accum and accelerator.num_processes > 1:
                    for entry in branch_prob_accum.values():
                        torch.distributed.all_reduce(entry["sum"], op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(entry["count"], op=torch.distributed.ReduceOp.SUM)

                # Collect norm bank counts (all ranks need to participate in all_reduce)
                norm_counts = {}
                unwrapped = accelerator.unwrap_model(model)
                for name, module in unwrapped.named_modules():
                    if isinstance(module, NormExpertBank):
                        counts = module.local_tokens_per_expert.clone()
                        if accelerator.num_processes > 1:
                            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                        label = "attn_norm" if "attn" in name else "mlp_norm"
                        norm_counts[label] = counts
                        module.local_tokens_per_expert.zero_()

                if accelerator.is_main_process:
                    scalar_stats = {}
                    if expert_count_accum is not None:
                        rstats = compute_routing_stats_from_counts(
                            expert_count_accum,
                            is_global=shared_mlp_pool,
                        )
                        scalar_stats.update({k: v for k, v in rstats.items() if not k.startswith("_hist/")})
                    scalar_stats.update(router_margin_accumulator_to_stats(expert_margin_accum))

                    for router_name, router_accum in sorted(attention_expert_count_accum.items()):
                        astats = compute_routing_stats_from_counts(
                            router_accum,
                            is_global=True,
                            prefix=f"routing/attention/{router_name}",
                        )
                        scalar_stats.update({k: v for k, v in astats.items() if not k.startswith("_hist/")})
                        scalar_stats.update(
                            router_margin_accumulator_to_stats(
                                attention_router_margin_accum.get(router_name),
                                prefix=f"routing/attention/{router_name}",
                            )
                        )

                    scalar_stats.update(branch_accumulator_to_stats(branch_prob_accum))

                    for label, counts in norm_counts.items():
                        active = int((counts > 0).sum().item())
                        scalar_stats[f"routing/{label}_global_active_experts"] = active

                    # Surface global active expert counts at top level for easy tracking
                    if "routing/global_pool_num_active" in scalar_stats:
                        scalar_stats["routing/mlp_global_active_experts"] = scalar_stats["routing/global_pool_num_active"]
                    for rname in ("q", "k", "v", "o", "attn"):
                        key = f"routing/attention/{rname}/global_pool_num_active"
                        if key in scalar_stats:
                            scalar_stats[f"routing/{rname}_global_active_experts"] = scalar_stats[key]

                    if log_with and scalar_stats:
                        accelerator.log(scalar_stats, step=global_step)

                    routing_dir = os.path.join(train_cfg.output_dir, "routing_logs")
                    step_dir = os.path.join(routing_dir, f"step_{global_step:08d}")
                    os.makedirs(step_dir, exist_ok=True)
                    snapshot = {"step": global_step}
                    snapshot.update(counts_accumulator_to_snapshot(expert_count_accum, is_global=shared_mlp_pool))
                    snapshot["attention"] = {
                        router_name: counts_accumulator_to_snapshot(router_accum, is_global=True)
                        for router_name, router_accum in sorted(attention_expert_count_accum.items())
                    }
                    snapshot["branch"] = branch_accumulator_to_snapshot(branch_prob_accum)
                    snapshot["norms"] = {
                        label: {
                            "token_counts": counts.cpu().tolist(),
                            "token_fracs": (counts / max(counts.sum().item(), 1.0)).cpu().tolist(),
                        }
                        for label, counts in norm_counts.items()
                    }

                    json_path = os.path.join(step_dir, "snapshot.json")
                    with open(json_path, "w") as f:
                        json.dump(snapshot, f)

                    plot_routing_snapshot(snapshot, step_dir, global_step)

                expert_count_accum = None
                expert_margin_accum = None
                attention_expert_count_accum = {}
                attention_router_margin_accum = {}
                branch_prob_accum = None

            if global_step % train_cfg.save_every == 0:
                ds_state = dataset.get_state()
                save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    global_step, train_cfg.output_dir, ds_state,
                    wandb_run_id=wandb_run_id,
                    tokens_seen=tokens_seen,
                )
                if args.max_checkpoints > 0 and accelerator.is_main_process:
                    cleanup_checkpoints(train_cfg.output_dir, args.max_checkpoints)

            loss_window_sum = 0.0
            ce_window_sum = 0.0
            aux_window_sum = 0.0
            seq_aux_window_sum = 0.0
            branch_aux_window_sum = 0.0
            microbatches_in_step = 0
            local_tokens_in_step = 0

    # Final checkpoint
    save_checkpoint(
        accelerator, model, optimizer, scheduler,
        global_step, train_cfg.output_dir,
        dataset.get_state(),
        wandb_run_id=wandb_run_id,
        tokens_seen=tokens_seen,
    )
    if args.max_checkpoints > 0 and accelerator.is_main_process:
        cleanup_checkpoints(train_cfg.output_dir, args.max_checkpoints)
    accelerator.end_training()
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
