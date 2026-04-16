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
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

def configure_liger_kernels(cfg: dict) -> str:
    """Apply the safe Liger subset for the current config and return a summary.

    Import-time patching leaks into unrelated CPU-only tests that import train.py,
    causing Triton kernels to run on CPU tensors.

    For `moe_everything`, the full Qwen3-MoE patch is not safe: the `swiglu`
    expert replacement breaks the learned-routing path and pushes fresh-init CE
    from ~12 to ~20. The fused linear CE patch also does not currently help
    `MoEverythingForCausalLM`, because that patch only replaces the stock
    `Qwen3MoeForCausalLM.forward`. Keep only the validated-safe RoPE and RMSNorm
    patches for that model family.
    """
    training_cfg = cfg.get("training", {})
    if training_cfg.get("disable_liger", False) or os.environ.get("MOE_DISABLE_LIGER", "0") == "1":
        return "disabled"

    if cfg["model"]["type"] == "gpt2_dense":
        return "disabled (unsupported for gpt2_dense)"

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

    if cfg["model"]["type"] == "moe_everything":
        apply_liger_kernel_to_qwen3_moe(
            rope=True,
            rms_norm=True,
            swiglu=False,
            fused_linear_cross_entropy=False,
            cross_entropy=False,
        )
        return "partial (rope+rms_norm only; swiglu/fused CE disabled for moe_everything)"

    apply_liger_kernel_to_qwen3_moe()
    return "full"

from src.data import (
    DataConfig as ParquetDataConfig,
    StatefulParquetDataset,
    StatefulTokenBinDataset,
    TokenBinConfig,
)
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
    SpeedrunMoEverythingConfig,
    SpeedrunMoEverythingForCausalLM,
)
from src.models.router import DeepSeekRouter
from src.models.load_balancing import (
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
)
from src.models.init_mapping import copy_global_to_alternating_sanity
from src.utils.training import (
    TrainingConfig,
    build_lr_scheduler,
    build_muon_optimizer,
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


def get_bias_update_routers(model, *, mlp_only: bool = False) -> list[DeepSeekRouter]:
    """Collect DeepSeek routers that should receive expert-bias updates."""
    return [router for group in get_bias_update_router_groups(model, mlp_only=mlp_only) for router in group]


def get_bias_update_router_groups(model, *, mlp_only: bool = False) -> list[list[DeepSeekRouter]]:
    """Collect DeepSeek routers grouped by routed expert pool.

    Global-style bias updates should pool counts only across routers that map
    into the same logical expert bank. For `moe_everything`, that means the
    MLP gate(s) are one group and each attention-router family (q/k/v/o or
    bundled variants) is its own group. Standard/global MoE fall back to a
    single flat group because all DeepSeek routers operate on the same MLP
    expert pool.
    """
    inner_model = getattr(model, "model", None)
    groups: list[list[DeepSeekRouter]] = []

    def _append_group(group) -> None:
        if group is None:
            return
        if isinstance(group, DeepSeekRouter):
            groups.append([group])
            return
        routers = [router for router in group if isinstance(router, DeepSeekRouter)]
        if routers:
            groups.append(routers)

    if inner_model is not None:
        mlp_bank = getattr(inner_model, "mlp_bank", None)
        if mlp_bank is not None:
            gates = getattr(mlp_bank, "gates", None)
            if gates is not None:
                _append_group(gates)
            else:
                _append_group(getattr(mlp_bank, "gate", None))

        if mlp_only:
            return groups

        attn_bank = getattr(inner_model, "attn_bank", None)
        if attn_bank is not None:
            for name in ("routers", "q_routers", "k_routers", "v_routers", "o_routers", "kv_routers", "qk_routers"):
                _append_group(getattr(attn_bank, name, None))
            for name in ("router", "q_router", "k_router", "v_router", "o_router", "kv_router", "qk_router"):
                _append_group(getattr(attn_bank, name, None))

    if groups:
        return groups

    fallback = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    return [fallback] if fallback else []


def _reduce_router_counts(router: DeepSeekRouter, accelerator) -> torch.Tensor:
    counts = router.local_tokens_per_expert.clone()
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    router.local_tokens_per_expert.zero_()
    return counts


def _apply_global_bias_update_to_group(
    router_group: list[DeepSeekRouter],
    accelerator,
    update_rate: float,
    alpha: float,
) -> None:
    per_router_counts = [_reduce_router_counts(router, accelerator) for router in router_group]
    global_counts = torch.stack(per_router_counts).sum(dim=0)
    global_avg = global_counts.mean()
    global_delta = torch.sign(global_avg - global_counts) * update_rate

    for router, counts in zip(router_group, per_router_counts):
        if alpha > 0:
            layer_avg = counts.mean()
            layer_delta = torch.sign(layer_avg - counts) * update_rate
            router.expert_bias += alpha * layer_delta + (1 - alpha) * global_delta
        else:
            router.expert_bias += global_delta


def _apply_per_router_bias_update(
    router_group: list[DeepSeekRouter],
    accelerator,
    update_rate: float,
) -> None:
    for router in router_group:
        counts = _reduce_router_counts(router, accelerator)
        avg = counts.mean()
        router.expert_bias += torch.sign(avg - counts) * update_rate


def _flatten_router_groups(router_groups: list[list[DeepSeekRouter]]) -> list[DeepSeekRouter]:
    return [router for group in router_groups for router in group]


def _bias_stats_for_routers(routers: list[DeepSeekRouter]) -> dict:
    stats = {}
    if not routers:
        return stats

    all_bias = torch.cat([r.expert_bias for r in routers])
    stats["routing/expert_bias_mean"] = all_bias.mean().item()
    stats["routing/expert_bias_std"] = all_bias.std().item()
    stats["routing/expert_bias_min"] = all_bias.min().item()
    stats["routing/expert_bias_max"] = all_bias.max().item()
    return stats


def update_expert_biases(
    model, update_rate: float, accelerator,
    is_global: bool = False, alpha: float = 0.0,
    routers: list[DeepSeekRouter] | None = None,
    router_groups: list[list[DeepSeekRouter]] | None = None,
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
    if router_groups is None:
        if routers is not None:
            router_groups = [routers] if routers else []
        else:
            router_groups = get_bias_update_router_groups(model)

    router_groups = [group for group in router_groups if group]
    if not router_groups:
        return {}

    if is_global:
        for group in router_groups:
            _apply_global_bias_update_to_group(group, accelerator, update_rate, alpha)
    else:
        for group in router_groups:
            _apply_per_router_bias_update(group, accelerator, update_rate)

    return _bias_stats_for_routers(_flatten_router_groups(router_groups))


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


def get_output_router_token_masks(output) -> tuple[torch.Tensor | None, ...] | None:
    masks = getattr(output, "router_token_masks", None)
    if masks:
        return tuple(masks)
    return None


def get_attention_router_topk(model_cfg, router_name: str) -> int:
    attn_mode = getattr(model_cfg, "attn_expert_mode", None)
    if attn_mode == "per_head_fully_independent":
        return model_cfg.num_key_value_heads if router_name in ("k", "v") else model_cfg.num_attention_heads
    if attn_mode == "per_head_precompute_kv":
        return model_cfg.num_attention_heads
    return model_cfg.num_attn_experts_per_tok


def compute_output_metrics(
    output,
    raw_model,
    model_cfg,
    input_ids: torch.Tensor,
    *,
    seq_aux_loss_coef: float,
) -> tuple[dict[str, float], tuple[torch.Tensor, ...] | None, tuple[torch.Tensor | None, ...] | None]:
    router_token_masks = get_output_router_token_masks(output)
    selected_experts = get_output_selected_experts(output, raw_model)

    aux = getattr(output, "aux_loss", None)
    aux_value = aux.detach().float().item() if isinstance(aux, torch.Tensor) else float(aux or 0.0)

    aux_normalized = None
    if getattr(output, "router_logits", None) is not None:
        aux_normalized = normalized_load_balancing_loss_func(
            output.router_logits,
            model_cfg.num_experts,
            model_cfg.num_experts_per_tok,
            token_masks=router_token_masks,
            selected_experts=selected_experts,
        )
    if isinstance(aux_normalized, torch.Tensor):
        aux_normalized_value = aux_normalized.detach().float().item()
    elif aux_normalized is not None:
        aux_normalized_value = float(aux_normalized)
    else:
        aux_normalized_value = 0.0

    total_value = output.loss.detach().float().item()

    ce_tensor = getattr(output, "ce_loss", None)
    if isinstance(ce_tensor, torch.Tensor):
        ce_value = ce_tensor.detach().float().item()
    elif ce_tensor is not None:
        ce_value = float(ce_tensor)
    else:
        ce_value = total_value - getattr(raw_model, "router_aux_loss_coef", 0.0) * aux_value

    seq_aux = getattr(output, "seq_aux_loss", None)
    if seq_aux is None and seq_aux_loss_coef > 0 and getattr(output, "router_logits", None) is not None:
        seq_aux = seq_load_balancing_loss_func(
            output.router_logits,
            model_cfg.num_experts,
            model_cfg.num_experts_per_tok,
            batch_size=input_ids.shape[0],
            selected_experts=selected_experts,
            token_masks=router_token_masks,
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

    attention_aux = getattr(output, "attention_aux_loss", None)
    if isinstance(attention_aux, torch.Tensor):
        attention_aux_value = attention_aux.detach().float().item()
    elif attention_aux is not None:
        attention_aux_value = float(attention_aux)
    else:
        attention_aux_value = 0.0

    if ce_tensor is None:
        ce_value -= seq_aux_loss_coef * seq_aux_value
        ce_value -= getattr(raw_model, "branch_router_aux_loss_coef", 0.0) * branch_aux_value

    metrics = {
        "loss": total_value,
        "ce_loss": ce_value,
        "aux_loss": aux_value,
        "aux_loss_normalized": aux_normalized_value,
        "seq_aux_loss": seq_aux_value,
        "branch_aux_loss": branch_aux_value,
        "attention_aux_loss": attention_aux_value,
    }
    return metrics, selected_experts, router_token_masks


def reduce_scalar(accelerator: Accelerator, value: float, reduction: str = "mean") -> float:
    tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        if reduction == "mean":
            tensor /= accelerator.num_processes
    return tensor.item()


@torch.no_grad()
def run_validation(
    *,
    accelerator: Accelerator,
    model,
    model_cfg,
    eval_dataloader,
    max_batches: int,
    is_dense: bool,
    seq_aux_loss_coef: float,
) -> dict[str, float]:
    if eval_dataloader is None:
        return {}

    was_training = model.training
    # Skip model.eval() when torch.compile is active — it triggers expensive
    # recompilation for a new graph.  Since dropout is 0 and we use RMSNorm
    # (no running stats), eval mode is a no-op for these models.
    _is_compiled = hasattr(accelerator.unwrap_model(model), "_orig_mod")
    if not _is_compiled:
        model.eval()
    raw_model = accelerator.unwrap_model(model)

    totals = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "aux_loss": 0.0,
        "aux_loss_normalized": 0.0,
        "seq_aux_loss": 0.0,
        "branch_aux_loss": 0.0,
        "attention_aux_loss": 0.0,
    }
    batches = 0

    for batch_idx, batch in enumerate(eval_dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"]
        labels = input_ids  # HF CausalLM models shift labels internally
        output = model(
            input_ids=input_ids,
            labels=labels,
            **({} if is_dense else {"output_router_logits": True}),
        )
        metrics, _, _ = compute_output_metrics(
            output,
            raw_model,
            model_cfg,
            input_ids,
            seq_aux_loss_coef=seq_aux_loss_coef,
        )
        for key in totals:
            totals[key] += metrics[key]
        batches += 1

    if was_training and not _is_compiled:
        model.train()

    if batches == 0:
        return {}

    averaged = {
        f"eval/{key}": reduce_scalar(accelerator, value / batches)
        for key, value in totals.items()
    }
    averaged["eval/perplexity"] = math.exp(min(20.0, averaged["eval/ce_loss"]))
    averaged["eval/num_batches"] = float(batches)
    return averaged


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


def get_data_format(cfg_dict: dict) -> str:
    return cfg_dict.get("format", "parquet")


def uses_tokenizer(cfg_dict: dict) -> bool:
    return get_data_format(cfg_dict) == "parquet"


def build_dataset_from_config(
    cfg_dict: dict,
    *,
    rank: int,
    world_size: int,
    seed: int,
    tokenizer=None,
):
    data_format = get_data_format(cfg_dict)
    if data_format == "parquet":
        data_cfg = ParquetDataConfig(
            data_dir=cfg_dict["data_dir"],
            text_column=cfg_dict.get("text_column", "text"),
            seq_len=cfg_dict.get("seq_len", 2048),
            tokenizer_name=cfg_dict.get("tokenizer_name", "gpt2"),
            num_workers=cfg_dict.get("num_workers", 4),
            split=cfg_dict.get("split", "all"),
            holdout_fraction=cfg_dict.get("holdout_fraction", 0.0),
        )
        return StatefulParquetDataset(
            config=data_cfg,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
    if data_format == "token_bin":
        data_cfg = TokenBinConfig(
            files_glob=cfg_dict["files_glob"],
            seq_len=cfg_dict.get("seq_len", 2048),
            header_bytes=cfg_dict.get("header_bytes", 1024),
            token_dtype=cfg_dict.get("token_dtype", "uint16"),
            max_tokens=cfg_dict.get("max_tokens"),
            shuffle_files=cfg_dict.get("shuffle_files", False),
            repeat=cfg_dict.get("repeat", False),
            align_to_bos=cfg_dict.get("align_to_bos", False),
        )
        return StatefulTokenBinDataset(
            config=data_cfg,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
    raise ValueError(f"Unknown data format: {data_format}")


def resolve_related_config_path(config_path: str, related_path: str) -> str:
    resolved = Path(related_path)
    if resolved.is_absolute():
        return str(resolved)
    return str((Path(config_path).resolve().parent / resolved).resolve())


def resolve_initialization_spec(
    cfg: dict,
    *,
    config_path: str,
    cli_source_config: str | None = None,
    cli_strategy: str | None = None,
) -> dict | None:
    spec = dict(cfg.get("initialization") or {})
    if cli_source_config is not None:
        spec["source_config"] = cli_source_config
    if cli_strategy is not None:
        spec["strategy"] = cli_strategy
    if not spec:
        return None
    if "source_config" not in spec or "strategy" not in spec:
        raise ValueError("initialization requires both 'source_config' and 'strategy'")
    spec["source_config"] = resolve_related_config_path(config_path, spec["source_config"])
    return spec


def build_model(cfg: dict):
    mtype = cfg["model"]["type"]
    mcfg = cfg["model"]
    attn_impl = mcfg.get("attn_implementation", "sdpa")

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
        config._attn_implementation = attn_impl
        # Dense model has no MoE fields — set dummies for compatibility
        config.num_experts = 0
        config.num_experts_per_tok = 0
        model = Qwen3ForCausalLM(config)
        return model, config

    if mtype == "gpt2_dense":
        config = GPT2Config(
            vocab_size=mcfg["vocab_size"],
            n_embd=mcfg["hidden_size"],
            n_layer=mcfg["num_hidden_layers"],
            n_head=mcfg["num_attention_heads"],
            n_positions=mcfg.get("max_position_embeddings", 1024),
            n_ctx=mcfg.get("max_position_embeddings", 1024),
            n_inner=mcfg.get("intermediate_size"),
            resid_pdrop=mcfg.get("resid_pdrop", 0.0),
            embd_pdrop=mcfg.get("embd_pdrop", 0.0),
            attn_pdrop=mcfg.get("attention_dropout", 0.0),
            layer_norm_epsilon=mcfg.get("layer_norm_epsilon", 1e-5),
            tie_word_embeddings=mcfg.get("tie_word_embeddings", True),
        )
        config.num_experts = 0
        config.num_experts_per_tok = 0
        model = GPT2LMHeadModel(config)
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
        seq_aux_loss_coef=mcfg.get("seq_aux_loss_coef", 0.0),
        norm_topk_prob=mcfg.get("norm_topk_prob", True),
        num_experts_per_tok=mcfg["num_experts_per_tok"],
        output_router_logits=True,
        attn_implementation=attn_impl,
    )

    def _set_router_params(config, mcfg):
        """Attach router params that Qwen3MoeConfig doesn't have natively."""
        config.router_exploration_rate = mcfg.get("router_exploration_rate", 0.0)

    def _set_deepseek_router_params(config, mcfg):
        """Attach DeepSeek V3 router params that Qwen3MoeConfig doesn't have natively."""
        _set_router_params(config, mcfg)
        config.topk_scaling_factor = mcfg.get("topk_scaling_factor", None)
        config.num_groups = mcfg.get("num_groups", None)
        config.group_topk = mcfg.get("group_topk", None)

    if mtype == "standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        _set_router_params(config, mcfg)
        model = StandardMoEModel(config)
    elif mtype == "deepseek_standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        _set_deepseek_router_params(config, mcfg)
        model = DeepSeekStandardMoEModel(config)
    elif mtype == "global_moe":
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        _set_router_params(config, mcfg)
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
            branch_router_aux_loss_coef=mcfg.get("branch_router_aux_loss_coef", 0.0),
            use_deepseek_routing=mcfg.get("use_deepseek_routing", False),
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            per_layer_router=mcfg.get("per_layer_router", False),
            per_layer_mlp_router=mcfg.get("per_layer_mlp_router", False),
            per_layer_attn_router=mcfg.get("per_layer_attn_router", False),
            routed_norm=mcfg.get("routed_norm", False),
            per_layer_norm=mcfg.get("per_layer_norm", False),
            per_layer_qk_norm=mcfg.get("per_layer_qk_norm", False),
            post_norm=mcfg.get("post_norm", False),
            dynamic_depth_min=mcfg.get("dynamic_depth_min", 1.0),
            dynamic_depth_max=mcfg.get("dynamic_depth_max", 1.0),
            depthwise_attention=mcfg.get("depthwise_attention", False),
            depthwise_block_size=mcfg.get("depthwise_block_size", 0),
            per_head_compute_mode=mcfg.get("per_head_compute_mode", "auto"),
            per_head_dense_fraction_threshold=mcfg.get(
                "per_head_dense_fraction_threshold", 0.75
            ),
            sanity_check_mode=mcfg.get("sanity_check_mode"),
            scale_attn_by_routing_weight=mcfg.get("scale_attn_by_routing_weight", True),
            scale_branch_by_routing_weight=mcfg.get("scale_branch_by_routing_weight", True),
            router_exploration_rate=mcfg.get("router_exploration_rate", 0.0),
            branch_router_exploration_rate=mcfg.get("branch_router_exploration_rate"),
            **common,
        )
        model = MoEverythingForCausalLM(config)
    elif mtype == "speedrun_moe_everything":
        config = SpeedrunMoEverythingConfig(
            num_experts=mcfg["num_experts"],
            num_attn_experts=mcfg.get("num_attn_experts", 4),
            num_attn_experts_per_tok=mcfg.get("num_attn_experts_per_tok", 1),
            attn_expert_mode=mcfg.get("attn_expert_mode", "bundled"),
            branch_router_aux_loss_coef=mcfg.get("branch_router_aux_loss_coef", 0.0),
            use_deepseek_routing=mcfg.get("use_deepseek_routing", False),
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            per_layer_router=mcfg.get("per_layer_router", False),
            per_layer_mlp_router=mcfg.get("per_layer_mlp_router", False),
            per_layer_attn_router=mcfg.get("per_layer_attn_router", False),
            routed_norm=mcfg.get("routed_norm", False),
            per_layer_norm=mcfg.get("per_layer_norm", False),
            per_layer_qk_norm=mcfg.get("per_layer_qk_norm", False),
            post_norm=mcfg.get("post_norm", False),
            dynamic_depth_min=mcfg.get("dynamic_depth_min", 1.0),
            dynamic_depth_max=mcfg.get("dynamic_depth_max", 1.0),
            depthwise_attention=mcfg.get("depthwise_attention", False),
            depthwise_block_size=mcfg.get("depthwise_block_size", 0),
            per_head_compute_mode=mcfg.get("per_head_compute_mode", "auto"),
            per_head_dense_fraction_threshold=mcfg.get(
                "per_head_dense_fraction_threshold", 0.75
            ),
            sanity_check_mode=mcfg.get("sanity_check_mode"),
            scale_attn_by_routing_weight=mcfg.get("scale_attn_by_routing_weight", True),
            scale_branch_by_routing_weight=mcfg.get("scale_branch_by_routing_weight", True),
            router_exploration_rate=mcfg.get("router_exploration_rate", 0.0),
            branch_router_exploration_rate=mcfg.get("branch_router_exploration_rate"),
            **common,
        )
        model = SpeedrunMoEverythingForCausalLM(config)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Path to checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-from-config",
        default=None,
        help="Optional source config for mapped initialization (resolved relative to --config if needed).",
    )
    parser.add_argument(
        "--init-strategy",
        choices=("global_to_alternating_sanity",),
        default=None,
        help="Optional mapped-initialization strategy override.",
    )
    parser.add_argument("--auto_resume", action="store_true",
                        help="Auto-find and resume from latest checkpoint in output_dir")
    parser.add_argument("--data_dir", default=None,
                        help="Override data.data_dir from config")
    parser.add_argument("--output_dir", default=None,
                        help="Override training.output_dir from config")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override training.max_steps from config")
    parser.add_argument("--max_checkpoints", type=int, default=0,
                        help="Max checkpoints to keep (0 = unlimited)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    initialization_spec = resolve_initialization_spec(
        cfg,
        config_path=args.config,
        cli_source_config=args.init_from_config,
        cli_strategy=args.init_strategy,
    )
    liger_mode = configure_liger_kernels(cfg)
    print(
        f"Liger kernels: {liger_mode} for model type '{cfg['model']['type']}'",
        flush=True,
    )
    tcfg_dict = cfg["training"]
    dcfg_dict = cfg.get("data", {})
    eval_cfg_dict = cfg.get("eval", {})

    # --- CLI overrides -------------------------------------------------------
    if args.data_dir:
        dcfg_dict["data_dir"] = args.data_dir
    if args.output_dir:
        tcfg_dict["output_dir"] = args.output_dir
    if args.max_steps is not None:
        tcfg_dict["max_steps"] = args.max_steps

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
    accelerator_mixed_precision = "no" if train_cfg.mixed_precision == "fp32" else train_cfg.mixed_precision
    ddp_kwargs = []
    if cfg["model"]["type"] == "moe_everything":
        # Shared-parameter MoE paths are not safe under DDP static_graph. In
        # practice this can silently corrupt training instead of raising the
        # usual unused-parameter error, especially in sanity modes with
        # deterministic branch routing.
        ddp_kwargs.append(DistributedDataParallelKwargs(static_graph=False))
    accelerator = Accelerator(
        mixed_precision=accelerator_mixed_precision,
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
    init_summary = None
    if initialization_spec is not None:
        set_seed(args.seed + accelerator.process_index)
        source_cfg = load_config(initialization_spec["source_config"])
        source_model, _ = build_model(source_cfg)
        strategy = initialization_spec["strategy"]
        if strategy == "global_to_alternating_sanity":
            pairs = copy_global_to_alternating_sanity(source_model, model)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        init_summary = {
            "strategy": strategy,
            "source_config": initialization_spec["source_config"],
            "num_pairs": len(pairs),
        }
        del source_model

    # --- torch.compile -------------------------------------------------------
    if tcfg_dict.get("torch_compile", False):
        compile_mode = tcfg_dict.get("torch_compile_mode", "default")
        accelerator.print(f"Compiling model with torch.compile(mode={compile_mode!r})")
        model = torch.compile(model, dynamic=False, mode=compile_mode)

    params = count_parameters(model)
    expert_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "gate_up_proj" in n or "down_proj" in n or "mlp.c_fc" in n or "mlp.c_proj" in n
    )
    is_dense = cfg["model"]["type"] in {"dense", "gpt2_dense"}
    is_global = cfg["model"]["type"] in ("global_moe", "deepseek_global_moe")
    is_moe_everything = cfg["model"]["type"] == "moe_everything"
    shared_mlp_pool = is_global or is_moe_everything
    global_router_update = cfg["model"].get("global_router_update", False)
    global_like_sanity = (
        is_moe_everything
        and cfg["model"].get("sanity_check_mode") == "alternating_global_moe"
    )
    bias_update_rate = cfg["model"].get("bias_update_rate", 0.0)
    bias_interpolation = cfg["model"].get("bias_interpolation", False)
    bias_interpolation_warmup_steps = cfg["model"].get("bias_interpolation_warmup_steps", 5000)
    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    router_exploration_rate = cfg["model"].get("router_exploration_rate", 0.0)
    eval_enabled = bool(eval_cfg_dict.get("enabled", False))
    eval_every = max(1, int(eval_cfg_dict.get("every", max(1, train_cfg.save_every))))
    eval_max_batches = int(eval_cfg_dict.get("max_batches", 0))

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
    if init_summary is not None:
        accelerator.print(
            "Mapped initialization enabled: "
            f"{init_summary['strategy']} from {init_summary['source_config']} "
            f"({init_summary['num_pairs']} mapped tensors)"
        )

    attn_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(
            x in n
            for x in [
                "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "init_k", "init_v",
                "attn.c_attn", "attn.c_proj",
            ]
        )
    )
    router_params = sum(
        p.numel() for n, p in model.named_parameters()
        if ("router" in n or "branch" in n or ("gate" in n and "gate_up" not in n and "gate_proj" not in n))
    )
    embed_params = sum(
        p.numel() for n, p in model.named_parameters()
        if ("embed" in n or ".wte." in n or ".wpe." in n or n.startswith("transformer.wte") or n.startswith("transformer.wpe"))
    )

    summary_lines = [
        f"\n{'='*60}",
        f"  Model     : {cfg['model']['type']}",
        f"  Params    : {params['total']/1e9:.3f}B total",
        f"    Embed   : {embed_params/1e6:.1f}M",
        f"    Attn    : {attn_params/1e6:.1f}M",
        f"    MLP     : {expert_params/1e6:.1f}M",
        f"    Router  : {router_params/1e6:.1f}M",
    ]

    mcfg = cfg["model"]
    summary_lines.append(f"  Architecture:")
    summary_lines.append(f"    hidden   : {mcfg['hidden_size']}")
    summary_lines.append(f"    heads    : {mcfg['num_attention_heads']}Q / {mcfg['num_key_value_heads']}KV")
    summary_lines.append(f"    head_dim : {mcfg.get('head_dim', mcfg['hidden_size'] // mcfg['num_attention_heads'])}")
    summary_lines.append(f"    layers   : {mcfg['num_hidden_layers']}")
    if is_dense:
        summary_lines.append(f"    MLP      : dense FFN (intermediate={mcfg['intermediate_size']})")
    else:
        summary_lines.append(f"    MLP exp  : {mcfg['num_experts']} pool, top-{mcfg['num_experts_per_tok']}")

    if is_moe_everything:
        attn_mode = mcfg.get("attn_expert_mode", "bundled")
        n_attn_exp = mcfg.get("num_attn_experts", 4)
        n_attn_top = mcfg.get("num_attn_experts_per_tok", 1)
        scale_attn = mcfg.get("scale_attn_by_routing_weight", True)
        summary_lines.append(f"    Attn mode: {attn_mode}")
        summary_lines.append(f"    Attn exp : {n_attn_exp} pool, top-{n_attn_top}")
        scale_branch = mcfg.get("scale_branch_by_routing_weight", True)
        summary_lines.append(f"    Scale attn by routing weight: {scale_attn}")
        summary_lines.append(f"    Scale branch by routing weight: {scale_branch}")
        if attn_mode == "per_head_fully_independent":
            num_heads = mcfg["num_attention_heads"]
            num_kv = mcfg["num_key_value_heads"]
            q_per_kv = num_heads // num_kv
            e_o = n_attn_exp * q_per_kv
            summary_lines.append(f"    Q pool   : {n_attn_exp} experts, top-{num_kv} (bundled per KV group)")
            summary_lines.append(f"    K/V pool : {n_attn_exp} experts, top-{num_kv} (per KV head)")
            summary_lines.append(f"    O pool   : {e_o} experts, top-{num_heads} (per query head)")
        elif attn_mode == "per_head_precompute_kv":
            num_kv = mcfg["num_key_value_heads"]
            summary_lines.append(f"    QKVO pool: {n_attn_exp} experts, top-{num_kv} (bundled per KV group)")
        if mcfg.get("per_layer_attn_router"):
            summary_lines.append(f"    Per-layer attn router: yes")
        if mcfg.get("per_layer_mlp_router"):
            summary_lines.append(f"    Per-layer MLP router: yes")
        if mcfg.get("per_layer_norm"):
            summary_lines.append(f"    Per-layer norm: yes")
        if mcfg.get("sanity_check_mode"):
            summary_lines.append(f"    Sanity   : {mcfg['sanity_check_mode']}")
    summary_lines.append(f"    Router exploration: {router_exploration_rate}")

    summary_lines.append(f"  Training:")
    summary_lines.append(f"    Dist type: {accelerator.distributed_type}")
    summary_lines.append(f"    Precision: {train_cfg.mixed_precision}")
    summary_lines.append(f"    GPUs     : {accelerator.num_processes}")
    data_format = get_data_format(dcfg_dict)
    summary_lines.append(f"    Data fmt : {data_format}")
    if eval_enabled:
        eval_format = get_data_format(eval_cfg_dict or dcfg_dict)
        if eval_format == "token_bin":
            eval_source = eval_cfg_dict.get("files_glob", dcfg_dict.get("files_glob"))
            eval_limit = eval_cfg_dict.get("max_tokens")
            batch_desc = "full dataset" if eval_max_batches <= 0 else f"{eval_max_batches} batches"
            summary_lines.append(
                f"    Eval     : every {eval_every} steps, {batch_desc}, "
                f"max_tokens={eval_limit} from {eval_source}"
            )
        else:
            eval_source = eval_cfg_dict.get("data_dir", dcfg_dict["data_dir"])
            eval_holdout = eval_cfg_dict.get("holdout_fraction", 0.05)
            summary_lines.append(
                f"    Eval     : every {eval_every} steps, {eval_max_batches} batches, "
                f"holdout={eval_holdout} from {eval_source}"
            )
    summary_lines.append(f"{'='*60}")

    accelerator.print("\n".join(summary_lines))

    # --- Dataset ------------------------------------------------------------
    tokenizer = None
    if uses_tokenizer(dcfg_dict):
        tokenizer_name = dcfg_dict.get("tokenizer_name", "gpt2")
        if accelerator.local_process_index == 0:
            print(
                f"[rank {accelerator.process_index}] Loading tokenizer {tokenizer_name}",
                flush=True,
            )
        with accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if accelerator.local_process_index == 0:
            print(f"[rank {accelerator.process_index}] Tokenizer ready", flush=True)

    train_source = dcfg_dict.get("data_dir", dcfg_dict.get("files_glob"))
    if accelerator.local_process_index == 0:
        print(
            f"[rank {accelerator.process_index}] Building dataset from {train_source}",
            flush=True,
        )

    train_data_cfg = dict(dcfg_dict)
    if get_data_format(train_data_cfg) == "parquet":
        train_data_cfg["split"] = "all"
        train_data_cfg["holdout_fraction"] = 0.0

    dataset = build_dataset_from_config(
        train_data_cfg,
        tokenizer=tokenizer,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=dcfg_dict.get("seed", 42),
    )
    if accelerator.local_process_index == 0:
        dataset_desc = getattr(dataset, "files", None)
        if dataset_desc is not None:
            print(
                f"[rank {accelerator.process_index}] Dataset ready: {len(dataset.files)} shard files",
                flush=True,
            )
        total_sequences = getattr(dataset, "total_sequences", None)
        if total_sequences is not None:
            print(
                f"[rank {accelerator.process_index}] Dataset ready: {len(dataset.files)} bin shards, "
                f"{total_sequences} sequences/rank",
                flush=True,
            )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=0,       # must be 0 for IterableDataset state tracking
        pin_memory=True,
    )
    eval_dataloader = None
    if eval_enabled:
        eval_data_cfg = dict(dcfg_dict)
        eval_data_cfg.update(eval_cfg_dict)
        if get_data_format(eval_data_cfg) == "token_bin":
            eval_data_cfg["repeat"] = bool(eval_cfg_dict.get("repeat", False))
        if get_data_format(eval_data_cfg) == "parquet":
            eval_data_dir = eval_cfg_dict.get("data_dir", dcfg_dict["data_dir"])
            eval_split = eval_cfg_dict.get("split")
            if eval_split is None:
                eval_split = "all" if "data_dir" in eval_cfg_dict else "val"
            eval_data_cfg["data_dir"] = eval_data_dir
            eval_data_cfg["split"] = eval_split
            eval_data_cfg["holdout_fraction"] = eval_cfg_dict.get(
                "holdout_fraction",
                0.05 if eval_split == "val" else 0.0,
            )
        eval_dataset = build_dataset_from_config(
            eval_data_cfg,
            tokenizer=tokenizer,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            seed=eval_cfg_dict.get("seed", 1234),
        )
        eval_batch_size = int(eval_cfg_dict.get("batch_size", train_cfg.batch_size))
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            num_workers=0,
            pin_memory=True,
        )

    # --- Optimizer & Scheduler ----------------------------------------------
    optimizer_type = tcfg_dict.get("optimizer", "adamw")
    if optimizer_type == "muon":
        muon_lr = tcfg_dict.get("muon_lr", 0.02)
        muon_wd = tcfg_dict.get("muon_weight_decay", 0.0)
        adam_lr = tcfg_dict.get("adam_lr", 3e-4)
        optimizer = build_muon_optimizer(
            model, train_cfg, muon_lr=muon_lr, muon_weight_decay=muon_wd, adam_lr=adam_lr,
        )
        accelerator.print(
            f"Using Muon optimizer: muon_lr={muon_lr}, muon_wd={muon_wd}, adam_lr={adam_lr}"
        )
    else:
        optimizer = build_optimizer(model, train_cfg)
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg)

    # --- Accelerate prepare (wraps model in DDP/FSDP) -----------------------
    accelerator.print("Calling accelerator.prepare()")
    if eval_dataloader is not None:
        model, optimizer, dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, eval_dataloader, scheduler
        )
    else:
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
    session_t0 = time.perf_counter()
    step_t0 = session_t0
    routing_log_every = tcfg_dict.get("routing_log_every", 50)
    expert_count_accum = None
    expert_margin_accum = None
    attention_expert_count_accum = {}
    attention_router_margin_accum = {}
    branch_prob_accum = None

    loss_window_sum = 0.0
    ce_window_sum = 0.0
    aux_window_sum = 0.0
    aux_normalized_window_sum = 0.0
    seq_aux_window_sum = 0.0
    branch_aux_window_sum = 0.0
    attention_aux_window_sum = 0.0
    microbatches_in_step = 0
    local_tokens_in_step = 0

    accelerator.print(f"Starting training from step {global_step}")

    while global_step < train_cfg.max_steps:
        if microbatches_in_step == 0:
            step_t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"]
        labels = input_ids  # HF CausalLM models shift labels internally

        with accelerator.accumulate(model):
            output = model(
                input_ids=input_ids,
                labels=labels,
                **({} if is_dense else {"output_router_logits": True}),
            )
            loss = output.loss
            raw_model = accelerator.unwrap_model(model)
            metrics, selected_experts, router_token_masks = compute_output_metrics(
                output,
                raw_model,
                model_cfg,
                input_ids,
                seq_aux_loss_coef=seq_aux_loss_coef,
            )

            loss_window_sum += metrics["loss"]
            ce_window_sum += metrics["ce_loss"]
            aux_window_sum += metrics["aux_loss"]
            aux_normalized_window_sum += metrics["aux_loss_normalized"]
            seq_aux_window_sum += metrics["seq_aux_loss"]
            branch_aux_window_sum += metrics["branch_aux_loss"]
            attention_aux_window_sum += metrics["attention_aux_loss"]
            microbatches_in_step += 1
            local_tokens_in_step += input_ids.numel()

            if getattr(output, "router_logits", None) is not None:
                expert_count_accum = accumulate_expert_counts(
                    output.router_logits,
                    num_experts_per_tok=model_cfg.num_experts_per_tok,
                    accumulator=expert_count_accum,
                    selected_experts=selected_experts,
                    token_masks=router_token_masks,
                )
                expert_margin_accum = accumulate_router_margins(
                    output.router_logits,
                    num_experts_per_tok=model_cfg.num_experts_per_tok,
                    accumulator=expert_margin_accum,
                    token_masks=router_token_masks,
                )

            attention_router_info = getattr(output, "attention_router_info", None)
            if attention_router_info is not None:
                router_names = sorted({name for depth_info in attention_router_info for name in depth_info})
                for router_name in router_names:
                    router_logits = []
                    router_selected = []
                    router_masks = []
                    for depth_info in attention_router_info:
                        info = depth_info.get(router_name)
                        if info is None:
                            continue
                        router_logits.append(info["router_logits"])
                        router_selected.append(info["selected_experts"])
                        router_masks.append(info.get("token_mask"))
                    attn_topk = get_attention_router_topk(model_cfg, router_name)
                    attention_expert_count_accum[router_name] = accumulate_expert_counts(
                        router_logits,
                        num_experts_per_tok=attn_topk,
                        accumulator=attention_expert_count_accum.get(router_name),
                        selected_experts=router_selected,
                        token_masks=router_masks,
                    )
                    attention_router_margin_accum[router_name] = accumulate_router_margins(
                        router_logits,
                        num_experts_per_tok=attn_topk,
                        accumulator=attention_router_margin_accum.get(router_name),
                        token_masks=router_masks,
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
                effective_global_bias = is_global or global_like_sanity or global_router_update
                bias_router_groups = None
                if global_like_sanity:
                    bias_router_groups = get_bias_update_router_groups(
                        accelerator.unwrap_model(model),
                        mlp_only=True,
                    )
                alpha = (
                    bias_alpha_schedule(global_step, warmup_steps=bias_interpolation_warmup_steps)
                    if (effective_global_bias and bias_interpolation)
                    else 0.0
                )
                bias_stats = update_expert_biases(
                    accelerator.unwrap_model(model),
                    bias_update_rate,
                    accelerator,
                    is_global=effective_global_bias,
                    alpha=alpha,
                    router_groups=bias_router_groups,
                )
                bias_stats["routing/bias_alpha"] = alpha
            else:
                bias_stats = {}

            scheduler.step()
            # Muon momentum warmup: ramp from 0.85 → 0.95 over first N steps
            if optimizer_type == "muon":
                from src.utils.muon import get_muon_momentum
                warmup = tcfg_dict.get("momentum_warmup_steps", 300)
                new_mom = get_muon_momentum(global_step, warmup_steps=warmup)
                for g in optimizer.param_groups:
                    if g.get("is_muon", False):
                        g["momentum"] = new_mom
            global_step += 1
            step_elapsed = time.perf_counter() - step_t0

            tokens_this_step = reduce_scalar(accelerator, float(local_tokens_in_step), reduction="sum")
            tokens_seen += tokens_this_step
            step_tokens_per_sec = tokens_this_step / max(step_elapsed, 1e-6)

            avg_total = reduce_scalar(accelerator, loss_window_sum / max(1, microbatches_in_step))
            avg_ce = reduce_scalar(accelerator, ce_window_sum / max(1, microbatches_in_step))
            avg_aux = reduce_scalar(accelerator, aux_window_sum / max(1, microbatches_in_step))
            avg_aux_normalized = reduce_scalar(
                accelerator, aux_normalized_window_sum / max(1, microbatches_in_step)
            )
            avg_seq_aux = reduce_scalar(accelerator, seq_aux_window_sum / max(1, microbatches_in_step))
            avg_branch_aux = reduce_scalar(accelerator, branch_aux_window_sum / max(1, microbatches_in_step))
            avg_attention_aux = reduce_scalar(
                accelerator, attention_aux_window_sum / max(1, microbatches_in_step)
            )
            avg_grad_norm = reduce_scalar(accelerator, grad_norm)

            if global_step % train_cfg.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                log_dict = {
                    "train/loss": avg_total,
                    "train/ce_loss": avg_ce,
                    "train/aux_loss": avg_aux,
                    "train/aux_loss_normalized": avg_aux_normalized,
                    "train/seq_aux_loss": avg_seq_aux,
                    "train/branch_aux_loss": avg_branch_aux,
                    "train/attention_aux_loss": avg_attention_aux,
                    "train/grad_norm": avg_grad_norm,
                    "train/lr": lr,
                    "train/tokens_per_sec": step_tokens_per_sec,
                    "train/sec_per_step": step_elapsed,
                    "train/tokens_seen_B": tokens_seen / 1e9,
                }
                accelerator.print(
                    f"step {global_step:6d}  "
                    f"loss={avg_total:.4f}  ce={avg_ce:.4f}  aux={avg_aux:.4f}  "
                    f"aux_n={avg_aux_normalized:.4f}  "
                    f"seq_aux={avg_seq_aux:.4f}  branch_aux={avg_branch_aux:.4f}  "
                    f"attn_aux={avg_attention_aux:.4f}  "
                    f"lr={lr:.2e}  tok/s={step_tokens_per_sec/1e3:.1f}k  "
                    f"sec/step={step_elapsed:.3f}  |g|={avg_grad_norm:.3f}"
                )
                if bias_stats:
                    log_dict.update(bias_stats)
                if log_with:
                    accelerator.log(log_dict, step=global_step)

            if eval_enabled and eval_dataloader is not None and global_step % eval_every == 0:
                eval_metrics = run_validation(
                    accelerator=accelerator,
                    model=model,
                    model_cfg=model_cfg,
                    eval_dataloader=eval_dataloader,
                    max_batches=eval_max_batches,
                    is_dense=is_dense,
                    seq_aux_loss_coef=seq_aux_loss_coef,
                )
                if eval_metrics:
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"eval {global_step:6d}  "
                            f"ce={eval_metrics['eval/ce_loss']:.4f}  "
                            f"ppl={eval_metrics['eval/perplexity']:.2f}  "
                            f"aux={eval_metrics['eval/aux_loss']:.4f}"
                        )
                    if log_with:
                        accelerator.log(eval_metrics, step=global_step)

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
            aux_normalized_window_sum = 0.0
            seq_aux_window_sum = 0.0
            branch_aux_window_sum = 0.0
            attention_aux_window_sum = 0.0
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
