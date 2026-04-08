"""
Torch-native pretraining entrypoint.

This is the first cut of a replacement for train.py that avoids Accelerate and
uses raw torch.distributed with DDP/FSDP options built in.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from src.data import (
    DataConfig as ParquetDataConfig,
    StatefulParquetDataset,
    StatefulTokenBinDataset,
    TokenBinConfig,
)
from src.models import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    StandardMoEModel,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    DeepSeekGlobalMoEForCausalLM,
    MoEverythingConfig,
    MoEverythingForCausalLM,
)
from src.models.speedrun_gpt import GPT as SpeedrunGPT
from src.models.speedrun_moe_gpt import SpeedrunMoEGPT
from src.models.init_mapping import copy_global_to_alternating_sanity
from src.models.load_balancing import (
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
)
from src.models.mixture_of_everything import NormExpertBank
from src.utils.training import (
    TrainingConfig,
    build_lr_scheduler,
    build_muon_optimizer,
    build_optimizer,
    count_parameters,
    get_grad_norm,
)

load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

torch.backends.cuda.preferred_blas_library("cublaslt")

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
except Exception:
    FSDP = None
    FullOptimStateDictConfig = None
    FullStateDictConfig = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None


def configure_liger_kernels(cfg: dict) -> str:
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_data_format(cfg_dict: dict) -> str:
    if "format" in cfg_dict:
        return cfg_dict["format"]
    if "files_glob" in cfg_dict:
        return "token_bin"
    return "parquet"


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

    if mtype == "speedrun_gpt":
        max_seq_len = max(mcfg.get("train_seq_len", 48 * 1024), mcfg.get("val_seq_len", 4 * 64 * 1024))
        model = SpeedrunGPT(
            vocab_size=mcfg.get("vocab_size", 50257),
            num_layers=mcfg.get("num_layers", 12),
            num_heads=mcfg.get("num_heads", 6),
            model_dim=mcfg.get("model_dim", 768),
            max_seq_len=max_seq_len,
        )
        # Return a simple namespace as config for compatibility
        class _Cfg:
            num_experts = 0
            num_experts_per_tok = 0
        return model, _Cfg()

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

    def _set_router_params(config, model_cfg):
        config.router_exploration_rate = model_cfg.get("router_exploration_rate", 0.0)

    def _set_deepseek_router_params(config, model_cfg):
        _set_router_params(config, model_cfg)
        config.topk_scaling_factor = model_cfg.get("topk_scaling_factor", None)
        config.num_groups = model_cfg.get("num_groups", None)
        config.group_topk = model_cfg.get("group_topk", None)

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
            per_head_dense_fraction_threshold=mcfg.get("per_head_dense_fraction_threshold", 0.75),
            sanity_check_mode=mcfg.get("sanity_check_mode"),
            scale_attn_by_routing_weight=mcfg.get("scale_attn_by_routing_weight", True),
            scale_branch_by_routing_weight=mcfg.get("scale_branch_by_routing_weight", True),
            router_exploration_rate=mcfg.get("router_exploration_rate", 0.0),
            branch_router_exploration_rate=mcfg.get("branch_router_exploration_rate"),
            **common,
        )
        model = MoEverythingForCausalLM(config)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    if hasattr(model, "set_experts_implementation"):
        experts_impl = mcfg.get("experts_implementation", "grouped_mm")
        try:
            model.set_experts_implementation(experts_impl)
        except Exception:
            model.set_experts_implementation("eager")

    return model, config


def get_selected_experts_for_seq_aux(model) -> tuple[torch.Tensor, ...] | None:
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
    branch_aux_value = branch_aux.detach().float().item() if isinstance(branch_aux, torch.Tensor) else float(branch_aux or 0.0)
    attention_aux = getattr(output, "attention_aux_loss", None)
    attention_aux_value = attention_aux.detach().float().item() if isinstance(attention_aux, torch.Tensor) else float(attention_aux or 0.0)

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


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def dist_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return dist_rank() == 0


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def reduce_scalar(value: float, reduction: str = "mean", device: torch.device | None = None) -> float:
    if not is_distributed():
        return value
    assert device is not None
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if reduction == "mean":
        tensor /= dist_world_size()
    return tensor.item()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else getattr(model, "_fsdp_wrapped_module", model)


def infer_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def build_fsdp_mixed_precision(mixed_precision_name: str):
    if FSDP is None or MixedPrecision is None:
        return None
    param_dtype = infer_dtype(mixed_precision_name)
    if param_dtype is None:
        return None
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=param_dtype,
        buffer_dtype=param_dtype,
    )


def wrap_model(model, *, strategy: str, local_rank: int, mixed_precision_name: str):
    world_size = dist_world_size()
    if strategy == "none" or world_size == 1:
        return model
    if strategy == "ddp":
        return DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=False)
    if strategy == "fsdp":
        if FSDP is None:
            raise RuntimeError("FSDP is unavailable in this torch install")
        return FSDP(
            model,
            device_id=torch.device("cuda", local_rank),
            mixed_precision=build_fsdp_mixed_precision(mixed_precision_name),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def save_checkpoint(
    *,
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
    os.makedirs(ckpt_dir, exist_ok=True)

    if FSDP is not None and isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy, optim_policy):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = unwrap_model(model).state_dict()
        optim_state = optimizer.state_dict()

    if is_main_process():
        torch.save(
            {
                "model": model_state,
                "optimizer": optim_state,
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(ckpt_dir, "trainer.pt"),
        )
        meta = {"step": step, "tokens_seen": tokens_seen}
        if dataset_state:
            meta["dataset_state"] = dataset_state
        if wandb_run_id:
            meta["wandb_run_id"] = wandb_run_id
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"Saved checkpoint to {ckpt_dir}", flush=True)
    barrier()


def load_checkpoint(model, optimizer, scheduler, resume_from: str) -> tuple[int, dict | None, float]:
    trainer_path = os.path.join(resume_from, "trainer.pt")
    payload = torch.load(trainer_path, map_location="cpu")

    if FSDP is not None and isinstance(model, FSDP):
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy, optim_policy):
            model.load_state_dict(payload["model"])
            optim_state = FSDP.optim_state_dict_to_load(model, optimizer, payload["optimizer"])
            optimizer.load_state_dict(optim_state)
    else:
        unwrap_model(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])

    scheduler.load_state_dict(payload["scheduler"])

    meta_path = os.path.join(resume_from, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("step", 0), meta.get("dataset_state"), meta.get("tokens_seen", 0.0)
    return 0, None, 0.0


def find_latest_checkpoint(output_dir: str) -> str | None:
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
    if max_keep <= 0 or not is_main_process():
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


@torch.no_grad()
def run_validation(
    *,
    model,
    model_cfg,
    eval_dataloader,
    max_batches: int,
    is_dense: bool,
    seq_aux_loss_coef: float,
    device: torch.device,
) -> dict[str, float]:
    if eval_dataloader is None:
        return {}

    was_training = model.training
    model.eval()
    raw_model = unwrap_model(model)
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
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        input_ids = batch["input_ids"]
        labels = input_ids
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

    if was_training:
        model.train()
    if batches == 0:
        return {}

    return {
        f"eval/{key}": reduce_scalar(value / batches, device=device)
        for key, value in totals.items()
    }


# ---------------------------------------------------------------------------
# Speedrun-specific data generator and training loop
# Ported from /tmp/modded-nanogpt-prefa3/train_gpt.py

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def _find_batch_starts(tokens, pos, seq_len, token_window):
    boundary_mask = tokens[pos:pos + token_window] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item()
        if end - start >= seq_len:
            starts.append(start)
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False, "increase token_window if necessary"


def _distributed_data_generator(filename_pattern, seq_len, grad_accum_steps, align_to_bos):
    import glob as _glob
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch_size = seq_len * world_size
    files = [Path(f) for f in sorted(_glob.glob(filename_pattern))]
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        token_window = grad_accum_steps * (2 * batch_size if align_to_bos else batch_size)
        if pos + token_window + 1 >= len(tokens):
            tokens = _load_data_shard(next(file_iter))
            pos = 0
        for _ in range(grad_accum_steps):
            if align_to_bos:
                batch_starts, tokens_consumed = _find_batch_starts(tokens, pos, seq_len, token_window)
                start_idx = batch_starts[rank]
            else:
                tokens_consumed = batch_size
                start_idx = pos + rank * seq_len
            buf = tokens[start_idx:][:seq_len + 1]
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            pos += tokens_consumed
            token_window -= tokens_consumed
            yield inputs, targets


def _next_multiple_of_n(v, *, n):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def _get_window_size_blocks(step, num_iterations):
    from functools import lru_cache

    @lru_cache(1)
    def _helper(window_size):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    x = step / num_iterations
    window_size = _next_multiple_of_n(1728 * x, n=128)
    return _helper(window_size)


def run_speedrun_training(args, cfg) -> None:
    """Full training loop for the speedrun_gpt model type."""
    import copy

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    master_process = rank == 0

    def print0(s, console=True):
        if master_process:
            if console:
                print(s, flush=True)

    tcfg_dict = cfg["training"]
    dcfg_dict = cfg.get("data", {})
    eval_cfg_dict = cfg.get("eval", {})
    mcfg = cfg["model"]

    num_iterations = tcfg_dict["max_steps"]
    cooldown_frac = tcfg_dict.get("cooldown_frac", 0.45)
    train_seq_len = dcfg_dict.get("seq_len", 48 * 1024)
    val_seq_len = eval_cfg_dict.get("seq_len", 4 * 64 * 1024)
    val_tokens = eval_cfg_dict.get("max_tokens", 10485760)
    val_loss_every = int(eval_cfg_dict.get("every", 125))
    grad_accum_steps = tcfg_dict.get("gradient_accumulation", 8 // world_size)
    train_align_to_bos = dcfg_dict.get("align_to_bos", True)
    output_dir = tcfg_dict.get("output_dir", "./outputs/plan/speedrun_gpt")

    os.makedirs(output_dir, exist_ok=True)

    # Build model
    mtype = mcfg["type"]
    max_seq_len = max(train_seq_len, val_seq_len)
    if mtype == "speedrun_gpt":
        model = SpeedrunGPT(
            vocab_size=mcfg.get("vocab_size", 50257),
            num_layers=mcfg.get("num_layers", 12),
            num_heads=mcfg.get("num_heads", 6),
            model_dim=mcfg.get("model_dim", 768),
            max_seq_len=max_seq_len,
        ).cuda()
    elif mtype in ("speedrun_moe_fully_independent", "speedrun_moe_precompute_kv"):
        mode_map = {
            "speedrun_moe_fully_independent": "per_head_fully_independent",
            "speedrun_moe_precompute_kv": "per_head_precompute_kv",
        }
        model = SpeedrunMoEGPT(
            vocab_size=mcfg.get("vocab_size", 50257),
            num_layers=mcfg.get("num_layers", 12),
            num_heads=mcfg.get("num_heads", 6),
            model_dim=mcfg.get("model_dim", 768),
            head_dim=mcfg.get("head_dim", 128),
            max_seq_len=max_seq_len,
            mode=mode_map[mtype],
            num_attn_experts=mcfg.get("num_attn_experts", 66),
            num_mlp_experts=mcfg.get("num_mlp_experts", 12),
        ).cuda()
    else:
        raise ValueError(f"Unknown speedrun model type: {mtype}")

    # Cast embeddings to bf16
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    # Broadcast parameters
    if distributed:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)

    params_total = sum(p.numel() for p in model.parameters())
    print0(f"{mtype}: {params_total / 1e6:.1f}M params, seq_len={train_seq_len}")

    # Build optimizers
    from src.utils.dist_optimizers import DistAdam, DistMuon

    # Collect hidden matrix params from blocks AND attn_bank (for MoE models)
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    if hasattr(model, "attn_bank"):
        hidden_matrix_params += [p for n, p in model.attn_bank.named_parameters() if p.ndim >= 2]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    adam_lr = tcfg_dict.get("adam_lr", 0.008)
    adam_betas = tuple(tcfg_dict.get("adam_betas", [0.8, 0.95]))
    adam_eps = tcfg_dict.get("adam_eps", 1e-10)
    muon_lr = tcfg_dict.get("muon_lr", 0.05)
    muon_wd = tcfg_dict.get("weight_decay", 0.0)

    optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=adam_lr, betas=adam_betas, eps=adam_eps, weight_decay=muon_wd)
    optimizer2 = DistMuon(hidden_matrix_params, lr=muon_lr, momentum=0.95, weight_decay=muon_wd)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # LR schedule: stable then decay
    def get_lr(step):
        x = step / num_iterations
        if x < 1 - cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / cooldown_frac
            return w * 1.0 + (1 - w) * 0.1

    # Compile model (MoE models use dynamic control flow, so skip fullgraph)
    is_moe = mtype in ("speedrun_moe_fully_independent", "speedrun_moe_precompute_kv")
    if tcfg_dict.get("torch_compile", True) and not is_moe:
        print0("Compiling model with torch.compile(dynamic=False, fullgraph=True)")
        model = torch.compile(model, dynamic=False, fullgraph=True)
    elif tcfg_dict.get("torch_compile", True) and is_moe:
        print0("Compiling model with torch.compile(dynamic=False) [no fullgraph for MoE]")
        model = torch.compile(model, dynamic=False)

    # WandB
    wandb_run = None
    if tcfg_dict.get("wandb_project") and master_process:
        import wandb
        wandb_run = wandb.init(
            project=tcfg_dict["wandb_project"],
            name=tcfg_dict.get("wandb_run_name", cfg.get("experiment_name")),
            config=cfg,
            dir=output_dir,
        )

    # Kernel warmup
    print0("Warming up kernels (10 steps)...")
    warmup_steps = 10
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
    )
    train_files = dcfg_dict.get("files_glob", "data/fineweb10B_gpt2/fineweb_train_*.bin")
    val_files = eval_cfg_dict.get("files_glob", "data/fineweb10B_gpt2/fineweb_val_*.bin")

    # For MoE models: use batched DataLoader; for speedrun_gpt: use 1D generator
    batch_size = tcfg_dict.get("batch_size", 1)
    if is_moe and batch_size > 1:
        # Build a DataLoader-based pipeline for batched MoE training
        from src.data import StatefulTokenBinDataset, TokenBinConfig
        _train_ds = StatefulTokenBinDataset(
            TokenBinConfig(files_glob=train_files, seq_len=train_seq_len,
                           header_bytes=1024, token_dtype="uint16",
                           shuffle_files=False, repeat=True),
            rank=rank, world_size=world_size, seed=42,
        )
        _train_dl = DataLoader(_train_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
        _val_ds = StatefulTokenBinDataset(
            TokenBinConfig(files_glob=val_files, seq_len=val_seq_len,
                           header_bytes=1024, token_dtype="uint16",
                           shuffle_files=False, repeat=False),
            rank=rank, world_size=world_size, seed=1234,
        )
        _val_dl = DataLoader(_val_ds, batch_size=batch_size, num_workers=0, pin_memory=True)

        def _moe_forward(model, inputs, targets, step):
            return model(inputs, targets)

        def _make_train_iter():
            return iter(_train_dl)

        def _make_val_iter():
            return iter(_val_dl)

        def _val_step_count():
            return max(1, val_tokens // (world_size * batch_size * val_seq_len))
    else:
        # Original 1D generator path for speedrun_gpt
        def _moe_forward(model, inputs, targets, step):
            return model(inputs, targets, _get_window_size_blocks(step, num_iterations))

        def _make_train_iter():
            return _distributed_data_generator(train_files, train_seq_len, grad_accum_steps, align_to_bos=train_align_to_bos)

        def _make_val_iter():
            return _distributed_data_generator(val_files, val_seq_len, grad_accum_steps, align_to_bos=False)

        def _val_step_count():
            val_batch_size = world_size * val_seq_len
            return val_tokens // val_batch_size

    train_loader = _make_train_iter()
    for _ in range(warmup_steps):
        batch = next(train_loader)
        if isinstance(batch, dict):
            inputs = batch["input_ids"].cuda(non_blocking=True)
            targets = batch["labels"].cuda(non_blocking=True)
        else:
            inputs, targets = batch
        _moe_forward(model, inputs, targets, 1).backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del train_loader, initial_state
    print0("Kernel warmup complete.")

    # Training
    train_loader = _make_train_iter()
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_steps = num_iterations
    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # Validation
        if last_step or (val_loss_every > 0 and step % val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_steps = _val_step_count()
            val_loader = _make_val_iter()
            val_loss = 0
            val_tokens_seen = 0
            with torch.no_grad():
                for vi, vbatch in enumerate(val_loader):
                    if vi >= val_steps:
                        break
                    if isinstance(vbatch, dict):
                        vinputs = vbatch["input_ids"].cuda(non_blocking=True)
                        vtargets = vbatch["labels"].cuda(non_blocking=True)
                    else:
                        vinputs, vtargets = vbatch
                    vloss = _moe_forward(model, vinputs, vtargets, step)
                    # Model uses mean-reduction in eval mode, so vloss is already per-token
                    val_loss += vloss.item()
                    val_tokens_seen += 1
            if val_tokens_seen > 0:
                val_loss /= val_tokens_seen
            del val_loader
            if distributed:
                val_loss_t = torch.tensor(val_loss, device=device)
                dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                val_loss = val_loss_t.item()
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            if wandb_run is not None:
                wandb_run.log({"eval/ce_loss": val_loss, "eval/perplexity": math.exp(min(20.0, val_loss))}, step=step)
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # Training step
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_loader)
            except StopIteration:
                train_loader = _make_train_iter()
                batch = next(train_loader)
            if isinstance(batch, dict):
                inputs = batch["input_ids"].cuda(non_blocking=True)
                targets = batch["labels"].cuda(non_blocking=True)
            else:
                inputs, targets = batch
            ce_loss = _moe_forward(model, inputs, targets, step)
            # Add aux loss for MoE models
            if is_moe and hasattr(model, '_aux_loss'):
                raw_model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model
                aux_coef = tcfg_dict.get("router_aux_loss_coef", 0.01)
                total_loss = ce_loss + aux_coef * raw_model_ref._aux_loss
            else:
                total_loss = ce_loss
            total_loss.backward()

        # Set LR
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        # Momentum warmup for Muon
        frac = min(step / 300, 1)
        for group in optimizer2.param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Step optimizers
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        # Logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        log_every = int(tcfg_dict.get("log_every", 1))
        if step % log_every == 0:
            print0(f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms")
            if wandb_run is not None:
                log_dict = {
                    "train/step_time_ms": approx_training_time_ms / (step + 1),
                }
                # Log routing stats for MoE models
                if is_moe:
                    raw_model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model
                    if hasattr(raw_model_ref, '_routing_stats'):
                        log_dict.update(raw_model_ref._routing_stats)
                    if hasattr(raw_model_ref, '_aux_loss'):
                        log_dict["train/aux_loss"] = raw_model_ref._aux_loss.item()
                wandb_run.log(log_dict, step=step)

        # Save checkpoint
        save_every = int(tcfg_dict.get("save_every", 125))
        if save_every > 0 and step > 0 and step % save_every == 0 and master_process:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            raw_model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                "model": raw_model_ref.state_dict(),
                "optimizers": [opt.state_dict() for opt in optimizers],
                "step": step,
            }, os.path.join(ckpt_dir, "trainer.pt"))
            print0(f"Saved checkpoint to {ckpt_dir}")

    # Final checkpoint
    if master_process:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        raw_model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            "model": raw_model_ref.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "step": step,
        }, os.path.join(ckpt_dir, "trainer.pt"))
        print0(f"Saved final checkpoint to {ckpt_dir}")

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max_checkpoints", type=int, default=0)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--dist-strategy", choices=("none", "ddp", "fsdp"), default="ddp")
    parser.add_argument("--init-from-config", default=None)
    parser.add_argument("--init-strategy", choices=("global_to_alternating_sanity",), default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Dispatch to speedrun training loop for speedrun model types
    if cfg["model"]["type"] in ("speedrun_gpt", "speedrun_moe_fully_independent", "speedrun_moe_precompute_kv"):
        return run_speedrun_training(args, cfg)

    initialization_spec = resolve_initialization_spec(
        cfg,
        config_path=args.config,
        cli_source_config=args.init_from_config,
        cli_strategy=args.init_strategy,
    )
    liger_mode = configure_liger_kernels(cfg)
    print(f"Liger kernels: {liger_mode} for model type '{cfg['model']['type']}'", flush=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        torch.cuda.set_device(0)
    device = torch.device("cuda", local_rank if distributed else 0)

    seed_everything(args.seed + rank)

    tcfg_dict = cfg["training"]
    dcfg_dict = cfg.get("data", {})
    eval_cfg_dict = cfg.get("eval", {})
    if args.data_dir:
        dcfg_dict["data_dir"] = args.data_dir
    if args.output_dir:
        tcfg_dict["output_dir"] = args.output_dir
    if args.max_steps is not None:
        tcfg_dict["max_steps"] = args.max_steps

    train_cfg = TrainingConfig(
        learning_rate=tcfg_dict["learning_rate"],
        weight_decay=tcfg_dict["weight_decay"],
        beta1=tcfg_dict.get("beta1", 0.9),
        beta2=tcfg_dict.get("beta2", 0.95),
        eps=tcfg_dict.get("eps", 1e-8),
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
    resume_from = args.resume or cfg.get("checkpoint", {}).get("resume_from")
    if args.auto_resume and not resume_from:
        resume_from = find_latest_checkpoint(train_cfg.output_dir)

    if is_main_process():
        print(
            f"[rank {rank}] host={socket.gethostname()} local_rank={local_rank} "
            f"world_size={world_size} device={device} strategy={args.dist_strategy}",
            flush=True,
        )

    model, model_cfg = build_model(cfg)
    base_model = model
    params = count_parameters(base_model)
    if initialization_spec is not None:
        source_cfg = load_config(initialization_spec["source_config"])
        source_model, _ = build_model(source_cfg)
        strategy = initialization_spec["strategy"]
        if strategy == "global_to_alternating_sanity":
            copy_global_to_alternating_sanity(source_model, model)
        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    if train_cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            if is_main_process():
                print("Gradient checkpointing enabled.", flush=True)

    model.to(device)
    if seq_aux_loss_coef := cfg["model"].get("seq_aux_loss_coef", 0.0):
        model._seq_aux_loss_coef = seq_aux_loss_coef

    if tcfg_dict.get("torch_compile", False):
        compile_mode = tcfg_dict.get("torch_compile_mode", "default")
        if compile_mode is True:
            compile_mode = "default"
        if is_main_process():
            print(f"Compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(model, mode=compile_mode, dynamic=False)

    model = wrap_model(model, strategy=args.dist_strategy, local_rank=local_rank, mixed_precision_name=train_cfg.mixed_precision)
    raw_model = base_model

    optimizer_type = tcfg_dict.get("optimizer", "adamw")
    if optimizer_type == "muon":
        optimizer = build_muon_optimizer(
            model,
            train_cfg,
            muon_lr=tcfg_dict.get("muon_lr", 0.02),
            muon_weight_decay=tcfg_dict.get("muon_weight_decay", 0.0),
            adam_lr=tcfg_dict.get("adam_lr", 3e-4),
        )
    else:
        optimizer = build_optimizer(model, train_cfg)
    scheduler_cfg = replace(train_cfg)
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg)

    is_dense = cfg["model"]["type"] in {"dense", "gpt2_dense"}
    if is_main_process():
        print("=" * 60, flush=True)
        print(f"  Model     : {cfg['model']['type']}", flush=True)
        print(f"  Params    : {params['total']/1e9:.3f}B total", flush=True)
        print(f"  Strategy  : {args.dist_strategy}", flush=True)
        print(f"  Precision : {train_cfg.mixed_precision}", flush=True)
        print("=" * 60, flush=True)

    tokenizer = None
    if get_data_format(dcfg_dict) == "parquet":
        tokenizer_name = dcfg_dict.get("tokenizer_name", "gpt2")
        if is_main_process():
            print(f"Loading tokenizer {tokenizer_name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        barrier()

    train_data_cfg = dict(dcfg_dict)
    if get_data_format(train_data_cfg) == "parquet":
        train_data_cfg["split"] = "all"
        train_data_cfg["holdout_fraction"] = 0.0
    dataset = build_dataset_from_config(
        train_data_cfg,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=dcfg_dict.get("seed", 42),
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, num_workers=0, pin_memory=True)

    eval_enabled = bool(eval_cfg_dict.get("enabled", False))
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
            rank=rank,
            world_size=world_size,
            seed=eval_cfg_dict.get("seed", 1234),
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=int(eval_cfg_dict.get("batch_size", train_cfg.batch_size)),
            num_workers=0,
            pin_memory=True,
        )

    global_step = 0
    dataset_state = None
    tokens_seen = 0.0
    if resume_from:
        global_step, dataset_state, tokens_seen = load_checkpoint(model, optimizer, scheduler, resume_from)
        if dataset_state:
            dataset.set_state(dataset_state)
        if is_main_process():
            print(f"Resumed from step {global_step}", flush=True)

    wandb_run = None
    if train_cfg.wandb_project and is_main_process():
        import wandb

        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.wandb_run_name,
            config=cfg,
            dir=train_cfg.output_dir,
        )

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    data_iter = iter(dataloader)
    autocast_dtype = infer_dtype(train_cfg.mixed_precision)
    autocast_enabled = autocast_dtype is not None
    model.train()

    if is_main_process():
        print(f"Starting training from step {global_step}", flush=True)

    while global_step < train_cfg.max_steps:
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        window_metrics = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "aux_loss": 0.0,
            "aux_loss_normalized": 0.0,
            "seq_aux_loss": 0.0,
            "branch_aux_loss": 0.0,
            "attention_aux_loss": 0.0,
        }
        local_tokens_in_step = 0
        grad_norm = 0.0

        for micro_idx in range(train_cfg.gradient_accumulation):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            input_ids = batch["input_ids"]
            labels = input_ids
            sync_context = (
                nullcontext()
                if micro_idx == train_cfg.gradient_accumulation - 1 or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        **({} if is_dense else {"output_router_logits": True}),
                    )
                loss = output.loss / train_cfg.gradient_accumulation
                loss.backward()

            metrics, _, _ = compute_output_metrics(
                output,
                raw_model,
                model_cfg,
                input_ids,
                seq_aux_loss_coef=seq_aux_loss_coef,
            )
            for key in window_metrics:
                window_metrics[key] += metrics[key]
            local_tokens_in_step += input_ids.numel()

        if train_cfg.max_grad_norm > 0:
            grad_norm = get_grad_norm(raw_model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if optimizer_type == "muon":
            from src.utils.muon import get_muon_momentum

            warmup = tcfg_dict.get("momentum_warmup_steps", 300)
            new_mom = get_muon_momentum(global_step, warmup_steps=warmup)
            for group in optimizer.param_groups:
                if group.get("is_muon", False):
                    group["momentum"] = new_mom

        global_step += 1
        tokens_seen += local_tokens_in_step * world_size
        elapsed = time.perf_counter() - step_start
        step_tokens = local_tokens_in_step * world_size
        tok_per_s = step_tokens / max(elapsed, 1e-9)

        reduced = {
            key: reduce_scalar(value / train_cfg.gradient_accumulation, device=device)
            for key, value in window_metrics.items()
        }
        reduced_grad = reduce_scalar(grad_norm, device=device)

        if global_step % train_cfg.log_every == 0 and is_main_process():
            print(
                f"step {global_step:6d}  "
                f"loss={reduced['loss']:.4f}  "
                f"ce={reduced['ce_loss']:.4f}  "
                f"aux={reduced['aux_loss']:.4f}  "
                f"aux_n={reduced['aux_loss_normalized']:.4f}  "
                f"seq_aux={reduced['seq_aux_loss']:.4f}  "
                f"branch_aux={reduced['branch_aux_loss']:.4f}  "
                f"attn_aux={reduced['attention_aux_loss']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"tok/s={tok_per_s/1e3:.1f}k  "
                f"sec/step={elapsed:.3f}  "
                f"|g|={reduced_grad:.3f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": reduced["loss"],
                        "train/ce": reduced["ce_loss"],
                        "train/aux": reduced["aux_loss"],
                        "train/aux_n": reduced["aux_loss_normalized"],
                        "train/seq_aux": reduced["seq_aux_loss"],
                        "train/branch_aux": reduced["branch_aux_loss"],
                        "train/attn_aux": reduced["attention_aux_loss"],
                        "train/grad_norm": reduced_grad,
                        "train/tok_per_s": tok_per_s,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/tokens_seen": tokens_seen,
                    },
                    step=global_step,
                )

        if eval_enabled and eval_dataloader is not None and global_step % int(eval_cfg_dict["every"]) == 0:
            eval_metrics = run_validation(
                model=model,
                model_cfg=model_cfg,
                eval_dataloader=eval_dataloader,
                max_batches=int(eval_cfg_dict.get("max_batches", 0)),
                is_dense=is_dense,
                seq_aux_loss_coef=seq_aux_loss_coef,
                device=device,
            )
            if is_main_process():
                ce = eval_metrics["eval/ce_loss"]
                eval_metrics["eval/perplexity"] = math.exp(min(20.0, ce))
                print(f"eval {global_step:6d}  ce={ce:.4f}  ppl={eval_metrics['eval/perplexity']:.2f}  aux={eval_metrics['eval/aux_loss']:.4f}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=global_step)

        if global_step % train_cfg.save_every == 0:
            dataset_state = getattr(dataset, "get_state", lambda: None)()
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                output_dir=train_cfg.output_dir,
                dataset_state=dataset_state,
                wandb_run_id=getattr(wandb_run, "id", None),
                tokens_seen=tokens_seen,
            )
            cleanup_checkpoints(train_cfg.output_dir, args.max_checkpoints)

    dataset_state = getattr(dataset, "get_state", lambda: None)()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        output_dir=train_cfg.output_dir,
        dataset_state=dataset_state,
        wandb_run_id=getattr(wandb_run, "id", None),
        tokens_seen=tokens_seen,
    )
    cleanup_checkpoints(train_cfg.output_dir, args.max_checkpoints)

    if wandb_run is not None:
        wandb_run.finish()
    if is_main_process():
        print("Training complete.", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
