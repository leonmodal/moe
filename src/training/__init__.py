"""Unified training library for MoE models.

Shared modules extracted from train_torch.py for use by a single CLI entrypoint.
Supports dense, standard_moe, global_moe, and moe_everything model families
with DDP, FSDP, and Modal multi-node training.
"""

from .config import TrainingConfig, load_config, build_training_config
from .model_factory import build_model, configure_liger_kernels
from .data import build_dataset_from_config, get_data_format
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    cleanup_checkpoints,
)
from .eval import run_validation
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    dist_rank,
    dist_world_size,
    barrier,
    reduce_scalar,
    wrap_model,
)
from .logging import setup_wandb, log_training_step, log_eval_metrics

__all__ = [
    "TrainingConfig",
    "load_config",
    "build_training_config",
    "build_model",
    "configure_liger_kernels",
    "build_dataset_from_config",
    "get_data_format",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "cleanup_checkpoints",
    "run_validation",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "dist_rank",
    "dist_world_size",
    "barrier",
    "reduce_scalar",
    "wrap_model",
    "setup_wandb",
    "log_training_step",
    "log_eval_metrics",
]
