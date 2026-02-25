"""Training utilities: LR scheduler, grad norm, parameter counting."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class TrainingConfig:
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    # Schedule
    lr_scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 2000
    max_steps: int = 100_000
    min_lr_ratio: float = 0.1
    # Batch
    batch_size: int = 4          # per GPU
    gradient_accumulation: int = 4
    # Mixed precision / memory
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    # Logging
    log_every: int = 10
    save_every: int = 5000
    output_dir: str = "./outputs"
    wandb_project: str | None = "moe-experiments"
    wandb_run_name: str | None = None


def build_lr_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LambdaLR:
    warmup = config.warmup_steps
    total = config.max_steps
    min_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if config.lr_scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(progress, 1.0)
        if config.lr_scheduler == "cosine":
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
        if config.lr_scheduler == "linear":
            return max(min_ratio, 1.0 - progress * (1 - min_ratio))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().pow(2).sum().item()
    return total_norm ** 0.5



def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    # Separate weight-decay and no-decay parameter groups
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
