"""Distributed training setup: DDP, FSDP, and utilities."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
except Exception:
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None


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


def reduce_scalar(
    value: float,
    reduction: str = "mean",
    device: torch.device | None = None,
) -> float:
    if not is_distributed():
        return value
    assert device is not None
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if reduction == "mean":
        tensor /= dist_world_size()
    return tensor.item()


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int, torch.device]:
    """Initialize distributed training.

    Returns (rank, local_rank, world_size, device).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        device = torch.device("cuda", local_rank)
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)

    if is_main_process():
        print(
            f"[rank {rank}] host={socket.gethostname()} local_rank={local_rank} "
            f"world_size={world_size} device={device}",
            flush=True,
        )

    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def infer_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def _build_fsdp_mixed_precision(mixed_precision_name: str):
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


def wrap_model(
    model,
    *,
    strategy: str,
    local_rank: int,
    mixed_precision_name: str = "bf16",
):
    """Wrap model with DDP or FSDP based on strategy."""
    world_size = dist_world_size()
    if strategy == "none" or world_size == 1:
        return model
    if strategy == "ddp":
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=False,
        )
    if strategy == "fsdp":
        if FSDP is None:
            raise RuntimeError("FSDP is unavailable in this torch install")
        # `use_orig_params=True` lets FSDP tolerate forward passes that do
        # not activate every parameter (MoE branch routers may skip either
        # the attention-expert bank or the MLP-expert bank on any given
        # step). Without it, FSDP's post-backward assertion fires when the
        # unused shard's gradient hook executes in the IDLE state.
        return FSDP(
            model,
            device_id=torch.device("cuda", local_rank),
            mixed_precision=_build_fsdp_mixed_precision(mixed_precision_name),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            use_orig_params=True,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def unwrap_model(model):
    """Get the underlying model from DDP/FSDP wrapper."""
    if isinstance(model, DDP):
        return model.module
    return getattr(model, "_fsdp_wrapped_module", model)


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
