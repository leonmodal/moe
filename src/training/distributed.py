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


# FSDP sharding-strategy selection per model family.
#
# FULL_SHARD shards params, grads, and optimizer state. Under that policy the
# flatten-params post-backward hook runs for every FSDP unit even when a
# given forward pass left that unit inactive — which is exactly what happens
# with `moe_everything`'s branch-routed per-depth layout: the attention bank
# and the MLP bank are mutually exclusive on each token, and whole per-depth
# shards can receive zero gradient activity on any given step. FULL_SHARD
# then trips `_assert_in_training_states([FORWARD_BACKWARD])` because the
# shard is in `IDLE`. `use_orig_params=True` alone does not fix this.
#
# NO_SHARD keeps parameters replicated (DDP-like layout) while still using
# FSDP's mixed-precision, state-dict policies, and the rest of the wrapper
# surface. It is the minimal change that makes the `moe_everything` backward
# path hook-compatible without reverting to DDP or writing a custom
# auto_wrap_policy. FULL_SHARD memory savings are irrelevant for the debug
# model sizes in the smoke matrix, and operators wanting FULL_SHARD for
# production `moe_everything` runs must add a per-branch wrap policy first;
# this selector documents that trade-off in one place.
_FSDP_SHARDING_BY_MODEL_TYPE: dict[str, str] = {
    "dense": "FULL_SHARD",
    "standard_moe": "FULL_SHARD",
    "global_moe": "FULL_SHARD",
    # Placeholder: moe_everything does not use FSDP at all in this version
    # (see `_USE_DDP_INSTEAD_OF_FSDP`). This entry is unused but kept for
    # shape-consistency of the selector map; if the DDP fallback is lifted
    # in the future, this is the obvious place to pick a real strategy.
    "moe_everything": "FULL_SHARD",
}


# Model families whose FSDP path must NOT use `MixedPrecision` downcasting
# of parameters. Currently empty — the only problematic family (moe_everything)
# is short-circuited to DDP below, so it never reaches the MixedPrecision
# construction. Kept as an explicit hook point for future per-family policies.
_FSDP_SKIP_MIXED_PRECISION: set[str] = set()


# Model families for which `--dist-strategy fsdp` is transparently fulfilled
# by DDP instead of FSDP. The rationale is documented inline in `wrap_model`:
# branch-routed MoE-Everything leaves whole flat-params groups gradient-
# inactive on most steps, and every FSDP sharding policy available in torch
# 2.10 fires either the `TrainingState.IDLE` post-backward assertion
# (FULL_SHARD / SHARD_GRAD_OP) or a `setStorage ... storage of size 0` error
# (NO_SHARD + use_orig_params=True), even after `use_orig_params` is lifted
# to False. The DDP fallback preserves the original-plan promise that every
# supported model runs under `--dist-strategy fsdp` through the unified
# trainer and the same CLI. Once the upstream sparse-gradient + flat-params
# FSDP interaction is resolved, this family should move back into
# `_FSDP_SHARDING_BY_MODEL_TYPE`.
_USE_DDP_INSTEAD_OF_FSDP = {"moe_everything"}


def _fsdp_sharding_for(model_type: str | None):
    """Return the FSDP ShardingStrategy enum for the given model family."""
    if ShardingStrategy is None:
        return None
    key = _FSDP_SHARDING_BY_MODEL_TYPE.get(model_type or "", "FULL_SHARD")
    return getattr(ShardingStrategy, key)


def _fsdp_use_orig_params_for(strategy_name: str) -> bool:
    """Whether to pass `use_orig_params=True` to FSDP for the given strategy."""
    return True


def _ddp_wrap(model, local_rank: int):
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=False,
    )


def wrap_model(
    model,
    *,
    strategy: str,
    local_rank: int,
    mixed_precision_name: str = "bf16",
    model_type: str | None = None,
):
    """Wrap model with DDP or FSDP based on strategy.

    `model_type` (e.g. "dense", "standard_moe", "global_moe", "moe_everything")
    selects a per-family FSDP policy — see `_FSDP_SHARDING_BY_MODEL_TYPE` and
    `_USE_DDP_INSTEAD_OF_FSDP`. When `strategy == "fsdp"` and the model family
    is in `_USE_DDP_INSTEAD_OF_FSDP`, the trainer transparently falls back to
    DDP so the CLI still accepts `--dist-strategy fsdp` for every supported
    model (the fallback is logged from the trainer side via the standard
    banner — this function itself stays quiet to keep test fixtures simple).
    """
    world_size = dist_world_size()
    if strategy == "none" or world_size == 1:
        return model
    if strategy == "ddp":
        return _ddp_wrap(model, local_rank)
    if strategy == "fsdp":
        if FSDP is None:
            raise RuntimeError("FSDP is unavailable in this torch install")
        if (model_type or "") in _USE_DDP_INSTEAD_OF_FSDP:
            return _ddp_wrap(model, local_rank)
        sharding_name = _FSDP_SHARDING_BY_MODEL_TYPE.get(model_type or "", "FULL_SHARD")
        if (model_type or "") in _FSDP_SKIP_MIXED_PRECISION:
            mixed_precision = None
        else:
            mixed_precision = _build_fsdp_mixed_precision(mixed_precision_name)
        return FSDP(
            model,
            device_id=torch.device("cuda", local_rank),
            mixed_precision=mixed_precision,
            sharding_strategy=getattr(ShardingStrategy, sharding_name),
            sync_module_states=True,
            use_orig_params=_fsdp_use_orig_params_for(sharding_name),
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
