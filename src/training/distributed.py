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

try:
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
except Exception:
    ModuleWrapPolicy = None


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
    # MoE-Everything's branch-routed forward leaves whole flat-params units
    # gradient-inactive on most steps; FULL_SHARD / SHARD_GRAD_OP fire the
    # `TrainingState.IDLE` post-backward-hook assertion on the inactive
    # shards. NO_SHARD keeps params replicated while still routing through
    # FSDP for consistent state-dict handling; combined with the
    # `auto_wrap_policy` below (which splits AttentionExpertBank,
    # MlpExpertBank, and BranchRouter into their own FSDP units) it
    # isolates sparse sub-module activity from the root unit's hooks.
    "moe_everything": "NO_SHARD",
}


# Model families whose FSDP path must NOT use `MixedPrecision` downcasting
# of parameters. MoE-Everything under NO_SHARD + bf16 MixedPrecision hits a
# `setStorage ... storage of size 0` error on the embedding flat-param in
# backward (torch 2.10) because NO_SHARD retains the fp32 "master" copy
# while FSDP materializes a bf16 shard; when the flat-param is resharded
# mid-backward, the fp32 storage is already freed. Skipping the FSDP-level
# MixedPrecision policy for this family leaves params in fp32; the
# trainer's outer `torch.autocast` still casts activations to bf16.
_FSDP_SKIP_MIXED_PRECISION: set[str] = {"moe_everything"}


def _moe_everything_auto_wrap_policy():
    """Build an `auto_wrap_policy` for MoE-Everything.

    Wraps each top-level sub-module class (`nn.Embedding`, `nn.Linear`,
    `AttentionExpertBank`, `MlpExpertBank`, `BranchRouter`, etc.) as its
    own FSDP unit. Under NO_SHARD this yields one FSDP unit per major
    sub-module so sparse gradient activity at the expert-bank level is
    isolated per-unit — the alternative of leaving the whole branch-routed
    stack in a single root FSDP unit trips the `TrainingState.IDLE`
    post-backward assertion because per-step inactive expert params live
    in the same flat-params group as always-active params.

    Returns `None` if the wrap-policy API or the target classes are
    unavailable, in which case the caller falls back to the default
    single-unit behavior.
    """
    if ModuleWrapPolicy is None:
        return None
    try:
        import torch.nn as nn
        from src.models.moe_everything.attention_bank import AttentionExpertBank
        from src.models.moe_everything.mlp_bank import MlpExpertBank
        from src.models.routing.routers import BranchRouter
        from src.models.modeling_qwen3_moe import Qwen3MoeRMSNorm
    except Exception:
        return None
    return ModuleWrapPolicy({
        AttentionExpertBank, MlpExpertBank, BranchRouter,
        nn.Embedding, nn.Linear, Qwen3MoeRMSNorm,
    })


def describe_wrapper(model) -> str:
    """Human-readable summary of the distributed wrapper on `model`.

    Used by the trainer banner so operators can audit the effective wrapper
    from the training log without cross-referencing CLI flags against the
    per-family policy table. For FSDP this reports the sharding strategy and
    whether an `auto_wrap_policy` was supplied; for DDP it just names the
    class; for unwrapped models it returns `"none"`.
    """
    if isinstance(model, DDP):
        return "DDP"
    if FSDP is not None and isinstance(model, FSDP):
        try:
            strat = model.sharding_strategy.name
        except Exception:
            strat = "unknown"
        has_auto_wrap = any(
            FSDP is not None and isinstance(m, FSDP)
            for m in model.modules() if m is not model
        )
        return f"FSDP({strat}, auto_wrap={'yes' if has_auto_wrap else 'no'})"
    return "none"


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


def _auto_wrap_policy_for(model_type: str | None):
    """Return the FSDP `auto_wrap_policy` callable for the given family.

    `moe_everything` needs its branch-routed sub-modules wrapped as separate
    FSDP units so sparse gradient activity does not trip the root unit's
    post-backward-hook assertion. Other families keep the default
    single-unit wrapping.
    """
    if model_type == "moe_everything":
        return _moe_everything_auto_wrap_policy()
    return None


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
    selects a per-family FSDP policy. All families use a real FSDP wrapper
    under `--dist-strategy fsdp`:
    - `dense` / `standard_moe` / `global_moe` → FULL_SHARD, single-unit wrap.
    - `moe_everything` → NO_SHARD with an `auto_wrap_policy` that makes
      `AttentionExpertBank`, `MlpExpertBank`, and `BranchRouter` separate
      FSDP units (see `_moe_everything_auto_wrap_policy`).
    """
    world_size = dist_world_size()
    if strategy == "none" or world_size == 1:
        return model
    if strategy == "ddp":
        return _ddp_wrap(model, local_rank)
    if strategy == "fsdp":
        if FSDP is None:
            raise RuntimeError("FSDP is unavailable in this torch install")
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
            auto_wrap_policy=_auto_wrap_policy_for(model_type),
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
