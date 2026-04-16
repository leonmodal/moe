# Distributed Training

## Strategies

The unified trainer (`scripts/train.py`) supports three distributed strategies via `--dist-strategy`:

### DDP (default)
```bash
torchrun --nproc_per_node=8 scripts/train.py --config config.yaml --dist-strategy ddp
```
- Standard `DistributedDataParallel` wrapping
- Each GPU holds a full model replica
- Gradients synchronized via all-reduce

### FSDP
```bash
torchrun --nproc_per_node=8 scripts/train.py --config config.yaml --dist-strategy fsdp
```
- `FullyShardedDataParallel` with `FULL_SHARD` strategy
- Model parameters sharded across GPUs
- Mixed-precision: all dtypes (param, reduce, buffer) set to the configured precision (default bf16)
- `sync_module_states=True` for consistent initialization

### None (single GPU)
```bash
python scripts/train.py --config config.yaml --dist-strategy none
```

## Modal Multi-Node

`modal_train.py` provides multi-node training on Modal cloud infrastructure:

```bash
modal run modal_train.py --config configs/scaling/m_standard.yaml
```

Configuration at top of `modal_train.py`:
- `N_NODES`: Number of containers
- `GPUS_PER_NODE`: GPUs per container (default 8)
- `GPU_TYPE`: B200, H200, or H100
- `TIMEOUT_HOURS`: Max wall-clock time

The launcher uses `torchrun` with RDMA-enabled NCCL communication.

## Implementation

`src/training/distributed.py` provides:

- `setup_distributed()` — Initialize process group, set devices, return rank/device info
- `cleanup_distributed()` — Destroy process group
- `wrap_model()` — Apply DDP or FSDP wrapping based on strategy
- `unwrap_model()` — Get underlying model from wrapper
- `barrier()` — Distributed barrier with CUDA device
- `reduce_scalar()` — All-reduce a scalar value with mean/sum
- `seed_everything()` — Deterministic seeding with per-rank offset

## Checkpoints in Distributed

- DDP: Checkpoint saved on rank 0 only; model state from `unwrap_model()`
- FSDP: Uses `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` for saving
- Resume: Both formats supported transparently via `load_checkpoint()`
