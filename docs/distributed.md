# Distributed Training

This document covers DDP, FSDP, multi-node setup, and deployment strategies.

## Table of Contents

- [1. Distributed Strategies](#1-distributed-strategies)
- [2. Multi-Node on Modal](#2-multi-node-on-modal)
- [3. Multi-Node on GCP](#3-multi-node-on-gcp)
- [4. Checkpointing](#4-checkpointing)

---

## 1. Distributed Strategies

### DDP (Distributed Data Parallel)

**Implementation**: Manual `torch.nn.parallel.DistributedDataParallel` wrapping in `train_torch.py`.

Each rank holds a full copy of the model. Gradients are all-reduced after each step (or accumulation boundary). Data is sharded across ranks.

```bash
torchrun --nproc_per_node=8 train_torch.py --config config.yaml --dist-strategy ddp
```

**Key detail for MoE-Everything**: Uses `static_graph=False` because shared parameters across depths are not safe with DDP's static graph optimization.

### FSDP (Fully Sharded Data Parallel)

**Implementation**: `torch.distributed.fsdp.FullyShardedDataParallel` in `train_torch.py`.

Shards model parameters, gradients, and optimizer states across ranks. Each rank holds only a fraction of the model. Parameters are gathered on-demand for forward/backward.

```bash
torchrun --nproc_per_node=8 train_torch.py --config config.yaml --dist-strategy fsdp
```

**Mixed precision**: FSDP uses `MixedPrecision` config for bf16 compute with fp32 parameter reduction.

**Checkpointing**: Uses `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` to gather full state on rank 0 for saving.

### Distributed Optimizers (Speedrun Path)

For speedrun models, `DistMuon` and `DistAdam` handle gradient communication internally via `reduce_scatter` / `all_gather`. No DDP wrapper is needed -- the optimizer IS the communication layer.

---

## 2. Multi-Node on Modal

**File**: `modal_train.py`

### Setup

```python
# In modal_train.py
N_NODES = 2
GPUS_PER_NODE = 8
GPU_TYPE = "H100"

# Volumes for persistent storage
data_volume = modal.Volume.from_name("moe-training-data")
ckpt_volume = modal.Volume.from_name("moe-checkpoints")
```

### How it works

1. Modal provisions `N_NODES` machines with `GPUS_PER_NODE` GPUs each
2. Each node runs `torchrun` with appropriate `--nnodes`, `--node-rank`, `--master-addr`
3. NCCL handles inter-node communication (RDMA when available)
4. Data and checkpoints are stored on Modal Volumes (persistent across runs)
5. On failure/timeout, the job restarts and auto-resumes from the latest checkpoint

### Running

```bash
# Upload data first
modal run modal_train.py::upload_data

# Launch training
modal run modal_train.py --config configs/scaling/xs_standard.yaml
```

See `MULTINODE_README.md` for full setup instructions.

---

## 3. Multi-Node on GCP

**File**: `gcp_setup.sh`

### Setup

The `gcp_setup.sh` script provisions GCE instances with:
- NVIDIA drivers and CUDA toolkit
- Python environment with all dependencies
- NCCL configuration for GCP networking

### Running

```bash
# On each node (adjust --node-rank):
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node-rank=0 \
  --master-addr=<master-ip> \
  --master-port=29500 \
  train_torch.py --config config.yaml
```

See `MULTINODE_README.md` for full GCP setup and networking instructions.

---

## 4. Checkpointing

### Save format

Each checkpoint is a directory `{output_dir}/checkpoint-{step}/` containing:

```
checkpoint-1000/
  model_state.pt      # model parameters (or FSDP full state dict)
  optimizer_state.pt   # optimizer state (or FSDP optimizer state dict)
  scheduler_state.pt   # LR scheduler state
  meta.json            # metadata:
    {
      "step": 1000,
      "tokens_seen": 134217728,
      "dataset_state": {...},     # StatefulParquetDataset state
      "wandb_run_id": "abc123"    # for WandB resume
    }
```

### Auto-resume

When `--auto_resume` is set, the training script scans `output_dir` for the highest `checkpoint-N` directory and resumes from it.

```python
def find_latest_checkpoint(output_dir):
    # Scans for checkpoint-(\d+) directories
    # Returns path to highest-numbered checkpoint
```

### Checkpoint cleanup

`max_checkpoints` controls how many checkpoints are retained. After each save, older checkpoints are deleted to save disk space.

```yaml
# Keep only 3 most recent checkpoints
--max_checkpoints 3
```

### Resume flow

1. Load model state dict
2. Load optimizer state dict (handles FSDP sharded states)
3. Load scheduler state dict
4. Restore dataset state from `meta.json`
5. Restore WandB run ID for logging continuity
6. Resume training from `step + 1`
