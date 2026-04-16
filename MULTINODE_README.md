# Multi-Node MoE Training on Modal

Train MoE models (dense, standard_moe, global_moe, moe_everything) on Modal with multi-node distributed training, automatic checkpoint resume, and fault tolerance.

## Prerequisites

1. **Modal account & CLI**

```bash
pip install modal
modal token set
```

2. **Create Modal secrets**

```bash
# HuggingFace token (for data download + tokenizer)
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here

# Weights & Biases (for experiment tracking)
modal secret create wandb-secret WANDB_API_KEY=your_key_here
```

## Quick Start

```bash
# Download training data
modal run modal_train.py::download_data --max-shards 64

# Launch training with default config
modal run modal_train.py

# Launch with specific config
modal run modal_train.py --config configs/scaling/m_standard.yaml
```

## Configuration

Edit `modal_train.py` top-level constants:

| Variable | Default | Description |
|----------|---------|-------------|
| `N_NODES` | 2 | Number of containers |
| `GPUS_PER_NODE` | 8 | GPUs per container |
| `GPU_TYPE` | "B200" | B200, H200, or H100 |
| `TIMEOUT_HOURS` | 24 | Max wall-clock time |
| `MAX_CHECKPOINTS` | 3 | Checkpoints to keep (0=unlimited) |
| `CONFIG_FILE` | configs/scaling/xs_standard.yaml | Default training config |

## How It Works

### Multi-node training

```
modal run modal_train.py
       │
       ▼
  Modal spawns N_NODES containers, each with GPUS_PER_NODE GPUs
       │
       ▼
  Each container runs torchrun with GPUS_PER_NODE processes
       │
       ▼
  Each process runs scripts/train.py with torch.distributed DDP
```

- `@modal.experimental.clustered(size=N_NODES, rdma=True)` provisions multi-node with RDMA
- `torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
- `torch.distributed.init_process_group()` creates the DDP process group
- NCCL handles all inter-node communication

### Auto-resume

`--auto_resume` (always on in Modal mode):

1. Scans output dir for `checkpoint-{step}` directories
2. Loads latest: model weights, optimizer state, scheduler, dataset position
3. Old checkpoints pruned to `MAX_CHECKPOINTS`

### Checkpoints

Saved as separate files per checkpoint directory:
- `model.pt` — model weights
- `optimizer_adam.pt` or `optimizer_muon.pt` — optimizer state
- `training_state.pt` — scheduler, step count, tokens seen
- `data_state.pt` — dataset position for deterministic resume
- `meta.json` — human-readable metadata

### Volumes

| Volume | Mount path | Purpose |
|--------|-----------|---------|
| `moe-training-data` | `/data` | Parquet training data |
| `moe-checkpoints` | `/checkpoints` | Checkpoints by experiment name |

## Standalone Mode (no Modal)

```bash
# Single GPU
python scripts/train.py --config configs/standard_moe.yaml

# Multi-GPU DDP
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml

# With auto-resume
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml --auto_resume

# FSDP
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml --dist-strategy fsdp
```

## Monitoring

### WandB

Metrics logged to WandB project (set via `wandb_project` in config). On resume, the same WandB run continues.

### Modal dashboard

```bash
modal container list
modal container exec <container-id> bash
```
