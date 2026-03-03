# Multi-Node MoE Training on Modal

Train Standard/Global MoE models on Modal with multi-node distributed training, automatic checkpoint resume, and fault tolerance.

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
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key_here
```

## Quick Start

```bash
cd moe/

# 1. Download data to Modal volume (one time — takes a while for all 8192 shards)
modal run modal_train.py::download_data --max-shards 64      # small test
modal run modal_train.py::download_data                       # full dataset

# 2. Launch training (--detach lets it run in background without Ctrl+C killing it)
modal run --detach modal_train.py
modal run --detach modal_train.py --config configs/scaling/xs_standard.yaml
modal run --detach modal_train.py --config configs/scaling/xs_global.yaml
modal run --detach modal_train.py --config configs/scaling/xs_deepseek_global.yaml
modal run --detach modal_train.py --config configs/scaling/xs_deepseek_standard.yaml



## Configuration

### Cluster settings

Edit the top of `modal_train.py`:

```python
N_NODES = 2              # number of containers (machines)
GPUS_PER_NODE = 8        # GPUs per container
GPU_TYPE = "B200"        # B200, H200, or H100
TIMEOUT_HOURS = 24       # max wall-clock time
MAX_CHECKPOINTS = 3      # checkpoints kept on volume (0 = unlimited)
CONFIG_FILE = "configs/scaling/xs_standard.yaml"  # default config
```

These are module-level constants because Modal decorators evaluate them at import time. Change them before running `modal run`.

### Training config

Pass any YAML config as a CLI arg:

```bash
modal run modal_train.py --config configs/scaling/xs_standard.yaml
modal run modal_train.py --config configs/scaling/m_standard.yaml
modal run modal_train.py --config configs/scaling/l_standard.yaml
```

Available configs in `configs/scaling/`:

| Config | Model | Total Params | Active Params |
|--------|-------|-------------|---------------|
| `xs_standard.yaml` / `xs_global.yaml` | Qwen3-0.6B backbone | ~0.6B | ~200M |
| `s_standard.yaml` / `s_global.yaml` | Custom | ~1.5B | ~400M |
| `m_standard.yaml` / `m_global.yaml` | Custom | ~9.7B | ~200M |
| `l_standard.yaml` / `l_global.yaml` | Qwen3-30B-A3B | ~30B | ~3B |

Training hyperparameters (lr, batch size, warmup, etc.) are in each YAML file under the `training:` key.

## How It Works

### Docker image / environment

The container image is built automatically by Modal — no Dockerfile needed. It's defined in `modal_train.py` lines 51-76:

- **Base**: `nvidia/cuda:12.4.0-devel-ubuntu22.04` with Python 3.11
- **System packages**: curl, git, vim, htop
- **Python packages**: torch, accelerate, transformers, liger-kernel, wandb, etc. (full list in `modal_train.py`)
- **Code**: `train.py`, `src/`, `configs/`, and `torchrun_util/` are copied into the container at `/root/moe/`

Modal builds and caches this image. It only rebuilds when you change the image definition or the local files it copies.

### Volumes (persistent storage)

Two Modal Volumes are created automatically:

| Volume | Mount path | Purpose |
|--------|-----------|---------|
| `moe-training-data` | `/data` | Parquet training data (persists across runs) |
| `moe-checkpoints` | `/checkpoints` | Checkpoints organized by experiment name |

Data written to these volumes persists across function calls and retries. All nodes in the cluster share the same volume mounts.

### Multi-node training

```
modal run modal_train.py
       │
       ▼
  Modal spawns N_NODES containers, each with GPUS_PER_NODE GPUs
       │
       ▼
  Each container runs torchrun.run() which spawns GPUS_PER_NODE processes
       │
       ▼
  Each process runs train.py with Accelerate (auto-detects torchrun env)
       │
       ▼
  Accelerate sets up DDP across all nodes via NCCL + RDMA
```

- `@modal.experimental.clustered(size=N_NODES, rdma=True)` provisions the multi-node cluster with RDMA networking
- `torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
- `Accelerator()` auto-detects these env vars and creates the DDP process group
- NCCL handles all inter-node communication (all-reduce, etc.)

### Auto-resume

When `--auto_resume` is passed (always on in Modal mode):

1. `find_latest_checkpoint()` scans the output dir for `checkpoint-{step}` directories
2. If found, loads the latest one via `accelerator.load_state()`
3. Restores: model weights, optimizer state, LR scheduler, dataset position
4. WandB run ID is saved in checkpoint metadata, so the same WandB run continues
5. Old checkpoints are pruned to keep only `MAX_CHECKPOINTS` most recent

### Fault tolerance

```python
retries=modal.Retries(initial_delay=0.0, max_retries=10)
```

If a node dies or the function fails, Modal restarts the entire cluster. On restart, auto-resume finds the last checkpoint on the volume and picks up where it left off. No manual intervention needed.

## Monitoring

### WandB

If `wandb-secret` is configured, metrics are logged to your WandB project (set via `wandb_project` in the YAML config). On resume, the same WandB run continues.

### Modal dashboard

```bash
# List running containers
modal container list

# Shell into a running container
modal container exec <container-id> bash

# Check GPU status inside container
nvidia-smi
```

## Standalone Mode (no Modal)

The modified `train.py` still works without Modal:

```bash
# Single-node with accelerate (original workflow)
./scripts/train.sh configs/scaling/xs_standard.yaml

# With auto-resume
./scripts/train.sh configs/scaling/xs_standard.yaml --auto_resume --max_checkpoints 3

# Manual resume from specific checkpoint
./scripts/train.sh configs/scaling/xs_standard.yaml --resume outputs/xs_standard_moe/checkpoint-5000
```

## File Structure

```
moe/
├── modal_train.py              # Modal launcher (multi-node + data download)
├── train.py                    # Training script (works with Modal and standalone)
├── torchrun_util/              # torchrun Python wrapper
│   ├── __init__.py
│   ├── torchrun.py             # torchrun.run() function
│   ├── argparse_to_pydantic.py # argparse → pydantic conversion
│   └── argparse_util.py        # env var argparse actions
├── configs/scaling/            # Training configs (model + hyperparams)
├── src/                        # Model code, data loading, utilities
├── scripts/                    # Standalone launch scripts
└── accelerate_configs/         # Accelerate configs (standalone mode only)
```


