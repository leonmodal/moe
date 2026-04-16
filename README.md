# MoE Training

Unified training codebase for Mixture-of-Experts language models built on custom Qwen3-based architecture.

## Supported Models

| Model Type | Description |
|-----------|-------------|
| `dense` | Standard dense transformer (Qwen3-based) |
| `standard_moe` | Per-layer MoE with softmax or DeepSeek routing (`router_type: deepseek`) |
| `global_moe` | Global shared expert pool across all layers, with softmax or DeepSeek routing |
| `moe_everything` | Branch-routed model with attention + MLP expert banks, per-head routing |

All models use Bagel-style custom Qwen3 components (GQA, QK norm, RoPE, SwiGLU) with HuggingFace `PreTrainedModel` compatibility for checkpoint save/load.

DeepSeek routing is a pluggable config option (`router_type: deepseek`) available to any MoE model — it is not a separate model type.

## Setup

```bash
uv sync
```

Add credentials to `.env`:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

## Data

Download parquet shards:

```bash
uv run python scripts/download_data.py --max_shards 64
```

## Training

### Single GPU
```bash
python scripts/train.py --config configs/standard_moe.yaml
```

### Multi-GPU DDP
```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml
```

### FSDP
```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml --dist-strategy fsdp
```

### Modal Multi-Node
```bash
modal run modal_train.py --config configs/scaling/m_standard.yaml
```

## Model Families

### `dense`
- Standard Qwen3 dense transformer
- No routing, no expert dispatch

### `standard_moe`
- Dense attention in every layer
- One routed MLP expert pool per layer
- Supports `router_type: softmax` (default) or `router_type: deepseek`

### `global_moe`
- Dense attention in every layer
- One shared global MLP expert pool across all layers
- Each layer has its own router into the shared pool

### `moe_everything`
- Custom depth loop with branch routing (attention vs MLP per token)
- Attention experts and MLP experts in shared banks across all depths
- Per-head attention routing: H separate top-1 routers per projection type
- Modes: `per_head_fully_independent` and `per_head_precompute_kv`

## Configuration

Configs use YAML format. Validate configs with:

```bash
python scripts/validate_configs.py
```

Key config fields:
- `model.type`: One of `dense`, `standard_moe`, `global_moe`, `moe_everything`
- `model.router_type`: `softmax` (default) or `deepseek` for DeepSeek-style sigmoid routing
- `training.optimizer`: `adamw` (default) or `muon` (Muon + Adam hybrid)
- `training.bias_update_rate`: Expert bias update rate for DeepSeek routing (0 = disabled)

## Checkpoints

Checkpoints are saved as separate files:
- `model.pt` — model weights
- `optimizer_adam.pt` or `optimizer_muon.pt` — optimizer state (type-specific)
- `training_state.pt` — scheduler, step count, tokens seen
- `data_state.pt` — dataset position for deterministic resume

Convert to safetensors:
```bash
python scripts/convert_checkpoint.py path/to/checkpoint-1000
```

## Project Structure

```
scripts/
├── train.py              # Unified training CLI entrypoint
├── convert_checkpoint.py  # Safetensors conversion utility
└── validate_configs.py    # Config validator/linter

src/
├── training/             # Shared training library
│   ├── config.py         # TrainingConfig, config loading
│   ├── model_factory.py  # Model builder with normalized taxonomy
│   ├── data.py           # Parquet dataset construction
│   ├── checkpoint.py     # Separate model/optimizer/data saves
│   ├── eval.py           # Validation loop with loss checks
│   ├── routing.py        # Expert bias updates during training
│   ├── distributed.py    # DDP/FSDP setup
│   ├── logging.py        # WandB and console logging
│   ├── metrics.py        # Output metrics computation
│   └── trainer.py        # Main training loop
├── models/
│   ├── base/             # Custom Qwen3 components (Bagel-style)
│   │   ├── attention.py  # GQA with QK norm
│   │   ├── mlp.py        # SwiGLU MLP
│   │   ├── normalization.py  # RMSNorm + FunctionalRMSNorm
│   │   ├── embeddings.py # RoPE
│   │   └── output_head.py # LM head (optional FP8, softcapping)
│   ├── routing/          # Unified routing module
│   │   ├── routers.py    # DeepSeekRouter, ExplorationTopKRouter, BranchRouter
│   │   ├── load_balancing.py  # Loss functions
│   │   ├── bias.py       # Global bias management
│   │   ├── stats.py      # Per-forward-pass routing stats
│   │   ├── helpers.py    # Routing dispatch helpers
│   │   └── fp32_ops.py   # FP32 index ops for stability
│   ├── moe_everything/   # Split MoE-Everything package
│   ├── standard_moe.py
│   ├── global_moe.py
│   ├── mixture_of_everything.py
│   ├── router.py
│   └── load_balancing.py
├── data/
│   └── parquet_dataset.py  # StatefulParquetDataset (sole data path)
└── utils/
    ├── routing_stats.py  # Count-based routing statistics
    ├── routing_plots.py  # Routing visualization/graphs
    ├── muon.py           # Muon optimizer
    └── dist_optimizers.py # Distributed optimizers

legacy/speedrun/          # Archived speedrun models and configs
configs/                  # Active training configs
```

## Legacy

Speedrun model architectures (speedrun_gpt, speedrun_moe_gpt) have been archived to `legacy/speedrun/`. Reusable components were extracted into active code before archival. See `legacy/speedrun/README.md` for details.
