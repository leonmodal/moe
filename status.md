# Status

Updated: 2026-04-16

## Current State

The codebase has been reorganized into a unified training architecture.

### Active Training Path

- **Entrypoint**: `scripts/train.py` (unified CLI for all model types)
- **Training library**: `src/training/` (config, model factory, data, checkpoint, eval, distributed, logging, routing, metrics, trainer)
- **Distributed**: DDP and FSDP via `--dist-strategy ddp|fsdp`
- **Modal**: `modal_train.py` uses `scripts/train.py` via torchrun

### Supported Models

| Type | Description |
|------|-------------|
| `dense` | Qwen3-based dense transformer |
| `standard_moe` | Per-layer MoE with softmax or DeepSeek routing |
| `global_moe` | Global shared expert pool across layers |
| `moe_everything` | Branch-routed with attention + MLP expert banks |

DeepSeek routing is a pluggable config option: `router_type: deepseek`.

### Data

- **Format**: Sharded parquet only (token-bin removed)
- **Dataset**: `src/data/parquet_dataset.py` (StatefulParquetDataset)
- **Tokenizer**: Configurable (default: Qwen/Qwen3-0.6B)

### Checkpoints

Saved as separate files: `model.pt`, `optimizer_adam.pt` / `optimizer_muon.pt`, `training_state.pt`, `data_state.pt`.

Convert to safetensors: `python scripts/convert_checkpoint.py path/to/checkpoint`

### Config Validation

```bash
python scripts/validate_configs.py
```

### Archived

Speedrun models (speedrun_gpt, speedrun_moe_gpt), Accelerator-based trainer, and token-bin dataset are in `legacy/speedrun/`. See `legacy/speedrun/README.md` for details.

## Historical Benchmark Results

The speedrun_gpt model (now archived) achieved the FineWeb GPT-2 target:
- Final validation CE: **3.2748** (target: <= 3.28)
- Throughput: ~6.0M tok/s on 8x B200
- This model is archived but its reusable components (FlexAttention, FP8 lm_head, etc.) were extracted into `src/models/base/`.
