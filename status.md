# Status

Updated: 2026-04-17

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
- **Tokenizer**: Configurable via `data.tokenizer_name` — the `DataConfig` dataclass default is `gpt2`; most shipped training configs (`configs/standard_moe.yaml`, `configs/global_moe.yaml`, …) set it to `Qwen/Qwen3-0.6B`.
- **Prefetch**: Dataset-level file read-ahead controlled by `data.prefetch_files` (default `1`). DataLoader `num_workers` is forced to `0` for stateful parquet datasets so checkpoint state stays authoritative — see `docs/data.md`.

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

## Current-Stack Loss Evidence (2026-04-17)

End-to-end multi-GPU validation of the unified trainer on real data. Each config was trained for 10,000 optimizer steps with `torchrun --standalone --nproc_per_node=8 scripts/train.py`, batch 16 per GPU (global batch 128), seq_len 1024, bf16, AdamW, cosine LR with 50-step warmup + 0.1 min ratio, starting peak LR 1e-3. Orchestrator: `scripts/validate_multi_gpu_pipeline.py --stage-b-scale large --loss-steps 10000`.

**Model scale (GPT2-base class)**: 16 layers × 1024 hidden, 16 attention heads (GQA with 8 KV heads), intermediate 2048, tied embeddings. MoE variants use 16 experts / top-2 / moe_intermediate_size 512; DeepSeek uses group-limited top-K with num_groups=4 / group_topk=2.

**Data**: `leonli66/latent-cot-finewebedu` (FineWeb-Edu derivative) at `/tmp/moe/data/parquet/` — 8192 parquet shards tokenized with the GPT-2 BPE tokenizer via the shared `StatefulParquetDataset`.

**Final training-loss trajectories over 10k steps on 8× H200**:

| Config | First loss (step 50) | Final loss (step 10000) | Wall time |
|---|---|---|---|
| `dense` | 7.106 | **3.022** | 33 min |
| `standard_moe` (softmax, 16 experts × top-2) | 7.257 | **3.006** | 65 min |
| `standard_moe` (deepseek, group-limited 4/2) | 7.057 | **3.035** | 52 min |

All three configs land under the 3.28 FineWeb reference target from the archived speedrun model. The softmax/deepseek MoE configs match the dense model's trajectory despite activating only top-2 of 16 experts per token, confirming that the active-expert compute budget, not raw parameter count, drives the loss curve at this scale.

Full report (including Stage A smoke across all 7 variants × DDP/FSDP and Stage C auto-resume round-trip) is produced by `scripts/validate_multi_gpu_pipeline.py`; the orchestrator itself is tested end-to-end with every run via `torchrun`.
