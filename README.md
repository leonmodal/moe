# MoE Training

This repo contains three model families:

- `standard_moe`
- `global_moe`
- `moe_everything`

The codebase is centered on pretraining and debugging routed language models built on Qwen3/Qwen3-MoE components.

## Setup

```bash
uv sync
```

Add credentials to `.env` if you want Hugging Face and WandB access:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

## Data

Download parquet shards into `data/parquet`:

```bash
uv run python scripts/download_data.py --max_shards 64
uv run python scripts/download_data.py
```

## Training

Single-node training:

```bash
./scripts/train.sh configs/standard_moe.yaml
./scripts/train.sh configs/global_moe.yaml
./scripts/train.sh configs/moe_everything_per_head_precompute_kv.yaml
```

`scripts/train.sh` uses `torchrun` by default when multiple GPUs are visible.

- `NPROC_PER_NODE=8 ./scripts/train.sh ...` runs 8-way DDP on one node
- `LAUNCHER=accelerate ./scripts/train.sh ...` uses the older Accelerate launcher path

Modal sweep for the current per-head runs:

```bash
bash scripts/launch_all.sh
```

That launches:

- `configs/moe_everything_per_head_independent_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_independent_perlayer_bothnorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_bothnorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_sanity.yaml`

## Root Configs

The root `configs/` directory now only contains actual train configs, not temporary debug compare configs.

Main baselines:

- `configs/standard_moe.yaml`
- `configs/global_moe.yaml`
- `configs/global_moe_nointerp.yaml`

Main MoE-Everything configs:

- `configs/moe_everything_bundled.yaml`
- `configs/moe_everything_kv_paired.yaml`
- `configs/moe_everything_qk_paired.yaml`
- `configs/moe_everything_fully_independent.yaml`
- `configs/moe_everything_precompute_kv.yaml`
- `configs/moe_everything_precompute_kv_perlayer.yaml`

Main per-head configs:

- `configs/moe_everything_per_head_fully_independent.yaml`
- `configs/moe_everything_per_head_precompute_kv.yaml`
- `configs/moe_everything_per_head_independent_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_independent_perlayer_bothnorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_bothnorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_sanity.yaml`

## Logging

The main training logger reports:

- `train/loss`
- `train/ce_loss`
- `train/aux_loss`
- `train/aux_loss_normalized`
- `train/seq_aux_loss`
- `train/branch_aux_loss`
- `train/tokens_per_sec`
- `train/sec_per_step`
- `train/tokens_seen_B`

For DeepSeek-style sigmoid routers:

- `train/aux_loss` is the raw batch aux diagnostic
- `train/aux_loss_normalized` is the easier-to-read normalized version
- `train/seq_aux_loss` is the more meaningful routing regularization metric

Branch routing is intentionally unconstrained in the per-head configs:

- `branch_router_aux_loss_coef: 0.0`
- no forced attention/MLP balance

## Docs

- [status.md](/tmp/moe/status.md): current status and recent fixes
- [MIXTURE_OF_EVERYTHING.md](/tmp/moe/MIXTURE_OF_EVERYTHING.md): model-family overview and MoE-Everything behavior
- [MULTINODE_README.md](/tmp/moe/MULTINODE_README.md): Modal multi-node usage
