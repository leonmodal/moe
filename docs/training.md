# Training

## Unified Trainer

All supported models use a single training entrypoint: `scripts/train.py`.

```bash
# Single GPU
python scripts/train.py --config configs/standard_moe.yaml

# Multi-GPU DDP
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml

# FSDP
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml --dist-strategy fsdp
```

The trainer supports all model types: `dense`, `standard_moe`, `global_moe`, `moe_everything`.

## Training Library

Shared modules in `src/training/`:

| Module | Responsibility |
|--------|---------------|
| `config.py` | TrainingConfig, YAML loading, config resolution |
| `model_factory.py` | Model construction with normalized taxonomy |
| `data.py` | Parquet dataset construction (sole data path) |
| `checkpoint.py` | Separate model/optimizer/data/training state saves |
| `eval.py` | Validation loop |
| `routing.py` | Expert bias updates during training |
| `distributed.py` | DDP/FSDP setup, barrier, reduce |
| `logging.py` | WandB + console logging, routing plots |
| `metrics.py` | Output metrics computation |
| `trainer.py` | Main training loop |

## Optimizer

- `optimizer: adamw` (default) — Standard AdamW with separate weight-decay groups
- `optimizer: muon` — Hybrid Muon (for weight matrices) + Adam (for scalars/embeddings)
  - Muon uses Newton-Schulz iteration for update direction
  - Momentum warmup over configurable steps (default 300)

## Learning Rate Schedule

Supported schedules via `lr_scheduler`:
- `cosine` — Cosine annealing with warmup
- `linear` — Linear decay with warmup
- `constant` — Constant after warmup
- `stable_decay` — Constant then linear cooldown (no warmup)

## Expert Bias Updates

For DeepSeek-style routing (`router_type: deepseek`), expert biases are updated after each step:

1. Count tokens per expert (all-reduced across workers)
2. Compute load imbalance: `sign(load - expected)`
3. Zero-sum update: `bias -= (s - s.mean()) * rate`
4. Clamp to ±16

Configuration (see `TrainingConfig` in `src/training/config.py`):
- `bias_update_rate`: Base rate, default `0.0` (disabled). The bias update pass is skipped entirely when this is `0`; set it to `1e-3` (DeepSeek V3 reference) on MoE configs to enable.
- `bias_warmup_start`: Initial rate for linear warmup, default `0.0`.
- `bias_warmup_steps`: Steps to ramp from `warmup_start` to `bias_update_rate`, default `0` (no warmup).

## Router-Exploration Warmup

Linearly schedule the effective `router_exploration_rate` from a lower start value to the model-configured target over the first N steps, then hold at the target. Useful for stabilizing DeepSeek routers during the first few hundred steps before the router has learned; see `docs/research/external_moe_techniques.md` §"Implemented here with benchmark evidence".

Configuration:
- `router_exploration_warmup_start`: Rate at step 0, default `0.0`.
- `router_exploration_warmup_steps`: Steps to ramp from `warmup_start` to the model's `router_exploration_rate`, default `0` (feature disabled; rate stays at the model target every step).

## Loss Computation

- Cross-entropy loss with model-handled label shifting (`labels = input_ids`)
- Auxiliary losses computed per model type:
  - `aux_loss`: Router auxiliary loss (batch-level)
  - `seq_aux_loss`: Sequence-level auxiliary loss
  - `branch_aux_loss`: Branch router loss (not used — BranchRouter has no load-balancing loss by design)
  - `attention_aux_loss`: Attention routing auxiliary loss
  - `aux_loss_normalized`: Normalized load-balancing metric

## Reference Loss Trajectories

Loss sanity checks from a 10,000-step multi-GPU run against `leonli66/latent-cot-finewebedu` (FineWeb-Edu derivative). Each config uses the GPT-2-base-class scale (16 layers × 1024 hidden, seq_len 1024, batch 128 global, bf16 AdamW). See `status.md` §"Current-Stack Loss Evidence" for the full table and `scripts/validate_multi_gpu_pipeline.py` for the orchestration code.

| Config | Final CE at step 10000 |
|---|---|
| `dense` | **3.022** |
| `standard_moe` softmax (16 experts × top-2) | **3.006** |
| `standard_moe` deepseek (group-limited 4/2) | **3.035** |

All three land under the 3.28 FineWeb GPT-2 reference target. A loss stuck at ~4.0+ after 1k steps on this data is a regression signal, not the normal trajectory.

## Checkpoints

Saved as separate files per checkpoint directory:
- `model.pt` — model weights only
- `optimizer_adam.pt` or `optimizer_muon.pt` — optimizer state (type-specific)
- `training_state.pt` — scheduler, step, tokens_seen, wandb_run_id
- `data_state.pt` — dataset position for deterministic resume
- `meta.json` — human-readable metadata

Convert to safetensors: `python scripts/convert_checkpoint.py path/to/checkpoint-1000`

## Liger Kernels

Optional acceleration via `liger-kernel`:
- Full mode for standard MoE models
- Partial mode for moe_everything (rope + rms_norm only)
- Disabled via `training.disable_liger: true` or `MOE_DISABLE_LIGER=1`

## Modal Multi-Node

`modal_train.py` launches multi-node training on Modal cloud:

```bash
modal run modal_train.py --config configs/scaling/m_standard.yaml
```
