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

Configuration:
- `bias_update_rate`: Base rate (default 0.001)
- `bias_warmup_start`: Initial rate for linear warmup
- `bias_warmup_steps`: Steps to ramp from warmup_start to base rate

## Loss Computation

- Cross-entropy loss with model-handled label shifting (`labels = input_ids`)
- Auxiliary losses computed per model type:
  - `aux_loss`: Router auxiliary loss (batch-level)
  - `seq_aux_loss`: Sequence-level auxiliary loss
  - `branch_aux_loss`: Branch router loss (not used — BranchRouter has no load-balancing loss by design)
  - `attention_aux_loss`: Attention routing auxiliary loss
  - `aux_loss_normalized`: Normalized load-balancing metric

## Checkpoints

Saved as separate files per checkpoint directory:
- `model.pt` — model weights only
- `optimizer.pt` — optimizer state
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
