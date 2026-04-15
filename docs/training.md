# Training

This document covers the training loop, optimizers, LR schedulers, loss composition, and the comparison between the two current training scripts.

## Table of Contents

- [1. Training Scripts Overview](#1-training-scripts-overview)
- [2. train.py (Accelerate-based) -- TO BE REMOVED](#2-trainpy-accelerate-based----to-be-removed)
- [3. train_torch.py (Torch-native) -- UNIFIED TARGET](#3-train_torchpy-torch-native----unified-target)
- [4. Comparison: What train_torch.py Needs](#4-comparison-what-train_torchpy-needs)
- [5. Optimizers](#5-optimizers)
- [6. LR Schedulers](#6-lr-schedulers)
- [7. Loss Composition](#7-loss-composition)
- [8. Evaluation](#8-evaluation)
- [9. modal_train.py (Launcher)](#9-modal_trainpy-launcher)

---

## 1. Training Scripts Overview

| Script | Backend | Models Supported | Status |
|--------|---------|-----------------|--------|
| `train.py` | Accelerate (DDP) | dense, standard_moe, global_moe, moe_everything | **To be removed** |
| `train_torch.py` | Raw torch.distributed (DDP/FSDP) | speedrun models + all standard models | **Unified target** |
| `modal_train.py` | Modal launcher | Dispatches to train.py or train_torch.py | **Keep** |

**Goal**: Consolidate into a single `train_torch.py` that supports all model types with DDP and FSDP.

---

## 2. train.py (Accelerate-based) -- TO BE REMOVED

**File**: `train.py` (1785 lines)

Uses HuggingFace Accelerate for distributed training. Handles DDP wrapping, mixed precision, gradient accumulation, and checkpointing automatically.

**What it has that train_torch.py needs**:

### Sophisticated Expert Bias Update System (lines 120-281)
The most important missing piece. Includes:
- `bias_alpha_schedule()`: Cosine decay for per-layer vs global interpolation
- `get_bias_update_router_groups()`: Groups routers by shared expert pool
- `update_expert_biases()`: Main entry point with alpha interpolation for global MoE
- Per-projection bias rates (`bias_rate_q`, `bias_rate_k`, etc.)

### Accumulated Routing Statistics (lines 1637-1747)
- Per-layer expert count accumulation with all-reduce
- Router margin tracking (sum, count, min per layer)
- Attention expert count per router type
- Branch probability accumulation
- NormExpertBank token counting
- `compute_routing_stats_from_counts()` integration

### torch.compile eval safety (lines 446-448)
Skips `model.eval()` when torch.compile is active to avoid triggering recompilation.

### WandB resume support (lines 1097-1116)
Captures `wandb_run.id` into checkpoint metadata, restores on resume with `resume="allow"`.

---

## 3. train_torch.py (Torch-native) -- UNIFIED TARGET

**File**: `train_torch.py` (1399 lines)

Uses raw `torch.distributed` for DDP and FSDP. Has two code paths:

1. **Speedrun training** (`run_speedrun_training`, line 829): Custom loop with DistMuon/DistAdam, kernel warmup, dynamic window sizing, FlexAttention support
2. **Standard training** (`run_standard_training`, line ~1450): General loop for HF-based models with AdamW/Muon

**Dispatch** (line ~1393):
```python
if model_type.startswith("speedrun"):
    run_speedrun_training(config)
else:
    run_standard_training(config)
```

**What it already has**:
- DDP and FSDP support with manual wrapping
- Manual gradient accumulation with `model.no_sync()` for efficiency
- FSDP checkpointing with `FullStateDictConfig`
- Both Parquet and TokenBin data loading
- Basic expert bias updates (per-router, no alpha interpolation)
- WandB logging
- Auto-resume from latest checkpoint
- Kernel warmup (speedrun path)

---

## 4. Comparison: What train_torch.py Needs

To fully replace train.py, the following must be ported:

| Feature | train.py | train_torch.py | Gap |
|---------|----------|----------------|-----|
| All model types | Yes | Yes | None |
| DDP | Via Accelerate | Manual torch.distributed | None |
| FSDP | Via Accelerate config | Manual FSDP wrapping | None |
| Mixed precision | Via Accelerate | Manual torch.autocast | None |
| Gradient accumulation | Accelerate context | Manual no_sync loop | None |
| Checkpointing | Accelerate save/load | Manual torch.save/load + FSDP | None |
| **Expert bias (full system)** | Lines 120-281 | Basic only (1234-1263) | **Must port** |
| **Routing statistics** | Lines 1637-1747 | Partial | **Must port** |
| **WandB resume** | Via meta.json + wandb_run.id | Basic | **Must port** |
| **torch.compile eval** | Skip model.eval() | Not handled | **Must port** |
| Dataset state resume | Via meta.json | Via meta.json | None |
| Eval loop | Lines 429-495 | Lines 690-750 | Minor differences |

### Critical gaps to fill:

1. **Expert bias update system**: Port the full `update_expert_biases()` with alpha interpolation, router grouping, and per-projection rates
2. **Routing statistics accumulation**: Port `accumulate_expert_counts`, `accumulate_router_margins`, and the full logging block
3. **WandB run ID tracking**: Save/restore wandb run ID in checkpoint metadata
4. **torch.compile guard**: Check for compiled model before calling `model.eval()`

---

## 5. Optimizers

### AdamW

**File**: `src/utils/training.py:84-104`
**Config**: `optimizer: adam`

Standard PyTorch AdamW with parameter group splitting:
- **Decay group**: All weight matrices (default `weight_decay=0.1`)
- **No-decay group**: Biases, LayerNorm/RMSNorm weights, embeddings

```yaml
training:
  optimizer: adam
  learning_rate: 1.0e-3
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
```

### Muon (Momentum + Newton-Schulz Orthogonalization)

**File**: `src/utils/muon.py`
**Config**: `optimizer: speedrun` (triggers Muon + Adam hybrid)

Muon applies Newton-Schulz iteration (5 steps) to compute the polar decomposition of 2D weight matrices, then uses the orthogonalized gradient for the update. Non-2D parameters (biases, norms, embeddings) fall back to Adam.

**Newton-Schulz iteration**: Approximates `U` from the polar decomposition `W = U * S` using:
```
X_{k+1} = a * X_k + b * X_k @ X_k^T @ X_k + c * (X_k @ X_k^T)^2 @ X_k
```
with coefficients `a=3.4445, b=-4.7750, c=2.0315` (5 iterations).

**Why**: Orthogonalized gradients prevent weight matrix collapse and maintain condition number, leading to faster convergence for transformer weight matrices.

**Parameter splitting**:
- Muon params: 2D weight matrices (attention projections, MLP weights)
- Adam params with decay: Other matrices
- Adam params without decay: Biases, norms, embeddings

### Distributed Optimizers

**File**: `src/utils/dist_optimizers.py`

- **DistMuon**: Distributed Muon with `reduce_scatter` / `all_gather` for gradient sync. Handles `lr_mul` and `wd_mul` per-parameter attributes. No DDP wrapper needed.
- **DistAdam**: Distributed Adam with parameter sharding across ranks.

These are used in the speedrun training path where the optimizer handles communication instead of DDP.

---

## 6. LR Schedulers

**File**: `src/utils/training.py:40-66`
**Function**: `build_lr_scheduler()`

All schedulers include a linear warmup phase from 0 to peak LR over `warmup_steps`.

| Scheduler | Formula (post-warmup) | Use Case |
|-----------|----------------------|----------|
| **Cosine** | `min_ratio + 0.5*(1-min_ratio)*(1+cos(pi*progress))` | Default, standard LLM training |
| **Linear** | `max(min_ratio, 1 - progress*(1-min_ratio))` | Simple alternative |
| **Constant** | `1.0` | Fine-tuning, debugging |
| **Stable Decay** | Constant, then linear cooldown in final `cooldown_frac` | WSD-style schedule |

```yaml
training:
  lr_scheduler: cosine
  warmup_steps: 1000
  min_lr_ratio: 0.1      # minimum LR as fraction of peak
  cooldown_frac: 0.45     # only for stable_decay
```

---

## 7. Loss Composition

The total loss is a weighted sum of components:

```
total_loss = ce_loss
           + router_aux_loss_coef * batch_aux_loss
           + seq_aux_loss_coef * seq_aux_loss
           + branch_router_aux_loss_coef * branch_aux_loss
           + attention_aux_losses (per-projection)
```

| Component | Source | Typical Coeff | Description |
|-----------|--------|---------------|-------------|
| `ce_loss` | Model forward | 1.0 | Cross-entropy language modeling loss |
| `batch_aux_loss` | `load_balancing_loss_func()` | 0.001 | Switch-style MLP expert balance |
| `seq_aux_loss` | `seq_load_balancing_loss_func()` | 0.0001 | Per-sequence expert balance (DeepSeek V3) |
| `branch_aux_loss` | Branch router | 0.0 | Balance between attention/MLP branches |
| `attention_aux_loss` | Attention expert routers | varies | Per-head expert balance |

---

## 8. Evaluation

Periodic evaluation runs every `eval_every` steps:

1. Switch to eval mode (skip if torch.compile active)
2. Run forward pass on validation data (separate dataloader or holdout split)
3. Collect: CE loss, all aux loss components
4. All-reduce across ranks
5. Compute perplexity: `exp(min(20.0, ce_loss))`
6. Log to WandB

```yaml
eval:
  enabled: true
  every: 5000
  max_batches: 50
  data_dir: ./data/val    # or uses holdout from training data
  batch_size: 32
```

---

## 9. modal_train.py (Launcher)

**File**: `modal_train.py`

Modal cloud launcher for multi-node training. Key settings:

```python
N_NODES = 2
GPUS_PER_NODE = 8
GPU_TYPE = "H100"  # or "B200", "H200"
TIMEOUT_HOURS = 24
MAX_CHECKPOINTS = 3
```

- Mounts Modal volumes for data (`moe-training-data`) and checkpoints (`moe-checkpoints`)
- Uses torchrun for distributed process management
- Auto-resumes from latest checkpoint on failure/restart
- Dispatches to `train_torch.py` (or `train.py` for legacy configs)
