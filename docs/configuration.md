# Configuration Reference

This document covers the YAML config schema, parameter reference, and example configurations.

## Table of Contents

- [1. Config File Structure](#1-config-file-structure)
- [2. Model Parameters](#2-model-parameters)
- [3. Training Parameters](#3-training-parameters)
- [4. Data Parameters](#4-data-parameters)
- [5. Eval Parameters](#5-eval-parameters)
- [6. Config Directory Layout](#6-config-directory-layout)
- [7. Example Configs](#7-example-configs)

---

## 1. Config File Structure

All configs are YAML files with three main sections:

```yaml
model:
  type: standard_moe
  # ... model architecture params

training:
  learning_rate: 1.0e-3
  # ... optimizer and training params

data:
  format: parquet
  # ... data loading params

eval:                    # optional
  enabled: true
  # ... evaluation params
```

---

## 2. Model Parameters

### Common (all models)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | required | Model type (see below) |
| `vocab_size` | int | 151936 | Vocabulary size |
| `hidden_size` | int | 1024 | Hidden dimension |
| `num_hidden_layers` | int | 16 | Number of transformer layers |
| `head_dim` | int | 128 | Attention head dimension |
| `num_attention_heads` | int | 16 | Number of attention heads |
| `num_key_value_heads` | int | 2 | GQA key-value heads |
| `intermediate_size` | int | 2816 | Dense MLP intermediate size |
| `rms_norm_eps` | float | 1e-6 | RMSNorm epsilon |
| `rope_theta` | float | 1000000 | RoPE base frequency |
| `max_position_embeddings` | int | 32768 | Max sequence length |
| `tie_word_embeddings` | bool | true | Share input/output embeddings |

### Model types

| Type | Description |
|------|-------------|
| `dense` | Standard dense Qwen3 transformer |
| `gpt2_dense` | GPT-2 style dense transformer |
| `standard_moe` | Per-layer MoE with softmax routing |
| `deepseek_standard_moe` | Per-layer MoE with DeepSeek sigmoid routing |
| `global_moe` | Shared global expert pool with softmax routing |
| `deepseek_global_moe` | Shared global expert pool with DeepSeek routing |
| `moe_everything` | Branch routing + attention/MLP expert banks |

### MoE-specific (standard_moe, global_moe)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | int | 16 | Experts per layer (standard) or total (global) |
| `num_experts_per_tok` | int | 4 | Top-K experts selected per token |
| `moe_intermediate_size` | int | 768 | Expert FFN intermediate size |
| `router_aux_loss_coef` | float | 0.001 | Batch load-balancing loss coefficient |
| `seq_aux_loss_coef` | float | 0.0 | Sequence-level aux loss coefficient |
| `output_router_logits` | bool | true | Return router logits for loss computation |
| `norm_topk_prob` | bool | true | Normalize top-K routing weights |

### DeepSeek routing (deepseek_* variants)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_deepseek_routing` | bool | true | Enable sigmoid + bias routing |
| `router_exploration_rate` | float | 0.02 | Random expert exploration probability |
| `bias_update_rate` | float | 0.001 | Expert bias learning rate |
| `topk_scaling_factor` | float | 2.5 | Learned scaling for top-K weights |
| `num_groups` | int | 0 | Group-limited routing groups (0=disabled) |
| `group_topk` | int | 0 | Groups selected per token |

### MoE-Everything specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_attn_experts` | int | 4 | Attention expert pool size |
| `num_attn_experts_per_tok` | int | 1 | Attention experts per token |
| `attn_expert_mode` | str | `per_head_fully_independent` | Attention routing mode |
| `branch_router_aux_loss_coef` | float | 0.0 | Branch routing balance loss |
| `per_layer_router` | bool | false | Per-depth vs shared routers |
| `per_layer_mlp_router` | bool | false | Per-depth MLP routers |
| `per_layer_attn_router` | bool | false | Per-depth attention routers |
| `scale_branch_by_routing_weight` | bool | true | Scale output by branch probability |
| `router_exploration_rate` | float | 0.0 | Random expert selection rate |
| `dynamic_depth_min` | float | 1.0 | Min depth multiplier (random) |
| `dynamic_depth_max` | float | 1.0 | Max depth multiplier (random) |

---

## 3. Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | str | `adam` | Optimizer: `adam` or `speedrun` (Muon+Adam) |
| `learning_rate` | float | 3e-4 | Peak learning rate |
| `weight_decay` | float | 0.1 | Weight decay coefficient |
| `beta1` | float | 0.9 | Adam/Muon beta1 |
| `beta2` | float | 0.95 | Adam beta2 |
| `eps` | float | 1e-8 | Adam epsilon |
| `max_grad_norm` | float | 1.0 | Gradient clipping norm (0=disabled) |
| `lr_scheduler` | str | `cosine` | Schedule: cosine, linear, constant, stable_decay |
| `warmup_steps` | int | 2000 | LR warmup steps |
| `max_steps` | int | 100000 | Total training steps |
| `min_lr_ratio` | float | 0.1 | Min LR as fraction of peak |
| `cooldown_frac` | float | 0.45 | Cooldown fraction (stable_decay only) |
| `batch_size` | int | 4 | Per-GPU batch size |
| `gradient_accumulation` | int | 4 | Gradient accumulation steps |
| `mixed_precision` | str | `bf16` | Precision: fp32, bf16, fp16 |
| `gradient_checkpointing` | bool | false | Activation checkpointing |
| `torch_compile` | bool | false | Enable torch.compile |
| `log_every` | int | 10 | Steps between log prints |
| `routing_log_every` | int | 50 | Steps between routing stats |
| `save_every` | int | 5000 | Steps between checkpoints |
| `heatmap_every` | int | 500 | Steps between routing heatmaps |
| `output_dir` | str | `./outputs` | Checkpoint directory |
| `wandb_project` | str | `moe-experiments` | WandB project (null=disabled) |
| `wandb_run_name` | str | null | WandB run name |
| `disable_liger` | bool | false | Disable Liger kernel optimizations |

### Expert bias update parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bias_update_rate` | float | 0.0 | Main bias update rate (0=disabled) |
| `bias_rate_q` | float | null | Per-Q-projection rate override |
| `bias_rate_k` | float | null | Per-K-projection rate override |
| `bias_rate_v` | float | null | Per-V-projection rate override |
| `bias_rate_o` | float | null | Per-O-projection rate override |
| `bias_rate_mlp` | float | null | Per-MLP rate override |
| `bias_rate_branch` | float | null | Per-branch rate override |
| `bias_warmup_start` | float | 0.0 | Initial bias rate |
| `bias_warmup_steps` | int | 0 | Bias rate warmup steps |
| `bias_interpolation` | bool | false | Enable per-layer/global interpolation |
| `bias_interpolation_warmup_steps` | int | 5000 | Interpolation schedule duration |

---

## 4. Data Parameters

### Parquet format (primary)

```yaml
data:
  format: parquet
  data_dir: ./data/fineweb
  text_column: text
  seq_len: 2048
  tokenizer_name: Qwen/Qwen3-0.6B
  num_workers: 4
  split: train           # train | val | all
  holdout_fraction: 0.05
```

### Token bin format (speedrun/legacy)

```yaml
data:
  format: token_bin
  files_glob: data/fineweb10B/*.bin
  seq_len: 2048
  header_bytes: 1024
  token_dtype: uint16
  max_tokens: null
  shuffle_files: false
  repeat: true
  align_to_bos: false
  bos_token_id: 50256
```

---

## 5. Eval Parameters

```yaml
eval:
  enabled: true
  every: 5000             # steps between evals
  max_batches: 50         # batches per eval (0=all)
  data_dir: ./data/val    # override data dir (optional)
  split: val              # override split
  holdout_fraction: 0.05  # file holdout for val
  batch_size: 32          # eval batch size
```

---

## 6. Config Directory Layout

```
configs/
  scaling/                    # Production configs at different scales
    xs_standard.yaml          # XS scale, standard MoE
    xs_global.yaml            # XS scale, global MoE
    s_standard.yaml           # Small scale
    m_standard.yaml           # Medium scale
    l_standard.yaml           # Large scale
    debug8_xs_deepseek_*.yaml # Debug/test configs
    
  depth_matched/              # Controlled experiments matching depth
    4_layers/                 # 4-layer variants
    8_layers/                 # 8-layer variants
    16_layers/                # 16-layer variants
    
  plan/                       # Planned/experimental configs
    
  speedrun/                   # Speedrun-specific configs (to be removed)
```

---

## 7. Example Configs

### Minimal Standard MoE

```yaml
model:
  type: standard_moe
  vocab_size: 151936
  hidden_size: 1024
  num_hidden_layers: 16
  num_attention_heads: 16
  num_key_value_heads: 2
  head_dim: 128
  intermediate_size: 2816
  num_experts: 16
  num_experts_per_tok: 4
  moe_intermediate_size: 768
  router_aux_loss_coef: 0.01
  output_router_logits: true

training:
  learning_rate: 1.0e-3
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 1000
  max_steps: 100000
  batch_size: 64
  gradient_accumulation: 1
  mixed_precision: bf16
  log_every: 10
  save_every: 5000
  output_dir: ./outputs/xs_standard

data:
  format: parquet
  data_dir: ./data/fineweb
  seq_len: 1024
  tokenizer_name: Qwen/Qwen3-0.6B
```

### MoE-Everything with DeepSeek Routing

```yaml
model:
  type: moe_everything
  vocab_size: 151936
  hidden_size: 1024
  num_hidden_layers: 16
  num_attention_heads: 16
  num_key_value_heads: 2
  head_dim: 128
  num_experts: 16
  num_experts_per_tok: 4
  num_attn_experts: 8
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_fully_independent
  use_deepseek_routing: true
  bias_update_rate: 0.001
  router_exploration_rate: 0.02
  branch_router_aux_loss_coef: 0.001
  router_aux_loss_coef: 0.001
  seq_aux_loss_coef: 0.0001

training:
  learning_rate: 1.0e-3
  weight_decay: 0.01
  max_steps: 100000
  batch_size: 32
  gradient_accumulation: 2
  mixed_precision: bf16
  save_every: 5000

data:
  format: parquet
  data_dir: ./data/fineweb
  seq_len: 2048
  tokenizer_name: Qwen/Qwen3-0.6B
```
