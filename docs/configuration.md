# Configuration Reference

Training configs are YAML files with these sections: `model`, `training`, `data`, `eval`, `checkpoint`.

## Validation

```bash
python scripts/validate_configs.py                        # all active configs
python scripts/validate_configs.py configs/standard_moe.yaml  # specific config
```

## Model Section

### Common Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | required | Model type: `dense`, `standard_moe`, `global_moe`, `moe_everything` |
| `vocab_size` | int | required | Vocabulary size |
| `hidden_size` | int | required | Hidden dimension |
| `num_hidden_layers` | int | required | Number of transformer layers |
| `head_dim` | int | required | Attention head dimension |
| `num_attention_heads` | int | required | Number of attention heads |
| `num_key_value_heads` | int | required | Number of KV heads (GQA) |
| `intermediate_size` | int | auto | Dense MLP intermediate size |
| `max_position_embeddings` | int | 32768 | Max sequence length |
| `rope_theta` | float | 1000000.0 | RoPE theta |
| `rms_norm_eps` | float | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | bool | false | Tie input/output embeddings |
| `attn_implementation` | string | "sdpa" | Attention implementation |

### MoE Fields (standard_moe, global_moe, moe_everything)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_experts` | int | required | Number of experts per layer |
| `num_experts_per_tok` | int | required | Top-K experts selected per token |
| `moe_intermediate_size` | int | required | Expert MLP intermediate size |
| `router_type` | string | "softmax" | `softmax` or `deepseek` |
| `router_aux_loss_coef` | float | 0.001 | Auxiliary loss coefficient |
| `seq_aux_loss_coef` | float | 0.0 | Sequence-level aux loss coefficient |
| `norm_topk_prob` | bool | true | Normalize top-k routing probabilities |
| `router_exploration_rate` | float | 0.0 | Random exploration rate during training |

### DeepSeek Routing Fields

Used when `router_type: deepseek`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topk_scaling_factor` | float | null | Post-normalization scaling factor |
| `num_groups` | int | null | Expert groups for group-limited routing |
| `group_topk` | int | null | Groups selected per token |
| `use_deepseek_routing` | bool | false | Legacy flag (use `router_type` instead) |

### MoE-Everything Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_attn_experts` | int | 4 | Number of attention experts |
| `num_attn_experts_per_tok` | int | 1 | Attention experts per token |
| `attn_expert_mode` | string | `per_head_fully_independent` | Attention expert routing. Valid values: `per_head_fully_independent` (H routers per Q/K/V/O projection, each top-1) or `per_head_precompute_kv` (H bundled QKVO routers, each top-1). Any other value is rejected with `ValueError`. |
| `per_layer_router` | bool | false | Per-layer vs shared router |
| `branch_router_aux_loss_coef` | float | 0.0 | Branch aux loss (0 = no branch balancing) |

## Training Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | 3e-4 | Base learning rate |
| `weight_decay` | float | 0.1 | AdamW weight decay |
| `max_grad_norm` | float | 1.0 | Gradient clipping |
| `lr_scheduler` | string | "cosine" | `cosine`, `linear`, `constant`, `stable_decay` |
| `warmup_steps` | int | 2000 | LR warmup steps |
| `max_steps` | int | 100000 | Total training steps |
| `min_lr_ratio` | float | 0.1 | Minimum LR as fraction of peak |
| `batch_size` | int | 4 | Per-GPU batch size |
| `gradient_accumulation` | int | 4 | Gradient accumulation steps |
| `mixed_precision` | string | "bf16" | `bf16`, `fp16`, or blank for fp32 |
| `optimizer` | string | "adamw" | `adamw` or `muon` |
| `output_dir` | string | required | Output directory for checkpoints |
| `wandb_project` | string | null | WandB project name |
| `bias_update_rate` | float | 0.0 | Expert bias update rate (0 = disabled) |
| `bias_warmup_start` | float | 0.0 | Initial bias rate at step 0 (linear ramp to `bias_update_rate`) |
| `bias_warmup_steps` | int | 0 | Steps to ramp `bias_warmup_start` → `bias_update_rate`; 0 = no warmup |
| `router_exploration_warmup_start` | float | 0.0 | Initial router-exploration rate at step 0; see `docs/training.md` §Router-Exploration Warmup |
| `router_exploration_warmup_steps` | int | 0 | Steps to ramp to the model's `router_exploration_rate`; 0 = feature disabled |
| `momentum_warmup_steps` | int | 300 | Muon optimizer momentum warmup horizon |
| `torch_compile` | bool | false | Enable torch.compile |
| `disable_liger` | bool | false | Disable liger kernels |

## Data Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_dir` | string | required | Path to parquet shards directory |
| `text_column` | string | "text" | Column name in parquet files |
| `seq_len` | int | 2048 | Sequence length |
| `tokenizer_name` | string | "gpt2" | HuggingFace tokenizer name |
| `num_workers` | int | 4 | DataLoader workers. **Overridden to 0 for stateful parquet datasets** — the trainer forces `num_workers=0` whenever the dataset exposes `get_state`/`set_state` so that checkpointed `data_state.pt` stays authoritative for deterministic resume (see `docs/data.md`). Throughput work is tracked separately under AC-8 as dataset-level read-ahead. |

## Eval Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable validation |
| `every` | int | required | Eval every N steps |
| `max_batches` | int | 0 | Max eval batches (0 = all) |
| `holdout_fraction` | float | 0.05 | Fraction of data held out for validation |
