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
| `norm_topk_prob` | bool | true | Normalize top-k routing probabilities |
| `router_exploration_rate` | float | 0.0 | Random exploration rate during training |
| `router_score_function` | string | `softmax` | Softmax-family router scoring: `softmax`, `sigmoid`, `sqrtsoftplus`. Ignored by DeepSeek router. See `docs/routing.md` §1.5.1. |
| `softmax_position` | string | `pre_topk` | DEC-17 (RESOLVED → AC-1 task38): when to apply the score function relative to top-K. `pre_topk` (default) = score function on all experts → top-K → gather (softmax happens BEFORE the top-K selection). `post_topk` = top-K on raw logits → score function only on the K selected (softmax happens AFTER the top-K selection). Ignored by DeepSeek router (which always uses sigmoid + biased top-K). Deprecated alias `router_topk_ordering` ∈ `{post, pre}` is accepted with a `DeprecationWarning`; mapping is `post → pre_topk`, `pre → post_topk`. The runtime rejects `softmax_position=post_topk` with `top_k=1` because softmax of a single selected logit yields a constant `1.0` weight (gradient kill). See `docs/routing.md` §1.5.2. |
| `router_z_loss_coef` | float | 0.0 | Logit-magnitude regularizer summed across routers and added to the output loss pre-backward. `0.0` = disabled (the `collect_router_z_loss` walk short-circuits). See `docs/routing.md` §1.5.4. |

### DeepSeek Routing Fields

Used when `router_type: deepseek`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topk_scaling_factor` | float | null | Post-normalization scaling factor |
| `num_groups` | int | null | Expert groups for group-limited routing (now also honoured by the softmax-family router) |
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

#### Branch-router exploration_only (preferred nested form)

When using exploration_only branch routing, the preferred config form
is the nested `branch_router` block:

```yaml
model:
  type: moe_everything
  branch_router:
    balancing: exploration_only        # "none" | "exploration_only"
    exploration_rate: 1.0              # initial p_explore at step 0
    exploration_decay: cosine          # "constant" | "linear" | "cosine"
    exploration_min: 0.01              # floor reached at warmup_steps
    exploration_warmup_steps: 1000     # decay length in steps
```

| Field (under `model.branch_router`) | Type | Default | Description |
|-------------------------------------|------|---------|-------------|
| `balancing` | string | `none` | `none` keeps the regular argmax routing; `exploration_only` flips every BranchRouter on the model into rate-driven uniform Bernoulli random routing for masked tokens. Branch aux loss and DeepSeek branch bias state are inert under `exploration_only`. |
| `exploration_rate` | float | 0.0 | Initial `p_explore` at step 0. Default 0.0 so a `balancing="none"` build never accidentally enables exploration_only via the BranchRouter's auto-promote rule. Set this explicitly when opting into `exploration_only`. |
| `exploration_decay` | string | `constant` | Decay shape applied by the trainer's per-step pre-forward schedule helper. `constant` ignores `exploration_warmup_steps` and `exploration_min`; `linear` and `cosine` decay from `exploration_rate` to `exploration_min` over `exploration_warmup_steps` steps. |
| `exploration_min` | float | 0.0 | Floor `p_explore` reached at `step >= exploration_warmup_steps`. Returned for every step thereafter. |
| `exploration_warmup_steps` | int | 0 | Decay length in steps. 0 with a non-constant schedule returns `exploration_min` immediately. |

##### Flat-bridge fields (legacy yamls)

The flat fields below are accepted on the `model:` block as a bridge
for legacy yamls until the nested-schema migration lands. The nested
form wins when both are present.

| Flat field | Maps to | Notes |
|------------|---------|-------|
| `branch_balancing` | `branch_router.balancing` | |
| `branch_exploration_rate` | `branch_router.exploration_rate` | |
| `branch_exploration_decay` | `branch_router.exploration_decay` | |
| `branch_exploration_min` | `branch_router.exploration_min` | |
| `branch_exploration_warmup_steps` | `branch_router.exploration_warmup_steps` | |

##### Trainer telemetry keys

When `branch_router.balancing == "exploration_only"` is active on
the model, the trainer's `log_training_step` writes the following
keys to console + W&B (omitted on inactive models so they incur no
extra payload cost):

| Key | Meaning |
|-----|---------|
| `train/branch_explore_rate` | `p_explore(global_step)` applied before this step's forward. |
| `train/branch_attn_fraction` | Global-mean fraction of branch tokens that chose ATTN (`selected_experts == 0`) on this step's forward. |
| `train/branch_attn_fraction/depth_<i>` | Per-router ATTN fraction for `per_layer_router=True` builds. Emitted only when there are >= 2 routers. |
| `train/branch_explore_mask_fraction` | Diagnostic-only fraction of branch tokens routed via the random override mask. NOT the same as `% ATTN`. |

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
| `bias_update_rate` | float | 0.0 | Expert bias update rate (0 = disabled). **Canonical block: `training:` per DEC-3b.** Legacy `model:` placement is deprecated; the resolver emits a `DeprecationWarning` and run [`scripts/migrate_balancing_fields_to_training.py`](../scripts/migrate_balancing_fields_to_training.py) to migrate. |
| `bias_warmup_start` | float | 0.0 | Initial bias rate at step 0 (linear ramp to `bias_update_rate`). Canonical block: `training:`. |
| `bias_warmup_steps` | int | 0 | Steps to ramp `bias_warmup_start` → `bias_update_rate`; 0 = no warmup. Canonical block: `training:`. |
| `bias_update_zero_sum` | bool | true | DEC-2 mode selector for the DeepSeek-style bias update. `true` (default) uses the nmoe / DeepSeek-V3 zero-sum formulation `bias -= (s - s.mean()) * rate` so the cumulative bias mean is pinned at zero. `false` uses the Megatron-LM plain-sign update `bias += sign(avg_load - load) * rate` — the mean is allowed to drift up to ±`bias_update_rate` per step under asymmetric loads (still bounded by the ±16 clamp). See [`docs/routing.md` § DEC-2 update modes](routing.md#dec-2-update-modes). Canonical block: `training:`. |
| `router_aux_loss_coef` | float | 0.001 | Switch-Transformer batch-level auxiliary loss coefficient (consumed by every model family's `forward`). **Canonical block: `training:` per DEC-3b** (was previously `model:` — same migration as `bias_update_rate`). |
| `seq_aux_loss_coef` | float | 0.0 | DeepSeek V3 sequence-level aux loss coefficient. **Canonical block: `training:` per DEC-3b**. |
| `load_balancing_method` | string | `aux_loss` | `aux_loss \| seq_aux_loss \| deepseek_bias \| quantile \| none`. Canonical block: `training:`. **Authoritative dispatch (Round 5+):** `load_config()` runs `normalize_balancing_config()` which AUTO-ZEROS legacy coefficients outside the resolved method's active set (with `DeprecationWarning`). `build_model()` stamps the resolved method onto `model._load_balancing_method` and `model_config.load_balancing_method`. Each model family's `forward` reads `_load_balancing_method` and gates aux/seq-aux additions; `update_expert_biases()` no-ops for any method outside `{deepseek_bias}`. Active sets per method: `aux_loss → router_aux_loss_coef`; `seq_aux_loss → seq_aux_loss_coef`; `deepseek_bias → bias_update_rate + bias_warmup_* + bias_rate_*`; `quantile → ∅` (implementation pending Milestone D); `none → ∅` (everything zeroed). |
| `bias_rate_q` / `bias_rate_k` / `bias_rate_v` / `bias_rate_o` / `bias_rate_mlp` / `bias_rate_branch` | float | `bias_update_rate` | Per-projection DeepSeek bias-update rate overrides. Each defaults to the global `bias_update_rate` if not set. **Canonical block: `training:` per DEC-3b**. |
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
