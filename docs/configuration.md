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
| `attn_expert_mode` | string | `per_head_no_recompute` | Sequence-side attention recompute mode: `per_head_no_recompute`, `per_head_recompute_k`, or `per_head_recompute_kv`. |
| `attn_routing_bundle` | string | mode-dependent | Projection routing bundle. `per_head_no_recompute` supports `q_k_v_o`; recompute modes support `qk_v_o`, `qk_vo`, `qkv_o`, and `qkvo`. |
| `attn_router_context` | string | `none` | Attention-router context mode. `ema_qk_v` makes QK and V routers consume `concat(normed_h_t, causal_prefix_ema_t)`; O routing is unchanged. |
| `attn_router_context_decay` | float | `0.95` | EMA decay for `attn_router_context: ema_qk_v`; must be in `[0, 1)`. |
| `use_fused_linear_ce` | bool | true | MoE-Everything computes LM loss with model-native fused linear CE on CUDA and does not materialize full vocab logits in trainer/eval loss calls. |
| `per_layer_router` | bool | false | Per-layer vs shared router |
| `branch_router_aux_loss_coef` | float | 0.0 | Branch aux loss (0 = no branch balancing) |
| `prelude_layers` | int | 0 | Standard-MoE-style decoder blocks prepended **before** the recurrent depth loop. 0 = pure MoE-Everything (no boundary). See "Prelude / Coda Hybrid" below. |
| `coda_layers` | int | 0 | Standard-MoE-style decoder blocks appended **after** the recurrent depth loop. 0 = pure MoE-Everything (no boundary). |

#### Prelude / Coda Hybrid

When either `prelude_layers > 0` or `coda_layers > 0`, the model
prepends / appends standard-MoE-style decoder blocks around the
recurrent MoE-Everything bank. Each boundary block owns its own dense
GQA + its own per-layer MLP expert pool (no sharing with the bank, no
sharing across boundary blocks). Boundary attention reuses the main
config's `hidden_size`, `num_attention_heads`, `num_key_value_heads`,
`head_dim`, RoPE, and QK-norm settings. Boundary MLP geometry is
configured via the nested `prelude_coda` block:

```yaml
model:
  type: moe_everything
  prelude_layers: 4
  coda_layers: 4
  prelude_coda:
    num_experts: 16
    num_experts_per_tok: 4
    moe_intermediate_size: 768
    num_groups: 8
    group_topk: 4
    router_type: deepseek            # 'deepseek' or 'softmax'
    norm_topk_prob: true
    topk_scaling_factor: 2.5         # optional
    bias_update_rate: 0.001          # only used by router_type=deepseek
    bias_update_zero_sum: true
```

| Field (under `model.prelude_coda`) | Type | Default | Description |
|------------------------------------|------|---------|-------------|
| `num_experts` | int | bank's `num_experts` | Per-layer MLP expert pool size for every prelude/coda block. |
| `num_experts_per_tok` | int | bank's `num_experts_per_tok` | Top-K for prelude/coda MLP routers. |
| `moe_intermediate_size` | int | bank's `moe_intermediate_size` | SwiGLU intermediate dim for boundary experts. |
| `num_groups` / `group_topk` | int | bank's values | Group-limited top-K knobs for boundary routers. |
| `router_type` | string | `deepseek` | `deepseek` (DeepSeekRouter + bias updates) or `softmax` (ExplorationTopKRouter). |
| `norm_topk_prob` | bool | bank's `norm_topk_prob` | Normalize top-K weights to sum to 1 (boundary). |
| `topk_scaling_factor` | float | bank's `topk_scaling_factor` | DeepSeek scaling factor applied at the boundary. |
| `bias_update_rate` | float | 0.001 | Boundary deepseek-bias rate. Currently informational — the trainer's bias-update walker uses the bank's `mlp_router.bias_update_rate` for boundary MLP owners. |
| `bias_update_zero_sum` | bool | true | Boundary zero-sum mode (informational; same caveat as above). |

Each boundary block runs attn + MLP sequentially (= 1 standard layer
= 2 substeps). Each recurrent depth runs one substep (branch picks
attn OR MLP), so two recurrent depths equal one standard layer. The
"layer-equivalent" total is therefore:

```
layer_equivalents = prelude_layers + num_hidden_layers / 2 + coda_layers
substeps          = prelude_layers * 2 + num_hidden_layers + coda_layers * 2
```

To match the existing 16-standard-layer MoE-Everything baselines, set
`prelude_layers=4`, `num_hidden_layers=16`, `coda_layers=4` →
`4 + 8 + 4 = 16` layer-equivalents (= 32 substeps).

The trainer's deepseek-bias walker picks up every boundary MLP router
via `MoEverythingForCausalLM.get_all_balancing_owners` with label
`"mlp"`; no extra plumbing required.

See `configs/16_layers/moe_everything_prelude4_recurrent16_coda4_branch_*.yaml`
for working examples.

#### Branch Router

The preferred config form is the nested `branch_router` block:

```yaml
model:
  type: moe_everything
  branch_router:
    balancing: sampling_entropy        # none | sampling_entropy | exploration_only | fixed_alternating | aux_loss | seq_aux_loss | deepseek_bias | quantile
    entropy_coef: 0.01                 # only used by sampling_entropy
    entropy_decay: cosine
    entropy_min: 0.0
    entropy_decay_steps: 1000
```

| Field (under `model.branch_router`) | Type | Default | Description |
|-------------------------------------|------|---------|-------------|
| `balancing` | string | `none` | `none`, `aux_loss`, and `seq_aux_loss` use the regular softmax branch gate. `sampling_entropy` samples the branch categorical during training and adds a decaying entropy bonus to the loss. `exploration_only` keeps top-1 argmax as the default path, but applies a scheduled random override to a fraction of tokens. `fixed_alternating` is parameterless: even depths run attention, odd depths run MLP. `deepseek_bias` uses two independent sigmoid scores plus persistent ATTN/MLP bias. `quantile` enables branch quantile state. |
| `exploration_rate` | float | 0.0 | Initial `p_explore` at step 0. Default 0.0 so a `balancing="none"` build never accidentally enables exploration_only via the BranchRouter's auto-promote rule. Set this explicitly when opting into `exploration_only`. |
| `exploration_decay` | string | `constant` | Decay shape applied by the trainer's per-step pre-forward schedule helper. `constant` ignores `exploration_warmup_steps` and `exploration_min`; `linear` and `cosine` decay from `exploration_rate` to `exploration_min` over `exploration_warmup_steps` steps. |
| `exploration_min` | float | 0.0 | Floor `p_explore` reached at `step >= exploration_warmup_steps`. Returned for every step thereafter. |
| `exploration_warmup_steps` | int | 0 | Decay length in steps. 0 with a non-constant schedule returns `exploration_min` immediately. |
| `entropy_coef` | float | 0.0 | Initial branch entropy-bonus coefficient for `sampling_entropy`. The model subtracts `entropy_coef * entropy(branch_probs)` from the training loss. |
| `entropy_decay` | string | `constant` | Entropy coefficient decay shape: `constant`, `linear`, or `cosine`. |
| `entropy_min` | float | 0.0 | Final entropy coefficient after decay. |
| `entropy_decay_steps` | int | 0 | Number of steps for `entropy_coef -> entropy_min`. |

For `branch_router.balancing: deepseek_bias`, the branch gate is not a
Switch-style two-way softmax. It computes:

```python
scores = sigmoid(gate(hidden_states))      # two independent scores
choice = argmax(scores + expert_bias)
```

The selected branch output is still multiplied by the unbiased selected
sigmoid score so the branch gate receives task-loss gradient.

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
| `branch_entropy_coef` | `branch_router.entropy_coef` | |
| `branch_entropy_decay` | `branch_router.entropy_decay` | |
| `branch_entropy_min` | `branch_router.entropy_min` | |
| `branch_entropy_decay_steps` | `branch_router.entropy_decay_steps` | |

##### Trainer telemetry keys

The trainer logs branch telemetry when the corresponding data is
available. `sampling_entropy` rows emit `train/branch_entropy`;
`exploration_only` rows also emit the exploration-mask diagnostics.

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
| `fsdp_sharding_strategy` | string | null | Optional FSDP override used when `--dist-strategy fsdp` is active: `auto`, `full_shard`, `shard_grad_op`, `no_shard`, or `hybrid_shard`. CLI `--fsdp-sharding-strategy` wins over this field. |
| `optimizer` | string | "adamw" | `adamw` or `muon` |
| `output_dir` | string | required | Output directory for checkpoints |
| `wandb_project` | string | null | WandB project name |
| `bias_update_rate` | float | 0.0 | Expert bias update rate (0 = disabled). **Canonical block: `training:` per DEC-3b.** Legacy `model:` placement is deprecated; the resolver emits a `DeprecationWarning` and run [`scripts/migrate_balancing_fields_to_training.py`](../scripts/migrate_balancing_fields_to_training.py) to migrate. |
| `bias_warmup_start` | float | 0.0 | Initial bias rate at step 0 (linear ramp to `bias_update_rate`). Canonical block: `training:`. |
| `bias_warmup_steps` | int | 0 | Steps to ramp `bias_warmup_start` → `bias_update_rate`; 0 = no warmup. Canonical block: `training:`. |
| `bias_update_zero_sum` | bool | true | DEC-2 mode selector for the DeepSeek-style bias update. `true` (default) uses the nmoe-style zero-sum formulation `bias -= (s - s.mean()) * rate` so the cumulative bias mean is pinned at zero. `false` uses the Megatron-LM plain-sign update `bias += sign(avg_load - load) * rate` -- the mean is allowed to drift up to +/-`bias_update_rate` per step under asymmetric loads (still bounded by the +/-16 clamp). See [`docs/routing.md` § DEC-2 update modes](routing.md#dec-2-update-modes). Canonical block: `training:`. |
| `router_aux_loss_coef` | float | 0.001 | Switch-Transformer batch-level auxiliary loss coefficient (consumed by every model family's `forward`). **Canonical block: `training:` per DEC-3b** (was previously `model:` — same migration as `bias_update_rate`). |
| `seq_aux_loss_coef` | float | 0.0 | DeepSeek V3 sequence-level aux loss coefficient. **Canonical block: `training:` per DEC-3b**. |
| `load_balancing_method` | string | `aux_loss` | `aux_loss \| seq_aux_loss \| deepseek_bias \| quantile \| none`. Canonical block: `training:`. **Authoritative dispatch (Round 5+):** `load_config()` runs `normalize_balancing_config()` which AUTO-ZEROS legacy coefficients outside the resolved method's active set (with `DeprecationWarning`). `build_model()` stamps the resolved method onto `model._load_balancing_method` and `model_config.load_balancing_method`. Each model family's `forward` reads `_load_balancing_method` and gates aux/seq-aux additions; `update_expert_biases()` no-ops for any method outside `{deepseek_bias}`. Active sets per method: `aux_loss → router_aux_loss_coef`; `seq_aux_loss → seq_aux_loss_coef`; `deepseek_bias → bias_update_rate + bias_warmup_* + bias_rate_*`; `quantile → ∅` (implementation pending Milestone D); `none → ∅` (everything zeroed). |
| `bias_rate_q` / `bias_rate_k` / `bias_rate_v` / `bias_rate_o` / `bias_rate_mlp` / `bias_rate_branch` | float | `bias_update_rate` | Per-projection DeepSeek bias-update rate overrides. Each defaults to the global `bias_update_rate` if not set. **Canonical block: `training:` per DEC-3b**. |
| `router_exploration_warmup_start` | float | 0.0 | Initial router-exploration rate at step 0; see `docs/training.md` §Router-Exploration Warmup |
| `router_exploration_warmup_steps` | int | 0 | Steps to ramp to the model's `router_exploration_rate`; 0 = feature disabled |
| `momentum_warmup_steps` | int | 300 | Muon optimizer momentum warmup horizon |
| `torch_compile` | bool | false | Enable torch.compile |
| `disable_liger` | bool | false | Disable global Liger monkeypatch kernels. MoE-Everything fused linear CE is controlled by `model.use_fused_linear_ce` and defaults on. |

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
