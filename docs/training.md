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

# FSDP with replicated parameters, used by the Modal launcher by default
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml --dist-strategy fsdp --fsdp-sharding-strategy no_shard
```

The trainer supports all model types: `dense`, `standard_moe`, `global_moe`, `moe_everything`. MoE-Everything gradient checkpointing is covered under both DDP and FSDP in the distributed smoke tests.

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
  - `branch_aux_loss`: Branch router auxiliary/sequence loss when `model.branch_router.balancing` selects it
  - `attention_aux_loss`: Attention routing auxiliary loss
  - `aux_loss_normalized`: Normalized load-balancing metric

## Eval And Routing Artifacts

Every enabled validation run writes local eval metrics in addition to console
and WandB logging:

```text
<output_dir>/
  eval_logs/
    eval_metrics.jsonl
    step_00000250/
      metrics.json
      branch_patterns/
        token_routes.csv
        token_routes.md
        top_patterns.csv
        depth_summary.csv
        summary.md
        top_patterns.png
        top_pattern_matrix.png
        branch_depth_ratios.png
      load_balancing/
        summary.csv
        expert_load.csv
        bias.csv
        summary.md
        summary.png
        attn/
          <route>/
            summary.md
            summary.csv
            expert_load.csv
            bias.csv
            heatmap.png
            global_histogram.png
            per_layer_histograms.png
            bias_by_expert.png
        mlp/
          summary.md
          summary.csv
          expert_load.csv
          bias.csv
          heatmap.png
          global_histogram.png
          per_layer_histograms.png
          bias_by_expert.png
        branch/
          summary.md
          summary.csv
          expert_load.csv
          heatmap.png
          global_histogram.png
          per_layer_histograms.png
```

`metrics.json` contains the step plus `eval/*` values such as
`eval/ce_loss`, `eval/perplexity`, `eval/aux_loss`,
`eval/branch_aux_loss`, `eval/branch_entropy_loss`, and
`eval/attention_aux_loss`. For MoE-Everything eval forwards, the step folder
also contains `branch_patterns/` and `load_balancing/` for the final
validation batch on rank 0.

MoE-Everything training routing snapshots also write token-preserving
branch-route artifacts:

```text
<output_dir>/
  routing_logs/
    step_00000250/
      branch_patterns/
        token_routes.csv
        token_routes.md
        top_patterns.csv
        depth_summary.csv
        summary.md
        top_patterns.png
        top_pattern_matrix.png
        branch_depth_ratios.png
      load_balancing/
        summary.csv
        expert_load.csv
        bias.csv
        summary.md
        summary.png
        attn/<route>/
        mlp/
        branch/
```

The branch pattern legend is `A = attention branch`, `M = MLP branch`, and
pattern characters are ordered by increasing depth. `token_routes.csv` is the
canonical token-level table: it includes full-depth pattern counts/shares,
per-token score summaries, and per-layer branch/attention/MLP expert columns.
`token_routes.md` is the same sampled token table formatted for direct review,
with compact per-depth route lines. `branch_patterns/summary.md` is the
human-readable overview.
`top_patterns.png` and `top_pattern_matrix.png` show the most common full-depth
branch paths; `branch_depth_ratios.png` shows the attention/MLP ratio by depth.

`load_balancing/summary.csv` is the high-level table for MLP, attention-router,
and branch pools: active experts, peak-load-over-ideal, coefficient of
variation, and normalized entropy. `load_balancing/summary.md` is the
human-readable overview; `expert_load.csv` is the per-expert raw table.
Detailed pool outputs are structured as `load_balancing/attn/<route>/`,
`load_balancing/mlp/`, and `load_balancing/branch/`. Fixed-alternating branch
routing does not create a `branch/` load-balancing folder because there is no
learned branch router/bias to inspect; the deterministic A/M pattern still
appears under `branch_patterns/`.

Inside each pool folder, `global_histogram.png` shows aggregate expert usage,
`per_layer_histograms.png` shows one expert histogram per depth, and
`heatmap.png` plots per-depth expert fractions. If the pool owns DeepSeek- or
quantile-style `expert_bias` buffers, `bias.csv` plus `bias_by_expert.png`
show the current non-gradient router bias state. The bias plot is indexed by
expert id and uses red bars for negative bias and green bars for positive
bias. Under `global_router_update: true`, the bias is one shared vector per
pool; per-layer routers store a broadcast copy for runtime use, but artifacts
collapse it back to the conceptual global vector. The root
`load_balancing/bias.csv` is the union across all routed pools.

A small current-format static example is checked in under
`docs/example_routing_outputs/`.

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
- Partial patch mode for moe_everything (rope + rms_norm only). The trainer
  uses MoE-Everything's model-native fused linear CE path by default for the
  LM loss, so it does not materialize full vocab logits during train/eval loss
  calls. SwigLU stays separate because MoE-Everything uses a custom grouped
  expert bank.
- The global Liger patch is disabled via `training.disable_liger: true` or
  `MOE_DISABLE_LIGER=1`; MoE-Everything fused CE is its own model loss path.

## Modal Multi-Node

`modal_train.py` launches multi-node training on Modal cloud:

```bash
modal run modal_train.py --config configs/scaling/m_standard.yaml
```
