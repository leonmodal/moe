# MoE Pretraining: Standard vs Global Mixture of Experts

## Setup

```bash
uv sync
```

Add HF and wandb keys to `.env`:

```
HF_TOKEN=hf_...
WANDB_API_KEY=wandb_...
```

## Download Data

```bash
# All 8192 shards (~3.4TB)
uv run python scripts/download_data.py

# Or limit shards for testing
uv run python scripts/download_data.py --max_shards 64
```

## Model Configs

| | XS (~50M) | S (~300M) | M (~10B) |
|---|---|---|---|
| `hidden_size` | 512 | 1024 | 2048 |
| `num_hidden_layers` | 8 | 12 | 16 |
| `num_attention_heads` | 8 | 16 | 16 |
| `num_key_value_heads` | 2 | 4 | 4 |
| `head_dim` | 64 | 64 | 128 |
| `moe_intermediate_size` | 256 | 512 | 768 |
| Standard `num_experts` | 8/layer (64 total) | 32/layer (384 total) | 128/layer (2048 total) |
| Global `num_experts` | 64 shared | 384 shared | 2048 shared |
| `num_experts_per_tok` | 2 | 4 | 8 |

All scales: `lr=1e-3`, `batch_size=16/gpu`, `warmup=100k`, `max_steps=1M`, `cosine decay`, `bf16`.

Config files: `configs/scaling/{xs,s,m}_{standard,global}.yaml`

## Run

```bash
# Smoke test (synthetic data, no download needed)
./scripts/train.sh xs standard --smoke_test
./scripts/train.sh xs global --smoke_test

# XS
./scripts/train.sh xs standard
./scripts/train.sh xs global

# S
./scripts/train.sh s standard
./scripts/train.sh s global

# M
./scripts/train.sh m standard
./scripts/train.sh m global

# Resume from checkpoint
./scripts/train.sh m global --resume outputs/m_global_moe/checkpoint-5000
```

## Logging

### Per-step metrics

```
step    182  loss=10.3080  ce=10.3059  aux=2.0113  lr=1.46e-05  tok/s=446.9k  |g|=1.841
```

**`loss`** — Total loss = `ce + 0.1 × aux`. What the optimizer minimizes.

**`ce`** — Cross-entropy loss. The language modeling objective. At init ~11.93 (`ln(151936)`, uniform over vocab). Below ~3.0 is decent.

**`aux`** — Switch Transformer load-balancing loss = `num_experts × Σ(f_i × p_i)` where `f_i` = fraction of tokens routed to expert `i`, `p_i` = mean router probability for expert `i`. Perfect balance = 1.0. Values near `num_experts_per_tok` mean the router hasn't differentiated yet.

**`lr`** — Learning rate. Linear warmup 0 → 1e-3 over 100k steps, then cosine decay to 1e-4 (10% of peak).

**`tok/s`** — Tokens per second across all 8 GPUs.

**`|g|`** — Gradient L2 norm before clipping (threshold = 1.0). Spikes to 100+ = instability. Steady <10 = healthy.

### Routing stats (per layer, logged every `routing_log_every` steps to wandb)

**`load_imbalance`** — `max(expert_token_count) / ideal_count`. 1.0 = every expert got the same number of tokens. 5.0 = busiest expert got 5× ideal. Watch for this growing — means router is collapsing.

**`load_cv`** — Coefficient of variation = `std(token_counts) / mean(token_counts)`. 0.0 = perfectly uniform. Smoother signal than `load_imbalance`.

**`utilization`** — Fraction of experts that received ≥1 token. 1.0 = all active. If 0.5, half your experts are dead weight with no gradients.

**`entropy`** — Normalized entropy of mean routing probs, scaled to [0,1]. 1.0 = uniform. 0.0 = collapsed. Should start near 1.0. Some drop is fine (specialization), below 0.5 is a red flag.

**`top_expert`** — Index of the most-loaded expert. Same index across many layers/steps = "sink" expert absorbing too many tokens.

### Global MoE only

**`cross_layer_sim_mean`** — Cosine similarity between each pair of layers' expert-load vectors. High (>0.8) = all layers picking the same experts, no depth specialization. Low (<0.3) = different layers prefer different experts, good.

**`expert_depth_coverage_mean`** — For each expert, count how many layers use it above uniform threshold, averaged across all experts. If this equals `num_hidden_layers`, every expert is used by every layer — no specialization. Lower = experts are specializing by depth.
