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

### Scaling experiments

| | XS (Qwen3-0.6B backbone) | S | M | L (Qwen3-30B-A3B) |
|---|---|---|---|---|
| `hidden_size` | 1024 | 1024 | 2048 | 2048 |
| `num_hidden_layers` | 8 | 12 | 16 | 48 |
| `num_attention_heads` | 16 | 16 | 16 | 32 |
| `num_key_value_heads` | 8 | 4 | 4 | 4 |
| `head_dim` | 128 | 64 | 128 | 128 |
| `moe_intermediate_size` | 768 | 768 | 768 | 768 |
| Standard `num_experts` | 16/layer (128 total) | 32/layer (384 total) | 128/layer (2048 total) | 128/layer (6144 total) |
| Global `num_experts` | 128 shared | 384 shared | 2048 shared | 6144 shared |
| `num_experts_per_tok` | 4 | 4 | 8 | 8 |

### Retrofit configs (dense → MoE upcycling)

| | Retrofit 1.7B (Qwen3-1.7B backbone) |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `moe_intermediate_size` | 768 |
| Standard `num_experts` | 64/layer (1792 total) |
| Global `num_experts` | 1792 shared |
| `num_experts_per_tok` | 8 |

Active FFN per token matches the original dense model's `intermediate_size` (e.g. `top_k × 768 = 6144` for 1.7B).

Config files: `configs/scaling/`

## Run

```bash
./scripts/train.sh configs/scaling/xs_standard.yaml
./scripts/train.sh configs/scaling/xs_global.yaml
./scripts/train.sh configs/scaling/l_standard.yaml

# Resume from checkpoint
./scripts/train.sh configs/scaling/m_global.yaml --resume outputs/m_global_moe/checkpoint-5000
```

## Logging

All metrics are logged to stdout and wandb (if `WANDB_API_KEY` is set).

### Training metrics (every `log_every` steps)

Stdout format:
```
step    182  loss=10.3080  ce=10.3059  aux=2.0113  lr=1.46e-05  tok/s=446.9k  |g|=1.841
```

#### `train/ce_loss` — Cross-entropy loss

The language modeling objective. This is the primary metric to compare runs.
- At init: ~11.93 (`ln(151936)`, uniform over vocab).
- Below ~3.0 is decent for pretraining.
- Plot this against `tokens_seen_B` (not steps) to fairly compare configs with different batch sizes.

#### `train/aux_loss` — Load-balancing loss

Switch Transformer auxiliary loss = `num_experts × Σ(f_i × p_i)` where `f_i` = fraction of tokens routed to expert `i`, `p_i` = mean router probability for expert `i`.
- Perfect balance = 1.0.
- Values near `num_experts_per_tok` at init mean the router hasn't differentiated yet.
- This doesn't reflect model quality — it only measures how evenly tokens are distributed across experts. A run can have great aux_loss but bad ce_loss (or vice versa).

#### `train/grad_norm` — Gradient L2 norm

Computed before clipping (clip threshold = `max_grad_norm`, default 1.0).
- Steady below ~10 = healthy.
- Occasional spikes to 50-100 are normal, especially early in training.
- Sustained spikes >100 = instability. Consider lowering LR or checking data.

#### `train/lr` — Learning rate

Linear warmup from 0 → peak over `warmup_steps`, then cosine decay to `min_lr_ratio × peak`.

#### `train/tokens_per_sec` — Throughput

Tokens processed per second across all GPUs. Use this to compare hardware efficiency across configs.

#### `train/tokens_seen_B` — Cumulative tokens (billions)

Total tokens processed across all GPUs since training started. This is the x-axis you should use when comparing ce_loss across different configs, since different batch sizes / gradient accumulation make step counts non-comparable.

---

### Routing metrics (every `routing_log_every` steps)

Computed per layer from `output.router_logits` (the softmax probabilities the router produces during the forward pass — zero extra compute cost).

#### `routing/layer_NN_load_imbalance` — Max expert load vs ideal

`max(expert_token_count) / (total_routing_slots / num_experts)`.
- 1.0 = every expert got the same number of tokens (perfect).
- 5.0 = the busiest expert got 5x the ideal share.
- Watch for this growing over training — it means the router is collapsing onto a few experts.

#### `routing/layer_NN_load_cv` — Coefficient of variation

`std(token_counts) / mean(token_counts)`.
- 0.0 = perfectly uniform distribution across experts.
- Smoother signal than `load_imbalance` (which is sensitive to a single outlier expert).

#### `routing/layer_NN_utilization` — Expert utilization fraction

Fraction of experts that received at least 1 token in this batch.
- 1.0 = all experts active.
- 0.5 = half your experts are dead weight (no tokens, no gradients, wasted params).

#### `routing/layer_NN_num_active_experts` — Active expert count

Same as utilization but as a raw count. With 16 experts, if this drops to 8, half are dead.

#### `routing/layer_NN_entropy` — Routing entropy

Normalized Shannon entropy of mean routing probabilities, scaled to [0, 1].
- 1.0 = perfectly uniform (router hasn't learned any preferences).
- 0.0 = fully collapsed (all tokens go to one expert).
- Some drop from 1.0 is expected and healthy (specialization). Below 0.5 is a red flag.

#### `routing/layer_NN_router_margin_mean` — Router confidence (mean)

Average gap between the k-th selected expert's probability and the (k+1)-th (first rejected) expert's probability, across all tokens in the batch.
- Near 0 = the router is barely distinguishing between selected and rejected experts — routing is essentially random noise.
- Large = the router is confident in its top-k choices and has learned clear expert preferences.
- This is one of the best signals for whether routing is actually meaningful vs just load-balanced noise.

#### `routing/layer_NN_router_margin_min` — Router confidence (worst case)

Minimum margin across all tokens. If this is 0, at least some tokens have essentially tied expert scores — the router is guessing for those tokens.

#### `routing/layer_NN_top_expert` — Most-loaded expert index

Index of the expert that received the most tokens. If the same index appears across many layers and steps, you have a "sink" expert absorbing disproportionate traffic.

#### `hist/layer_NN_expert_load_frac` — Expert load histogram

Per-expert fraction of total routing slots, logged as a `wandb.Histogram`.
- Ideal: every bar at `top_k / num_experts` (e.g. `4/16 = 0.25` for XS).
- Spiky = some experts overloaded, others starved.

---

### Global MoE metrics (every `routing_log_every` steps)

These are only logged for `type: global_moe` and measure how the shared expert pool is used across layers.

#### `routing/cross_layer_sim_mean` — Cross-layer similarity

Mean cosine similarity between each pair of layers' expert-load vectors.
- High (>0.8) = all layers are picking the same experts — no depth specialization, the global pool isn't being used effectively.
- Low (<0.3) = different layers prefer different experts — good, the model is developing depth-wise specialization.

#### `routing/cross_layer_sim_min` / `cross_layer_sim_max`

Min and max pairwise cosine similarity. Large gap between min and max means some layer pairs share experts while others don't — could indicate natural grouping (e.g. early vs late layers).

#### `routing/layer_NN_global_experts_used` — Unique experts per layer

How many distinct experts from the global pool this layer routed to in this batch.
- With top-4 and batch_size=16, seq_len=2048: each layer makes `16 × 2048 × 4 = 131072` routing decisions, spread across however many unique experts.
- If this number is much smaller than the pool size, the layer is only using a small slice of the pool.

#### `hist/global_expert_layer_count` — Expert reuse histogram

For each expert in the global pool, how many layers routed at least 1 token to it. Logged as a `wandb.Histogram`.
- Bin at 0 = dead experts (no layer uses them — wasted params).
- Bin at 1 = specialized experts (used by exactly one layer).
- Bin at L = "universal" experts (every layer uses them).
- The sum of all values in this histogram equals the sum of `layer_NN_global_experts_used` across all layers (same matrix, summed along different axes).

---

### Checkpoints

Saved every `save_every` steps to `output_dir/checkpoint-{step}/`. Includes model, optimizer, scheduler, and dataset state for exact resumption.
