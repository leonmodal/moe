# MoE Pretraining: Standard vs Global Mixture of Experts

Pretraining experiment comparing two MoE routing strategies at multiple scales.
Architecture base: **Qwen3-MoE** (GQA + QK-norm + SwiGLU + RoPE).

---

## The Experiment

### Standard MoE
Each layer owns its own independent pool of experts.
A token at layer `l` routes top-k from **layer `l`'s 128 experts only**.

```
Layer 0 : Router_0  →  top-8 from {E_0_0 … E_0_127}
Layer 1 : Router_1  →  top-8 from {E_1_0 … E_1_127}
...
Layer 15: Router_15 →  top-8 from {E_15_0 … E_15_127}

Total experts: 16 × 128 = 2048 (all distinct)
```

### Global MoE
All layers share one expert pool.
A token at layer `l` routes top-k from the **global 2048 experts**.

```
Shared pool: {E_0 … E_2047}

Layer 0 : Router_0  →  top-8 from {E_0 … E_2047}
Layer 1 : Router_1  →  top-8 from {E_0 … E_2047}
...
Layer 15: Router_15 →  top-8 from {E_0 … E_2047}

Total experts: 2048 (shared)
```

Same total expert parameter count. Same number of routers.
Router width: Standard = `[128, H]` per layer; Global = `[2048, H]` per layer.

---

## Architecture (Qwen3-MoE base)

| Component | Detail |
|---|---|
| Attention | Grouped-Query Attention (GQA) |
| QK-Norm | Per-head RMSNorm on Q and K before RoPE (training stability) |
| Activation | SwiGLU (gate_proj × silu + up_proj → down_proj) |
| Norm | RMSNorm with fp32 upcast, eps=1e-6 |
| Position | RoPE, θ=1,000,000 |
| Router | Softmax → top-k → renorm (norm_topk_prob=True) |
| Aux loss | Switch-Transformer load-balancing, coef=0.001 |

---

## Model Scales

### XS — ~50M total params

| Field | Value |
|---|---|
| `hidden_size` | 512 |
| `num_hidden_layers` | 8 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 2 (GQA: 4 queries per KV) |
| `head_dim` | 64 |
| `moe_intermediate_size` | 256 (per expert) |
| **Standard** `num_experts` | 8 per layer → 64 total |
| **Global** `num_experts` | 64 shared |
| `num_experts_per_tok` | 2 (top-k active) |
| `max_position_embeddings` | 4096 |

Config: `configs/scaling/xs_standard.yaml` / `xs_global.yaml`

---

### S — ~300M total params

| Field | Value |
|---|---|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 (GQA: 4 queries per KV) |
| `head_dim` | 64 |
| `moe_intermediate_size` | 512 (per expert) |
| **Standard** `num_experts` | 32 per layer → 384 total |
| **Global** `num_experts` | 384 shared |
| `num_experts_per_tok` | 4 (top-k active) |
| `max_position_embeddings` | 8192 |

Config: `configs/scaling/s_standard.yaml` / `s_global.yaml`

---

### M — ~9.7B total params (original target)

| Field | Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 16 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 (GQA: 4 queries per KV) |
| `head_dim` | 128 |
| `moe_intermediate_size` | 768 (per expert) |
| **Standard** `num_experts` | 128 per layer → 2048 total |
| **Global** `num_experts` | 2048 shared |
| `num_experts_per_tok` | 8 (top-k active) |
| `max_position_embeddings` | 32768 |

Config: `configs/scaling/m_standard.yaml` / `m_global.yaml`

---

## Training Setup

| Setting | Value |
|---|---|
| Hardware | 8× NVIDIA B200 (183 GB each), single node |
| Distribution | DDP (`accelerate_configs/ddp_8gpu.yaml`) |
| Precision | BF16 |
| Gradient accumulation | 1 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR schedule | Cosine decay with warmup, min_lr = 10% peak |
| Tokenizer | Qwen3 (vocab=151936) |
| Data | Stateful Parquet dataset (resumable) |

---

## Logging

Every step:

| Metric | Description |
|---|---|
| `train/loss` | Total loss (CE + aux) |
| `train/ce_loss` | Cross-entropy only |
| `train/aux_loss` | Load-balancing auxiliary loss |
| `train/grad_norm` | Gradient L2 norm |
| `train/lr` | Learning rate |
| `train/mfu` | Model FLOP utilization vs B200 BF16 peak (2.25 PFLOPS) |
| `train/tokens_per_sec` | Throughput |
| `train/tokens_seen_B` | Cumulative tokens |
| `sys/gpu_mem_alloc_GB` | GPU memory allocated |
| `sys/gpu_mem_res_GB` | GPU memory reserved |

Every `routing_log_every` steps (free — from `output.router_logits`):

| Metric | Description |
|---|---|
| `routing/layer_NN_entropy` | Normalised routing entropy [0=collapsed, 1=uniform] |
| `routing/layer_NN_utilization` | Fraction of experts receiving ≥1 token |
| `routing/layer_NN_top_expert` | Most-used expert index |
| `routing/cross_layer_sim_mean` | **Global only** — cosine similarity between layers' routing distributions; high = no depth specialisation |
| `routing/expert_depth_coverage_mean` | **Global only** — avg layers each expert is active in |
| `hist/expert_depth_coverage` | **Global only** — wandb histogram of per-expert depth coverage |

---

## Usage

```bash
# Install
uv sync

# Smoke test (synthetic data, no parquet needed)
accelerate launch --config_file accelerate_configs/ddp_8gpu.yaml \
    train.py --config configs/scaling/xs_standard.yaml --smoke_test

# Standard MoE (M scale)
accelerate launch --config_file accelerate_configs/ddp_8gpu.yaml \
    train.py --config configs/scaling/m_standard.yaml

# Global MoE (M scale)
accelerate launch --config_file accelerate_configs/ddp_8gpu.yaml \
    train.py --config configs/scaling/m_global.yaml

# Resume
accelerate launch --config_file accelerate_configs/ddp_8gpu.yaml \
    train.py --config configs/scaling/m_global.yaml \
    --resume outputs/m_global_moe/checkpoint-5000
```

---

## File Structure

```
configs/
  scaling/
    xs_standard.yaml / xs_global.yaml
    s_standard.yaml  / s_global.yaml
    m_standard.yaml  / m_global.yaml
accelerate_configs/
  ddp_8gpu.yaml
  fsdp_8gpu.yaml           # use if model outgrows DDP memory budget
src/
  models/
    standard_moe.py        # = Qwen3MoeForCausalLM (alias)
    global_moe.py          # GlobalMoEForCausalLM — ~90 lines on top of Qwen3
    modeling_qwen3_moe.py  # copied from transformers (reference)
    configuration_qwen3_moe.py
  data/
    parquet_dataset.py     # stateful, resumable parquet streaming
  utils/
    training.py            # MFU, optimizer, scheduler
    routing_stats.py       # per-layer entropy, utilization, cross-layer similarity
train.py
```
