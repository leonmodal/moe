# MoE Pre-training Documentation

## Overview

This repository implements Mixture-of-Experts (MoE) language model pre-training with multiple architectural variants, DeepSeek-style routing, and distributed training support.

## Documentation Index

| Document | Contents |
|----------|----------|
| [Architecture](architecture.md) | Model families, layer design, attention modes |
| [Routing & Experts](routing.md) | Router implementations, expert selection, load balancing, bias updates |
| [Training](training.md) | Unified training loop, optimizers, LR schedulers, loss functions |
| [Data](data.md) | Parquet dataset, tokenization, stateful resumption |
| [Distributed Training](distributed.md) | DDP, FSDP, multi-node setup on Modal |
| [Configuration](configuration.md) | Config file schema, parameter reference |
| [External Research](research/external_moe_techniques.md) | Findings from Megatron-LM, modal-nmoe, nmoe |

## Model Families

| Model | Config Type | Description |
|-------|------------|-------------|
| **Dense** | `dense` | Standard dense transformer (Qwen3 backbone) |
| **Standard MoE** | `standard_moe` | Per-layer routed MLP experts |
| **Global MoE** | `global_moe` | Single shared expert pool across all layers |
| **MoE-Everything** | `moe_everything` | Branch routing + per-head attention/MLP expert banks |

DeepSeek expert routing is available via `router_type: deepseek` for any
MoE model type. In `moe_everything`, the branch router is configured
separately with `model.branch_router.balancing`; `deepseek_bias` there
means two independent sigmoid scores for ATTN/MLP plus branch bias.

MoE-Everything attention has two explicit axes:
- `attn_expert_mode`: `per_head_no_recompute`, `per_head_recompute_k`, or `per_head_recompute_kv`
- `attn_routing_bundle`: `q_k_v_o`, `qk_v_o`, `qk_vo`, `qkv_o`, or `qkvo`
- `attn_router_context`: `none` or `ema_qk_v`; EMA context applies only to
  QK/V routers, while O remains a local post-attention route.

## Quick Start

```bash
# Install
uv sync

# Single-node training (8 GPUs)
torchrun --nproc_per_node=8 scripts/train.py --config configs/16_layers/standard_moe_deepseek_bias.yaml

# Multi-node on Modal
modal run modal_train.py --config configs/16_layers/moe_everything_per_head_recompute_k_qk_v_o_deepseek_bias.yaml
```
