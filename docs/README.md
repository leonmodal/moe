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

DeepSeek routing available via `router_type: deepseek` for any MoE model type.

MoE-Everything supports two attention expert modes:
- **Fully Independent** (`per_head_fully_independent`): Q, K, V, O each routed independently per head
- **Precompute KV** (`per_head_precompute_kv`): bundled QKVO routing per head

## Quick Start

```bash
# Install
uv sync

# Single-node training (8 GPUs)
torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml

# Multi-node on Modal
modal run modal_train.py
```
