# MoE Pre-training Documentation

## Overview

This repository implements Mixture-of-Experts (MoE) language model pre-training with multiple architectural variants, DeepSeek-style routing, and distributed training support.

## Documentation Index

| Document | Contents |
|----------|----------|
| [Architecture](architecture.md) | Model families, layer design, attention modes, and advanced features |
| [Routing & Experts](routing.md) | Router implementations, expert selection, load balancing, bias updates |
| [Training](training.md) | Training loop, optimizers, LR schedulers, loss functions, `train.py` vs `train_torch.py` comparison |
| [Data](data.md) | Dataset formats, data loading, tokenization, stateful resumption |
| [Distributed Training](distributed.md) | DDP, FSDP, multi-node setup on Modal and GCP |
| [Configuration](configuration.md) | Config file schema, parameter reference, example configs |
| [Functionality Inventory](functionality-inventory.md) | Complete feature inventory table with keep/drop decisions for cleanup |

## Model Families (Final)

| Model | Type | File | Description |
|-------|------|------|-------------|
| **Standard LLM** | Dense | `Qwen3ForCausalLM` | Standard dense transformer (Qwen3 backbone) |
| **Standard MoE** | Sparse | `src/models/standard_moe.py` | Per-layer routed MLP experts with fixed load-balancing loss |
| **Global MoE** | Sparse | `src/models/global_moe.py` | Single shared expert pool across all layers |
| **MoE-Everything** | Sparse | `src/models/mixture_of_everything.py` | Branch routing (attn vs MLP) + per-head attention expert banks + MLP expert banks, all shared across depths |

MoE-Everything supports two attention expert modes:
- **Fully Independent**: 4 routers per head (Q, K, V, O route independently)
- **Precompute KV**: 1 router per head (same expert for all QKVO, KV precomputed per expert)

## Quick Start

```bash
# Install
uv sync

# Single-node training (8 GPUs)
torchrun --nproc_per_node=8 train_torch.py --config configs/scaling/xs_standard.yaml

# Multi-node on Modal
modal run modal_train.py
```
