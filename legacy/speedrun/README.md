# Legacy Speedrun Models

This directory contains archived speedrun model code. These models were used for rapid experimentation
but have been superseded by the unified training stack with Qwen3-based custom model architecture.

## Archived Files

- `speedrun_gpt.py` - Dense GPT speedrun model with FlexAttention, FP8 lm_head, sigmoid softcapping
- `speedrun_moe_gpt.py` - MoE speedrun model with per-head attention routing (H routers, each top-1)
- `speedrun_mixture_of_everything.py` - Speedrun variant of MoE-Everything with FunctionalRMSNorm
- `configs/` - Speedrun model configurations
- `tests/` - Speedrun model tests

## Extracted Components

Before archival, the following reusable components were extracted into active code:

- **Per-head attention routing** (H separate top-1 routers) → `src/models/moe_everything/attention_bank.py`
- **RoutingStats dataclass** → `src/models/routing/stats.py`
- **Global bias system** (per-projection, zero-sum update, clamp) → `src/models/routing/bias.py`
- **BranchRouter DeepSeek variant** → merged into `src/models/routing/routers.py`
- **FunctionalRMSNorm** → `src/models/base/normalization.py`
- **FlexAttention with doc masking** → `src/models/base/attention.py`
- **FP8 lm_head** → `src/models/base/output_head.py`
- **Routing helpers** (_make_router, _route_top1, etc.) → `src/models/routing/helpers.py`

## Decoupled Utilities (Still Active)

These utilities were already decoupled from speedrun code and remain in active use:

- `src/utils/routing_stats.py` - Count-based routing statistics
- `src/utils/routing_plots.py` - Routing visualization/graphs
- `src/utils/muon.py` - Muon optimizer
- `src/models/triton_grouped_gemm.py` - Triton grouped GEMM kernel
- `src/utils/triton_newton_schulz.py` - Newton-Schulz iteration
- `src/utils/dist_optimizers.py` - Distributed optimizers (DistAdam, DistMuon)
- `src/utils/routing_loss.py` - Routing loss functions

## Why Archived

The active codebase now uses a Bagel-style approach: custom Qwen3 components copied into
`src/models/base/` with full customization control while preserving HuggingFace checkpoint
compatibility. The speedrun models used a different architecture (GPT-2 style) that is
incompatible with this approach.
