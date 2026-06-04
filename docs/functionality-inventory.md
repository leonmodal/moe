# Functionality Inventory

Post-reorganization inventory of active features and archived components.

## Active Model Architectures

| Model | Config Type | File | Notes |
|-------|------------|------|-------|
| Dense | `dense` | `Qwen3ForCausalLM` | Qwen3 backbone |
| Standard MoE | `standard_moe` | `src/models/standard_moe.py` | Per-layer MoE, softmax or DeepSeek routing |
| Global MoE | `global_moe` | `src/models/global_moe.py` | Shared expert pool |
| MoE-Everything | `moe_everything` | `src/models/moe_everything/` (config.py, attention_bank.py, mlp_bank.py, model.py) | Branch routing + expert banks |

## Active Training Infrastructure

| Component | File | Notes |
|-----------|------|-------|
| Unified trainer | `scripts/train.py` | Single CLI entrypoint for all models |
| Training library | `src/training/` | Config, model factory, data, checkpoint, eval, distributed, logging, routing, metrics |
| Modal launcher | `modal_train.py` | Multi-node via torchrun |
| Config validator | `scripts/validate_configs.py` | Schema + enum validation |
| Safetensors converter | `scripts/convert_checkpoint.py` | Model checkpoint conversion |

## Active Data

| Component | File | Notes |
|-----------|------|-------|
| Parquet dataset | `src/data/parquet_dataset.py` | Sole data path, stateful resume |

## Active Utilities

| Component | File | Notes |
|-----------|------|-------|
| Routing stats | `src/utils/routing_stats.py` | Count-based statistics |
| Routing plots | `src/utils/routing_plots.py` | Heatmaps and visualization |
| Muon optimizer | `src/utils/muon.py` | Muon + Adam hybrid |
| Distributed optimizers | `src/utils/dist_optimizers.py` | DistAdam, DistMuon |
| Triton grouped GEMM | `src/models/triton_grouped_gemm.py` | Expert dispatch kernel |
| Newton-Schulz | `src/utils/triton_newton_schulz.py` | For Muon optimizer |
| Routing loss | `src/utils/routing_loss.py` | Switch-style aux loss |

## Active Routing Package

| Component | File | Notes |
|-----------|------|-------|
| BranchRouter | `src/models/routing/routers.py` | Binary attn/MLP branch selection with speedrun features |
| DeepSeekRouter | `src/models/router.py` | Sigmoid + expert bias (re-exported via routing/) |
| ExplorationTopKRouter | `src/models/router.py` | Softmax top-k with exploration |
| Load balancing | `src/models/routing/load_balancing.py` | Batch + sequence-level aux losses |
| Bias management | `src/models/routing/bias.py` | Per-projection global bias update |
| Routing stats | `src/models/routing/stats.py` | Per-forward-pass accumulation |
| Routing helpers | `src/models/routing/helpers.py` | Top-1 router factory (`make_top1_router`) and grouped expert projection helpers; per-head dispatch logic still in `attention_bank.py` |

## Archived (no longer in active code)

All entries below have been **removed from their original paths** and moved into `legacy/speedrun/`. The "Pre-archive path" column records where each component used to live in the active tree; active code must not reference those paths. The "Archived location" column is where the file lives today.

| Component | Pre-archive path (no longer exists) | Archived location |
|-----------|--------------------------------------|-------------------|
| SpeedrunGPT | `src/models/speedrun_gpt.py` | `legacy/speedrun/speedrun_gpt.py` |
| SpeedrunMoEGPT | `src/models/speedrun_moe_gpt.py` | `legacy/speedrun/speedrun_moe_gpt.py` |
| SpeedrunMoEverything | `src/models/speedrun_mixture_of_everything.py` | `legacy/speedrun/speedrun_mixture_of_everything.py` |
| Accelerator trainer | `train.py` | `legacy/speedrun/train_accelerator.py` |
| Token-bin dataset | `src/data/token_bin_dataset.py` | archived alongside speedrun (used only by speedrun configs) |
| Accelerate configs | `accelerate_configs/` | removed; no Accelerate dependency remains |
| Speedrun configs | `configs/speedrun/` | `legacy/speedrun/configs/` |
| torch-native trainer extraction source | `train_torch.py` | superseded by `scripts/train.py` + `src/training/` |

Reusable components (`RoutingStats`, `BranchRouter` DeepSeek variant, `RoutedAttention*`, global bias helpers, routing helpers, `FunctionalRMSNorm`, FlexAttention, FP8 lm_head, sigmoid logit softcapping) were extracted out before archival and now live in active code — see `src/models/routing/`, `src/models/base/`, and `src/utils/`. See `legacy/speedrun/README.md` for the extraction ledger.
