# Functionality Inventory

Complete feature inventory with keep/drop decisions for the codebase cleanup.

## Model Architectures

| # | Model | File | Keep? | Notes |
|---|-------|------|-------|-------|
| A1 | Standard MoE | `src/models/standard_moe.py` | **Keep** | Per-layer routed MLP experts |
| A2 | DeepSeek Standard MoE | `src/models/standard_moe.py` | **Keep** | Sigmoid routing variant |
| A3 | Global MoE | `src/models/global_moe.py` | **Keep** | Shared global expert pool |
| A4 | MoE-Everything | `src/models/mixture_of_everything.py` | **Keep** | Branch routing + expert banks (both modes) |
| A5 | Standard LLM (Dense) | HF `Qwen3ForCausalLM` | **Keep** | Dense baseline |
| A6 | SpeedrunGPT | `src/models/speedrun_gpt.py` | **Drop** | Speedrun-only dense model |
| A7 | SpeedrunMoEGPT | `src/models/speedrun_moe_gpt.py` | **Drop** | Speedrun-only MoE |
| A8 | SpeedrunMoEverything | `src/models/speedrun_mixture_of_everything.py` | **Drop** | Speedrun-only variant |

## Routing & Expert Selection

| # | Component | File | Keep? | Notes |
|---|-----------|------|-------|-------|
| B1 | DeepSeek Router | `src/models/router.py` | **Keep** | Primary router |
| B2 | ExplorationTopKRouter | `src/models/router.py` | **Keep** | Softmax alternative |
| B3 | Branch Router | `src/models/mixture_of_everything.py` | **Keep** | MoE-Everything component |
| B4 | Per-head fully independent | `src/models/mixture_of_everything.py` | **Keep** | Attention expert mode |
| B5 | Per-head precompute KV | `src/models/mixture_of_everything.py` | **Keep** | Attention expert mode |
| B6 | Group-limited top-K | `src/models/router.py` | **Keep** | Megatron-LM routing |
| B7 | Expert bias updates | `train.py` | **Keep** | Port to train_torch.py |

## Loss Functions

| # | Loss | File | Keep? |
|---|------|------|-------|
| C1 | Cross-entropy | Model forward | **Keep** |
| C2 | Batch load-balancing | `src/models/load_balancing.py` | **Keep** |
| C3 | Sequence load-balancing | `src/models/load_balancing.py` | **Keep** |
| C4 | Normalized load-balancing | `src/models/load_balancing.py` | **Keep** |
| C5 | Branch routing aux | `src/models/mixture_of_everything.py` | **Keep** |
| C6 | Attention expert aux | `src/models/mixture_of_everything.py` | **Keep** |

## Optimizers

| # | Optimizer | File | Keep? |
|---|-----------|------|-------|
| D1 | AdamW | `src/utils/training.py` | **Keep** |
| D2 | Muon | `src/utils/muon.py` | **Keep** |
| D3 | DistMuon | `src/utils/dist_optimizers.py` | **Keep** |
| D4 | DistAdam | `src/utils/dist_optimizers.py` | **Keep** |

## LR Schedulers

| # | Scheduler | File | Keep? |
|---|-----------|------|-------|
| E1 | Cosine | `src/utils/training.py` | **Keep** |
| E2 | Linear | `src/utils/training.py` | **Keep** |
| E3 | Constant | `src/utils/training.py` | **Keep** |
| E4 | Stable Decay | `src/utils/training.py` | **Keep** |

## Training Loops

| # | File | Keep? | Notes |
|---|------|-------|-------|
| F1 | `train.py` (Accelerate) | **Drop** | Replace with unified train_torch.py |
| F2 | `train_torch.py` (torch.distributed) | **Keep** | Unified target, must port missing features |
| F3 | `modal_train.py` | **Keep** | Cloud launcher |

## Data Loading

| # | Component | File | Keep? | Notes |
|---|-----------|------|-------|-------|
| G1 | StatefulParquetDataset | `src/data/parquet_dataset.py` | **Keep** | Primary data backend |
| G2 | StatefulTokenBinDataset | `src/data/token_bin_dataset.py` | **Drop** | Speedrun-only format |

## Distributed Training

| # | Feature | Keep? | Notes |
|---|---------|-------|-------|
| H1 | DDP via torch.distributed | **Keep** | In train_torch.py |
| H2 | FSDP via torch.distributed | **Keep** | In train_torch.py |
| H3 | DDP via Accelerate | **Drop** | Remove with train.py |
| H4 | Multi-node on Modal | **Keep** | modal_train.py |
| H5 | Multi-node on GCP | **Keep** | gcp_setup.sh + docs |
| H6 | Accelerate configs | **Drop** | Remove accelerate_configs/ |

## Checkpointing

| # | Feature | Keep? | Notes |
|---|---------|-------|-------|
| I1 | Manual torch.save/load | **Keep** | In train_torch.py |
| I2 | FSDP state dict handling | **Keep** | In train_torch.py |
| I3 | Accelerate save/load | **Drop** | Remove with train.py |
| I4 | meta.json (step, tokens, dataset state) | **Keep** | |
| I5 | Auto-resume | **Keep** | |
| I6 | Checkpoint cleanup | **Keep** | |

## Logging & Monitoring

| # | Feature | Keep? |
|---|---------|-------|
| J1 | WandB integration | **Keep** |
| J2 | Routing statistics | **Keep** |
| J3 | Expert bias stats | **Keep** |
| J4 | Routing heatmaps | **Keep** |
| J5 | Evaluation loop | **Keep** |

## Custom Kernels & Precision

| # | Component | File | Keep? | Notes |
|---|-----------|------|-------|-------|
| K1 | FP32 routing ops | `src/models/fp32_routing.py` | **Keep** | Precision-critical |
| K2 | Triton grouped GEMM | `src/models/triton_grouped_gemm.py` | **Keep** | Expert dispatch |
| K3 | Triton Newton-Schulz | `src/utils/triton_newton_schulz.py` | **Keep** | Muon optimizer |
| K4 | FP8 lm_head | `src/models/speedrun_gpt.py` | **Drop** | Speedrun-only |

## Speedrun Features (Portability Assessment)

| # | Feature | Keep? | Notes |
|---|---------|-------|-------|
| L1 | FlexAttention | **Keep if possible** | General PyTorch feature, can be applied to any model |
| L2 | Dynamic window sizing | **Keep if possible** | Training strategy, not model-specific |
| L3 | Kernel warmup | **Keep** | Useful for any torch.compile workload |
| L4-L12 | Other speedrun features | **Drop** | Model-specific optimizations |

## Files to Remove

| Path | Reason |
|------|--------|
| `train.py` | Replaced by unified train_torch.py |
| `accelerate_configs/` | Accelerate removed |
| `src/models/speedrun_gpt.py` | Speedrun model |
| `src/models/speedrun_moe_gpt.py` | Speedrun model |
| `src/models/speedrun_mixture_of_everything.py` | Speedrun model |
| `configs/speedrun/` | Speedrun configs |
| `src/data/token_bin_dataset.py` | Speedrun data format |
| `bench_expert_attn.py` | Speedrun benchmark |
| `scripts/download_fineweb10b_gpt2_bins.py` | Speedrun data script |

## Files to Keep and Clean

| Path | Action |
|------|--------|
| `train_torch.py` | Port missing features from train.py, support all models |
| `modal_train.py` | Update to only use train_torch.py |
| `src/models/standard_moe.py` | Keep as-is |
| `src/models/global_moe.py` | Keep as-is |
| `src/models/mixture_of_everything.py` | Keep as-is |
| `src/models/router.py` | Keep as-is |
| `src/models/load_balancing.py` | Keep as-is |
| `src/models/__init__.py` | Remove speedrun exports |
| `src/data/parquet_dataset.py` | Keep as-is |
| `src/utils/` | Keep all (training.py, muon.py, dist_optimizers.py, routing_*) |
| `configs/scaling/` | Keep |
| `configs/depth_matched/` | Keep |
| `configs/plan/` | Remove speedrun configs, keep others |
