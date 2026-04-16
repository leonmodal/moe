# Functionality Inventory

Post-reorganization inventory of active features and archived components.

## Active Model Architectures

| Model | Config Type | File | Notes |
|-------|------------|------|-------|
| Dense | `dense` | `Qwen3ForCausalLM` | Qwen3 backbone |
| Standard MoE | `standard_moe` | `src/models/standard_moe.py` | Per-layer MoE, softmax or DeepSeek routing |
| Global MoE | `global_moe` | `src/models/global_moe.py` | Shared expert pool |
| MoE-Everything | `moe_everything` | `src/models/mixture_of_everything.py` | Branch routing + expert banks |

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

## Archived to legacy/speedrun/

| Component | Original Location | Notes |
|-----------|-------------------|-------|
| SpeedrunGPT | `src/models/speedrun_gpt.py` | Dense speedrun model |
| SpeedrunMoEGPT | `src/models/speedrun_moe_gpt.py` | MoE speedrun model |
| SpeedrunMoEverything | `src/models/speedrun_mixture_of_everything.py` | Speedrun MoE-Everything variant |
| Accelerator trainer | `train.py` | HF Accelerate-based trainer |
| Token-bin dataset | `src/data/token_bin_dataset.py` | Binary token format |
| Accelerate configs | `accelerate_configs/` | DDP/FSDP Accelerate configs |
| Speedrun configs | `configs/speedrun/` | Speedrun model configs |
| torch-native trainer | `train_torch.py` | Original extraction source (superseded by scripts/train.py) |
