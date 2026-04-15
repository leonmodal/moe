# MoE Training Codebase Reorganization and Cleanup

## Goal Description

Reorganize and clean up the MoE training codebase to achieve a clean, maintainable architecture. This involves:

1. **Documentation correction** — fix all docs to match actual codebase behavior, eliminating aspirational statements presented as current. Docs must include detailed coverage of: MoE-Everything model architecture, global load-imbalancing router, expert bias normalization, branch routing, and per-head attention routing modes.
2. **Unified torch-native training stack** — remove the Accelerator-based `train.py` entirely; extract shared modules from `train_torch.py` into a clean `src/training/` library with a single CLI entrypoint. Support DDP, FSDP, and Modal multi-node training. The training pipeline must include a loss sanity check: loss should converge to sensible values (~3.28 on FineWeb GPT-2 reference); loss stuck at ~4 after 1k steps indicates a bug (reference: the shifted-logits CE label bug fixed in commit `740f306`).
3. **Bagel-style custom model architecture** — copy and customize Qwen3 HuggingFace components (attention, MLP, RMSNorm, RoPE) into our own standalone model files. Keep HF `PreTrainedModel` base for checkpoint compatibility but do not depend on `transformers` for model forward-pass logic at runtime. This gives full customization control while preserving HF ecosystem compatibility. Supported models: dense, standard MoE, global MoE, and MoE-Everything (both `per_head_fully_independent` and `per_head_precompute_kv` modes). DeepSeek routing becomes a config option (`router_type`) within standard MoE and global MoE, not a separate model type.
4. **Speedrun archival** — move speedrun model architecture code (speedrun_gpt, speedrun_moe_gpt, speedrun_mixture_of_everything) and their configs/tests to a `legacy/` directory. **Decoupled speedrun utilities stay in active code**: routing stats (`routing_stats.py`), routing visualization/graphs (`routing_plots.py`), Muon optimizer (`muon.py`), Triton grouped GEMM (`triton_grouped_gemm.py`), Newton-Schulz iteration (`triton_newton_schulz.py`), distributed optimizers (`dist_optimizers.py`), and momentum warmup schedule.
5. **Data pipeline simplification** — remove token-bin data support entirely; sharded parquet is the sole data path. Optimize data loading performance (prefetching, parallel loading) for throughput.
6. **Code organization** — split `mixture_of_everything.py` (2311 lines) into coherent modules by API seam using inheritance patterns; normalize naming conventions across all model files.
7. **External MoE research** — study Megatron-LM, modal-nmoe, and nmoe for techniques that make training more stable and faster (grouped GEMM coverage, dispatcher improvements, expert-bias tuning, communication overlap); integrate only benchmark-validated techniques.
8. **Llama 3.1 not needed** — since we are writing our own model config and architecture files (Bagel-style), we can inherit from Qwen3 patterns (GQA, RoPE, QK norm) directly. Llama 3.1 backbone is not needed for this effort.

**Checkpoint requirements**: Remove Accelerator entirely. Checkpointing must save model state, optimizer state (both Adam and Muon), and data loading order separately — not as a single monolithic file. Provide a utility to convert model checkpoints to safetensors format. Resume from checkpoint must restore exact training state.

**Tracking**: Keep the current routing tracking and graph-saving capabilities (routing stats, expert heatmaps, branch routing curves, WandB integration) from the speedrun infrastructure — these are already decoupled from speedrun model code.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Documentation accuracy — all docs match the post-refactor codebase reality with detailed coverage
  - Positive Tests (expected to PASS):
    - Each section in `architecture.md`, `routing.md`, `training.md`, `configuration.md`, `data.md`, `distributed.md` verified against actual code paths
    - `architecture.md` includes detailed documentation of: MoE-Everything model (branch routing, attention/MLP expert banks, per-head modes), global load-imbalancing router, expert bias normalization, and DeepSeek routing
    - `routing.md` covers all active routing mechanisms with code-level accuracy
    - README quick-start commands work against the unified trainer
    - `status.md` reflects current state, not historical benchmarks
  - Negative Tests (expected to FAIL):
    - No docs reference `train.py`, Accelerator, speedrun models, or token-bin as current features
    - No aspirational features described as implemented

- AC-2: Unified training infrastructure — single torch-native trainer stack with shared modules
  - Positive Tests (expected to PASS):
    - All supported models (dense, standard_moe, global_moe, moe_everything with both per_head modes) train correctly through a single CLI entrypoint
    - Shared modules exist in `src/training/` for: model factory, dataset construction, checkpoint IO, eval, routing stats, WandB/resume, distributed setup
    - `modal_train.py` uses the unified trainer
  - Negative Tests (expected to FAIL):
    - No duplicated implementations of model build, dataset build, checkpoint, eval, routing stats, or WandB logic in active (non-legacy) code
    - No Accelerator imports or `accelerate` dependencies in active code
    - `train.py` does not exist in the project root
    - No `accelerate_configs/` directory exists

- AC-3: Supported model set — explicitly defined and tested with custom architecture
  - Positive Tests (expected to PASS):
    - Dense (Qwen3-based) trains through unified trainer
    - Standard MoE trains with softmax routing
    - Standard MoE trains with DeepSeek routing (`router_type: deepseek`)
    - Global MoE trains with softmax routing
    - Global MoE trains with DeepSeek routing (`router_type: deepseek`)
    - MoE-Everything trains with `per_head_fully_independent` mode
    - MoE-Everything trains with `per_head_precompute_kv` mode
  - Negative Tests (expected to FAIL):
    - Requesting `speedrun_moe_gpt` model type produces a clear error message
    - Requesting `speedrun_moe_everything` model type produces a clear error message
    - Requesting `gpt2_dense` model type produces a clear error message
  - AC-3.1: DeepSeek routing merged as config option
    - Positive: `router_type: deepseek` in standard_moe/global_moe config activates DeepSeek sigmoid+expert-bias routing
    - Negative: `deepseek_standard_moe` and `deepseek_global_moe` are not valid model types
  - AC-3.2: Bagel-style custom model architecture
    - Positive: Model files contain copied+customized Qwen3 components (attention, MLP, RMSNorm, RoPE) in standalone files
    - Positive: Models inherit from HF `PreTrainedModel` for checkpoint compatibility (`from_pretrained`/`save_pretrained` work)
    - Negative: No runtime import of model layers from `transformers` in model forward pass

- AC-4: Speedrun models archived, utilities preserved
  - Positive Tests (expected to PASS):
    - `legacy/` directory contains all speedrun model code, configs, and tests
    - `legacy/` has its own README explaining the archive
    - Routing stats (`routing_stats.py`), routing plots (`routing_plots.py`), Muon optimizer (`muon.py`), Triton grouped GEMM, Newton-Schulz, and distributed optimizers remain in active `src/utils/`
  - Negative Tests (expected to FAIL):
    - No speedrun model imports in `src/models/__init__.py` or any active source file
    - No speedrun model configs in `configs/` (only in `legacy/`)
    - No speedrun model references in active documentation

- AC-5: Code organization — coherent module boundaries with inheritance
  - Positive Tests (expected to PASS):
    - `mixture_of_everything.py` decomposed into separate modules by API seam (config/types, routing, attention bank, MLP bank, model assembly)
    - Modules use inheritance patterns for shared behavior (base expert bank, base router, base model)
    - Each module has a clear single responsibility
    - All imports between modules are acyclic
  - Negative Tests (expected to FAIL):
    - No circular dependencies between split modules (verified by import analysis)
    - No single active model file exceeds coherent single-responsibility scope

- AC-6: Config normalization — active configs follow consistent schema
  - Positive Tests (expected to PASS):
    - All active configs in `configs/` follow a consistent naming convention
    - A config validator/linter command exists and passes on all active configs
    - Model type field uses the normalized taxonomy (dense, standard_moe, global_moe, moe_everything)
  - Negative Tests (expected to FAIL):
    - No active configs use deprecated model types (deepseek_standard_moe, deepseek_global_moe, gpt2_dense, speedrun_*)
    - No duplicate or conflicting active configs exist

- AC-7: Routing and load-balancing consistency
  - Positive Tests (expected to PASS):
    - Single routing stats implementation used by all supported models
    - Load-balancing loss computation follows one code path for all models
    - Expert bias updates use the same mechanism across standard_moe, global_moe, and moe_everything
  - Negative Tests (expected to FAIL):
    - No model-specific routing stat implementations in active code
    - No scattered load-balancing loss functions (single source in `src/models/load_balancing.py`)

- AC-8: Data pipeline — sharded parquet only, with performance optimization
  - Positive Tests (expected to PASS):
    - `StatefulParquetDataset` works for all supported models through unified trainer
    - Dataset construction, train/val splitting, and distributed sharding are correct
    - Data loading includes prefetching or parallel loading for throughput
  - Negative Tests (expected to FAIL):
    - No `token_bin_dataset.py` in active `src/data/`
    - No token-bin references in active configs or trainer code

- AC-9: Distributed training — DDP, FSDP, and Modal multi-node
  - Positive Tests (expected to PASS):
    - Single-node DDP training works for all supported models
    - FSDP training works for all supported models
    - Modal multi-node training works through `modal_train.py` using the unified trainer
  - Negative Tests (expected to FAIL):
    - No Accelerator-based distributed setup exists
    - No `accelerate launch` commands in scripts or docs

- AC-10: External optimization research — focused on training stability and throughput
  - Positive Tests (expected to PASS):
    - Research document exists with findings from Megatron-LM, modal-nmoe, and nmoe
    - Each finding includes: technique description, applicability assessment, and benchmark data (if integrated)
    - Findings specifically address training stability improvements and throughput optimization
    - Grouped GEMM coverage gaps identified and addressed where benchmarks justify
  - Negative Tests (expected to FAIL):
    - No unverified performance claims in code or docs
    - No optimizations integrated without supporting benchmark data
  - AC-10.1: Research findings documented separately from code integration
    - Positive: Clear separation between "observed in external codebases" and "implemented here"
    - Negative: No speculative optimization code without benchmark backing

- AC-11: No regression — trainer smoke tests pass with loss sanity check
  - Positive Tests (expected to PASS):
    - Deterministic smoke test passes for each supported model variant through unified trainer
    - Smoke test matrix: dense, standard_moe (softmax), standard_moe (deepseek), global_moe (softmax), global_moe (deepseek), moe_everything (fully_independent), moe_everything (precompute_kv)
    - Each smoke test: forward pass, backward pass, optimizer step, checkpoint save/load
    - Loss decreases to sensible values during training (not stalling at unreasonably high values)
  - Negative Tests (expected to FAIL):
    - No unsupported model type passes through the trainer without error
    - Loss stuck at ~4+ after 1k steps on reference data triggers investigation (potential label-shift or loss computation bug)

- AC-12: Checkpoint robustness — separate saves, safetensors, resume
  - Positive Tests (expected to PASS):
    - Model state saved separately from optimizer state (not a single monolithic file)
    - Optimizer state saved correctly for both Adam and Muon optimizers
    - Data loading order/position saved for deterministic resume
    - Safetensors conversion utility exists and converts model checkpoints correctly
    - Resume from checkpoint restores exact training state (model, optimizer, data position, step count)
  - Negative Tests (expected to FAIL):
    - No monolithic `train.pt` or single-file checkpoint format
    - Resume after conversion to safetensors produces identical training continuation

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes complete documentation rewrite with detailed coverage of MoE-Everything, global routing, and bias normalization; full trainer unification with clean module extraction into `src/training/`; Bagel-style custom model architecture with copied+customized Qwen3 components; all 4 model families plus DeepSeek routing variants tested through deterministic smoke tests with loss sanity checks; speedrun models fully archived with all decoupled utilities preserved in active code; `mixture_of_everything.py` split into 5+ coherent modules using inheritance patterns; all active configs normalized with a schema validator/linter; external optimizations (grouped GEMM coverage, dispatcher improvements) integrated where benchmarks justify stability and throughput gains; DDP, FSDP, and Modal multi-node all validated; token-bin fully removed; checkpoint saves model/optimizer/data-state separately with safetensors conversion utility; data loading optimized for throughput.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes docs corrected for factual errors with key topics covered; trainer unification using `train_torch.py` as base with shared modules extracted to `src/training/`; Bagel-style custom model files for at minimum the base components (attention, MLP, RMSNorm, RoPE); all 4 model families trainable through unified CLI entrypoint; speedrun models moved to `legacy/` with decoupled utilities remaining active; `mixture_of_everything.py` split into at minimum 3 modules (model assembly, attention bank, MLP bank) with inheritance; active configs given consistent naming and model type taxonomy; external research documented with recommendations; DDP tested, FSDP and Modal multi-node smoke-tested; token-bin removed; checkpoint saves model and optimizer separately with resume support; safetensors conversion at minimum for model weights.

### Allowed Choices

- Can use: `torch.distributed` (DDP, FSDP), Triton kernels, WandB, liger_kernel (optional), Muon optimizer, HF `PreTrainedModel` base class for checkpoint compat
- Cannot use: `accelerate` library, speedrun model architectures in main training path, token-bin data format in active code, runtime imports of `transformers` model layers in forward pass
- Trainer structure: shared modules in `src/training/` with a thin CLI wrapper — not a monolithic training function
- Model architecture: Bagel-style — copy Qwen3 components locally and customize, inheriting from HF `PreTrainedModel`
- Model taxonomy: DeepSeek routing as `router_type` config option within standard_moe/global_moe, not as separate model classes
- Architecture scope: Qwen3-based custom models; Llama 3.1 not needed

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Start from `train_torch.py`** as the base — it is the newer, torch-native trainer with DDP/FSDP support and no Accelerator dependency. Extract its shared logic into modules:

```
src/training/
├── __init__.py
├── config.py         # TrainingConfig, model factory, config parsing/validation
├── data.py           # Dataset construction (parquet only), train/val split, distributed sharding, prefetch
├── checkpoint.py     # Separate model/optimizer/data-state saves, resume, safetensors conversion
├── eval.py           # Validation loop, compile-safe eval behavior, loss sanity checks
├── routing.py        # Unified routing stats, load-balancing loss, expert bias updates
├── distributed.py    # DDP/FSDP setup, gradient handling, communication
└── logging.py        # WandB logging, console output, metrics formatting, routing graphs
```

2. **Bagel-style model architecture**: Copy Qwen3 components from HuggingFace into our own files:

```
src/models/
├── base/
│   ├── __init__.py
│   ├── attention.py      # Copied+customized Qwen3 attention (GQA, QK norm)
│   ├── mlp.py            # Copied+customized Qwen3 MLP
│   ├── normalization.py  # RMSNorm
│   ├── embeddings.py     # Token embeddings, RoPE
│   └── config.py         # Base model configuration
├── dense.py              # Dense model (inherits from base)
├── standard_moe.py       # Standard MoE with router_type config
├── global_moe.py         # Global MoE with shared expert pool
├── moe_everything/       # Split by API seam with inheritance
│   ├── __init__.py
│   ├── config.py         # Config dataclass, type definitions
│   ├── routing.py        # Branch router, routing logic
│   ├── attention_bank.py # AttentionExpertBank, per-head modes
│   ├── mlp_bank.py       # MlpExpertBank
│   └── model.py          # MoEverythingForCausalLM assembly
├── router.py             # DeepSeekRouter, ExplorationTopKRouter
├── load_balancing.py     # Unified loss functions
├── fp32_routing.py       # Numerical stability helpers
└── triton_grouped_gemm.py # Expert dispatch kernel
```

3. **Create unified CLI entrypoint** (`train.py` or `scripts/train.py`) that imports from `src/training/` and provides the main training loop.

4. **Checkpoint module**: Save model state, optimizer state, and data state as separate files:
   - `model.safetensors` (or `model.pt`) — model weights only
   - `optimizer_adam.pt` / `optimizer_muon.pt` — optimizer states
   - `data_state.pt` — dataset position, token buffer state, epoch
   - `training_state.pt` — step count, LR scheduler, RNG states
   - Conversion utility: `convert_checkpoint.py` to export model weights to safetensors format

5. **Merge DeepSeek routing**: In `standard_moe.py` and `global_moe.py`, accept a `router_type` config field. When `router_type: deepseek`, use `DeepSeekRouter`; when `router_type: softmax` (or default), use `ExplorationTopKRouter`.

6. **Archive speedrun models**: Move `speedrun_gpt.py`, `speedrun_moe_gpt.py`, `speedrun_mixture_of_everything.py` and their configs/tests to `legacy/speedrun/`. Keep all decoupled utilities (`routing_stats.py`, `routing_plots.py`, `muon.py`, `dist_optimizers.py`, `triton_grouped_gemm.py`, `triton_newton_schulz.py`) in active `src/utils/` and `src/models/`.

7. **Update `modal_train.py`** to use the unified trainer entrypoint instead of dispatching between `train.py` and `train_torch.py`.

### Relevant References

- `train_torch.py` — base for unified trainer (1768 lines of logic to extract)
- `train.py` — Accelerator-based trainer to remove (reference for any features not yet in train_torch.py)
- `/tmp/moe/Bagel/` — reference implementation for Bagel-style copy+customize approach
- `src/models/modeling_qwen3_moe.py` — locally shadowed HF code with bug fixes (basis for custom components)
- `src/models/standard_moe.py` — standard MoE wrapper (88 lines, needs DeepSeek routing merge)
- `src/models/global_moe.py` — global MoE with shared expert pool (214 lines, needs DeepSeek routing merge)
- `src/models/mixture_of_everything.py` — largest model file to split (2311 lines)
- `src/models/router.py` — `DeepSeekRouter` and `ExplorationTopKRouter` (251 lines)
- `src/models/load_balancing.py` — loss functions to unify (274 lines)
- `src/data/parquet_dataset.py` — sole surviving data path (needs performance optimization)
- `src/utils/training.py` — existing shared training utilities (132 lines)
- `src/utils/routing_stats.py` — routing stats collector, decoupled from speedrun (13.5KB)
- `src/utils/routing_plots.py` — routing visualization/graphs, decoupled from speedrun (14.3KB)
- `src/utils/muon.py` — Muon optimizer, decoupled from speedrun (8.3KB)
- `modal_train.py` — Modal multi-node launcher to update (9KB)
- `configs/` — 116+ config files, active ones need normalization

## Dependencies and Sequence

### Milestones

1. **M1: Documentation Audit and Initial Corrections**
   - Phase A: Inventory all docs, identify factual errors vs actual code
   - Phase B: Mark planned removals (speedrun models, Accelerator, token-bin) as planned, not current
   - Phase C: Defer comprehensive doc rewrite until after codebase stabilizes (M12)

2. **M2: Define Supported Surface and Archive Policy**
   - Phase A: Freeze supported model set: dense, standard_moe, global_moe, moe_everything
   - Phase B: Define legacy/archive structure for speedrun models (utilities stay active)
   - Phase C: Define config naming convention and normalized model taxonomy
   - Depends on: M1

3. **M3: Extract Shared Trainer Library**
   - Phase A: Extract model factory from `train_torch.py` into `src/training/config.py`
   - Phase B: Extract dataset construction into `src/training/data.py` with performance optimization
   - Phase C: Extract checkpoint IO into `src/training/checkpoint.py` with separate model/optimizer/data saves, Adam+Muon support
   - Phase D: Extract eval loop into `src/training/eval.py` with loss sanity checks
   - Phase E: Extract routing stats/bias updates into `src/training/routing.py`
   - Phase F: Extract DDP/FSDP setup into `src/training/distributed.py`
   - Phase G: Extract WandB/logging into `src/training/logging.py` with routing graph support
   - Phase H: Create unified CLI entrypoint using shared modules
   - Phase I: Create safetensors conversion utility
   - Depends on: M2

4. **M4: Remove Accelerator Path**
   - Phase A: Delete `train.py` (Accelerator-based trainer)
   - Phase B: Delete `accelerate_configs/` directory
   - Phase C: Remove `accelerate` from dependencies
   - Depends on: M3

5. **M5: Archive Speedrun Models to Legacy**
   - Phase A: Create `legacy/speedrun/` directory structure
   - Phase B: Move speedrun model files (`speedrun_moe_gpt.py`, `speedrun_mixture_of_everything.py`, `speedrun_gpt.py`)
   - Phase C: Move speedrun configs from `configs/speedrun/`
   - Phase D: Move speedrun-specific tests
   - Phase E: Remove speedrun models from `src/models/__init__.py` exports and active imports
   - Phase F: Verify decoupled utilities (`routing_stats.py`, `routing_plots.py`, `muon.py`, etc.) remain functional
   - Phase G: Add `legacy/README.md` explaining archive
   - Depends on: M3

6. **M6: Remove Token-Bin Data**
   - Phase A: Remove `token_bin_dataset.py` from active `src/data/`
   - Phase B: Remove token-bin references from active configs and trainer code
   - Phase C: Archive to `legacy/` if any speedrun configs depend on it
   - Depends on: M3

7. **M7: Model Architecture Rewrite and Surface Cleanup**
   - Phase A: Copy and customize Qwen3 base components (attention, MLP, RMSNorm, RoPE) into `src/models/base/`
   - Phase B: Merge DeepSeek routing into `router_type` config option in standard_moe and global_moe
   - Phase C: Split `mixture_of_everything.py` by API seams with inheritance patterns (config, routing, attention bank, MLP bank, model)
   - Phase D: Normalize model class naming conventions
   - Phase E: Update model factory to use normalized taxonomy and custom base components
   - Depends on: M4, M5, M6

8. **M8: Config Normalization**
   - Phase A: Define schema for active configs with normalized model types
   - Phase B: Rename and normalize all active configs following convention
   - Phase C: Add config validation/linter command
   - Phase D: Remove or archive deprecated configs
   - Depends on: M7

9. **M9: Routing and Loss Unification**
   - Phase A: Unify routing stats collection into single implementation
   - Phase B: Unify load-balancing loss computation to single code path
   - Phase C: Unify expert bias update mechanism across models
   - Phase D: Ensure consistent metrics for WandB logging and routing graph generation
   - Depends on: M7

10. **M10: External Optimization Research** *(can proceed in parallel with M3–M9)*
    - Phase A: Research Megatron-LM MoE training techniques (grouped GEMM, dispatcher selection, communication overlap)
    - Phase B: Research modal-nmoe and nmoe techniques (RDEP, CUDA-IPC, expert dispatch strategies)
    - Phase C: Document findings with focus on training stability and throughput improvements
    - Phase D: Benchmark candidate optimizations
    - Phase E: Integrate only benchmark-validated improvements

11. **M11: Testing and Validation**
    - Phase A: Create deterministic smoke test for each supported model variant with loss sanity check
    - Phase B: Validate DDP training for all supported models
    - Phase C: Validate FSDP training for all supported models
    - Phase D: Validate Modal multi-node training through updated `modal_train.py`
    - Phase E: Verify routing stats consistency across models
    - Phase F: Verify checkpoint save/load/resume with separate files
    - Depends on: M7, M8, M9

12. **M12: Final Documentation**
    - Phase A: Rewrite `architecture.md` with detailed coverage of MoE-Everything, global routing, bias normalization, per-head modes
    - Phase B: Rewrite `routing.md`, `training.md`, `configuration.md`, `data.md`, `distributed.md` to match final codebase
    - Phase C: Update `README.md` with correct quick-start, model families, and training commands
    - Phase D: Clean up `status.md` to reflect current state
    - Phase E: Update `MULTINODE_README.md` and `EFFICIENCY_NOTES.md`
    - Depends on: M11

M3–M9 are sequential with dependencies as shown. M10 (external research) can run in parallel. M1 is first, M12 is last.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Audit all docs for factual errors vs actual code behavior | AC-1 | analyze | - |
| task2 | Research Megatron-LM MoE training techniques (grouped GEMM, dispatcher, communication overlap, stability) | AC-10 | analyze | - |
| task3 | Research modal-nmoe and nmoe training techniques (RDEP, CUDA-IPC, expert dispatch, throughput) | AC-10 | analyze | - |
| task4 | Initial doc corrections: mark planned removals as planned, fix factual errors | AC-1 | coding | task1 |
| task5 | Define supported model surface and create `legacy/` directory structure | AC-3, AC-4 | coding | task4 |
| task6 | Extract model factory module from `train_torch.py` into `src/training/config.py` | AC-2 | coding | task5 |
| task7 | Extract dataset construction into `src/training/data.py` with performance optimization (prefetch, parallel loading) | AC-2, AC-8 | coding | task5 |
| task8 | Extract checkpoint IO into `src/training/checkpoint.py` with separate model/optimizer/data saves, Adam+Muon support | AC-2, AC-12 | coding | task5 |
| task9 | Extract eval loop into `src/training/eval.py` with loss sanity checks | AC-2, AC-11 | coding | task5 |
| task10 | Extract routing stats and bias updates into `src/training/routing.py` | AC-2, AC-7 | coding | task5 |
| task11 | Extract DDP/FSDP distributed setup into `src/training/distributed.py` | AC-2, AC-9 | coding | task5 |
| task12 | Extract WandB/logging module into `src/training/logging.py` with routing graph support | AC-2 | coding | task5 |
| task13 | Create unified CLI entrypoint using extracted `src/training/` modules | AC-2 | coding | task6, task7, task8, task9, task10, task11, task12 |
| task14 | Remove `train.py` (Accelerator-based) and `accelerate_configs/` directory | AC-2, AC-9 | coding | task13 |
| task15 | Move speedrun model code, configs, and tests to `legacy/speedrun/`; verify decoupled utilities remain functional | AC-4 | coding | task13 |
| task16 | Remove `token_bin_dataset.py` and all token-bin references from active code | AC-8 | coding | task13 |
| task17 | Copy and customize Qwen3 base components (attention, MLP, RMSNorm, RoPE) into `src/models/base/` with HF PreTrainedModel inheritance | AC-3, AC-3.2 | coding | task14, task15, task16 |
| task18 | Merge DeepSeek routing variants into `router_type` config option in standard_moe and global_moe | AC-3, AC-3.1 | coding | task17 |
| task19 | Split `mixture_of_everything.py` into modules by API seam with inheritance patterns (config, routing, attention, mlp, model) | AC-5 | coding | task18 |
| task20 | Normalize model class naming conventions and update model factory to use custom base components | AC-5 | coding | task19 |
| task21 | Define config schema with normalized model types and naming convention | AC-6 | coding | task20 |
| task22 | Normalize all active configs to follow new schema and naming | AC-6 | coding | task21 |
| task23 | Create config validation/linter command | AC-6 | coding | task21 |
| task24 | Unify routing stats into single implementation across all supported models | AC-7 | coding | task19 |
| task25 | Unify load-balancing loss computation to single code path | AC-7 | coding | task19 |
| task26 | Unify expert bias update mechanism across models | AC-7 | coding | task19 |
| task27 | Document external research findings with focus on stability and throughput | AC-10 | coding | task2, task3 |
| task28 | Benchmark candidate external optimizations and integrate validated ones | AC-10, AC-11 | coding | task27, task24 |
| task29 | Update `modal_train.py` to use unified trainer entrypoint | AC-9 | coding | task13, task11 |
| task30 | Create safetensors conversion utility for model checkpoints | AC-12 | coding | task8 |
| task31 | Optimize data loading performance (prefetching, parallel loading) | AC-8 | coding | task7 |
| task32 | Create deterministic smoke tests for all supported model variants with loss sanity checks | AC-11 | coding | task24, task25, task26 |
| task33 | Validate DDP and FSDP training for all supported models | AC-9, AC-11 | coding | task32 |
| task34 | Validate Modal multi-node training | AC-9, AC-11 | coding | task29, task32 |
| task35 | Validate checkpoint save/load/resume with separate files (model, optimizer, data state) | AC-12 | coding | task30, task32 |
| task36 | Rewrite all documentation with detailed coverage of MoE-Everything, routing, bias normalization | AC-1 | coding | task33, task34, task35 |
| task37 | Update README.md, status.md, MULTINODE_README.md, EFFICIENCY_NOTES.md | AC-1 | coding | task36 |
| task38 | Git push final changes | - | coding | task37 |

## Claude-Codex Deliberation

### Agreements

- Documentation correction is a valid first-class milestone; the repo is visibly drifted (docs describe both old Accelerate path and torch-native rewrite in transition)
- Unifying the trainer stack with shared modules (not a monolithic function) is the correct approach
- Narrowing the active model surface to dense, standard_moe, global_moe, and moe_everything is well-justified
- Archiving speedrun models to `legacy/` (not deleting) is the safest approach — preserves reproducibility and history
- Decoupled speedrun utilities (routing stats, plots, Muon, Triton GEMM) should remain in active code
- Splitting `mixture_of_everything.py` by API seams with inheritance is justified given its 2311-line single-file complexity
- Unifying routing stats and load-balancing across supported models reduces 4 scattered implementations to 1
- Config normalization should be scoped to active configs only, not historical/legacy configs
- External research should be documented separately from code integration; only benchmark-validated optimizations should be merged
- `train_torch.py` is the right base for the unified trainer (newer, no Accelerator dependency)

### Resolved Disagreements

- **File size threshold for AC-5**: Claude originally proposed an `~800 lines` hard threshold. Codex argued the real criterion is coherent module boundaries and no circular imports. **Resolution**: adopted module boundary criterion (split by API seam with inheritance) over arbitrary line count.

- **AC-8 (external research) scope**: Claude had an open-ended "integrate applicable optimizations where feasible." Codex required splitting into documented findings and benchmark-validated integration. **Resolution**: split into AC-10 (research documented) with sub-gate AC-10.1 (only benchmark-backed integration).

- **AC-2 duplication absolutism**: Claude originally stated "no duplicated implementations exist." Codex argued `legacy/` should be excluded since archived code inherently duplicates. **Resolution**: scoped to active (non-legacy) code only.

- **Trainer target**: Claude initially proposed generic "unified trainer." Codex identified that `train.py` vs `train_torch.py` is a real decision. **Resolution**: user decided to remove Accelerator entirely and use torch-native as the sole path.

### Convergence Status

- Final Status: `partially_converged`
- All technical items converged after gen-plan Round 1 revision
- 6 items required user decisions during gen-plan — all resolved
- Refine-plan processed 10 user comments resolving model architecture direction (Bagel-style), speedrun utility retention, checkpoint requirements, and other clarifications
- 1 pending decision remains (DEC-6: hardware scope)

## Pending User Decisions

- DEC-1: Llama 3.1 backbone scope
  - Claude Position: Defer to later phase after Qwen cleanup
  - Codex Position: Could be in-scope, but combining two major refactors reduces debuggability
  - Tradeoff Summary: In-scope now means backbone abstraction work interleaved with cleanup; deferred means cleaner separation of concerns but later delivery
  - Decision Status: **Not needed** — user confirmed writing own model configs (Bagel-style), can inherit from Qwen3. Llama not required.

- DEC-2: DeepSeek routing variant treatment
  - Claude Position: Merge into standard_moe/global_moe as `router_type` config option
  - Codex Position: Merge is reasonable but migration path must be explicit
  - Tradeoff Summary: Merged reduces model surface area; separate preserves explicitness but adds maintenance
  - Decision Status: **Keep as routing options** — `router_type: deepseek` config flag within standard_moe/global_moe

- DEC-3: Checkpoint backward compatibility
  - Claude Position: Clean break, no migration
  - Codex Position: Must define policy either way
  - Tradeoff Summary: Clean break simplifies refactor; migration preserves continuity for running experiments
  - Decision Status: **Clean break** — user wants separate model/optimizer/data saves, safetensors conversion, Adam+Muon support. No backward compat with old checkpoints.

- DEC-4: Token-bin data support
  - Claude Position: Archive with speedrun to legacy/
  - Codex Position: Could keep for dense benchmarking/reproducibility
  - Tradeoff Summary: Removing simplifies data path; keeping preserves dense benchmark reproducibility
  - Decision Status: **Remove entirely** — keep only sharded parquet dataset

- DEC-5: Distributed training target
  - Claude Position: DDP first, FSDP and Modal multi-node as follow-up
  - Codex Position: Must define full scope
  - Tradeoff Summary: DDP-only first is simpler; full distributed is more work but delivers complete stack
  - Decision Status: **DDP + FSDP + Modal multi-node all in scope** — Accelerator removed, torch-native distributed only

- DEC-6: Hardware scope
  - Claude Position: N/A — not addressed in draft
  - Codex Position: Must define to inform kernel choices (H100 vs B200 implications)
  - Tradeoff Summary: Explicit hardware targeting affects kernel selection; unspecified means general compatibility
  - Decision Status: `PENDING` — user did not specify hardware targets; plan proceeds without hardware-specific constraints

- DEC-7: Model architecture strategy
  - Claude Position: Keep wrapping Qwen3 HF components
  - Codex Position: N/A (not raised during gen-plan)
  - Tradeoff Summary: HF wrappers are simpler but limit customization; Bagel-style copy+customize gives full control while keeping HF compatibility
  - Decision Status: **Bagel-style copy + customize HF components** — user confirmed after reviewing Bagel repo approach

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Key Technical Notes
- The existing `modeling_qwen3_moe.py` (locally shadowed HF code with double-softmax fix) serves as the starting point for Bagel-style custom components. The bug fixes must be preserved when copying components into `src/models/base/`.
- The shifted-logits CE label bug (commit `740f306`) was caused by datasets pre-shifting labels AND HF models shifting them again internally. The fix sets `labels = input_ids` and lets the model handle the shift once. The unified trainer must preserve this fix and include a loss sanity check.
- `mixture_of_everything.py` contains the `BranchRouter` which has NO load-balancing loss by design — do not add one during routing unification.
- Expert bias updates in `DeepSeekRouter` are non-gradient (buffer-based) — this must be preserved when unifying bias update logic.
- Triton grouped GEMM kernel (`src/models/triton_grouped_gemm.py`) is already used by moe_everything and stays in active code — verify it continues to work after the model architecture rewrite.
- `modal_train.py` currently dispatches non-speedrun jobs to `train.py` (Accelerator); it must be updated to use the unified trainer.
- Speedrun tracking capabilities (routing_stats.py, routing_plots.py) are already fully decoupled from speedrun model code — they work with any model that outputs routing statistics. Keep them in active `src/utils/`.
- The `RoutingStats` dataclass in `speedrun_moe_gpt.py` is separate from `src/utils/routing_stats.py` — only the latter survives in active code. The speedrun-specific dataclass moves to legacy.
- Checkpoint module must handle both Adam and Muon optimizer states. Current Muon implementation is in `src/utils/muon.py` (already decoupled). The hybrid optimizer (Muon for weight matrices, Adam for scalars) requires saving both optimizer states separately.

--- Original Design Draft Start ---

First take a look and docs and the codebase. correct anything thats not correct in the docs.

Now the second thing is that i think now the codebase is really messy and not organized. I want you to really organize the code in a clean way and there is only one unified training function and stuff. We also want to stay away from speedrun model for training and now just train our normal models like moe and stuff.

the models we need are just 
standard llm
standard moe
gloabl moe
mixture of everything, include fully independent and precompute kv

and we want to stay away from the speedrun archs. Instead we will be using qwen 3 / llama 3.1 archs

a few other codebases to take a look at are:
Megatron-LM
modal-nmoe
nmoe

take a look at those moe training and see any tricks can help us to train better. include stuff like gemm to have faster throughput and etc.

When you finish make a git push please.
--- Original Design Draft End ---
