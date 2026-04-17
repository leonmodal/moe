# Distributed Training

## Strategies

The unified trainer (`scripts/train.py`) supports three distributed strategies via `--dist-strategy`:

### DDP (default)

```bash
torchrun --nproc_per_node=8 scripts/train.py --config config.yaml --dist-strategy ddp
```

- Standard `DistributedDataParallel` wrapping for every supported model family.
- Each GPU holds a full model replica.
- Gradients synchronized via NCCL all-reduce.

### FSDP

```bash
torchrun --nproc_per_node=8 scripts/train.py --config config.yaml --dist-strategy fsdp
```

`src/training/distributed.py::wrap_model` selects an FSDP policy per model family. Every family gets a real `FullyShardedDataParallel` wrapper — `--dist-strategy fsdp` never silently substitutes DDP.

| Model family | Sharding strategy | `use_orig_params` | FSDP `MixedPrecision` | `auto_wrap_policy` |
|--------------|-------------------|-------------------|-----------------------|---------------------|
| `dense` | `FULL_SHARD` | `True` | `MixedPrecision(param_dtype=reduce_dtype=training.mixed_precision, buffer_dtype=fp32)` | default (single root unit) |
| `standard_moe` | `FULL_SHARD` | `True` | same | default |
| `global_moe` | `FULL_SHARD` | `True` | same | default |
| `moe_everything` | `NO_SHARD` | `True` | **not set** (see §MoE-Everything notes) | `ModuleWrapPolicy({AttentionExpertBank, MlpExpertBank, BranchRouter, nn.Embedding, nn.Linear, Qwen3MoeRMSNorm})` |

Shared across all FSDP paths: `sync_module_states=True`, `device_id=torch.device("cuda", local_rank)`, `use_orig_params=True`.

The per-family selection tables live in `_FSDP_SHARDING_BY_MODEL_TYPE` and `_FSDP_SKIP_MIXED_PRECISION`; the MoE-Everything `auto_wrap_policy` is built by `_moe_everything_auto_wrap_policy()`.

#### MoE-Everything notes

MoE-Everything's branch-routed forward leaves whole FSDP flat-params units with no gradient activity on most steps (the branch router sends each token either to the attention-expert bank or to the MLP-expert bank, so exactly one bank sees zero gradient on every token). Two concrete symptoms on torch 2.10 forced the per-family split:

- `FULL_SHARD` / `SHARD_GRAD_OP` fire `ValueError: expected [FORWARD_BACKWARD] but current state is IDLE` from the post-backward hook of whichever flat-params unit was inactive for the step. `use_orig_params=True` alone does not suppress it.
- `NO_SHARD` + `MixedPrecision(param_dtype=bf16)` trips `RuntimeError: setStorage: ... storage of size 0` on the embedding flat-param during backward because the FSDP-internal fp32 master storage is freed while the bf16 shard is still in use.

The per-family policy sidesteps both: `NO_SHARD` keeps parameters replicated (DDP-like layout) so sharded post-backward hooks never fire on inactive shards; the `auto_wrap_policy` splits the expert banks and router into their own FSDP units so any residual sparse gradient activity stays localised; and skipping the FSDP-level `MixedPrecision` leaves embedding params in fp32 while the trainer's outer `torch.autocast` still runs the forward in `training.mixed_precision`. Both `moe_everything_fully_independent` and `moe_everything_precompute_kv` pass the 7-variant × 2-strategy subprocess smoke (`tests/test_trainer_distributed_smoke.py`) under `--dist-strategy fsdp`.

### None (single GPU)

```bash
python scripts/train.py --config config.yaml --dist-strategy none
```

## Trainer banner — auditing the effective wrapper

At startup the trainer prints (on rank 0):

```
============================================================
  Model     : <model_type>
  Params    : <N.NNN>B total
  Strategy  : <cli-flag>        # the `--dist-strategy` value the user passed
  Wrapper   : <effective>       # the wrapper actually attached to the model
  Precision : <bf16|fp16|...>
============================================================
```

- `Strategy` is the raw CLI flag (`ddp` / `fsdp` / `none`).
- `Wrapper` is the source of truth for the effective distributed runtime. Possible values:
  - `DDP` — `DistributedDataParallel` attached.
  - `FSDP(<STRATEGY>, auto_wrap=yes|no)` — a real FSDP wrapper with the named `ShardingStrategy`; `auto_wrap=yes` iff any nested module is itself an FSDP unit (i.e., a non-default `auto_wrap_policy` was supplied).
  - `none` — the model runs unwrapped (single-GPU / `--dist-strategy none`).

The `Wrapper` line is produced by `src/training/distributed.py::describe_wrapper(model)`. Operators should use the `Wrapper` value, not `Strategy`, when auditing a run — `Strategy` only records the user's intent, whereas `Wrapper` reports what `wrap_model(...)` actually constructed. The subprocess smoke at `tests/test_trainer_distributed_smoke.py` asserts the banner's `Wrapper` line against the expected value for every (family, strategy) pair so a silent policy regression fails fast.

## Modal Multi-Node

`modal_train.py` provides multi-node training on Modal cloud infrastructure:

```bash
modal run modal_train.py --config configs/scaling/m_standard.yaml
```

Configuration at top of `modal_train.py`:

- `N_NODES`: Number of containers
- `GPUS_PER_NODE`: GPUs per container (default 8)
- `GPU_TYPE`: B200, H200, or H100
- `TIMEOUT_HOURS`: Max wall-clock time

The launcher uses `torchrun` with RDMA-enabled NCCL communication. The exact command shape is built by `modal_train.build_torchrun_invocation(...)` and pinned by `tests/test_modal_launcher.py`.

## Implementation

`src/training/distributed.py` provides:

- `setup_distributed()` — Initialize process group, set devices, return rank/device info
- `cleanup_distributed()` — Destroy process group
- `wrap_model(model, *, strategy, local_rank, mixed_precision_name, model_type)` — Apply DDP or FSDP wrapping; selects the per-family FSDP policy shown in §FSDP above
- `unwrap_model(model)` — Get underlying model from the wrapper
- `describe_wrapper(model)` — Human-readable wrapper summary for the trainer banner (see §Trainer banner)
- `barrier()` — Distributed barrier with CUDA device
- `reduce_scalar()` — All-reduce a scalar value with mean/sum
- `seed_everything()` — Deterministic seeding with per-rank offset

## Checkpoints in Distributed

- DDP: Checkpoint saved on rank 0 only; model state taken from `unwrap_model()`.
- FSDP: Uses `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` + `FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)` so the on-disk checkpoint is a single consolidated state dict regardless of sharding strategy. Under `NO_SHARD` the state dict is already unsharded; under `FULL_SHARD` FSDP gathers before writing.
- Resume: `src/training/checkpoint.py::load_checkpoint` supports the current separate-files layout (`model.pt` + `optimizer_{adam,muon}.pt` + `training_state.pt` + `data_state.pt`), a safetensors variant (`model.safetensors`), and the legacy monolithic `trainer.pt` for backward compatibility.
