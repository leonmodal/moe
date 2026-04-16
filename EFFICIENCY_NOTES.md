# Efficiency And Correctness Notes

This note summarizes correctness fixes and throughput work done during the codebase reorganization. Sections labelled "historical" describe the state at the time of the original benchmark run and may have been superseded by later rounds — see `docs/architecture.md` and `docs/training.md` for the current state.

## 1. Correctness fixes

### GQA restoration across experiments

The baseline configs are back on their original GQA ratios:

- `standard_moe` and `global_moe`: `16` attention heads / `8` KV heads
- `xs` and retrofit configs: `16 / 8`
- `m` and `s` scaling configs: `16 / 4`

The per-head configs were also moved onto GQA so the comparisons are no longer mixing baseline GQA with per-head MHA.

### `per_head_precompute_kv` now routes KV groups, not individual Q heads

The main architectural fix is in `per_head_precompute_kv`.

Old behavior:

- route `num_attention_heads` slots
- use one routed Q-head per KV group as the KV representative

New behavior:

- route `num_key_value_heads` slots
- each selected expert emits:
  - one KV head
  - the full grouped-query slice associated with that KV head
- `repeat_kv` expands the routed KV heads back over the grouped query heads

This is the correct GQA-shaped analog of the baseline Qwen attention layout.

### Fresh-init loss verification

Fresh-init CE was rechecked in clean processes because Liger kernel patching is stateful. If a full Liger patch for standard/global is applied in the same interpreter before evaluating `moe_everything`, the `swiglu` monkeypatch can leak into the MoE-Everything run and produce a misleadingly bad loss.

Measured fresh-init losses on one batch:

| Config | Liger mode | Loss | CE |
| --- | --- | ---: | ---: |
| `standard_moe.yaml` | full | 12.1851 | 12.1851 |
| `global_moe.yaml` | full | 12.1550 | 12.1550 |
| `moe_everything_per_head_precompute_kv_sanity.yaml` | partial | 12.2261 | 12.2260 |
| `moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml` | partial | 12.1318 | 12.1316 |
| `moe_everything_per_head_independent_perlayer_prenorm.yaml` | partial | 12.1689 | 12.1684 |

These are all in the expected `~12.x` range.

## 2. Throughput work

### Grouped GEMM for expert projections

The per-expert Python loop in `AttentionExpertBank` was replaced with a grouped matmul path:

- uses `torch._grouped_mm` when available on CUDA with fp16/bf16 inputs
- falls back to the old per-expert loop if grouped GEMM is unavailable
- applies to the shared projection path and the pair-input projection path

This removes a major Python-side bottleneck when many experts are active.

### Faster MLP expert dispatch

The shared `Qwen3MoeExperts` MLP block now uses the same sort-by-expert dispatch structure:

- flatten `(token, topk_slot)` assignments once
- sort them by expert id
- run gate/up and down projections on contiguous per-expert chunks
- accumulate back with a single `index_add_`
- try `torch._grouped_mm` first, then fall back to chunked `F.linear` calls

On the current B200 stack, the runtime did not accept the `torch._grouped_mm` path for this MLP shape during the benchmark probe, so the observed speedup came from the better dispatch structure rather than from a confirmed grouped kernel hit.

### Faster sparse `per_head_precompute_kv`

The sparse precompute-KV path no longer loops over KV heads and launches a separate SDPA for each one.

Instead it now:

- builds fresh sparse grouped-Q and KV projections once
- expands routed K/V with `repeat_kv`
- runs a single SDPA per batch slice over all query heads
- applies the expanded KV routing weights afterward

This keeps the grouped-query semantics while removing the redundant per-KV-head attention work.

### Better sparse/dense auto dispatch

Under the restored GQA geometry, both per-head modes now use the same auto cutoff:

- `per_head_fully_independent`: sparse below `0.75`, dense at higher density
- `per_head_precompute_kv`: sparse below `0.75`, dense at higher density

The earlier MHA-tuned thresholds no longer matched the GQA benchmark results.

For the current experiment YAMLs, the per-head configs are pinned to:

- `per_head_compute_mode: dense`

That keeps the actual training runs on one execution path. Dense was pinned for simplicity and lower implementation risk, not because it is always the fastest option at every routing density. The sparse path remains available for benchmarking and regression tests.

### Consistent router masking

Dense mixed-path execution attaches the token mask back into `last_router_info` and zeroes masked entries consistently. This keeps dense and sparse router logging aligned for tests and debugging.

## 3. Benchmarks

Commands used:

```bash
uv run python scripts/benchmark_per_head_attention_dispatch.py --mode per_head_precompute_kv --batch-size 2 --seq-len 512 --hidden-size 1024 --head-dim 128 --num-heads 16 --num-kv-heads 8 --num-attn-experts 256 --warmup 2 --iters 4
uv run python scripts/benchmark_per_head_attention_dispatch.py --mode per_head_fully_independent --batch-size 2 --seq-len 512 --hidden-size 1024 --head-dim 128 --num-heads 16 --num-kv-heads 8 --num-attn-experts 256 --warmup 2 --iters 4
```

### `per_head_precompute_kv`, batch 2, seq 512, GQA `16/8`

| attn_frac | sparse_sec | dense_sec | faster |
| ---: | ---: | ---: | --- |
| 0.125 | 0.2163 | 0.2343 | sparse |
| 0.250 | 0.2344 | 0.2400 | sparse |
| 0.500 | 0.2358 | 0.2385 | sparse |
| 0.750 | 0.2468 | 0.2422 | dense |
| 1.000 | 0.2444 | 0.2404 | dense |

### `per_head_fully_independent`, batch 2, seq 512, GQA `16/8`

| attn_frac | sparse_sec | dense_sec | faster |
| ---: | ---: | ---: | --- |
| 0.125 | 0.1155 | 0.1505 | sparse |
| 0.250 | 0.1428 | 0.1468 | sparse |
| 0.500 | 0.1313 | 0.1358 | sparse |
| 0.750 | 0.1361 | 0.1351 | dense |
| 1.000 | 0.1298 | 0.1267 | dense |

### `Qwen3MoeExperts` MLP block, 2048 tokens, 256 experts, top-4 routing

Command used:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY'
# benchmark script omitted here for brevity; see shell history in this workspace
PY
```

Observed results:

| path | sec_per_iter | note |
| --- | ---: | --- |
| legacy per-expert loop | 0.0553 | one-hot + `where` + per-expert loop |
| new grouped dispatch | 0.0168 | same outputs, faster dispatch |

The measured max difference against the legacy reference was `0.0`.

## 4. What was not implemented in the benchmark pass

### Expert parallelization

Expert parallel all-to-all / sharded expert execution was not implemented during this benchmarking work. TorchTitan and Megatron-LM references were inspected for the design patterns, but the active repo runs experts locally rather than with true expert-parallel dispatch. This remains open follow-up work.

## 5. Distributed training (current state)

The original benchmark pass ran under HuggingFace Accelerate. Post-reorganization, the active trainer is torch-native and the Accelerate dependency is gone:

- Entrypoint: `scripts/train.py`
- Distributed wiring: `src/training/distributed.py` (DDP, FSDP, single-GPU) selected via `--dist-strategy ddp|fsdp|none`
- Modal multi-node: `modal_train.py` launches `scripts/train.py` via `torchrun` under `@modal.experimental.clustered(size=N_NODES, rdma=True)`

See `docs/distributed.md` for the active API surface and `MULTINODE_README.md` for the Modal launcher specifics.

## 6. Regression checks

The active test suite covers the per-head attention and router geometry items in this note:

- the 7-variant trainer smoke matrix (`tests/test_unified_trainer.py`): forward / backward / optimizer / checkpoint round-trip / loss decrease for dense, standard_moe (softmax/deepseek), global_moe (softmax/deepseek), moe_everything (`per_head_fully_independent`, `per_head_precompute_kv`)
- aux-loss semantics (`tests/test_aux_loss_fix.py`, `tests/test_seq_loss_vs_paper.py`, `tests/test_integration.py`)
- routing stats and plots (`tests/test_routing_stats.py`, `tests/test_routing_plots.py`)

Command:

```bash
uv run pytest -q
```
