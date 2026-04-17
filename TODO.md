# Outstanding TODO

Carry-over from the multi-audit pass (commit `a96f9ef` closed the first batch of 5 items). Each entry below is a concrete bug or perf win surfaced by the trainer / data / distributed audits. Tier = priority, not an SLA.

## Tier 1 — correctness / silent-bug risks

- [ ] **`exploration_rate_schedule` cache ineffective during warmup** (`src/training/trainer.py`, ~l. 283–293). The `if current_rate != last_applied_exploration_rate` guard is almost always true during the ramp because `frac = step/warmup_steps` produces a distinct float each step. Effect: `apply_router_exploration_rate` walks `model.modules()` every step until warmup converges. Fix: skip the walk when `step >= warmup_steps and warmup_steps > 0` and the rate has already plateaued at `target`; additionally, avoid the walk when nothing has actually changed.

- [ ] **`collect_router_z_loss` leaves stale `_last_z_loss` after forward** (`src/models/router.py`, `src/training/routing.py::collect_router_z_loss`). With activation checkpointing or when a router's forward is skipped on a micro-batch, the previously cached tensor is still attached and double-counts. Fix: router forward clears `_last_z_loss` before recomputation; the trainer consumes and detaches after `collect_router_z_loss`.

- [ ] **`get_selected_experts_for_seq_aux` swallows all exceptions** (`src/training/metrics.py:14-33`). A typo or refactor that moves `layer.mlp.gate` silently degrades `seq_aux_loss` to 0 without a warning. Fix: narrow to `(AttributeError, TypeError)` and log-once when it fires so refactors don't silently break telemetry.

- [ ] **FSDP `buffer_dtype=bf16` drifts RoPE positions** (`src/training/distributed.py:108-118`). `_build_fsdp_mixed_precision` casts buffers to the param dtype, which includes RoPE cos/sin and RMSNorm epsilons. Megatron keeps buffers fp32 as a default stability knob. Fix: always pass `buffer_dtype=torch.float32`.

- [ ] **Safetensors optim load silently restarts** (`src/training/checkpoint.py:179-180`). Missing optimizer state during safetensors resume prints a warning and starts fresh — momentum and moment estimates are lost. Fix: require the user to pass an explicit `--reset-optimizer` flag (or similar) to opt into the fresh-start path; default should fail loud.

- [ ] **Eval shard drift / rank stall under unequal parquet files** (`src/data/parquet_dataset.py`). Ranks get distinct file counts and will reach the `reduce_scalar` barrier at different batch counts; with very few shards a rank can legitimately see zero files. Fix: round-robin pad to `ceil(N/W)*W` (repeat files) or truncate to `(N // W) * W`.

## Tier 2 — per-step perf

- [ ] **Batch `reduce_scalar` into one collective per log step** (`src/training/trainer.py:367-377`, `src/training/distributed.py::reduce_scalar`). Currently 7–8 separate `all_reduce` + `.item()` calls per step (one per window_metric + grad_norm). Fix: concat into a single pre-allocated fp32 tensor, one `all_reduce`, one `.tolist()`; additionally gate on `step % log_every == 0` so non-log steps do no cross-rank sync at all.

- [ ] **Batch `.item()` calls in `compute_output_metrics`** (`src/training/metrics.py:66, 78, 87, 91, 104, 111, 113`). 7 host↔device syncs per micro-batch. Fix: stack into one tensor, `.tolist()` once; or defer entirely until the log boundary.

- [ ] **`update_expert_biases` redundant `.sum()` + host-branch** (`src/training/routing.py:59-60, 73-74`). `counts.sum() > 0` is a host-side bool on a device tensor — each call is a CPU↔GPU sync. Fix: compute `total = counts.sum()` once per call and reuse; avoid the host-side branch where possible (e.g. `loads * (total > 0).float()`).

- [ ] **Parquet `token_buffer[:seq_len]` front-delete is O(n)** (`src/data/parquet_dataset.py:291-303`). Repeated front-delete shifts the remaining tokens every chunk. Fix: use `collections.deque`, or maintain an offset index and only compact when offset exceeds half the buffer.

- [ ] **Hoist `per_proj_rates` and `window_metrics` init above the step loop** (`src/training/trainer.py:~296-300, ~388-395`). Minor: avoid per-step dict rebuild.

## Out of scope for this cleanup

- `tests/test_logits_differ.py::test_param_init_comparison` `PytestReturnNotNoneWarning` — pre-existing, unchanged.
- `get_worker_info()` sub-sharding for `StatefulParquetDataset` — requires a new sharding protocol for multi-worker DataLoader; deferred since we force `num_workers=0` for stateful datasets anyway.

## Completed in `a96f9ef`

- Per-rank `data_state.pt` save/load (was silently duplicating rank 0's data across ranks).
- Atomic checkpoint write (`.tmp/` + rename; crash-safe resume).
- `run_validation` wrapped in `try/finally` so eval-path exceptions don't leave the model in `.eval()`.
- Trainer reuses `clip_grad_norm_`'s return value instead of a second parameter walk via `get_grad_norm`.
- GPU-smoke router-options test made deterministic (seed before `build_model`) and assertion loosened to the real AC-11 invariant (bounded + finite loss, not head-vs-tail trend on random-token data).
