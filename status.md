# Status

Updated: 2026-04-01 UTC

## Current State

- The main correctness and stability issues surfaced during the per-head MoE-Everything work are fixed locally.
- The three target per-head configs from `scripts/launch_all.sh` have now been validated on 8 GPUs with the exact model settings, using DDP, `bf16`, gradient checkpointing, and W&B disabled.
- The codepath is in a materially better state than the older notes in this repo suggested.
- I would still not claim that every config in the repo is fully proven; the validation below is specific to the tested configs and codepaths.

## Key Fixes

- `src/models/mixture_of_everything.py`
  - restored proper GQA handling for the per-head precompute-KV path
  - pinned the main per-head experiment configs to `per_head_compute_mode: dense`
  - added grouped expert matmul dispatch for the custom attention expert projections
  - added router fp32 enforcement under CUDA autocast for attention-bank routing paths
- `src/models/modeling_qwen3_moe.py`
  - router math now stays in fp32 under autocast
- `src/models/router.py`
  - DeepSeek router gating math now runs with autocast disabled so router logits/probs stay fp32 in `bf16` training
- `train.py`
  - fixed the global bias-update crash caused by mixing routers with different expert-pool sizes
  - global bias updates now pool counts per logical router family instead of stacking unrelated `[256]` and `[128]` tensors together
  - `moe_everything` still uses only the validated-safe subset of Liger patches
- `.venv/.../transformers/models/qwen3_moe/modeling_qwen3_moe.py`
  - patched the runtime Qwen3-MoE router path to keep router math in fp32 under autocast
- `tests/test_models.py`
  - added regression coverage for router fp32 behavior under CUDA autocast
  - added regression coverage for grouped MLP dispatch parity
  - added regression coverage for mixed-size global bias updates
  - updated per-head routing and config expectations

## Correctness Status

What is validated:

- router logits/probabilities are computed in fp32 during `bf16` mixed-precision training
- grouped expert dispatch changes preserve model outputs within normal floating-point tolerance
- the mixed `[256]` / `[128]` global bias-update failure is fixed
- the three target per-head configs run on 8 GPUs without OOM or crashes for at least 10 exact steps each

What is not fully proven:

- every config in the repo
- a full 16-GPU run in this local environment
- long-horizon training quality beyond the short exact runs listed below
- expert parallelism, which is still not implemented

## Exact 8-GPU Validation

All runs below used the real launch-config model shape:

- `num_hidden_layers: 32`
- `hidden_size: 1024`
- `num_attention_heads: 16`
- `num_key_value_heads: 8`
- `num_experts: 256`
- `batch_size: 32`
- `gradient_accumulation: 2`
- `seq_len: 1024`
- `mixed_precision: bf16`
- `gradient_checkpointing: true`

Runs:

- `configs/scaling/moe_everything_per_head_independent_perlayer_prenorm_8gpu_100_nowandb.yaml`
  - reached step `21`
  - no OOM
  - no distributed crash
  - loss moved from about `12.1308` to about `11.9740`
- `configs/scaling/moe_everything_per_head_precompute_kv_perlayer_prenorm_8gpu_100_nowandb.yaml`
  - reached step `10`
  - no OOM
  - no distributed crash
  - loss moved from about `12.1307` to about `12.0899`
- `configs/scaling/moe_everything_per_head_precompute_kv_sanity_8gpu_100_nowandb.yaml`
  - reached step `14`
  - no OOM
  - no distributed crash
  - loss moved from about `12.1562` to about `11.3326`
  - final `KeyboardInterrupt` traces were from manual stop after the target step count, not a model failure

Operational summary:

- DDP/NCCL startup worked
- `bf16` mixed precision worked
- gradient checkpointing worked
- no NaNs were observed in these runs
- the tested runs fit on local `NVIDIA B200` GPUs without the broken Liger `swiglu` patch

## Performance Notes

The exact configs are slow because they are large real runs, not debug runs:

- global tokens per optimizer step are high:
  - `32 batch/gpu * 8 gpu * 1024 seq_len * 2 grad_accum = 524,288 tokens/step`
- gradient checkpointing trades memory for substantial recompute cost
- per-layer routed attention and MoE dispatch add real indexing/sorting/scatter overhead
- `moe_everything` only uses the safe subset of Liger patches
  - `rope` and `rms_norm` are enabled
  - `swiglu` is disabled
  - fused CE is not wired into `MoEverythingForCausalLM`

Observed behavior on the exact runs:

- the two learned per-head runs were roughly `100 sec/step`
- the sanity config was much faster because its alternating-global schedule is simpler than the full learned branch-routing path

## Liger Status

- Full Liger on learned-routing `moe_everything` is still broken
- The bad component is the `swiglu` expert replacement
- The validated setup for `moe_everything` remains:
  - keep Liger `rope`
  - keep Liger `rms_norm`
  - disable Liger `swiglu`
  - do not rely on the stock fused linear CE patch for this model family

## Validation

- `uv run pytest -q tests/test_models.py -k 'stores_seq_aux_coef_from_config or alternating_global_sanity_uses_parameterless_branch_router or alternating_global_sanity_bf16_attention_path_matches_global_moe'`
- result: `4 passed, 140 deselected`
- `uv run python` single-process mapped-init check on one CUDA batch:
  - global and sanity total loss match exactly after the seq-aux fix
  - embed grad max diff: `3.65e-07`
- `uv run python` manual 8-shard parquet average (no DDP, no bias updates), 2 optimizer steps:
  - step 0: exact fp32 loss match, embed grad diff `1.19e-07`
  - step 1: fp32 loss diff `9.54e-07`, embed grad diff `4.74e-06`
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 2 --data-mode parquet --batch-size 1 --seq-len 128 --bias-update-rate 0 --static-graph off`
- result (`fp32`):
  - step 0: exact loss match, grad diff `9.97e-08`
  - step 1: loss diff `2.9e-05`, zero selected-expert mismatches on rank 0
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 3 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --bias-update-rate 0 --static-graph off`
- result (`bf16`):
  - step 0: `|g-s|=0.002599`
  - step 1: `|g-s|=0.004048`
  - step 2: `|g-s|=0.001905`
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 1000 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing config --report-every 100`
- result (`bf16`, config checkpointing settings):
  - no catastrophic blow-up through 1000 DDP steps
  - representative CE diffs: step 100 `0.055992`, step 500 `0.010468`, step 900 `0.074997`, step 999 `0.012322`
  - worst observed CE diff in that run: `2.535876` at step 15
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config /tmp/moe/tmp_configs/interp_4_layers/global_moe.yaml --sanity-config /tmp/moe/tmp_configs/interp_4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 500 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 50`
- result (`4_layers`, `bf16`, real parquet data, interpolation restored with warmup `5000`):
  - step 0: `12.126092` vs `12.127457` (CE diff `0.001365`)
  - step 50: `7.744940` vs `7.673828` (CE diff `0.071112`)
  - step 100: `8.201558` vs `8.174387` (CE diff `0.027171`)
  - step 150: `7.815928` vs `7.811861` (CE diff `0.004067`)
  - step 200: `7.640862` vs `7.655036` (CE diff `0.014175`)
  - step 250: `7.710170` vs `7.687034` (CE diff `0.023136`)
  - step 300: `7.395101` vs `7.311156` (CE diff `0.083945`)
  - step 350: `7.696858` vs `7.631919` (CE diff `0.064939`)
  - step 400: `7.619305` vs `7.556596` (CE diff `0.062709`)
  - step 450: `7.654131` vs `7.618386` (CE diff `0.035745`)
  - step 499: `7.613384` vs `7.516917` (CE diff `0.096467`)
  - worst observed CE diff: `0.653481` at step `12`
  - note: this parity run uses the real 4-layer configs and real parquet data, but keeps the standard parity-harness batch shape (`batch_size=1`, `seq_len=128`) rather than the full training config token load
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing off --bias-update-rate 0 --report-every 20`
- result (`bf16`, checkpointing off, bias updates off):
  - step 99 CE diff: `0.078086`
  - worst observed CE diff in that run: `1.916025` at step 16
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 1 --seq-len 128 --gradient-checkpointing off --bias-update-rate 0 --report-every 20`
- result (`fp32`, checkpointing off, bias updates off):
  - step 0: exact CE/loss match
  - step 99 CE diff: `0.032928`
  - worst observed CE diff in that run: `2.108144` at step 11
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/8_layers/global_moe.yaml --sanity-config configs/depth_matched/8_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 2 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --report-every 1`
- result (`8_layers`, `bf16`):
  - step 0 CE diff: `0.002402`
  - step 1 CE diff: `0.016558`
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/8_layers/global_moe.yaml --sanity-config configs/depth_matched/8_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 10`
- result (`8_layers`, `bf16`, mapped init, pure global bias updates with `alpha=0`):
  - step 0: `12.095649` vs `12.093246` (CE diff `0.002402`)
  - step 10: `9.093688` vs `9.015924` (CE diff `0.077764`)
  - step 20: `8.482841` vs `8.366564` (CE diff `0.116278`)
  - step 30: `8.297062` vs `8.420064` (CE diff `0.123002`)
  - step 40: `8.384623` vs `8.397344` (CE diff `0.012721`)
  - step 50: `7.737661` vs `7.892931` (CE diff `0.155270`)
  - step 60: `7.416779` vs `7.483973` (CE diff `0.067194`)
  - step 70: `8.026033` vs `7.991763` (CE diff `0.034270`)
  - step 80: `7.730869` vs `7.693813` (CE diff `0.037056`)
  - step 90: `7.917434` vs `7.901228` (CE diff `0.016205`)
  - step 99: `7.850127` vs `7.856890` (CE diff `0.006763`)
  - worst observed CE diff: `1.080865` at step `17`
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/16_layers/global_moe.yaml --sanity-config configs/depth_matched/16_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 2 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --report-every 1`
- result (`16_layers`, `bf16`):
  - step 0 CE diff: `0.002459`
  - step 1 CE diff: `0.012911`
- real 8-GPU `train.py` smoke runs on temporary 2-step copies of the production configs under `configs/depth_matched/4_layers`:
  - `global_moe.yaml` completed
  - `standard_moe.yaml` completed
  - `moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml` completed
  - `moe_everything_per_head_independent_perlayer_prenorm.yaml` completed
  - `moe_everything_per_head_precompute_kv_sanity.yaml` completed

## Alternating Global Sanity Investigation

- Fixed a real bf16 mismatch in the per-head bank path:
  - the custom Q/K RMSNorm path did not match `Qwen3MoeRMSNorm` numerics
  - the grouped-matmul fallback also upcast bf16 projection outputs into a fp32 buffer before norm, which hid the true projection dtype and made the mismatch worse
- Fixed an objective mismatch outside `train.py`:
  - `build_model()` was not propagating `seq_aux_loss_coef` into the standard/global wrappers
  - `moe_everything` already had it, so side-by-side diagnostics were comparing different total losses
- Removed the unused learned branch router from `alternating_global_moe`:
  - branch routing is deterministic in this sanity mode
  - leaving a trainable `branch_router.gate.weight` registered created an unused DDP parameter
- Added `scripts/compare_global_sanity_stepwise_ddp.py` for side-by-side DDP parity checks on real 8-GPU runs, with overrides for `static_graph`, LR, and bias-update rate.
- Aligned the parity-harness config to the baseline training mode:
  - `moe_everything_per_head_precompute_kv_sanity.yaml` now keeps `gradient_checkpointing: false` by default, matching `global_moe.yaml`
- Applied the same checkpointing alignment to the `8_layers` and `16_layers` sanity harness configs.
- After that fix, mapped-init alternating-global sanity now matches the global model exactly through:
  - Q projection
  - K projection
  - V projection
  - attention output
- The remaining non-equivalence is no longer a DDP correctness failure, but it is still enough to create transient CE spikes during longer parity runs.

## Ready To Run

- The per-head model error that caused the catastrophic 8-GPU drift is resolved.
- The fix is not limited to the sanity harness:
  - the bf16 Q/K bank fix is in the shared per-head implementation
  - the DDP `static_graph=False` fix applies to real `moe_everything` training
- The configs under `configs/depth_matched/4_layers` are runnable with the current codebase.
- `moe_everything_per_head_precompute_kv_sanity.yaml` is still only a parity harness, not the real target architecture.
- The sanity config now defaults to the same checkpointing mode as the global baseline so parity runs are cleaner.
- For real per-head training, the main configs to use are:
  - `moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`
  - `moe_everything_per_head_independent_perlayer_prenorm.yaml`

## 8-GPU DDP Findings

- The huge step-1 split was a DDP configuration bug, not a forward-path parity failure.
- Manual 8-shard averaging in one process stayed aligned, which ruled out the model math, optimizer, and bias updates.
- With `static_graph=False`, mapped-init DDP checks stay aligned through the first steps:
  - exact at step 0 in `fp32`
  - close in early-step `bf16`
- With `static_graph=True`, the same mapped-init run diverged immediately after step 0.
- Disabling `static_graph` also exposed the real DDP contract violation:
  - the alternating-global sanity model still had an unused `branch_router.gate.weight`
  - after removing that parameter, DDP with `static_graph=False` runs cleanly
- On longer 100-1000 step runs, the pair remains stable but not stepwise-identical:
  - later CE values stay in the same ballpark
  - transient early CE spikes can still appear, even with checkpointing off and bias updates off

## Current Read

- Gradient checkpointing is not the primary cause here.
- Liger is not the primary cause of the mapped-init DDP failure.
- The fixed bank bug was real and removed the Q/K/V/attention mismatch.
- The catastrophic loss drift came from running `moe_everything` under DDP with `static_graph=True`.
- The deterministic sanity mode also incorrectly registered an unused learned branch router, which made the DDP setup invalid.
- After fixing the objective mismatch, removing the unused branch router, and disabling `static_graph` for `moe_everything`, the catastrophic DDP failure is gone.
- The alternating-global sanity harness is now good for implementation/regression checks, but not for requiring near-identical CE at every training step over long runs.
- In practice the global baseline and sanity harness remain in the same training regime over 1k DDP steps, but they can show transient CE gaps that are much larger than the first-step mapped-init deltas.

## Same-Init And Bias-Update Policy

- Mapped init is now a first-class training option instead of living only in debug scripts:
  - added `src/models/init_mapping.py` with the shared `global -> alternating_global_moe sanity` weight copy
  - `train.py` now supports:
    - config-driven initialization via a top-level `initialization:` block
    - CLI overrides via `--init-from-config` and `--init-strategy`
- The depth-matched sanity configs now default to mapped init from their sibling global configs:
  - `configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml`
  - `configs/depth_matched/8_layers/moe_everything_per_head_precompute_kv_sanity.yaml`
  - `configs/depth_matched/16_layers/moe_everything_per_head_precompute_kv_sanity.yaml`
- Bias updates are now pinned to pure pooled/global behavior across all depth-matched `4_layers`, `8_layers`, and `16_layers` configs:
  - `bias_interpolation: false`
  - `global_router_update: true`
- In other words:
  - global pooling is always enabled for these configs
  - alpha is always `0`
  - there is no longer any per-layer/global interpolation path in the depth-matched runs

## 8-Layer Launch Validation

- Environment: single node, `8 x NVIDIA B200`, real `train.py`, parquet data, actual `8_layers` config settings (`batch_size=32`, `gradient_accumulation=2`, `seq_len=1024`, `bf16`)
- Added `--max-steps` override to `train.py` so these smokes can exit cleanly after a real optimizer step count.
- `uv run pytest -q tests/test_models.py -k 'stores_seq_aux_coef_from_config or alternating_global_sanity_matches_global_moe or alternating_global_sanity_bf16_attention_path_matches_global_moe or alternating_global_sanity_uses_parameterless_branch_router'`
  - result: `5 passed`
- `uv run python` config-resolution check:
  - all three sanity configs resolve `initialization.source_config` to their matching absolute `global_moe.yaml` path
- Real 8-GPU smokes (`--max-steps 2`) all completed without OOM:
  - `configs/depth_matched/8_layers/global_moe.yaml`
    - step 1 CE `12.1447`
    - step 2 CE `12.1474`
    - observed memory: about `46.8 GiB/GPU` (`46790 MiB`)
  - `configs/depth_matched/8_layers/standard_moe.yaml`
    - step 1 CE `12.1431`
    - step 2 CE `12.1452`
    - observed memory: about `35.9 GiB/GPU` (`36772 MiB`)
  - `configs/depth_matched/8_layers/moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`
    - step 1 CE `12.1501`
    - step 2 CE `12.1508`
    - observed memory: about `116.0 GiB/GPU` (`118800 MiB`)
  - `configs/depth_matched/8_layers/moe_everything_per_head_independent_perlayer_prenorm.yaml`
    - step 1 CE `12.1615`
    - step 2 CE `12.1600`
    - observed memory: about `119.0 GiB/GPU` (`121844 MiB`)
  - `configs/depth_matched/8_layers/moe_everything_per_head_precompute_kv_sanity.yaml`
    - mapped init log: `85 mapped tensors`
    - step 1 CE `12.1306`
    - step 2 CE `12.1316`
    - observed memory: about `167.5 GiB/GPU` (`171532 MiB`)
- Read on `8_layers`:
  - the real training configs are comfortably below the B200 limit
  - the sanity harness also fits on B200, but it is much closer to the ceiling because it doubles depth to emulate 8 logical layers
  - if the choice is between `4_layers` and `8_layers` for real per-head training on this hardware, `8_layers` is good to run
  - the `8_layers` sanity config is also runnable here, but it has much less headroom than the real 8-layer models

## Remaining Open Items

- No expert parallel implementation yet
- No full 16-GPU local validation in this environment
- No claim yet that every repo config is fully covered
- If we need faster real runs, the next likely gains are:
  - reduce per-step logging/synchronization overhead
  - integrate a proper fused CE path for `MoEverythingForCausalLM`
  - continue improving expert dispatch efficiency
