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
- `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config /tmp/moe/tmp_configs/interp_4_layers/global_moe.yaml --sanity-config /tmp/moe/tmp_configs/interp_4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 2 --data-mode parquet --batch-size 32 --seq-len 1024 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 1`
- result (`4_layers`, `bf16`, real parquet data, interpolation restored, full training token load):
  - step 0: `12.083158` vs `12.083319` (CE diff `0.000160`)
  - step 1: `10.861225` vs `10.861650` (CE diff `0.000425`)
  - this paired `global + sanity` DDP run did fit on local `8 x B200` at the full config token load
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

## Important Parity-Check Scope Note

- Do not rely only on the reduced-load parity harness shape (`batch_size=1`, `seq_len=128`) when judging whether global-vs-sanity training parity is "good enough".
- That reduced-load check is still useful for implementation debugging, but it can materially understate drift relative to the actual training token load.
- For real parity calls, run:
  - real parquet data from `data/parquet`
  - the actual depth-matched config family
  - the actual training token load (`batch_size=32`, `seq_len=1024` for the current depth-matched configs)
- Evidence for why this matters:
  - the reduced-load `4_layers` interpolation-restored 500-step DDP run stayed in a modest-gap regime (for example step 100 CE diff `0.027171`, step 499 CE diff `0.096467`)
  - the full-load `4_layers` interpolation-restored DDP run on real parquet data showed much larger gaps early:
    - step 0 CE diff `0.000160`
    - step 50 CE diff `0.120424`
    - step 100 CE diff `0.343344`
    - step 150 CE diff `0.441594`
    - step 200 CE diff `0.449409`
- So the smaller parity-harness token load should be treated as a debugging signal, not the final answer for training-match quality.

## Full-Load Dataset Sensitivity Check

- I checked whether the large full-load `4_layers` parity gap is just a model-side effect or is strongly data-sensitive.
- The local packed parquet corpus does not look obviously malformed on a basic inspection:
  - `8192` parquet shards under `data/parquet`
  - sampled rows are normal natural-language documents
  - sampled row-length stats are in the same ballpark as an official FineWebEdu sample:
    - local packed parquet: char-length `p50 ~= 2998`, `p95 ~= 13567`
    - official FineWebEdu sample parquet: char-length `p50 ~= 2935`, `p95 ~= 13840`
- So this is not a trivial "empty rows" or "completely broken text column" issue.

- I then ran three full-load controls on `8 x B200`, `batch_size=32`, `seq_len=1024`, interpolation restored (`bias_interpolation: true`, `bias_interpolation_warmup_steps: 5000`, `global_router_update: true`):
  - current packed parquet (`data/parquet`):
    - step 0 CE diff `0.000160`
    - step 50 CE diff `0.120424`
    - step 100 CE diff `0.343344`
    - step 150 CE diff `0.441594`
    - step 200 CE diff `0.449409`
  - official FineWebEdu sample parquet generated from `HuggingFaceFW/fineweb-edu` `sample-10BT`:
    - added `scripts/create_official_finewebedu_sample_parquet.py` to materialize a clean local parquet sample
    - generated `data/parquet_finewebedu_official_sample_8x10k` with `8` shards x `10000` rows
    - step 0 CE diff `0.000099`
    - step 50 CE diff `0.008281`
    - step 100 CE diff `0.286312`
    - the run later died with a separate OOM / peer-memory failure after step 100, so there is not yet a clean step-200 number on this dataset
  - synthetic tokens:
    - step 0 CE diff `0.000331`
    - step 50 CE diff `0.000346`
    - step 99 CE diff `0.003066`
    - summary max CE diff `0.011969 @ step 9`

- Current read:
  - the large full-load drift is strongly data-sensitive
  - it is not explained purely by "large batch/sequence length" because synthetic stays tightly matched
  - it is also not explained purely by "the packed parquet is corrupted" because the official FineWebEdu sample still drifts under real text, just less severely early
  - the local packed parquet corpus is therefore a likely amplifier of the parity gap, not yet proven to be the sole root cause
  - the next data-side audit should focus on corpus composition / ordering / duplication differences in `leonli66/latent-cot-finewebedu`, not just row-format correctness

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

## 4-Layer Shared-Attention BF16 Parity Fix

- Root cause of the remaining `global_moe` vs `alternating_global_moe` sanity drift:
  - the shared `AttentionExpertBank._run_attention(...)` helper in `mixture_of_everything` was not using the same backend semantics as baseline Qwen attention
  - on the exact `global_moe` vs `alternating_global_moe` sanity path, that meant bf16 backward was going through a different SDPA/GQA path than the baseline model, which left step-0 forward exact but introduced small `v_proj` / `o_proj` gradient differences that amplified after the first optimizer step under real DDP
  - independently, the same shared helper had a real CUDA bug for sparse-query calls: when `query_positions` was provided and `attention_mask` was `None`, the GPU path ignored `query_positions`, so sparse per-head attention could silently lose causal masking
- Fix applied in `src/models/mixture_of_everything.py`:
  - `_run_attention(...)` is now the shared source of truth and always uses the Transformers attention backend selection (`ALL_ATTENTION_FUNCTIONS.get_interface(..., eager_attention_forward)`) instead of maintaining a second custom SDPA implementation inside `mixture_of_everything`
  - `_run_attention(...)` now builds an explicit causal mask from `query_positions` before dispatch, so the sparse per-head CUDA paths preserve causality correctly
  - the backend helper normalizes outputs back to the internal `mixture_of_everything` layout, so the rest of the bank logic stays unchanged
  - the alternating-global logical-dense sanity branch still uses `mixture_of_everything`’s shared helper; it is no longer special-cased onto a separate attention runner
- Sanity/unit coverage:
  - `uv run pytest -q tests/test_models.py -k 'per_head_dense_and_sparse_dispatch_match or run_attention_respects_query_positions_on_cuda or alternating_global_sanity_matches_global_moe or alternating_global_sanity_bf16_attention_path_matches_global_moe or alternating_global_sanity_mixed_precision_bf16_stays_close_to_global_moe'`
  - result: `6 passed, 139 deselected`
- Single-process bf16 train-mode check after the fix:
  - step 0: loss diff `0.0`, logits diff `0.0`, grad diffs `0.0` for all tracked mapped pairs
  - step 1: loss diff `0.0`, logits diff `0.0`, grad diffs `0.0` for all tracked mapped pairs
- Real 8-GPU DDP parity is exact again on this machine (`8 x NVIDIA B200`) with the shared helper fix:
  - note: the side-by-side parity harnesses (`scripts/compare_global_sanity_stepwise_ddp.py` and `scripts/compare_global_sanity_stepwise_zero1.py`) build models directly and do not call `configure_liger_kernels()`, so the matched parity results below are already the no-Liger case
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 20 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 10`
    - steps `0`, `10`, and `19`: CE diff `0`, loss diff `0`, logits diff `0`, grad diff `0`, param diff `0`, optimizer diff `0`, selected-expert mismatches `0`
    - summary: `max_ce_diff=0.000000`, `max_loss_diff=0.000000`
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 20`
    - steps `0`, `20`, `40`, `60`, `80`, and `99`: CE diff `0`, loss diff `0`, logits diff `0`, grad diff `0`, param diff `0`, optimizer diff `0`, selected-expert mismatches `0`
    - summary: `max_ce_diff=0.000000`, `max_loss_diff=0.000000`
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 20 --data-mode parquet --batch-size 1 --seq-len 128 --amp-bf16 --gradient-checkpointing on --static-graph off --report-every 10`
    - steps `0`, `10`, and `19`: CE diff `0`, loss diff `0`, logits diff `0`, grad diff `0`, param diff `0`, optimizer diff `0`, selected-expert mismatches `0`
    - summary: `max_ce_diff=0.000000`, `max_loss_diff=0.000000`
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 2 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 1`
    - step `0`: CE diff `0`, loss diff `0`, logits diff `0`, selected-expert mismatches `0`, but grad/param/optimizer state were already slightly nonzero (`grad_max=1.02e-4`, `param_max=0.002`, `opt_max=1.02e-5`)
    - step `1`: CE diff `3.2e-5`, loss diff `3.4e-5`, logits max diff `0.635254`, grad max diff `5.01e-4`, param max diff `0.00395`, optimizer diff `5.06e-5`, selected-expert mismatches `5685`
    - local-rank-0 mismatches by layer at step `1`: `[123, 193, 262, 125]`
    - summary: `max_ce_diff=0.000032`, `max_loss_diff=0.000034`
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 300 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 25 --capture-logits off`
    - the `seq_len=1024` drift persists over a few hundred real matched steps
    - selected-expert mismatches climb to roughly `124k`-`130k` by the later checkpoints
    - representative CE diffs: step `25` `0.022995`, step `75` `0.239699`, step `150` `0.107307`, step `299` `0.073700`
    - summary: `max_ce_diff=4.398703@step12`, `max_loss_diff=4.398718@step12`
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 300 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 25 --capture-logits off --bias-update-rate 0`
    - disabling expert-bias updates does not eliminate the remaining `seq_len=1024` bf16 drift, but it makes the matched runs materially closer
    - representative CE diffs: step `25` `0.004215`, step `75` `0.015125`, step `150` `0.101271`, step `299` `0.050544`
    - summary: `max_ce_diff=0.683054@step12`, `max_loss_diff=0.683065@step12`
    - read: bias updates are a strong amplifier, not the sole root cause
  - added a separate ZeRO-1 parity harness at `scripts/compare_global_sanity_stepwise_zero1.py` so DeepSpeed can be tested side-by-side without the `train.py` Liger asymmetry
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_zero1.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 30 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing config --report-every 5 --capture-logits off`
    - ZeRO-1 does not fix the matched `seq_len=1024` drift on this node
    - representative CE diffs: step `5` `0.000879`, step `10` `0.001350`, step `15` `0.062858`, step `25` `0.032105`, step `29` `0.023605`
    - summary: `max_ce_diff=9.095173@step12`, `max_loss_diff=9.095204@step12`
    - note: this harness uses DeepSpeed's built-in bf16 path rather than the outer `torch.autocast`, so the internal numerical path is not identical to the DDP script; the practical result is still that ZeRO-1 did not remove the spike
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 30 --data-mode parquet --batch-size 1 --seq-len 1024 --gradient-checkpointing config --static-graph off --report-every 5 --capture-logits off`
    - fp32 removes the observed `seq_len=1024` drift on the tested window
    - steps `0`, `5`, `10`, `15`, `20`, `25`, and `29`: CE diff `0`, loss diff `0`, grad diff `0`, param diff `0`, optimizer diff `0`, selected-expert mismatches `0`
    - summary: `max_ce_diff=0.000000`, `max_loss_diff=0.000000`
  - added a symmetric safe-Liger wrapper at `scripts/compare_global_sanity_stepwise_ddp_safe_liger.py`
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp_safe_liger.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 30 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing config --static-graph off --report-every 5 --capture-logits off`
    - this applies the validated-safe Liger subset (`rope+rms_norm`, no `swiglu`, no fused CE) to both models symmetrically
    - unlike the no-Liger harness, exact parity is already lost at step `0`: CE diff `0.000161`, grad diff `7.65e-4`, selected-expert mismatches `1001`
    - later representative CE diffs: step `5` `0.001675`, step `10` `0.006875`, step `15` `0.007095`, step `25` `0.000382`, step `29` `0.012565`
    - summary: `max_ce_diff=0.180929@step12`, `max_loss_diff=0.180935@step12`
    - read: the safe Liger subset reduces the large mid-run bf16 spike relative to the no-Liger bf16 run, but it destroys exact step-0 parity, so it is not suitable for the sanity/global equivalence harness
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp_safe_liger.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 30 --data-mode parquet --batch-size 1 --seq-len 1024 --gradient-checkpointing off --static-graph off --report-every 5 --capture-logits off`
    - fp32 with the same symmetric safe-Liger subset is also not exact
    - step `0`: CE/loss still match, but grad diff `3.26e-08`, param diff `8.75e-05`, optimizer diff `3.26e-09`
    - representative CE diffs: step `5` `4.1e-05`, step `10` `9.85e-04`, step `15` `0.013474`, step `20` `0.015665`, step `29` `0.022394`
    - summary: `max_ce_diff=0.048881@step14`, `max_loss_diff=0.048882@step14`
    - read: unlike the no-Liger fp32 harness, safe-Liger fp32 is not exact, so Liger itself introduces a parity mismatch independent of bf16
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 4 --seq-len 1024 --amp-bf16 --gradient-checkpointing off --static-graph off --report-every 10 --capture-logits off`
    - larger micro-batching does not fix the long-run bf16 drift
    - representative CE diffs: step `10` `0.013202`, step `50` `0.043298`, step `80` `0.173048`, step `99` `0.232381`
    - summary: `max_ce_diff=11.774851@step12`, `max_loss_diff=11.774869@step12`
    - read: increasing per-GPU micro-batch from `1` to `4` makes late-step parity worse on the tested 100-step window
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing on --static-graph off --report-every 10 --capture-logits off`
    - gradient checkpointing does not remove the bf16 drift, but it reduces the early transient spike
    - representative CE diffs: step `10` `0.007825`, step `50` `0.015156`, step `60` `0.135933`, step `99` `0.237543`
    - summary: `max_ce_diff=3.154278@step12`, `max_loss_diff=3.154270@step12`
    - read: checkpointing helps the worst step-12 excursion relative to the no-checkpointing bf16 baseline, but late-step CE drift remains
  - added `--grad-accum-steps` to `scripts/compare_global_sanity_stepwise_ddp.py` for true parity-harness accumulation tests
  - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 100 --data-mode parquet --batch-size 1 --grad-accum-steps 4 --seq-len 1024 --amp-bf16 --gradient-checkpointing off --static-graph off --report-every 10 --capture-logits off`
    - true gradient accumulation is materially better than increasing micro-batch to `4` for the same effective batch size
    - representative CE diffs: step `10` `0.000192`, step `30` `0.077217`, step `60` `0.020740`, step `99` `0.059988`
    - summary: `max_ce_diff=1.357561@step12`, `max_loss_diff=1.357558@step12`
    - read: if a larger effective batch is needed, `batch_size=1` plus accumulation is significantly less harmful to parity than `batch_size=4`
  - added `MOE_EVERYTHING_DISABLE_GROUPED_MM=1` kill-switches in:
    - `src/models/mixture_of_everything.py` for the attention-bank grouped-matmul fast path
    - `src/models/modeling_qwen3_moe.py` for the shared `Qwen3MoeExperts` grouped-matmul fast path
  - apples-to-apples current bf16 baseline (no kill-switch):
    - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 30 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing off --static-graph off --report-every 5 --capture-logits off`
    - summary: `max_ce_diff=7.688423@step12`, `max_loss_diff=7.688437@step12`
  - same run with both grouped-matmul fast paths disabled:
    - `MOE_EVERYTHING_DISABLE_GROUPED_MM=1 uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 30 --data-mode parquet --batch-size 1 --seq-len 1024 --amp-bf16 --gradient-checkpointing off --static-graph off --report-every 5 --capture-logits off`
    - representative CE diffs improved at the reported checkpoints: step `10` `0.001384`, step `20` `0.011292`, step `25` `0.006100`, step `29` `0.043180`
    - summary: `max_ce_diff=0.729905@step12`, `max_loss_diff=0.729906@step12`
  - however, the grouped-matmul kill-switch is not a full fix:
    - focused reruns remained unstable and still showed catastrophic bf16 spikes
    - with bias updates enabled:
      - `MOE_EVERYTHING_DISABLE_GROUPED_MM=1 ... --steps 15 --report-every 1`
      - summary: `max_ce_diff=9.808236@step13`, `max_loss_diff=9.808259@step13`
    - with bias updates disabled:
      - `MOE_EVERYTHING_DISABLE_GROUPED_MM=1 ... --steps 15 --report-every 1 --bias-update-rate 0`
      - summary: `max_ce_diff=6.138495@step12`, `max_loss_diff=6.138523@step12`
    - read: grouped matmul is a major bf16 contributor/amplifier, but there is still at least one additional bf16-sensitive mismatch elsewhere
  - split the grouped-matmul kill-switch into:
    - `MOE_EVERYTHING_DISABLE_ATTN_GROUPED_MM=1` for the `mixture_of_everything` attention-bank grouped matmuls
    - `MOE_EVERYTHING_DISABLE_MLP_GROUPED_MM=1` for the shared `Qwen3MoeExperts` grouped matmuls
  - on the harsher `8 GPU`, `seq_len=1024`, `batch_size=4`, `steps=15`, bf16 no-checkpointing harness:
    - baseline:
      - `uv run torchrun --standalone --nproc_per_node 8 scripts/compare_global_sanity_stepwise_ddp.py --global-config configs/depth_matched/4_layers/global_moe.yaml --sanity-config configs/depth_matched/4_layers/moe_everything_per_head_precompute_kv_sanity.yaml --steps 15 --data-mode parquet --batch-size 4 --seq-len 1024 --amp-bf16 --gradient-checkpointing off --static-graph off --report-every 1 --capture-logits off`
      - summary: `max_ce_diff=12.763151@step12`, `max_loss_diff=12.763174@step12`
    - attention grouped-matmul disabled only:
      - `MOE_EVERYTHING_DISABLE_ATTN_GROUPED_MM=1 ...`
      - summary: `max_ce_diff=0.138773@step12`, `max_loss_diff=0.138775@step12`
      - read: the `mixture_of_everything` per-head attention grouped-matmul path is a major bf16 mismatch source
    - shared MLP grouped-matmul disabled only:
      - `MOE_EVERYTHING_DISABLE_MLP_GROUPED_MM=1 ...`
      - summary: `max_ce_diff=0.062117@step11`, `max_loss_diff=0.062118@step11`
      - read: the shared `Qwen3MoeExperts` grouped-matmul path is also a strong bf16 amplifier, even though it is common to both models
    - both grouped-matmul paths disabled together:
      - `MOE_EVERYTHING_DISABLE_ATTN_GROUPED_MM=1 MOE_EVERYTHING_DISABLE_MLP_GROUPED_MM=1 ...`
      - summary: `max_ce_diff=6.096264@step12`, `max_loss_diff=6.096291@step12`
      - read: the bf16 instability is not additive in a simple way; disabling both kernels does not restore parity
    - attention grouped-matmul disabled with router bias updates turned off:
      - `MOE_EVERYTHING_DISABLE_ATTN_GROUPED_MM=1 ... --bias-update-rate 0`
      - summary: `max_ce_diff=11.765869@step12`, `max_loss_diff=11.765894@step12`
      - read: `bias_update_rate=0` is not a robust fix and does not explain the root cause; the bf16 execution path is still the primary issue
  - added real fp32/no-Liger train configs under `configs/depth_matched_fp32_no_liger/{4_layers,8_layers,16_layers}`
    - each copied depth-matched config now sets `training.mixed_precision: fp32`
    - each sets `training.disable_liger: true`
    - `train.py` now honors `training.disable_liger: true` (and `MOE_DISABLE_LIGER=1`) so these configs really skip Liger patching
  - `train.py` also maps repo-facing `mixed_precision: fp32` onto Accelerate full precision (`mixed_precision="no"`), so the fp32 configs actually launch under DDP
  - all bf16 configs under `configs/depth_matched/{4_layers,8_layers,16_layers}` now also set `training.disable_liger: true`, so the depth-matched folders no longer rely on implicit Liger patching by default
  - validated 8-layer fp32/no-Liger DDP batch safety on the real folder configs in `configs/depth_matched_fp32_no_liger/8_layers`
    - target was `64` effective per rank
    - `batch_size=64`, `gradient_accumulation=1` is too large: on the 8-layer sanity config it OOMs during CE loss after using about `166 GiB` per GPU and trying to allocate another `37.09 GiB`
    - `batch_size=32`, `gradient_accumulation=2` gives the same `64` effective per rank and is stable
  - real 8-GPU DDP, `max_steps=10`, `batch_size=32`, `gradient_accumulation=2`, fp32/no-Liger:
    - `configs/depth_matched_fp32_no_liger/8_layers/global_moe.yaml`
      - passed through step `10`
      - representative logs: step `1` CE `12.1337`, step `10` CE `11.7065`
    - `configs/depth_matched_fp32_no_liger/8_layers/moe_everything_per_head_independent_perlayer_prenorm.yaml`
      - passed through step `10`
      - representative logs: step `1` CE `12.1227`, step `10` CE `11.9779`
    - `configs/depth_matched_fp32_no_liger/8_layers/moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`
      - passed through step `10`
      - representative logs: step `1` CE `12.1081`, step `10` CE `12.0191`
    - `configs/depth_matched_fp32_no_liger/8_layers/moe_everything_per_head_precompute_kv_sanity.yaml`
      - passed through step `10`
      - representative logs: step `1` CE `12.1337`, step `10` CE `11.7065`
    - `configs/depth_matched_fp32_no_liger/8_layers/standard_moe.yaml`
      - passed through step `10`
      - representative logs: step `1` CE `12.1399`, step `10` CE `11.6869`
    - read: the whole `8_layers` fp32/no-Liger folder is currently validated for at least `10` optimizer steps at `batch_size=32`, `gradient_accumulation=2`
- Current read:
  - on this 8-GPU node, short-context (`seq_len=128`) DDP parity is fixed end-to-end on the tested 4-layer harness
  - the `global_moe` and `alternating_global_moe` sanity models are now logically aligned in forward and in bf16 backward for the short-context harness that originally failed
  - this is no longer a sanity-only workaround; the fix lives in the shared `mixture_of_everything` attention runner used by the per-head codepaths
  - the previous long-run blow-up at short context was a real implementation mismatch in the shared attention backend path, not just random bf16 noise
  - however, full `seq_len=1024` dense parity on real parquet data is still not fully reconciled: step `0` forward stays exact, but optimizer-facing drift reappears by step `1` and still shows large transient CE spikes over longer runs
  - `bias_update_rate=0` is not a robust fix; it helped some earlier probes but worsened the newer `batch_size=4` attention-isolated run
  - fp32 does restore exact parity on the tested `30`-step `seq_len=1024` window, so the remaining issue is specifically in the bf16 path rather than in the mapped initialization or routing logic
  - for parity/debugging, Liger should stay off: even the symmetric safe subset breaks exact step-0 equivalence
  - the fp32 Liger result confirms this is not just a mixed-precision artifact: safe-Liger itself is not parity-preserving on the mapped global-vs-sanity harness
  - among the bf16 training-shape knobs tested so far:
    - larger micro-batch (`batch_size=4`) is harmful
    - gradient accumulation (`batch_size=1`, `grad_accum_steps=4`) is much less harmful than larger micro-batch
    - gradient checkpointing helps the early spike but does not solve long-run drift
  - the router path itself already runs in fp32; the remaining no-Liger mismatch is more likely in other bf16 execution paths
  - the strongest MoEverything-specific bf16 suspect now is the attention-bank grouped-matmul path in `mixture_of_everything`
  - the shared `Qwen3MoeExperts` grouped-matmul path is also a major bf16 amplifier once trajectories begin to separate
  - those grouped-matmul kernels are not the entire story, because disabling both together still leaves large step-12 spikes
  - for practical training safety, the 8-layer fp32/no-Liger configs are currently the most validated path:
    - `64` effective batch per rank via `batch_size=32`, `gradient_accumulation=2`
    - validated on all five 8-layer configs for `10` optimizer steps on this 8-GPU B200 node
  - DeepSpeed ZeRO-1 is not a fix for this issue on the current harness

## Remaining Open Items

- No expert parallel implementation yet
- No full 16-GPU local validation in this environment
- No claim yet that every repo config is fully covered
- If we need faster real runs, the next likely gains are:
  - reduce per-step logging/synchronization overhead
  - integrate a proper fused CE path for `MoEverythingForCausalLM`
  - continue improving expert dispatch efficiency
