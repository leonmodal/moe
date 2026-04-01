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

- `uv run pytest -q tests/test_models.py`
- result: `134 passed`

## Remaining Open Items

- No expert parallel implementation yet
- No full 16-GPU local validation in this environment
- No claim yet that every repo config is fully covered
- If we need faster real runs, the next likely gains are:
  - reduce per-step logging/synchronization overhead
  - integrate a proper fused CE path for `MoEverythingForCausalLM`
  - continue improving expert dispatch efficiency
