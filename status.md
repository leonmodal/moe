# Status

Updated: 2026-03-31 UTC

## Current State

- The per-head MoE-Everything codepath is in much better shape than the older notes in this repo suggested.
- The biggest recent bug was a real router-initialization problem in `MoEverythingForCausalLM`, and that is now fixed.
- The sanity path is now useful again as an implementation check for the custom MoE-Everything stack.
- The learned per-head MoE-Everything configs are not compatible with the full Liger Qwen3-MoE patch bundle.
- On the current B200 setup, the learned per-head configs fit without the broken Liger pieces.

## Recent Fixes

- `src/models/mixture_of_everything.py`
  - `MoEverythingForCausalLM` now uses `Qwen3MoePreTrainedModel.post_init()`
  - DeepSeek/Qwen router weights are initialized correctly
  - experts implementation dispatch now works on the MoE-Everything path
  - sanity MLP routers now map by logical layer, not physical depth
- `train.py`
  - logs raw batch aux and normalized batch aux separately
  - `train/aux_loss_normalized` is now available for DeepSeek runs
  - Liger handling is now selective by model family
  - `moe_everything` keeps only the validated-safe Liger `rope` and `rms_norm` patches
  - Liger `swiglu` is disabled for `moe_everything`
- `src/models/load_balancing.py`
  - added normalized batch-aux helper for sigmoid-router diagnostics
- `tests/test_models.py`
  - added regression coverage for router initialization and experts dispatch
  - updated sanity tests for logical-layer MLP gates
- `tests/test_aux_loss_fix.py`
  - added normalized-aux regression coverage

## Sanity Compare Result

The broken sanity run used to show:

- `seq_aux_loss` pinned near `1.0`
- raw `aux_loss` around `512`
- large loss gap vs `global_moe`

That was caused by zero-initialized DeepSeek routers in the MoE-Everything path.

After the fix:

- the sanity run no longer shows the router-init failure signature
- `seq_aux_loss` now tracks the `global_moe` run much more closely
- the 30-step frozen-snapshot sanity gap shrank substantially

Reference result from the fixed 8-GPU frozen-snapshot compare:

- step 10 gap vs `global_moe`: about `+0.024`
- step 20 gap vs `global_moe`: about `+0.060`
- step 30 gap vs `global_moe`: about `+0.082`

## Intentional Design Choices

- Branch routing is not load-balanced.
- `branch_router_aux_loss_coef` should remain `0.0` in the per-head configs.
- The branch router is free to learn its own attention/MLP ratio.
- For DeepSeek routing, the important regularization signal is `seq_aux_loss`, not raw batch aux.

## Liger Status

- Full Liger on learned-routing `moe_everything` is broken.
- The failure signature is immediate: fresh-init CE jumps from about `12.13` to about `20.0`.
- The bad Liger component is the `swiglu` expert replacement.
- The `rope` patch is fine.
- The `rms_norm` patch is fine.
- The stock Liger fused linear CE patch does not currently help `MoEverythingForCausalLM`, because it patches the stock `Qwen3MoeForCausalLM.forward`, not the custom MoE-Everything forward.
- Current training behavior for `moe_everything` is:
  - keep Liger `rope`
  - keep Liger `rms_norm`
  - disable Liger `swiglu`
  - disable Liger fused linear CE for this model family

Reference local checks on the real per-head learned configs:

- no Liger: CE about `12.13`
- full Liger: CE about `20.0`
- partial Liger (`rope + rms_norm` only): CE stays about `12.13`

## Memory Status

Current memory probes were run on local `NVIDIA B200` GPUs with the real config settings:

- `seq_len = 1024`
- `batch_size = 32` per GPU
- `gradient_accumulation = 2`
- `bf16`
- gradient checkpointing on

Observed results without the broken Liger pieces:

- single GPU, `moe_everything_per_head_precompute_kv_perlayer_prenorm`:
  - peak allocated about `69.7 GB`
  - peak reserved about `89.0 GB`
- single GPU, `moe_everything_per_head_independent_perlayer_prenorm`:
  - peak allocated about `71.8 GB`
  - peak reserved about `90.8 GB`
- 8 GPU DDP, `moe_everything_per_head_independent_perlayer_prenorm`, one optimizer step:
  - peak allocated about `73.6 GB` per rank max
  - peak reserved about `92.8 GB` per rank max
- 8 GPU DDP, same config, `gradient_accumulation = 2`, `3` optimizer steps:
  - peak allocated about `91.2 GB` per rank max
  - peak reserved about `130.6 GB` per rank max

Conclusion:

- On the current B200 setup, learned per-head MoE-Everything does not need the broken Liger `swiglu` patch to fit.
- If we later need fused CE memory savings on a smaller GPU target, the right path is to integrate Liger's lower-level `LigerForCausalLMLoss` directly into `MoEverythingForCausalLM` instead of turning on the full Qwen3-MoE monkey-patch bundle.

## Still Open

- The remaining sanity-vs-global gap is smaller, but not zero.
- The sanity path still instantiates deterministic branch and attention-router structures that are bypassed in practice.
- Older non-per-head attention modes still execute dense attention.
- MoE-Everything attention is still the custom attention stack, not the standard Qwen attention path.

## Validation

- `uv run pytest -q tests/test_aux_loss_fix.py tests/test_models.py`
- result: `129 passed`
