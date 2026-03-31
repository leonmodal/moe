# Status

Updated: 2026-03-31 UTC

## Current State

- The per-head MoE-Everything codepath is in much better shape than the older notes in this repo suggested.
- The biggest recent bug was a real router-initialization problem in `MoEverythingForCausalLM`, and that is now fixed.
- The sanity path is now useful again as an implementation check for the custom MoE-Everything stack.

## Recent Fixes

- `src/models/mixture_of_everything.py`
  - `MoEverythingForCausalLM` now uses `Qwen3MoePreTrainedModel.post_init()`
  - DeepSeek/Qwen router weights are initialized correctly
  - experts implementation dispatch now works on the MoE-Everything path
  - sanity MLP routers now map by logical layer, not physical depth
- `train.py`
  - logs raw batch aux and normalized batch aux separately
  - `train/aux_loss_normalized` is now available for DeepSeek runs
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

## Still Open

- The remaining sanity-vs-global gap is smaller, but not zero.
- The sanity path still instantiates deterministic branch and attention-router structures that are bypassed in practice.
- Older non-per-head attention modes still execute dense attention.
- MoE-Everything attention is still the custom attention stack, not the standard Qwen attention path.

## Validation

- `uv run pytest -q tests/test_aux_loss_fix.py tests/test_models.py`
- result: `129 passed`
