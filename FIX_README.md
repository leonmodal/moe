# Fix Summary

This commit fixes two issues that could make `standard` vs `global` MoE
comparisons look more similar than they should.

## 1. `train/ce_loss` logging fix

File:
- `train.py`

Problem:
- The model forward path already computes `seq_aux_loss` using the actual biased
  routed experts from `DeepSeekRouter`.
- But the training logger recomputed `seq_aux_loss` without those selected
  experts, so logged `train/ce_loss` was only an approximation for DeepSeek
  routing.

Fix:
- Added `get_selected_experts_for_seq_aux()`.
- Logging now uses the same routed expert assignments when recomputing
  `seq_aux_loss` for `train/ce_loss`.

Impact:
- Training/backprop behavior is unchanged.
- Logged `train/ce_loss` is now more faithful.
- Old `ce_loss` dashboards may be misleading for DeepSeek runs.

## 2. Comparison script fix

File:
- `scripts/compare_training_steps.py`

Problems:
- The script did not call `update_expert_biases()`.
- That made `global` and `global_nointerp` effectively the same in that script.
- The script also double-added `seq_aux_loss`.

Fix:
- Added bias updates with the real `bias_alpha_schedule()` and
  `update_expert_biases()` path.
- Removed the duplicate `seq_aux_loss` addition.

Impact:
- The script is now a valid comparison for:
  - `xs_deepseek_standard.yaml`
  - `xs_deepseek_global.yaml`
  - `xs_deepseek_global_nointerp.yaml`

## 3. Tiny matched-capacity diagnostic

File:
- `scripts/debug_tiny_global_vs_perlayer.py`

Purpose:
- Fast GPU diagnostic that isolates the main question:
  same total expert capacity, but:
  - per-layer private experts vs
  - global shared experts

Observed result:
- `standard` vs `global`: small but real loss differences
- `global` vs `global_nointerp`: much smaller early differences

## Verification

Verified with:

```bash
uv run python -m py_compile train.py scripts/debug_tiny_global_vs_perlayer.py scripts/compare_training_steps.py
uv run python scripts/debug_tiny_global_vs_perlayer.py
```

## Recommended interpretation

- `loss` used for optimization was already the main trustworthy metric.
- Old logged `train/ce_loss` for DeepSeek routing should be treated cautiously.
- If long-run `standard` vs `global` curves are still very similar after this
  fix, that likely reflects a genuinely small architecture effect rather than
  an obvious forward-pass bug.
