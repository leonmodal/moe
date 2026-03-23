# Status

Updated: 2026-03-23 UTC

## Current State
- Core MoE-Everything training/runtime fixes are in the repo.
- Canonical XS configs now live at `configs/standard_moe.yaml`, `configs/global_moe.yaml`, and `configs/global_moe_nointerp.yaml`.
- `ARCHITECTURE.md` was updated to match the current configs and MoE-Everything variants.
- All 5 shipped `configs/moe_everything_*.yaml` files were changed to the memory-safe training settings for the 8xB200 box while preserving effective global batch 64:
  - `batch_size: 1`
  - `gradient_accumulation: 8`
  - `gradient_checkpointing: true`

## Code Fixes Landed
- `train.py`
  - Scheduler uses world-size-scaled steps only for the prepared scheduler, not for loop control.
  - Loss/tokens logging is accumulation-safe and only logs on real optimizer steps.
  - Gradient checkpointing is actually enabled when requested.
  - MoE-Everything routing stats now include MLP routing, attention routing, and branch ratios.
- `src/models/mixture_of_everything.py`
  - DeepSeek routing options are wired through.
  - Real gradient checkpointing support added for the 32-step shared-depth loop.
  - Forward output now exposes `ce_loss`, `aux_loss`, `seq_aux_loss`, `branch_aux_loss`, selected experts, branch probabilities, and attention-router info.
- `src/models/router.py`
  - Router accounting is checkpoint-safe and does not double-count during recompute.
- `src/utils/routing_stats.py` and `src/utils/routing_plots.py`
  - Added attention-router stats/plots and branch-ratio tracking.
- `tests/test_models.py`
  - Added focused MoE-Everything output/checkpoint-routing tests.

## Runtime Verification Completed
- Verified on real 8xB200 hardware with `uv run accelerate launch`.
- Original shipped MoE-Everything config (`batch_size: 64`, no checkpointing) OOMed before step 1.
- Bundled mode was verified to run successfully with the fitting settings:
  - `batch_size: 1`
  - `gradient_checkpointing: true`
- Bundled mode was also verified with effective global batch 64:
  - 8 GPUs
  - `batch_size: 1`
  - `gradient_accumulation: 8`
  - printed/logged only optimizer steps
  - LR advanced once per optimizer step
  - attention routing and branch-ratio metrics were logged successfully

## Remaining Work For Tomorrow
- Run the 4 remaining shipped MoE-Everything configs on the real 8xB200 setup with the new settings:
  - `configs/moe_everything_kv_paired.yaml`
  - `configs/moe_everything_qk_paired.yaml`
  - `configs/moe_everything_fully_independent.yaml`
  - `configs/moe_everything_precompute_kv.yaml`
- Re-run bundled once from the shipped config path after cleanup if you want a fresh canonical check instead of the temporary probe config.
- If any variant still OOMs at `batch_size: 1`, the next lever is sequence length or model-side memory reduction; gradient accumulation will not change memory.

## Notes
- MoE-Everything attention is still custom matmul attention, not FlashAttention.
- Liger is still applied for the supported Qwen3-MoE pieces; the custom MoE-Everything attention bank is separate.
- No live training processes were left running at cleanup time.
