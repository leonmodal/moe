# Status

Updated: 2026-03-27 UTC

## Current State
- The per-head MoE-Everything work is in a materially better state than the old note in this file.
- The base per-head configs now use the intended attention expert pool size of `256`, not the broken `16`.
- PHFI `O` routing now uses the actual attention output, not the pre-attention hidden state.
- Sparse execution is implemented for the per-head attention modes and for the MLP branch, so non-selected tokens no longer pay full compute there.
- Router/load-balancing stats now respect token masks, which matters once branch routing is actually sparse.
- The training logger now reports real windowed throughput instead of only cumulative throughput.

## What Was Fixed
- `configs/moe_everything_per_head_fully_independent.yaml`
  - `num_attn_experts` is now `256`.
- `configs/moe_everything_per_head_independent_prenorm.yaml`
  - `num_attn_experts` is `256`.
- `configs/moe_everything_per_head_independent_bothnorm.yaml`
  - `num_attn_experts` is `256`.
- `configs/moe_everything_per_head_precompute_kv.yaml`
  - corrected to the intended `256`-expert setup and comments now match the code.
- `configs/moe_everything_per_head_precompute_kv_prenorm.yaml`
  - `num_attn_experts` is `256`.
- `configs/moe_everything_per_head_precompute_kv_bothnorm.yaml`
  - `num_attn_experts` is `256`.
- `src/models/mixture_of_everything.py`
  - PHFI `O` router input dim is `q_dim`.
  - PHFI `O` routing is computed from the attention output.
  - Sparse per-head PHFI attention path added.
  - Sparse per-head PHPKV attention path added.
  - MLP branch respects token masks.
  - Attention router info now carries `token_mask`.
  - GPU sparse writeback dtype mismatch was fixed.
- `src/models/load_balancing.py`
  - batch/sequence aux helpers accept token masks.
- `src/utils/routing_stats.py`
  - routing stats/counts/margins accept token masks.
- `train.py`
  - logs both `train/tokens_per_sec` (windowed) and `train/tokens_per_sec_cumulative`.
  - logs `train/ms_per_step`.

## Slowdown Investigation
- The main reported slowdown was real in the logs, but mostly not real in the runtime.
- The old `tok/s` metric was cumulative from process start, so startup + warmup costs kept dragging the number down even when late-step runtime was stable.
- That logger bug is now fixed in `train.py`.

### 8-GPU Findings
- Real 8-GPU Accelerate DDP runs were executed with `accelerate_configs/ddp_8gpu.yaml`.
- PHFI 40-step run completed successfully on 8 GPUs:
  - no OOM
  - checkpoint saved at `outputs/debug8_xs_deepseek_moe_everything_per_head_fully_independent_ddp40/checkpoint-40`
  - late steps were stable at about `5.0k-5.7k tok/s` windowed
  - late step time was about `11.4s-13.2s/step`
- PHPKV 40-step run also completed successfully on 8 GPUs:
  - no OOM
  - checkpoint saved at `outputs/debug8_xs_deepseek_moe_everything_per_head_precompute_kv_ddp40/checkpoint-40`
  - late steps were stable at about `2.7k tok/s`
- A later PHFI rerun reached step 31 and matched the same late-step band, which supports the same conclusion.

### Conclusion
- I do not see evidence of a continuing throughput collapse at steps `30-40`.
- I do see early-run warmup instability through roughly the first `10-15` steps.
- The user-visible "it keeps slowing down" signal was primarily caused by the cumulative throughput metric, and that is now fixed.

## Validation Completed
- Unit tests:
  - `uv run python -m pytest tests/test_models.py -q`
  - `uv run python -m pytest tests/test_routing_stats.py tests/test_seq_loss_vs_paper.py -q`
- Added/updated coverage includes:
  - PHFI `O` router uses attention output
  - branch token masks are reported correctly
  - single-branch all-attn / all-MLP backward remains safe
  - masked routing stats
  - masked sequence aux loss
  - mixed-dtype sparse attention writeback on GPU paths
- Real GPU validation:
  - single-GPU CUDA forward/backward smoke tests passed
  - 8-GPU Accelerate DDP runs passed for PHFI and PHPKV debug configs

## Previous Questions: Fixed Or Not
- "Why are both branches executed?"
  - fixed for the per-head attention modes and for the MLP branch
  - not fixed for the older non-per-head attention modes
- "Are we enforcing 50/50 attention vs MLP balance?"
  - no, intentionally
  - the model is currently allowed to learn its own attention/MLP ratio
  - `branch_router_aux_loss_coef` exists in config, but branch balancing is not being used
- "If `per_layer_norm` is enabled, is that per layer?"
  - yes
  - unchanged
- "Change `O` routing to use attention output"
  - fixed
- "Should PHFI use 256 experts instead of 16?"
  - fixed
- "Do both models use GQA?"
  - yes
  - unchanged
- "Can attention use DeepSeek bias routing?"
  - yes
  - already true in code
- "Are we doing Switch-style batch aux for attention?"
  - no
  - attention uses sequence aux only
  - the per-head configs keep MLP batch aux off with `router_aux_loss_coef: 0.0`

## Still Open
- The older non-per-head attention modes still execute dense attention.
- The early `10-15` step warmup variance is not fully explained; it does not currently look like a late-run regression.
- MoE-Everything attention is still custom attention, not FlashAttention.

## Intentional Design Choices
- There is no branch-balance loss forcing a 50/50 attention/MLP split.
- The branch router is intentionally free to learn its own ratio.

## Temporary Files
- Temporary 40-step probe configs exist:
  - `configs/scaling/debug8_xs_deepseek_moe_everything_per_head_fully_independent_ddp40.yaml`
  - `configs/scaling/debug8_xs_deepseek_moe_everything_per_head_precompute_kv_ddp40.yaml`
