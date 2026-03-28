# Architecture Overview

This repo has three model families:

1. `standard_moe`
2. `global_moe`
3. `moe_everything`

All three use the Qwen3/Qwen3-MoE backbone pieces: RoPE, RMSNorm, SwiGLU, and a causal LM head.
Standard/global configs keep the usual Qwen GQA layouts; the current per-head launch configs use MHA.

## Standard MoE

File: `src/models/standard_moe.py`

- Standard transformer stack.
- Every layer has dense attention.
- Every layer owns its own MLP expert pool.
- Routing is only on the MLP branch.

Use this when you want the normal per-layer MoE baseline.

## Global MoE

File: `src/models/global_moe.py`

- Standard transformer stack.
- Every layer still has dense attention.
- MLP experts are shared globally across all layers.
- Each layer has its own router into the shared MLP pool.

Use this when you want to test whether sharing the expert pool across depth helps relative to standard per-layer experts.

## MoE-Everything

File: `src/models/mixture_of_everything.py`

MoE-Everything is the experimental architecture in this repo.

- The model keeps a persistent KV state across depth.
- At each depth, a branch router chooses `attention` or `mlp` per token.
- The selected branch output is added through a residual update.
- Attention weights and MLP experts live in shared banks instead of being owned by a fixed layer.

### Depth Loop

For each depth:

1. Branch route tokens to `attention` or `mlp`.
2. Attention-selected tokens update KV state.
3. MLP-selected tokens keep the old KV state.
4. Residual update is applied with hard branch masks and probability-weighted outputs.

### Attention Modes

The attention bank supports:

- `bundled`
- `kv_paired`
- `qk_paired`
- `fully_independent`
- `per_head_fully_independent`
- `precompute_kv`
- `per_head_precompute_kv`

The two per-head modes now have:

- grouped per-expert batching for Q/K/V/O projection work
- sparse execution when only a subset of tokens take the attention branch
- optional dense fallback controlled by:
  - `per_head_compute_mode: auto | sparse | dense`
  - `per_head_dense_fraction_threshold`

`auto` uses sparse attention when the attention-token fraction is low and dense attention when it is high.

### Per-Layer Routing

`per_layer_router: true`

- one branch router per depth

`per_layer_attn_router: true`

- one attention-router set per depth
- now supported for all attention modes, not just the per-head modes

### Norm Options

- shared norm
- `per_layer_norm: true`
- `routed_norm: true`
- optional `post_norm: true`

### Depthwise Attention

`depthwise_attention: true`

- caches depth outputs
- learns a depth-mixing query
- supports block mode with `depthwise_block_size`

### Sanity Routing Mode

`sanity_check_mode: alternating_global_moe`

This is a deterministic debugging mode for `per_head_precompute_kv`.

- Depth 0, 2, 4, ... are forced to attention.
- Depth 1, 3, 5, ... are forced to MLP.
- Attention experts are fixed per logical layer and per head.
- MLP routing still uses the normal learned gate.

This is meant for “does this reduce to a global-MoE-like stack when routing is hardcoded?” checks.

The launch config `configs/moe_everything_per_head_precompute_kv_sanity.yaml` is now aligned with the main
per-head sweep on MLP capacity and routing settings:

- MHA: `num_attention_heads = num_key_value_heads = 16`
- attention bank: `num_attn_experts = 256`
- MLP bank: `num_experts = 256`
- active MLP experts: `num_experts_per_tok = 4`

## Training

Main entrypoint: `train.py`

Local launcher: `scripts/train.sh`

- local multi-GPU runs use `torchrun` by default
- single-GPU runs use `uv run python`
- `LAUNCHER=accelerate` keeps the old `accelerate launch` path
- inside the process, training still uses `Accelerator` for wrapping, checkpointing, logging, and distributed utilities

## Logging

Training logs now report exact per-step throughput:

- `train/tokens_per_sec`
- `train/sec_per_step`
- `train/tokens_seen_B`

These are last-step measurements, not windowed or cumulative throughput estimates.

Routing logs are written under `output_dir/routing_logs/step_XXXXXXXX/`.

## Useful Configs

Representative configs:

- `configs/standard_moe.yaml`
- `configs/global_moe.yaml`
- `configs/moe_everything_precompute_kv.yaml`
- `configs/moe_everything_precompute_kv_perlayer.yaml`
- `configs/moe_everything_per_head_precompute_kv.yaml`
- `configs/moe_everything_per_head_precompute_kv_sanity.yaml`
- `configs/scaling/debug8_xs_deepseek_moe_everything_precompute_kv_perlayer.yaml`

## Benchmarks

To compare sparse vs dense per-head attention dispatch:

```bash
uv run python scripts/benchmark_per_head_attention_dispatch.py --mode per_head_precompute_kv
uv run python scripts/benchmark_per_head_attention_dispatch.py --mode per_head_fully_independent
```
