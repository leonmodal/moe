# Efficiency And Correctness Notes

This note records the current MoE-Everything attention execution decision.
See `docs/architecture.md` for the full model description.

## Dense Attention Execution

MoE-Everything attention uses dense execution only.

The branch router still selects attention vs MLP per token, but the attention
bank computes full-token attention tensors and applies the branch mask to the
residual output and, for no-recompute mode, to the KV state update.

Sparse gather / packed-query attention was removed after H100
forward/backward throughput checks showed dense execution was consistently
faster in the current PyTorch/SDPA implementation.

Measured on Modal H100 80GB, `B=4`, `T=512`, `4` layers, `hidden=512`,
SDPA attention, bf16 autocast, forward plus backward:

| Mode | Dense ms/step | Sparse ms/step before removal |
|------|--------------:|-------------------------------:|
| `per_head_no_recompute` | 61.82 | 88.31 |
| `per_head_recompute_k` | 64.64 | 190.64 |
| `per_head_recompute_kv` | 72.94 | 515.18 |

The sparse path reduced theoretical query rows, but the gather/scatter,
packing, and smaller attention launches dominated wall-clock time. Dense
execution keeps fewer, larger GPU kernels and was the better practical default.

## Recompute Cost

`per_head_recompute_k` recomputes full-sequence K tables for active Q/K expert
routes. `per_head_recompute_kv` recomputes full-sequence K and V tables for
active route pairs. The attention calls are standard attention calls over dense
query tensors; rows for other routes are masked after each expert-table call.

This is intentionally not the theoretical minimum number of QK rows. A fused
ragged grouped-attention kernel could revisit that tradeoff later, but the
current code keeps the measured-faster dense path.
