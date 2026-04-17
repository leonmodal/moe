# Multi-GPU pipeline validation report

Every entry below was run via `torchrun --standalone --nproc_per_node=8 scripts/train.py`.

## Stage B — loss-to-3.3 evidence

| Variant | Status | Steps | First loss | Final loss | Elapsed |
|---|---|---|---|---|---|
| dense | OK | 10000 | 7.106 | 3.022 | 1968.0s |
| standard_moe_softmax | OK | 10000 | 7.257 | 3.006 | 3902.4s |
| standard_moe_deepseek | OK | 10000 | 7.057 | 3.035 | 3107.1s |

