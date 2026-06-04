# Current Launch Plan

The active launch set is the qkvo + recompute-KV branch-router ablation.
Attention and MLP expert routing stay fixed across the three rows; only the
branch policy changes.

## Configs

1. `configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_sampling_entropy.yaml`
2. `configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_top1_explore_decay.yaml`
3. `configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_fixed_alternating.yaml`

## Architecture Questions

- Q/K/V/O route together through `attn_routing_bundle: qkvo`.
- The attention bank uses `attn_expert_mode: per_head_recompute_kv`.
- O is still token-local; recompute-KV affects the sequence-side K/V tables.
- MoE-Everything rows use per-depth branch, MLP, and attention routers.
- MoE-Everything rows use per-depth RMSNorm and per-depth Q/K norm weights.

## Defaults

- MLP router balancing: `deepseek_bias`
- Attention router balancing: `deepseek_bias`
- Branch rows:
  - `sampling_entropy`: categorical branch sampling plus decaying entropy bonus.
  - `top1_explore_decay`: top-1 softmax branch gate with random override decayed from 0.10 to 0 over 1000 steps.
  - `fixed_alternating`: even depths attention, odd depths MLP; no branch-router parameters.
- Distributed default: DDP
- MoE-Everything LM loss: model-native fused linear CE on CUDA
- Modal launch should pass `--disable-attn-grouped-mm` until the H200 illegal-instruction issue is isolated.

## Verification

Run:

```bash
python scripts/validate_configs.py
python -m pytest tests/test_nested_configs_drift.py tests/test_recompute_attention_execution.py
```
