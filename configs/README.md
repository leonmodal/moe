# MoE Launch Configs

The active training set lives only under `configs/16_layers/` and is locked by
`tests/test_nested_configs_drift.py` plus `scripts/validate_configs.py`.

## Active Set

- `standard_moe_deepseek_bias.yaml`
- `moe_everything_per_head_recompute_k_qk_v_o_deepseek_bias.yaml`
- `moe_everything_per_head_recompute_k_qk_v_o_ema_qk_v_deepseek_bias.yaml`
- `moe_everything_per_head_recompute_kv_qk_v_o_deepseek_bias.yaml`
- `moe_everything_per_head_recompute_kv_qk_v_o_ema_qk_v_deepseek_bias.yaml`

All five configs use `model.num_hidden_layers: 16`.

All four `moe_everything` configs use fully per-depth router/norm state:

```yaml
per_layer_router: true
per_layer_mlp_router: true
per_layer_attn_router: true
per_layer_norm: true
per_layer_qk_norm: true
```

## Attention Variants

The four `moe_everything` rows all use:

```yaml
attn_routing_bundle: qk_v_o
```

That means Q/K share one route, V routes separately, and O routes separately
after attention. O is token-local and is never part of sequence-side
recompute.

The two recompute modes are:

- `per_head_recompute_k`: recompute K by the current token's Q/K route; V is
  projected once from its token-routed V table.
- `per_head_recompute_kv`: recompute K by Q/K route and V by the current
  token's V route.

The EMA rows additionally set:

```yaml
attn_router_context: ema_qk_v
attn_router_context_decay: 0.95
```

This changes only QK and V router inputs. Those routers see:

```text
concat(normed_h_t, ema(normed_h_<t))
```

The EMA is causal: token `t` never sees itself or future tokens in the prefix
summary. Projection inputs remain the normal hidden states.

## Router Balancing

Every active row uses DeepSeek-style bias balancing for MLP routing. The
`moe_everything` rows also use DeepSeek-style bias balancing for attention
routers:

```yaml
mlp_router:
  balancing: deepseek_bias
  bias_update_rate: 0.001
  bias_update_zero_sum: true
attn_router:
  balancing: deepseek_bias
  bias_update_rate: 0.001
  bias_update_zero_sum: true
```

The focused qkvo + recompute-KV branch-router ablation uses three branch
contracts over the same attention/MLP expert setup:

```yaml
branch_router:
  balancing: sampling_entropy
  entropy_coef: 0.01
  entropy_decay: cosine
  entropy_min: 0.0
  entropy_decay_steps: 1000
```

```yaml
branch_router:
  balancing: exploration_only
  exploration_rate: 0.10
  exploration_decay: cosine
  exploration_min: 0.0
  exploration_warmup_steps: 1000
```

```yaml
branch_router:
  balancing: fixed_alternating
```

The corresponding configs are:

- `16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_sampling_entropy.yaml`
- `16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_top1_explore_decay.yaml`
- `16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_fixed_alternating.yaml`

## Validation

Run:

```bash
python scripts/validate_configs.py
python -m pytest tests/test_nested_configs_drift.py
```
