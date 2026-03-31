# Mixture Of Everything

This file documents the model families in this repo and how `moe_everything` differs from the standard MoE baselines.

## Model Families

## `standard_moe`

Files:

- `src/models/standard_moe.py`

Behavior:

- standard transformer layer structure
- dense attention in every layer
- one routed MLP expert pool per layer
- routing only happens on the MLP branch

Use this as the normal per-layer MoE baseline.

## `global_moe`

Files:

- `src/models/global_moe.py`

Behavior:

- standard transformer layer structure
- dense attention in every layer
- one shared global MLP expert pool across all layers
- each layer has its own router into that shared pool

Use this when you want shared MLP experts across depth without changing the attention stack.

## `moe_everything`

Files:

- `src/models/mixture_of_everything.py`

Behavior:

- custom depth loop instead of a standard decoder-layer stack
- a branch router chooses `attention` or `mlp` per token at each depth
- attention experts and MLP experts live in shared banks
- KV state persists across depth
- tokens that take MLP keep their old KV state

This is the experimental architecture in the repo.

## Attention Modes

`moe_everything` supports:

- `bundled`
- `kv_paired`
- `qk_paired`
- `fully_independent`
- `per_head_fully_independent`
- `precompute_kv`
- `per_head_precompute_kv`

The current work is mainly around the per-head modes:

- `per_head_fully_independent`
- `per_head_precompute_kv`

## Router Structure

There are three separate router concepts in `moe_everything`:

- branch router: chooses `attention` vs `mlp`
- attention routers: choose attention experts
- MLP router: chooses MLP experts

Important switches:

- `per_layer_router`
  one branch router per depth
- `per_layer_attn_router`
  one attention-router set per depth
- `per_layer_mlp_router`
  one MLP router per depth
- `global_router_update`
  use global-style DeepSeek bias updates across routed expert pools

These are different axes:

- `per-layer` means separate router weights by depth
- `global_router_update` means pooled bias updates, not shared router weights

## Load Balancing

For the current per-head configs:

- `branch_router_aux_loss_coef: 0.0`
- `router_aux_loss_coef: 0.0`
- `seq_aux_loss_coef: 0.0001`
- `bias_update_rate: 0.001`

That means:

- no branch-balance loss
- no Switch-style batch aux term in the training objective
- sequence-level aux is active
- DeepSeek-style expert-bias updates are active

For DeepSeek sigmoid routers, logging now exposes both:

- raw batch aux: `train/aux_loss`
- normalized batch aux diagnostic: `train/aux_loss_normalized`

The normalized value is easier to compare across runs.

## Per-Head Root Configs

Learned-routing configs:

- `configs/moe_everything_per_head_fully_independent.yaml`
- `configs/moe_everything_per_head_precompute_kv.yaml`
- `configs/moe_everything_per_head_independent_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_independent_perlayer_bothnorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_bothnorm.yaml`

Sanity config:

- `configs/moe_everything_per_head_precompute_kv_sanity.yaml`

The per-head configs now use the same GQA geometry as the baseline configs:

- `num_attention_heads = 16`
- `num_key_value_heads = 8`

## Per-Head Attention Picture

For the current root per-head configs:

- there are `8` fixed KV groups
- there are `16` query heads arranged as `2` query heads per KV group
- there are `256` attention experts in the shared pool
- each token routes those `8` KV groups into that `256`-expert pool

The important distinction is:

- a KV group is a fixed position such as `group 0`, `group 1`, ..., `group 7`
- an attention expert is a reusable parameter set from the shared expert pool

So an expert is not permanently tied to one KV group id. For each token, the router fills the `8` KV-group slots with selected experts from the `256`-expert pool.

### `per_head_precompute_kv`

In `per_head_precompute_kv`, one selected expert provides the grouped-query slice plus the shared `K`, `V`, and `O` behavior for that KV group.

```text
Config:
  num_attention_heads = 16
  num_key_value_heads = 8
  num_attn_experts = 256

For one token x: [1024]

router(x) -> 8 expert ids, one for each KV group

kv group:     0    1    2    3   ...    7
expert id:   37   91    4  144   ...    8

Then for each KV group g:

  e = expert_id[g]

  Q_g = q_norm[e](x @ Wq[e])   -> [2, 128]
  K_g = k_norm[e](x @ Wk[e])   -> [128]
  V_g =            x @ Wv[e]   -> [128]

Collect over 8 KV groups:

  Q -> [16, 128]
  K, V -> [8, 128]

Run GQA attention:

  attn(Q, K, V) -> Y -> [16, 128]

Project back with the selected output experts per KV group:

  out_g = Y_g @ Wo[e]          -> [1024]
  sum over KV groups           -> final token output [1024]
```

Batch-shaped view:

```text
hidden_states         [B, T, 1024]
pre-norm              [B, T, 1024]
flat tokens           [N, 1024]          where N = B * T

router idx, weights   [N, 16], [N, 16]

Q_heads               [N, 16, 128]
K_heads               [N, 16, 128]
V_heads               [N, 16, 128]

reshape to attention:
Q, K, V               [B, 16, T, 128]

attention output      [B, 16, T, 128]

O projection result   [N, 1024]
final output          [B, T, 1024]
```

### Q/K Norms

There are two different norm ideas in the per-head path:

- input pre-norm on the token hidden state before routing/projection
- expert-specific `q_norm_weight` and `k_norm_weight` on the projected head vectors

The expert-specific Q/K norms follow the selected expert, not the fixed head id.

```text
token A:
  head 0 -> expert 37 -> uses q_norm_weight[37], k_norm_weight[37]

token B:
  head 0 -> expert 12 -> uses q_norm_weight[12], k_norm_weight[12]
```

So the right mental model is:

- `8` fixed KV-group positions
- `16` query heads derived from those `8` groups
- `256` candidate attention experts
- each token chooses which experts fill those `8` KV-group positions
- the selected expert brings its own grouped-Q and K norm weights

### `per_head_fully_independent`

This mode is more flexible than `per_head_precompute_kv`.

- `Q`, `K`, `V`, and `O` route separately
- the same head slot can choose different experts for `Q`, `K`, `V`, and `O`

So in `per_head_fully_independent`, the head slot is still fixed, but the component experts are not tied together as one shared `QKVO` package.

## Sanity Mode

`configs/moe_everything_per_head_precompute_kv_sanity.yaml` uses:

- `sanity_check_mode: alternating_global_moe`

Meaning:

- even depths are forced to attention
- odd depths are forced to MLP
- attention expert assignment is deterministic by logical layer
- MLP routing is learned
- MLP bias updates use the global shared-pool style

The sanity path is meant to catch implementation bugs in the MoE-Everything stack while staying on the custom MoE-Everything codepath.

Recent sanity fixes:

- per-layer MLP routers added
- global-style MLP bias updates added
- branch aux kept off
- router initialization fixed through `Qwen3MoePreTrainedModel.post_init()`
- sanity MLP gates now map by logical layer instead of carrying dead extra MLP gates

## What Sanity Does Not Mean

The sanity config is not literally `global_moe`.

It still uses:

- the custom MoE-Everything depth loop
- persistent KV across depth
- the custom per-head attention bank

So the sanity run is useful for finding bugs in our implementation, but it is still not a literal replacement for the standard global-MoE stack.
