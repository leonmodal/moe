# Speedrun Model Configurations

Models compared on the modded-nanogpt 10K step speedrun benchmark using FineWeb-10B (GPT-2 tokenizer).

## Architecture Overview

All MoE models share the same base:
- 12 layers -> 24 depth steps (each step routes to attention OR MLP via branch router)
- Shared expert projection banks across all depths (66 attn experts, 12 MLP experts)
- Per-depth routers (separate weights per depth, but can share bias via global load balancing)
- DeepSeek V3 style sigmoid routing + expert bias updates
- Gradient checkpointing with use_reentrant=True (required for shared parameter banks)

## Models

### 1. Baseline `speedrun_gpt` (Dense)

```
Input -> Embed -> norm
         |
   Dense Block (x24 depths):
     Merged QKVO -> FlexAttention + ReLU^2 MLP
     + U-net skip connections + learnable residual
         |
   norm -> lm_head (FP8) -> sigmoid logits
```
- 275.7M params, bs=1, seq=49152, 1 node 8x H200

### 2. `speedrun_moe_fully_independent` (Token-level branch, token-level Q/K/V/O)

```
Per depth:
  branch_router(token) -> attn or mlp?      [per-token decision]
  Attn: Q/K/V/O each routed per-token       [4 x 6 head routers]
  MLP: routed per-token                      [1 router]
```
- Per-depth routers, per-router load balancing

### 3. `speedrun_moe_precompute_kv` (Token-level branch, bundled QKVO)

```
Per depth:
  branch_router(token) -> attn or mlp?      [per-token decision]
  Attn: one router per head -> same expert for Q/K/V/O
        K/V precomputed for entire sequence per expert
  MLP: routed per-token
```

### 4. `speedrun_moe_fi_branch_sampling` (Token-level branch with sampling)

Same as #2 but branch router samples from softmax instead of argmax during training.

### 5. `speedrun_moe_fi_global_lb` (Token-level, global load balancing)

Same as #2 but with global load balancing:
- One shared expert_bias for all attention routers across all depths
- One shared expert_bias for all MLP routers across all depths
- Bias updated from globally pooled token counts

### 6. `speedrun_moe_fi_seq_branch` (Seq-level branch, token-level Q/K/V/O)

```
Per depth:
  seq_repr = mean_pool(tokens)
  branch_router(seq_repr) -> attn or mlp?   [one decision per SEQUENCE]
  Attn: Q/K/V/O each routed per-token       [4 x 6 head routers]
  MLP: routed per-token
```
- Global load balancing

### 7. `speedrun_moe_fi_seq_branch_seq_qkvo` (Seq-level branch, seq-level Q/K/V/O)

```
Per depth:
  seq_repr = mean_pool(tokens)
  branch_router(seq_repr) -> attn or mlp?   [one decision per SEQUENCE]
  Attn: per-head router(seq_repr) -> one expert per head
        ALL tokens use same expert for Q, K, V, O
  MLP: routed per-token
```
- Global load balancing
- Entire sequence agrees on attention pattern AND value extraction

### 8. `speedrun_moe_fi_seq_branch_seq_qk` (Seq-level branch, seq-level Q/K, token-level V/O)

```
Per depth:
  seq_repr = mean_pool(tokens)
  branch_router(seq_repr) -> attn or mlp?   [one decision per SEQUENCE]
  Attn: Q/K routed at seq-level (one expert per head for all tokens)
        V/O routed per-token (each token picks its own expert)
  MLP: routed per-token
```
- Global load balancing
- "Where to look" (Q/K) is a sequence-level decision
- "What to extract" (V/O) is a token-level decision

## Key Design Dimensions

```
                        Branch      Q/K       V/O       MLP       Load Balance
---------------------------------------------------------------------------------
baseline                -           -         -         -         -
fi                      token soft  token     token     token     per-router
fi_branch_sampling      token samp  token     token     token     per-router
fi_global_lb            token soft  token     token     token     global (single)
fi_seq_branch           seq soft    token     token     token     global (single)
fi_seq_branch_qkvo      seq soft    seq       seq       token     global (single)
fi_seq_branch_qk        seq soft    seq       token     token     global (single)
fi_alternating_seq_qk   alternating seq       token     token     global (single)
precompute_kv           token soft  bundled   bundled   token     per-router
ds_branch_token_attn    seq DS      token     token     token     global (per-proj)
ds_branch_seq_qkvo      seq DS      seq       seq       token     global (per-proj)
ds_branch_seq_qk        seq DS      seq       token     token     global (per-proj)
```

Legend:
- "soft" = softmax + argmax, "samp" = softmax + sampling, "DS" = DeepSeek sigmoid + bias
- "global (single)" = one shared attn bias (bug — fixed in later runs)
- "global (per-proj)" = separate bias per Q/K/V/O/MLP, all-reduced across ranks

## Config Options

Model config fields:
- `branch_level`: "token" (default) or "seq" (mean-pool sequence representation)
- `branch_deepseek`: true/false (DeepSeek-style sigmoid + bias for branch router)
- `branch_sampling`: true/false (sample vs argmax, only for softmax branch)
- `branch_mode`: "router" (default) or "alternating" (hardcoded attn/mlp/attn/mlp)
- `attn_routing_level`: "token" (default), "seq" (all Q/K/V/O seq-level), or "seq_qk" (Q/K seq, V/O token)
- `global_load_balancing`: true/false (per-projection-type shared bias across all depths)
- `exploration_rate`: fraction of tokens with random expert assignment (default 0.02)

Bias update config:
- `bias_update_rate`: default fallback rate (default 0.001)
- `bias_rate_q`: Q router bias rate (default = bias_update_rate)
- `bias_rate_k`: K router bias rate
- `bias_rate_v`: V router bias rate
- `bias_rate_o`: O router bias rate
- `bias_rate_mlp`: MLP router bias rate
- `bias_rate_branch`: branch router bias rate
- `bias_warmup_start`: initial bias rate for warmup schedule (default = bias_update_rate)
- `bias_warmup_steps`: steps to decay from warmup_start to bias_update_rate (default 0)

## Router Gradient Flow

Expert routers (Q/K/V/O/MLP) use DeepSeek-style sigmoid scoring with top-1 selection.
For top-1, normalization is skipped (weight = raw sigmoid score ~0.5) so gradients flow
from the CE loss through the routing weight back to the router linear layer.
For top-K (K>1), normalization is preserved (weights sum to 1).

The gradient only flows through the selected expert's score — it cannot tell the router
to switch to a different expert. Actual routing decisions are steered by the bias mechanism.

## Bias Update Mechanism

Uses the nmoe-style zero-sum update (prevents drift):
```
loads = counts / counts.sum()           # normalize to fractions
s = sign(loads - 1/num_experts)         # overloaded = +1, underloaded = -1
bias -= (s - s.mean()) * rate           # zero-sum: update sums to 0 every step
bias.clamp_(-16, 16)                    # safety bound
```

Key properties:
- Bias sum stays at exactly 0 (no drift) because `s - s.mean()` sums to 0
- Per-projection rates: Q/K can have higher rate than V/O/MLP
- All-reduced across ranks before update (all ranks stay synchronized)
- Counts accumulate across gradient accumulation micro-batches
- Branch bias is per-depth (24 separate biases), expert bias is global per projection type

## Plots Generated (every heatmap_every steps)

```
routing_logs/step_XXXXXXXX/
  branch_routing.png              # 4-row: token fraction, attn/mlp ratio, sigmoid weights, branch bias
  branch_histogram.png            # histogram of attn fracs across depths
  global_expert_biases.png        # Q/K/V/O/MLP bias bar charts (red=neg, green=pos)
  global_projection_histograms.png # expert usage per projection type
  q/h0/ h1/ ... h5/              # per-head: heatmap + histograms
  k/h0/ ... 
  v/h0/ ...
  o/h0/ ...
  mlp/                            # MLP heatmap + histograms
```

## Seq-Level Routing and Expert Utilization

With seq-level routing (attn_routing_level="seq"), the number of routing decisions per step
equals batch_size (not batch_size * seq_len). With 66 experts and bs=64, at most 64/66 experts
can be selected per step. Per-step utilization will be lower than token-level routing, but over
multiple steps all experts get used. This is expected behavior, not a bug.
