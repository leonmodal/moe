# Mixture-of-Everything Architecture

## Core Idea

A standard MoE transformer has 16 independent layers, each owning its own attention weights and MLP experts. Mixture-of-Everything takes ALL those weights and puts them into shared banks — one bank of attention head experts, one bank of MLP experts — then loops over 32 depth iterations, routing into the same banks every time. This is a Universal Transformer with heterogeneous expert routing.

The key difference from a standard transformer: instead of each layer doing attention THEN MLP in sequence, each token at each depth picks EITHER attention OR MLP via a branch router. Only the selected branch computes.

## Standard Transformer Comparison

A standard 16-layer MoE transformer (our XS baseline):

```
Layer 0:  RMSNorm → Attention(Q₀,K₀,V₀,O₀) → RMSNorm → MLP(Router₀ → top-4 of 16 experts)
Layer 1:  RMSNorm → Attention(Q₁,K₁,V₁,O₁) → RMSNorm → MLP(Router₁ → top-4 of 16 experts)
...
Layer 15: RMSNorm → Attention(Q₁₅,K₁₅,V₁₅,O₁₅) → RMSNorm → MLP(Router₁₅ → top-4 of 16 experts)
```

Total attention head parameters across all layers:
- Q: 16 layers × 16 heads = 256 head projections `(1024 → 128)` each
- K: 16 layers × 8 KV heads = 128 head projections `(1024 → 128)` each
- V: 16 layers × 8 KV heads = 128 head projections `(1024 → 128)` each
- O: 16 layers × 16 heads = 256 head projections `(128 → 1024)` each

Total MLP experts: 16 layers × 16 experts = 256 SwiGLU experts

In MoE-Everything, we collect all of these into shared banks:
- Q bank: 256 expert head projections
- K bank: 128 expert head projections
- V bank: 128 expert head projections
- O bank: 256 expert head projections
- MLP bank: 256 SwiGLU experts

Then loop 32 times, routing into these banks at every depth.

## Forward Pass

### 1. Input Processing

```
input_ids (B, T)
    → Embedding(151936, 1024)
    → hidden_states (B, T, 1024)

Initial KV state (so attention works from depth 0):
    K_init = Linear(1024 → 1024) → RMSNorm per head → RoPE
    V_init = Linear(1024 → 1024)
    Shape: (B, 8 kv_heads, T, 128 head_dim)

Position embeddings:
    cos, sin = RotaryEmbedding()    # computed once, reused all 32 depths
```

### 2. Depth Loop (32 iterations)

All weight banks are shared. Only activations change. The state carried across depths is `(hidden_states, K, V)`.

For each depth `d` in 0..31:

#### Step A: Branch Router — ATTENTION or MLP (hard routing)

```
BranchRouter[d]: Linear(1024 → 2) → softmax → (p_attn, p_mlp)
    choice = argmax(probs)              # each token picks 0 (attention) or 1 (MLP)
    weight = probs[chosen]              # differentiable scaling weight
```

Each token computes ONLY its chosen branch. The unselected branch does not run for that token. The output is scaled by the softmax probability of the chosen branch, so gradients flow through the router (same mechanism as MoE expert routing).

#### Step B: ATTENTION Branch (tokens that chose attention)

**Pre-norm** — one of three modes:
- `routed_norm`: NormExpertBank with 32 norm weight vectors, router picks 1 per token
- `per_layer_norm`: RMSNorm[d], one per depth (standard transformer style)
- default: shared RMSNorm (same norm every depth)

**Per-head expert routing** (`per_head_fully_independent` mode):

Each head independently selects its expert from the shared bank:

```
normed = PreNorm(hidden_states)                     # (B, T, 1024)

Q_router[d](normed) → select 16 experts from 256   # one per Q head
K_router[d](normed) → select 8 experts from 128    # one per KV head (GQA 2:1)
V_router[d](normed) → select 8 experts from 128    # one per KV head
O_router[d](normed) → select 16 experts from 256   # one per output head
```

All 4 routers are per-depth (32 copies each when `per_layer_attn_router=True`).

**Per-head projection through shared weight banks:**

```
For each Q head h in 0..15:
    expert_id = Q_router_selection[h]
    Q_h = Q_bank[expert_id] · normed                # (1024 → 128)
    Q_h = QK_norm(Q_h, weight=q_norm_weight[expert_id])
    Q_h = RoPE(Q_h)
    Q_h *= routing_weight[h]                         # scale by router prob

For each KV head h in 0..7:
    K_h = K_bank[K_expert_id[h]] · normed            # (1024 → 128)
    K_h = QK_norm(K_h) → RoPE(K_h)
    K_h *= routing_weight[h]
    V_h = V_bank[V_expert_id[h]] · normed            # (1024 → 128)
    V_h *= routing_weight[h]
```

**GQA expansion:**
```
Q: (B, 16 heads, T, 128)
K: (B, 8 kv_heads, T, 128) → repeat_kv(groups=2) → (B, 16, T, 128)
V: (B, 8 kv_heads, T, 128) → repeat_kv(groups=2) → (B, 16, T, 128)
```

**KV state handling:**

Tokens that chose attention get fresh K,V. Tokens that chose MLP keep their old K,V from the previous depth:
```
K_blend = where(chose_attn, K_fresh, K_old)
V_blend = where(chose_attn, V_fresh, V_old)
```

**Attention computation:**
```
scores = Q @ K_blend^T / sqrt(128)
scores += causal_mask                               # upper triangle = -inf
attn_weights = softmax(scores)
attn_output = attn_weights @ V_blend                # (B, 16, T, 128)
```

**O projection:**
```
For each head h in 0..15:
    out_h = O_bank[O_expert_id[h]] · attn_head_h    # (128 → 1024)
    out_h *= routing_weight[h]
attn_out = sum(out_h for h in 0..15)                # (B, T, 1024)
```

**Residual:**
```
hidden_states += branch_weight_attn * attn_out
```

#### Step C: MLP Branch (tokens that chose MLP)

**Pre-norm** — same three modes as attention (separate norm instance).

**DeepSeek V3 expert routing:**
```
normed = MLP_PreNorm(hidden_states)
DeepSeekRouter(normed):
    logits = Linear(1024 → 256) in FP32
    scores = sigmoid(logits)                         # NOT softmax
    biased_scores = scores + expert_bias             # non-gradient bias for load balancing
    group_limited_topk:                              # 8 groups, pick top-4 groups, top-4 experts
        → select top-4 experts from 256
    gather unbiased scores → normalize → scale by 2.5
```

**SwiGLU expert computation:**
```
For each selected expert e (4 per token):
    gate_up = gate_up_proj[e] · normed               # (1024 → 1536)
    gate, up = split(gate_up)                        # each (768,)
    out_e = silu(gate) * up                          # SwiGLU activation
    out_e = down_proj[e] · out_e                     # (768 → 1024)
mlp_out = Σ routing_weight[e] × out_e
```

**Residual:**
```
hidden_states += branch_weight_mlp * mlp_out
```

#### Step D: KV State Update

```
K_new = where(chose_attn, K_fresh, K_old)
V_new = where(chose_attn, V_fresh, V_old)
```

Tokens that picked MLP carry their KV state forward unchanged. Tokens that picked attention have fresh KV. This means a token's KV state persists until it next chooses the attention branch.

### 3. Output

```
hidden_states = RMSNorm(hidden_states)      # single final norm
logits = Linear(1024 → 151936)              # LM head (tied with embedding)
```

### 4. Losses

| Loss | Formula | Coefficient | Purpose |
|---|---|---|---|
| Cross-entropy | standard next-token prediction | 1.0 | Language modeling |
| MLP seq aux | sequence-level load balancing (DeepSeek V3) | 0.0001 | Expert utilization |
| Branch balance | `2 × Σ(mean_prob²)` | 0.01 | Prevent branch collapse |
| MLP batch aux | batch-level load balancing | 0.0 (disabled) | — |

The expert bias in DeepSeekRouter is updated non-gradient after each step:
```
bias += sign(average_load - expert_load) × rate
```

## `per_head_precompute_kv` Variant

Same architecture except the attention routing:

- **1 router per depth** (not 4): picks 16 experts from pool of 256
- Single routing decision binds Q, K, V, O for each head position
- K and V are **precomputed per-expert over all tokens**: expert `e` computes K and V for ALL tokens, then only tokens assigned to expert `e` attend through those KV tables
- This eliminates subspace mismatch: Q and K always come from the same expert
- GQA via rank selection: KV head for group `g` uses expert at rank `g × 2`
- Pool is 256/256/256/256 (bundled, since one expert provides Q+K+V+O)
- Active per token: 16 Q, 8 KV (GQA), 16 O — same compute as fully_independent

## Norm Variants

| Config flag | Attention pre-norm | MLP pre-norm | Total norms |
|---|---|---|---|
| (default) | 1 shared RMSNorm | 1 shared RMSNorm | 2 |
| `per_layer_norm` | 32 RMSNorms (one per depth) | 32 RMSNorms (one per depth) | 64 |
| `routed_norm` | NormExpertBank (32 weight vectors, top-1 router) | NormExpertBank (32 weight vectors, top-1 router) | 64 weight vectors + 2 routers |

`per_layer_norm` matches a standard 32-layer transformer: fixed depth-to-norm mapping.

`routed_norm` is more expressive: same parameter count but dynamic — different tokens at the same depth can use different norm weights.

## Router Summary

| Router | Per-depth? | Input → Output | Selection |
|---|---|---|---|
| Branch router | Yes (32 copies) | `(1024 → 2)` | argmax, hard 0/1 |
| Q expert router | Yes (32 copies) | `(1024 → 256)` | top-16 from 256 |
| K expert router | Yes (32 copies) | `(1024 → 128)` | top-8 from 128 |
| V expert router | Yes (32 copies) | `(1024 → 128)` | top-8 from 128 |
| O expert router | Yes (32 copies) | `(1024 → 256)` | top-16 from 256 |
| MLP expert router | Shared | `(1024 → 256)` DeepSeek sigmoid | top-4 from 256 |
| Attn norm router | Shared | `(1024 → 32)` | top-1 from 32 |
| MLP norm router | Shared | `(1024 → 32)` | top-1 from 32 |

For `per_head_precompute_kv`: single attn router per depth `(1024 → 256)` selecting 16, replaces the 4 separate Q/K/V/O routers.

## Shared vs Per-Depth

| Component | Shared | Per-depth |
|---|---|---|
| Q weight bank `[256, 1024, 128]` | Yes | — |
| K weight bank `[128, 1024, 128]` | Yes | — |
| V weight bank `[128, 1024, 128]` | Yes | — |
| O weight bank `[256, 128, 1024]` | Yes | — |
| QK norm weights `[256+128, 128]` | Yes | — |
| MLP experts `[256, 1024, 1536]` + `[256, 768, 1024]` | Yes | — |
| MLP router (DeepSeek) | Yes | — |
| Branch routers | — | 32 copies |
| Attn expert routers (Q,K,V,O) | — | 32 × 4 copies |
| Attn pre-norm | — | 32 (per_layer_norm or routed_norm) |
| MLP pre-norm | — | 32 (per_layer_norm or routed_norm) |
| Embedding / LM head (tied) | Single | — |
| Init KV projections | Single | — |
| Final RMSNorm | Single | — |

## Parameter Budget (XS scale, `num_attn_experts=256`)

| Component | Shape | Parameters |
|---|---|---|
| MLP gate_up bank | `[256, 1024, 1536]` | 402.7M |
| Embedding (tied w/ LM head) | `[151936, 1024]` | 311.2M |
| MLP down bank | `[256, 768, 1024]` | 201.3M |
| Q bank | `[256, 1024, 128]` | 33.6M |
| O bank | `[256, 128, 1024]` | 33.6M |
| Attn expert routers (32 × 4) | | 25.2M |
| K bank | `[128, 1024, 128]` | 16.8M |
| V bank | `[128, 1024, 128]` | 16.8M |
| Init KV projections | | 2.1M |
| MLP router, branch routers, norms | | ~0.6M |
| **Total** | | **1043.6M** |

## Config Reference

Four variants, all in `configs/`:

| Config | Attention | Norm |
|---|---|---|
| `moe_everything_phfi_perlayer_routednorm.yaml` | per_head_fully_independent (4 routers/depth) | NormExpertBank |
| `moe_everything_phfi_perlayer_perlayernorm.yaml` | per_head_fully_independent (4 routers/depth) | per-depth RMSNorm |
| `moe_everything_phpkv_perlayer_routednorm.yaml` | per_head_precompute_kv (1 router/depth) | NormExpertBank |
| `moe_everything_phpkv_perlayer_perlayernorm.yaml` | per_head_precompute_kv (1 router/depth) | per-depth RMSNorm |

All use: 32 depths, hidden 1024, 16 Q heads, 8 KV heads, 256 attn experts, 256 MLP experts top-4, DeepSeek routing, per-layer branch + attn routers, gradient checkpointing, batch 32, grad_accum 2, seq 1024.
