# Model Architecture

This document covers all model families, their layer designs, and advanced architectural features.

## Table of Contents

- [1. Standard LLM (Dense)](#1-standard-llm-dense)
- [2. Standard MoE](#2-standard-moe)
- [3. Global MoE](#3-global-moe)
- [4. MoE-Everything](#4-moe-everything)
  - [4a. No-Recompute Mode](#4a-no-recompute-mode)
  - [4b. Recompute Attention](#4b-recompute-attention)
- [5. Architecture Comparison](#5-architecture-comparison)
- [6. Advanced Features (extracted from archived speedrun models)](#6-advanced-features-extracted-from-archived-speedrun-models)

---

## 1. Standard LLM (Dense)

**Config type**: `dense`
**Implementation**: `Qwen3ForCausalLM`

A standard dense transformer with no expert routing. Every token passes through every parameter. Used as a baseline for comparing MoE efficiency.

**Architecture**:
- Multi-head or grouped-query attention (GQA)
- SwiGLU MLP
- RMSNorm
- RoPE positional encoding

**When to use**: Baseline comparisons and when you want a simple dense model without routing overhead.

---

## 2. Standard MoE

**Config type**: `standard_moe` (with optional `router_type: deepseek`)
**File**: `src/models/standard_moe.py`
**Base**: `Qwen3MoeForCausalLM` with fixes

Each transformer layer has its own set of MLP experts. Attention is dense (shared across all tokens), but the MLP is routed: each token selects top-K experts from the layer's pool.

```
Layer N:
  Input -> Attention (dense, shared) -> RMSNorm
        -> Router -> selects top-K from [Expert_0 ... Expert_E] (layer-local)
        -> Weighted sum of selected expert outputs -> Residual
```

**Key fix over HuggingFace**: The original HF implementation applies softmax twice to router logits (once in the router, once in the loss function), which flattens the probability distribution and makes load-balancing loss ineffective. Our version uses router probs directly.

**Variants**:
- `standard_moe`: Softmax router with optional exploration
- With `router_type: deepseek`: Sigmoid router + non-gradient expert bias updates (DeepSeek V3 style)

**Typical config**: 16 experts per layer, top-4 routing

---

## 3. Global MoE

**Config type**: `global_moe` (with optional `router_type: deepseek`)
**File**: `src/models/global_moe.py`

Instead of each layer having its own experts, all layers share a single global pool of experts. Each layer has only a router (no expert weights). The router selects from the shared pool.

```
Global Expert Pool: [Expert_0 ... Expert_2047]  (registered once at model root)

Layer N:
  Input -> Attention (dense) -> RMSNorm
        -> Router_N -> selects top-K from global pool
        -> Weighted sum -> Residual
```

**Why global pooling?**
- Parameter efficiency: total expert params = `num_global_experts * expert_size` regardless of layer count
- Emergent specialization: different layers can learn to reuse experts in different combinations
- Scaling: can have a very large expert pool (e.g., 2048) without per-layer duplication

**Expert bias interpolation**: For DeepSeek variants, bias updates can interpolate between per-layer and global statistics using a cosine schedule (alpha decays from 1.0 to 0.0 over warmup steps).

---

## 4. MoE-Everything

**Config type**: `moe_everything`
**Package**: `src/models/moe_everything/` (config.py, attention_bank.py, mlp_bank.py, model.py)
**Config class**: `MoEverythingConfig` (extends `Qwen3MoeConfig`)

The most advanced architecture. Routes **both** attention and MLP through shared expert banks, with a branch router deciding whether each token gets attention or MLP at each depth step. All expert weights are shared across depths (Universal Transformer style) -- only activations change per depth, not weights.

### 4.1 High-Level Forward Pass

For each depth step `d = 0, 1, ..., D-1`, the model first runs the
branch router. Tokens routed to attention run the attention bank; tokens
routed to MLP run the MLP bank.

```
hidden_states ─┬─> Branch Router ──> choice per token: ATTENTION (0) or MLP (1)
               │
               ├─ ATTENTION path (tokens with choice=0):
               │    no_recompute: project Q/K/V, refresh KV for attention tokens,
               │                  keep old KV for MLP tokens
               │    recompute_k:  project Q/V, recompute K tables from current hidden
               │    recompute_kv: project Q, recompute K/V tables from current hidden
               │    hidden += w_attn * attn_out
               │
               ├─ MLP path (tokens with choice=1):
               │    mlp_out = MlpExpertBank(hidden, token_mask=mlp_mask)
               │    hidden += w_mlp * mlp_out
               │
               └─> output: updated hidden_states
```

### 4.2 Three Core Components

The model has exactly three shared components, instantiated once and reused across all depth steps:

1. **Branch Router** (`BranchRouter`): Binary hard router -- ATTENTION vs MLP per token
2. **Attention Expert Bank** (`AttentionExpertBank`): Pool of attention projection experts (Q/K/V/O weight matrices)
3. **MLP Expert Bank** (`MlpExpertBank`): Pool of SwiGLU MLP experts (reuses `Qwen3MoeExperts`)

### 4.3 Branch Router -- Detailed

**Class**: `BranchRouter` (line 159)
**Parameters**: One linear layer `gate: Linear(hidden_size, 2, bias=False)`

Softmax branch routing (`branch_router.balancing: none`,
`sampling_entropy`, `aux_loss`, or `seq_aux_loss`) is a Switch-style
binary router:

```python
logits = gate(hidden_states)               # (B, T, 2) -- [attn, mlp]
scores = softmax(logits, dim=-1)           # scores sum to 1
choice = argmax(scores, dim=-1)            # hard decision: 0=attn, 1=mlp
# sampling_entropy uses Categorical(scores) during training instead.

attn_mask = (choice == 0).unsqueeze(-1)    # (B, T, 1) -- boolean mask
mlp_mask  = (choice == 1).unsqueeze(-1)

# Weights for scaling branch outputs (differentiable):
w_attn = scores[..., 0:1] * attn_mask
w_mlp  = scores[..., 1:2] * mlp_mask
```

DeepSeek branch routing (`branch_router.balancing: deepseek_bias`, or
the legacy `branch_deepseek: true`) uses two independent sigmoid scores,
not a two-class softmax:

```python
logits = gate(hidden_states)
scores = sigmoid(logits)                   # two independent scores
choice = argmax(scores + expert_bias, dim=-1)
attn_mask = (choice == 0).unsqueeze(-1)
mlp_mask = (choice == 1).unsqueeze(-1)

# Branch-output scaling uses the unbiased sigmoid score so gradients flow
# to the branch gate; expert_bias is a persistent buffer updated post-step.
w_attn = scores[..., 0:1] * attn_mask
w_mlp  = scores[..., 1:2] * mlp_mask
```

**How gradients flow through hard routing**: The `argmax` is
non-differentiable, but the output scaling multiplies the selected branch
output by the selected softmax probability or sigmoid score. This gives
the branch gate a gradient path even though the selection itself is hard.

**Branch exploration / ablations**: The focused qkvo + recompute-KV rows
compare `sampling_entropy`, `exploration_only`, and `fixed_alternating`.
`sampling_entropy` samples the ATTN/MLP categorical during training and
subtracts a small decaying entropy bonus from the loss. `exploration_only`
uses top-1 routing but randomly overrides a scheduled fraction of tokens.
`fixed_alternating` bypasses the learned branch gate entirely: even depths
run attention, odd depths run MLP.

**Shared vs per-layer**: By default, one `BranchRouter` is shared across all depths. With `per_layer_router=True`, each depth gets its own router (more parameters, more expressiveness).

### 4.4 KV State in No-Recompute Attention

KV state is carried across depth steps only in `per_head_no_recompute`.

- Tokens that chose **ATTENTION** at depth `d` get **fresh** K, V from the attention expert bank.
- Tokens that chose **MLP** at depth `d` **keep their old** K, V from the previous depth.

```python
# In _depth_step():
K_new = torch.where(attn_mask_kv, K_fresh, K_old)  # refresh only for attn tokens
V_new = torch.where(attn_mask_kv, V_fresh, V_old)  # MLP tokens keep stale KV
```

This means attention tokens from depth `d` attend using a mix of:
- Fresh KV from tokens that also chose attention at depth `d`
- Stale KV from tokens that chose MLP (their KV is from the last depth they chose attention)

For `per_head_recompute_k` and `per_head_recompute_kv`, this KV
persistence is not used. The recompute attention path has no carried
`kv_state`: it builds the needed K or K/V tables from the current hidden
state at each depth and returns only the updated hidden states.

### 4.5 Dense Execution Only

MoE-Everything attention now uses dense execution only. The branch router still
decides which tokens take the attention branch, but the attention bank computes
full-token attention tensors and then applies the branch mask to the residual
and KV update.

We removed the sparse gather/packed-query execution path after H100
forward/backward throughput tests showed dense was consistently faster. Sparse
had fewer theoretical query rows, but the extra gather/scatter, ragged packing,
and smaller attention launches dominated wall-clock time.

For no-recompute attention, dense execution means:

```text
project Q/K/V for all tokens
blend fresh K/V only for tokens that chose attention
run one full attention call
apply the attention residual only to attention-routed tokens
```

For recompute attention, the grouped execution is:

```text
recompute_k:
  for each active K expert:
    build K_e over the full sequence
    run full-query attention, then keep rows whose Q/K route is e

recompute_kv:
  for each active (K expert, V expert) pair:
    build/reuse K_e and V_v over the full sequence
    run full-query attention, then keep rows whose route is (e, v)
```

This does extra attention rows compared with an ideal fused ragged grouped
kernel, but H100 measurements favored dense full-query launches for the current
PyTorch/SDPA implementation. The extra model cost in recompute modes is
recomputing K, or K and V, tables for active expert routes.

---

### 4a. No-Recompute Mode

**Config**: `attn_expert_mode: per_head_no_recompute`

No-recompute attention keeps the standard transformer attention call. Q, K,
V, and O route independently, and the only supported bundle is:

```yaml
attn_routing_bundle: q_k_v_o
```

This is the most expressive routing layout: Q, K, V, and O can all choose
different experts for the same token/head slot. The cost is that Q and K can
come from different learned subspaces, so the attention score may compare
projections that were not selected as a pair.

The runtime keeps a KV state across depth calls. Tokens routed to attention
write fresh K/V; tokens routed to MLP keep their prior K/V.

### 4b. Recompute Attention

**Config**: `attn_expert_mode: per_head_recompute_k` or
`attn_expert_mode: per_head_recompute_kv`

Recompute attention is the aligned-Q/K path. Q and K always share a route, so
the dot product is computed between matched projections. Recompute has two
variants:

- `per_head_recompute_k`: recomputes the full-sequence K table for each active Q/K expert. V is token-routed once.
- `per_head_recompute_kv`: recomputes full-sequence K and V tables for each active routed pair.

O is never sequence-side; it is projected after attention from the local
attention output. It can share a routing decision with Q/K/V, but there is no
separate "recompute O" mode.

The active launch configs use `qk_v_o`: Q/K share the attention-metric route,
V routes separately, and O routes separately after attention.

The active launch configs also set all router and norm state per depth:
branch routers, MLP routers, attention routers, pre-RMSNorms, and Q/K norm
weights are depth-indexed.

The routing bundle controls which projections share a router:

| Bundle | Routing |
|--------|---------|
| `qk_v_o` | Q/K together, V separate, O separate |
| `qk_vo` | Q/K together, V/O together |
| `qkv_o` | Q/K/V together, O separate |
| `qkvo` | Q/K/V/O together |

`q_k_v_o` is intentionally not valid for recompute modes because Q and K
would no longer be aligned.

EMA router context is an optional router-input feature:

```yaml
attn_router_context: ema_qk_v
attn_router_context_decay: 0.95
```

When enabled, only the QK and V routers consume
`concat(normed_h_t, causal_prefix_ema_t)`. The projections still use
`normed_h_t`, and O routing still consumes the local attention output.

The recompute implementation is split into three phases:

1. `_build_per_head_recompute_tables()` routes Q/K, optional V, then builds token-local Q and, for `recompute_k`, the token-routed V table.
2. `_run_per_head_recompute_expert_tables()` builds/reuses active full-sequence K or K/V expert tables and runs attention only for the query rows assigned to each table.
3. `_project_recompute_o()` applies the configured O route from the local attention output.

#### Comparison

| Aspect | No recompute | Recompute K | Recompute KV |
|--------|--------------|-------------|--------------|
| Q/K route | Separate | Shared | Shared |
| V route | Separate | Bundle-dependent token route | Bundle-dependent query-pair route |
| O route | Separate | Bundle-dependent local route | Bundle-dependent local route |
| KV state across depths | Yes | No | No |
| Sequence-side recompute | None | K | K and V |
| Main tradeoff | Maximum routing freedom | Fixes Q/K mismatch with lower cost | Strongest alignment, highest cost |

### 4.5 MLP Expert Bank

**Class**: `MlpExpertBank` (uses `Qwen3MoeExperts` internally)

Standard top-K MoE over SwiGLU experts, reusing the HuggingFace `Qwen3MoeExperts` implementation. Each expert is a `gate_proj + up_proj -> SiLU gate -> down_proj` SwiGLU network.

The MLP bank has its own router (can be shared or per-layer via `per_layer_mlp_router`). Only tokens that chose MLP in the branch router are processed -- the token_mask from branch routing is passed to restrict computation.

### 4.6 MoE-Everything Configuration Reference

Key parameters in `MoEverythingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_attn_experts` | 4 | Size of attention expert pool (E) |
| `num_attn_experts_per_tok` | 1 | Attention experts selected per token |
| `attn_expert_mode` | `per_head_no_recompute` | `per_head_no_recompute`, `per_head_recompute_k`, or `per_head_recompute_kv` |
| `attn_routing_bundle` | `q_k_v_o` for no-recompute, `qkvo` for recompute | `q_k_v_o`, `qk_v_o`, `qk_vo`, `qkv_o`, `qkvo` |
| `attn_router_context` | `none` | `none` or `ema_qk_v`; EMA applies only to QK/V routers |
| `attn_router_context_decay` | 0.95 | Causal prefix EMA decay for `ema_qk_v` |
| `branch_router_aux_loss_coef` | 0.0 | Aux loss for branch routing balance |
| `use_deepseek_routing` | False | Sigmoid + bias vs softmax routing for expert routers |
| `per_layer_router` | False | Separate branch routers per depth vs shared |
| `per_layer_mlp_router` | False | Separate MLP routers per depth |
| `per_layer_attn_router` | False | Separate attention routers per depth |
| `per_layer_norm` | False | Separate RMSNorm per depth (standard transformer style) |
| `per_layer_qk_norm` | False | Shared per-depth Q/K norms instead of per-expert norms |
| `post_norm` | False | Apply RMSNorm to branch output before residual |
| `scale_branch_by_routing_weight` | True | Scale branch output by the selected softmax probability or sigmoid score |
| `scale_attn_by_routing_weight` | True | Scale expert projections by routing weight |
| `router_exploration_rate` | 0.0 | Random expert probability during training |
| `branch_router_exploration_rate` | None | Override exploration rate for branch router |
| `dynamic_depth_min/max` | 1.0 | Random depth perturbation range during training |
| `routed_norm` | False | Bank of RMSNorm experts with per-token routing |
| `depthwise_attention` | False | Learned weighted combination across depths |
| `sanity_check_mode` | None | `alternating_global_moe` for deterministic debugging |
| `prelude_layers` | 0 | Standard-MoE-style decoder blocks prepended before the recurrent bank loop (see §4.7) |
| `coda_layers` | 0 | Standard-MoE-style decoder blocks appended after the recurrent bank loop (see §4.7) |
| `boundary_num_experts` | None | Per-layer MLP pool size used by every prelude/coda block (falls back to bank's `num_experts` when unset) |
| `boundary_num_experts_per_tok` | None | Top-K for prelude/coda MLP routers (falls back to bank's `num_experts_per_tok`) |
| `boundary_moe_intermediate_size` | None | SwiGLU intermediate dim for prelude/coda experts (falls back to bank's `moe_intermediate_size`) |
| `boundary_num_groups` / `boundary_group_topk` | None | Group-limited top-K for prelude/coda routers (falls back to bank's values) |
| `boundary_router_type` | `deepseek` | `deepseek` (DeepSeekRouter + bias updates) or `softmax` (ExplorationTopKRouter) for prelude/coda gates |

### 4.7 Prelude / Recurrent / Coda Hybrid

`prelude_layers` and `coda_layers` opt-in to a hybrid layout: a stack
of standard-MoE-style decoder blocks runs **before** the recurrent
MoE-Everything bank, and another stack runs **after** it. The recurrent
loop in between is unchanged.

```
embeddings
   ↓
Prelude block 1..N      ← dense GQA + per-layer MLP expert pool
   ↓                       (each block has its own attention weights
   ...                      and its own MLP experts; nothing shared
   ↓                       with the bank)
Prelude block N
   ↓
Recurrent depth 0..D-1  ← MoE-Everything (shared attention bank +
   ↓                       shared MLP bank + branch router; weights
   ...                      shared across all D depths)
   ↓
Recurrent depth D-1
   ↓
Coda block 1..M         ← dense GQA + per-layer MLP expert pool
   ↓                       (own weights again, no sharing with bank
   ...                      or with prelude)
   ↓
Coda block M
   ↓
final RMSNorm → lm_head
```

Defaults of `prelude_layers=0`, `coda_layers=0` keep the model
bit-identical to a pure MoE-Everything build. When either is nonzero,
boundary blocks are built off a separate `Qwen3MoeConfig` derived from
the `boundary_*` fields (or from the bank's geometry when an explicit
boundary_* override is None). The boundary's MLP router gate is then
swapped to a `DeepSeekRouter` post-construction so the deepseek-bias
balancing walker registers it as a regular MLP balancing owner.

**Counting layer-equivalents**: each prelude/coda block runs one
attention substep + one MLP substep sequentially (= 1 standard layer
= 2 substeps); each recurrent depth runs exactly one substep (the
branch router picks attention OR MLP), so two recurrent depths equal
one standard layer. Compute totals are therefore:

```
layer_equivalents = prelude_layers + num_hidden_layers / 2 + coda_layers
substeps          = prelude_layers * 2 + num_hidden_layers + coda_layers * 2
```

A config that sets `prelude_layers=4, num_hidden_layers=16,
coda_layers=4` has `4 + 8 + 4 = 16` layer-equivalents (32 substeps),
matching the pure-16-layer (32-depth) MoE-Everything baselines.

**Use case**: ablations where the entry and exit transformations are
held to a standard MoE recipe while the middle of the stack explores
shared-weight branch routing — see
`configs/16_layers/moe_everything_prelude4_recurrent16_coda4_branch_*.yaml`.

---

## 5. Architecture Comparison

| Feature | Standard LLM | Standard MoE | Global MoE | MoE-Everything |
|---------|-------------|-------------|------------|----------------|
| **Attention** | Dense | Dense | Dense | Routed per-head experts |
| **MLP** | Dense | Routed (per-layer pool) | Routed (global pool) | Routed (shared bank) |
| **Expert ownership** | N/A | Per-layer | Global shared | Global shared |
| **Branch routing** | No | No | No | Yes (attn vs MLP) |
| **Depth sharing** | No | No | No | Yes (Universal Transformer style) |
| **Backbone** | Qwen3 | Qwen3MoE | Qwen3MoE (modified) | Custom (Qwen3MoE-based) |
| **Typical scales** | 0.6B-30B | 0.6B-30B | 0.6B-30B | 0.6B-30B |
| **Active params** | All | ~top-K/E fraction | ~top-K/E fraction | ~top-K/E fraction |

---

## 6. Advanced Features (from Speedrun)

These features originated in the modded-nanogpt speedrun models and may be ported to the unified codebase where beneficial.

### FlexAttention with Document Masking + Sliding Window

**Origin**: `legacy/speedrun/speedrun_gpt.py:277-315`

PyTorch's FlexAttention API allows defining custom block-level attention masks efficiently. Our implementation combines three masking strategies:

1. **Causal masking**: Standard autoregressive mask (token can only attend to earlier tokens)
2. **Document boundary masking**: When packing multiple documents into one sequence, prevents attention across document boundaries. Documents are detected by BOS token (id 50256) using `cumsum` to assign document IDs.
3. **Sliding window**: Limits each token's attention to a window of recent tokens, reducing memory from O(T^2) to O(T * window).

The implementation works at block granularity (128 tokens per block) for efficiency, with token-level refinement at block boundaries:

```python
docs = (input_seq == 50256).cumsum(0)  # per-token document ID
# Block-level: "any overlap" vs "fully contained" masks
# Token-level: exact causal + document mask applied within partial blocks
```

**Layer pattern**: Alternating long/short windows across layers: `[long, short, short, short, long, short, short, long, short, short, short, long]`. Early and late layers see global context; middle layers focus on local patterns.

**Portability**: FlexAttention is a general PyTorch feature. It can be applied to any model that uses packed multi-document sequences.

---

### Dynamic Window Sizing

**Origin**: `train_torch.py:817-826`

The sliding window size for FlexAttention grows linearly during training:

```python
window_size = next_multiple_of_128(1728 * (step / num_iterations))
```

- **Step 0**: Window = 0 blocks (purely local attention)
- **Step N/2**: Window = ~864 tokens (~6.75 blocks)
- **Step N**: Window = 1728 tokens (~13.5 blocks)

**Why**: Early in training, the model learns local patterns (n-grams, syntax). Gradually expanding the window introduces longer-range dependencies as the model matures. This is computationally cheaper than full-context attention from the start, and provides a natural curriculum from local to global context.

**Applicability**: This is a training strategy, not a model architecture feature. It can be applied to any model using FlexAttention with sliding windows.

---

### Sigmoid Logit Softcapping

**Origin**: `legacy/speedrun/speedrun_gpt.py:348-349`

```python
logits = 30 * torch.sigmoid(logits / 7.5)
```

This bounds the logits to the range [0, 30] before cross-entropy loss, following the Gemma 2 paper.

**What it solves**: During training, logits can grow to extreme values (100+), causing:
- Numerical instability in softmax/cross-entropy
- Sharp probability distributions that are hard to learn from
- Gradient explosion on incorrect high-confidence predictions

**How it works**:
- `logits / 7.5`: Rescales so sigmoid operates in its sensitive range (~[-7.5, 7.5])
- `sigmoid(...)`: Squashes to [0, 1], preventing extreme values
- `* 30`: Rescales to [0, 30], which is the typical useful range for vocabulary logits

**The constants**: 30 was chosen to match typical logit magnitudes for confident predictions. 7.5 controls the "sharpness" -- it's the half-width of the sigmoid's active zone. Together they create a smooth soft-clamp.

---

### Learnable Skip Connections (U-net Design)

**Origin**: `legacy/speedrun/speedrun_gpt.py:260-344`

The model learns scalar weights that control residual connections in a U-net pattern:

```
Layer 0 ──> save ─────────────────────────────────> Layer 6: x += skip_weight[0] * saved[0]
Layer 1 ──> save ───────────────────────> Layer 7:  x += skip_weight[1] * saved[1]
Layer 2 ──> save ─────────────> Layer 8:  x += skip_weight[2] * saved[2]
...
Layer 5 ──> save ──> Layer 11: x += skip_weight[5] * saved[5]
```

Three sets of learnable scalars per layer:

1. **Skip weights** (`num_layers/2` scalars, init=1.0): Control how much the U-net skip connection contributes. Layer `i` (for `i >= num_layers/2`) receives: `x += skip_weight[i-n] * hidden_state_from_layer[i-n]`

2. **Block lambdas** (2 per layer, init=[1.0, 0.0]): Blend current hidden state with original embeddings at each layer entry: `x = lambda[0] * x + lambda[1] * x0`. Initially passes through current state only; can learn to mix in raw embeddings.

3. **SA lambdas** (2 per layer, init=[0.5, 0.5]): Blend attention values with "value embeddings" (separate embedding tables): `v = sa_lambda[0] * v + sa_lambda[1] * ve`. Starts at 50/50 mix.

**Why U-net**: Allows later layers to directly access early representations, helping with gradient flow and feature reuse. The learnable weights let the model decide how much skip information to use.

**Learning rate**: These scalars use `lr_mul = 5.0` (5x the base learning rate) because they need to adapt quickly to find the right blending.

---

### Gated Attention

**Origin**: `legacy/speedrun/speedrun_gpt.py:183-201`

Each attention head has a learned gate that can suppress or amplify its output:

```python
# Project first 12 dims of input to per-head gates
gate = sigmoid(linear(x[..., :12]))  # shape: [B, T, num_heads]
# Multiply attention output by gates
y = attention_output * gate.unsqueeze(-1)  # per-head scaling
```

**Mechanism**:
1. Takes the first 12 dimensions of the input (a low-rank signal, ~1.6% of 768-dim hidden state)
2. Projects to `num_heads` values via a learned linear layer
3. Applies sigmoid to get gates in [0, 1]
4. Multiplies each head's output by its gate

**Initialization**: Gate weights start at zero, so `sigmoid(0) = 0.5` -- all heads pass through at half strength initially. The model learns to increase or decrease each head's contribution per token.

**Why**: Allows the model to dynamically suppress irrelevant attention heads for specific tokens. This is a lightweight form of conditional computation -- the model can effectively "turn off" heads that aren't useful for a given input, without the overhead of full expert routing.

---

### Kernel Warmup

**Origin**: `train_torch.py:1044-1062`

Before real training begins, runs a few forward-backward-optimizer steps with real data, then **throws away all the updates** and resets to the initial state.

```python
# Save initial state
initial_state = {model_state, optimizer_states}

# Run N warmup steps (real data, real computation)
for _ in range(warmup_steps):
    loss = model(batch).backward()
    optimizer.step()

# Discard everything and restore
model.load_state_dict(initial_state["model"])
optimizer.load_state_dict(initial_state["optimizers"])
```

**Why**: The first few iterations of training are abnormally slow because:
1. **CUDA kernel compilation**: PyTorch compiles and caches GPU kernels on first use. Complex kernels (FlexAttention, Triton grouped GEMM, FP8 matmul) take seconds to compile.
2. **GPU cache warming**: Memory allocator, L2 cache, and TLB need to see typical access patterns.
3. **torch.compile graphs**: If using `torch.compile`, the compilation happens on first call.

After warmup, the GPU is "hot" and all kernels are compiled. Resetting to initial state means the actual training measurements are clean and consistent, without compilation jitter in the first few steps.

**Cost**: A few seconds of discarded computation. Benefit: clean training metrics and consistent step times from step 1.
