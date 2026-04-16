# Model Architecture

This document covers all model families, their layer designs, and advanced architectural features.

## Table of Contents

- [1. Standard LLM (Dense)](#1-standard-llm-dense)
- [2. Standard MoE](#2-standard-moe)
- [3. Global MoE](#3-global-moe)
- [4. MoE-Everything](#4-moe-everything)
  - [4a. Fully Independent Mode](#4a-fully-independent-mode)
  - [4b. Precompute KV Mode](#4b-precompute-kv-mode)
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

For each depth step `d = 0, 1, ..., D-1`, the `_depth_step()` method runs:

```
hidden_states ─┬─> Branch Router ──> choice per token: ATTENTION (0) or MLP (1)
               │
               ├─ ATTENTION path (tokens with choice=0):
               │    hidden -> AttentionExpertBank.project() -> Q, K_fresh, V_fresh
               │    K_new = where(attn_mask, K_fresh, K_old)   # keep old KV for MLP tokens
               │    V_new = where(attn_mask, V_fresh, V_old)
               │    attn_out = AttentionExpertBank.attend(Q, K_new, V_new)
               │    hidden += w_attn * attn_out                # w_attn = softmax prob of attn choice
               │
               ├─ MLP path (tokens with choice=1):
               │    mlp_out = MlpExpertBank(hidden, token_mask=mlp_mask)
               │    hidden += w_mlp * mlp_out                  # w_mlp = softmax prob of mlp choice
               │
               └─> output: updated hidden_states, K_new, V_new
```

### 4.2 Three Core Components

The model has exactly three shared components, instantiated once and reused across all depth steps:

1. **Branch Router** (`BranchRouter`): Binary hard router -- ATTENTION vs MLP per token
2. **Attention Expert Bank** (`AttentionExpertBank`): Pool of attention projection experts (Q/K/V/O weight matrices)
3. **MLP Expert Bank** (`MlpExpertBank`): Pool of SwiGLU MLP experts (reuses `Qwen3MoeExperts`)

### 4.3 Branch Router -- Detailed

**Class**: `BranchRouter` (line 159)
**Parameters**: One linear layer `gate: Linear(hidden_size, 2, bias=False)`

```python
# Forward pass (simplified):
logits = gate(hidden_states.float())       # (B, T, 2) -- two logits: [attn_score, mlp_score]
probs = softmax(logits, dim=-1)            # (B, T, 2) -- probabilities sum to 1
choice = argmax(probs, dim=-1)             # (B, T) -- hard decision: 0=attn, 1=mlp

attn_mask = (choice == 0).unsqueeze(-1)    # (B, T, 1) -- boolean mask
mlp_mask  = (choice == 1).unsqueeze(-1)

# Weights for scaling branch outputs (differentiable):
w_attn = probs[..., 0:1] * attn_mask      # 0 for MLP tokens, prob for attn tokens
w_mlp  = probs[..., 1:2] * mlp_mask       # 0 for attn tokens, prob for mlp tokens
```

**How gradients flow through hard routing**: The `argmax` is non-differentiable, but the output scaling `w_attn * attn_out` multiplies the attention output by the softmax probability of the attention choice. This means the router receives gradients through the probability value, even though the selection itself is hard. This is the same pattern used in standard MoE expert routing.

**Exploration**: During training, `exploration_rate` fraction of tokens get random branch assignments instead of argmax, preventing routing collapse.

**Shared vs per-layer**: By default, one `BranchRouter` is shared across all depths. With `per_layer_router=True`, each depth gets its own router (more parameters, more expressiveness).

### 4.4 KV State Persistence Across Depths

A critical design detail: tokens carry (K, V) state across depth steps.

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

For `per_head_precompute_kv` mode, this KV persistence is not used -- K and V are precomputed per-expert from the full hidden state at each depth (see section 4b below).

### 4.5 Sparse vs Dense Execution

The attention bank dynamically chooses between sparse and dense execution based on what fraction of tokens chose attention:

```python
def should_use_sparse_path(token_mask):
    attn_fraction = token_mask.float().mean()
    if attn_fraction < 0.75:    # threshold (configurable)
        return True              # sparse: only compute for routed tokens
    else:
        return False             # dense: compute for all tokens (less overhead)
```

**Sparse path**: Only processes tokens that chose attention. Projects only selected tokens through expert weight banks, runs attention only on those positions. More efficient when < 75% of tokens chose attention.

**Dense path**: Processes all tokens through projection, then masks out MLP tokens. Simpler GPU kernel launches, better for high attention fractions.

---

### 4a. Fully Independent Mode (Deep Dive)

**Config**: `attn_expert_mode: per_head_fully_independent`
**Init method**: `_init_per_head_fully_independent()` (line 648)

This is the most expressive attention routing mode. Each projection type (Q, K, V, O) has its own independent router, and each router selects from its own expert pool.

#### Why it's called "Fully Independent"

Q, K, and V are routed **independently** — for the same head slot, Q might come from expert 7 while K comes from expert 12 and V from expert 3. This means Q and K can live in **different learned subspaces**. Token at position `t` with Q from expert 7 may attend against token at position `s` with K from expert 12 — the dot product is between weight matrices that were never trained together. This is the fundamental tradeoff: **maximum routing flexibility at the cost of Q-K subspace alignment**.

**Cost**: 1 standard attention call per depth — same as a normal transformer. Very cheap.

#### Router structure

Each head slot has its **own dedicated router** doing **top-1** from the expert pool. This is NOT one router picking top-K — it is H separate routers each picking top-1.

- Q: H routers (one per head), each selecting 1 expert → H experts total
- K: H routers (one per head), each selecting 1 expert → H experts total
- V: H routers (one per head), each selecting 1 expert → H experts total
- O: H routers (one per head, routing on attention output), each selecting 1 expert → H experts total
- Total: **4H routers** per depth

Each router is `nn.Linear(dim, num_experts)` → top-1. Different head-slot routers learn to specialize independently.

> **Implementation**: `src/models/moe_everything/attention_bank.py` uses per-head top-1 routers created via `src/models/routing/helpers.py:make_top1_router()`.

#### Weight Banks

Four separate expert weight banks, stored as 3D parameter tensors:

```python
self.q_proj = Parameter(E, hidden_size, q_group_dim)      # Q experts
self.k_proj = Parameter(E_kv, hidden_size, head_dim)       # K experts
self.v_proj = Parameter(E_kv, hidden_size, head_dim)       # V experts
self.o_proj = Parameter(E_o, head_dim, hidden_size)        # O experts
```

Where:
- `E` = `num_attn_experts` (e.g., 4)
- `E_kv` = `E` (same pool size for K, V)
- `E_o` = `E * q_heads_per_kv` (larger pool for O to match per-layer O projection parameter count)
- `q_group_dim` = `q_heads_per_kv * head_dim` (GQA: each Q expert produces queries for all Q heads in a KV group)

#### Per-Head Routers (4H total)

```python
# H routers per projection type, each doing top-1
q_routers = ModuleList([Router(hidden_size -> E) for _ in range(H)])   # H routers, each picks 1 expert
k_routers = ModuleList([Router(hidden_size -> E) for _ in range(H)])   # H routers, each picks 1 expert
v_routers = ModuleList([Router(hidden_size -> E) for _ in range(H)])   # H routers, each picks 1 expert
o_routers = ModuleList([Router(head_dim -> E) for _ in range(H)])      # H routers, each picks 1 expert
```

Each head slot has its own router that independently picks one expert. With H=6 heads, that's 24 routers per depth doing top-1, not 4 routers doing top-6.

#### Forward Flow (per token)

```
Token x (hidden_size=1024)
│
├─ Q path (H = num_kv_heads separate top-1 routers):
│   x -> q_pre_norm
│   For each head slot h in [0, H):
│     q_routers[h](x) -> picks 1 expert e_h with weight w_h
│     q_h = x @ q_proj[e_h]  # (hidden_size) -> (q_group_dim)
│     q_h = RMSNorm(q_h, q_norm_weight[e_h])
│     q_h *= w_h
│   Stack: Q = [q_0, q_1, ..., q_{H-1}]
│   Reshape: Q -> (num_heads, head_dim) via GQA unfolding
│
├─ K path (H separate top-1 routers):
│   x -> k_pre_norm
│   For each head slot h: k_routers[h](x) -> picks 1 expert
│   Stack: K = [k_0, ..., k_{H-1}] with per-expert RMSNorm
│
├─ V path (H separate top-1 routers):
│   x -> v_pre_norm
│   For each head slot h: v_routers[h](x) -> picks 1 expert
│   Stack: V = [v_0, ..., v_{H-1}]
│
├─ Apply RoPE to Q, K
├─ Run standard attention: attn_out = Attention(Q, K, V)
│
└─ O path (num_heads separate top-1 routers):
    attn_flat = concat(all head outputs) -> (num_heads * head_dim)
    For each head h: o_routers[h](attn_flat) -> picks 1 expert
    o_h = attn_heads[h] @ o_proj[e_h] * w_h  # head_dim -> hidden_size
    Sum across heads: output = sum(o_0, ..., o_{num_heads-1})
```

**Key insight**: Each head slot has its own dedicated router that picks exactly one expert via top-1 selection. This is NOT one router picking top-K — it is H separate routers each independently picking top-1. Different tokens route different experts to the same head slot.

#### Grouped Expert MatMul (Efficient Execution)

Instead of looping over experts one by one, the implementation sorts tokens by their expert assignment and uses Triton grouped GEMM:

```python
# Sort tokens by expert assignment
sort_order = argsort(expert_indices)
sorted_inputs = inputs[sort_order]
sorted_experts = expert_indices[sort_order]
unique_experts, counts = unique_consecutive(sorted_experts)

# Single batched matmul across all expert groups
proj = triton_grouped_gemm(sorted_inputs, weight_bank, unique_experts, counts)

# Unsort back to original token order
output[sort_order] = proj * expert_weights
```

This is much faster than per-expert loops on GPU because it maximizes parallelism.

---

### 4b. Precompute KV Mode (Deep Dive)

**Config**: `attn_expert_mode: per_head_precompute_kv`
**Init method**: `_init_per_head_precompute_kv()` (line 693)

This mode uses a **single router** to select experts for all four projections (Q, K, V, O). The key optimization: since all tokens in a batch see the same expert's K and V weights, K and V can be precomputed once per expert and reused across all tokens routed to that expert.

#### Why it's called "Precompute KV"

Because the same expert provides both Q and K, we know which K/V weight matrix each head slot will use before running attention. So we can **precompute the full K/V tables** for each active expert once, then run attention against those tables. This guarantees **Q-K subspace alignment** — Q and K always come from the same learned projection, so dot-product attention scores are always meaningful.

**Cost**: `num_active_experts` separate full attention passes per depth. More expensive than fully independent, but guarantees correctness of attention.

#### Router structure

Each head slot has its **own dedicated router** doing **top-1** — one routing decision per head that picks Q+K+V+O together from the same expert.

- H routers (one per head slot), each selecting 1 expert → H experts total
- Same expert index provides Q, K, V, and O for that head slot
- Total: **H routers** per depth

Each router is `nn.Linear(dim, num_experts)` → top-1. The bundled decision means Q and K always come from the same learned subspace.

> **Implementation**: `src/models/moe_everything/attention_bank.py` uses per-head top-1 routers for precompute-KV mode.

#### Weight Banks

```python
self.q_proj = Parameter(E, hidden_size, q_group_dim)    # Q experts (produces grouped Q)
self.k_proj = Parameter(E, hidden_size, head_dim)        # K experts
self.v_proj = Parameter(E, hidden_size, head_dim)        # V experts
self.o_proj = Parameter(E, q_group_dim, hidden_size)     # O experts (takes grouped Q dim)
```

Note: all four banks have the same number of experts `E`, and the O projection takes `q_group_dim` input (not `head_dim`) because it operates on the grouped-query output.

#### Per-Head Bundled Routers (H total)

```python
# H routers, each doing top-1 (one bundled QKVO decision per head)
routers = ModuleList([Router(hidden_size -> E) for _ in range(H)])   # H routers, each picks 1 expert
```

Each head slot has its own router that picks one expert. That single expert provides Q, K, V, and O projections together for that head. With H=6 heads, that's 6 routers per depth doing top-1, not 1 router doing top-6.

#### Forward Flow

The forward is split into two phases: table building and attention execution.

**Phase 1: Build Expert Tables** (`_build_per_head_precompute_kv_tables`, line 1127)

```
Token x (hidden_size=1024)
│
├─ x -> norm -> router -> selects num_kv_heads expert indices [e_0, e_1, ..., e_{num_kv_heads-1}]
│                         with routing weights [w_0, w_1, ..., w_{num_kv_heads-1}]
│
├─ For each selected expert e_i (using SAME index for Q, K, V):
│   q_group_i = x @ q_proj[e_i]  # (hidden_size) -> (q_group_dim = q_heads_per_kv * head_dim)
│   k_i       = x @ k_proj[e_i]  # (hidden_size) -> (head_dim)
│   v_i       = x @ v_proj[e_i]  # (hidden_size) -> (head_dim)
│   Apply RMSNorm to q (per head within group) and k
│   Scale by routing weight w_i
│
├─ Reshape Q from (num_kv_heads, q_group_dim) -> (num_heads, head_dim)
│  (Each KV group's q_group unfolds into q_heads_per_kv separate Q heads)
│
└─ Apply RoPE to Q and K
   Return: Q (B, num_heads, T, head_dim), K (B, num_kv_heads, T, head_dim), V same as K
```

**Phase 2: Per-Expert Attention** (`_run_per_head_precompute_kv_expert_tables`, line 1206)

This is where the "precompute KV" optimization happens. Instead of standard attention where all tokens share one K/V, we run **separate attention per active expert**:

```
For each active expert e in the batch:
  # Precompute K, V for ALL tokens using expert e's weights
  K_e = all_tokens @ k_proj[e]           # (B, T, head_dim) -- computed ONCE for expert e
  V_e = all_tokens @ v_proj[e]           # (B, T, head_dim) -- computed ONCE for expert e
  K_e = RMSNorm(K_e) then RoPE(K_e)

  # Run full attention: Q attends to K_e, V_e
  attn_e = Attention(Q, K_e, V_e, causal_mask)    # (B, num_heads, T, head_dim)

  # Mask: only keep results for (head, token) pairs that were actually routed to expert e
  group_mask = (token_expert_assignments == e)     # which KV groups selected this expert
  head_mask = expand group_mask to per-head        # repeat for all Q heads in each KV group
  attn_output += attn_e * head_mask                # accumulate masked results
```

**Why this is efficient**: Instead of running `num_tokens * num_kv_heads` separate small attention operations (one per token-head pair), we run at most `num_active_experts` full attention operations. Since `num_active_experts << num_tokens * num_kv_heads`, this is much faster on GPU. The "redundant" computation on tokens not routed to expert `e` is thrown away by the mask, but the GPU parallelism more than compensates.

**Phase 3: O Projection**

After attention, the O projection uses the same expert indices from the router:

```
attn_output (B, T, q_dim) -> for each expert e_i:
  o_i = attn_output_group_i @ o_proj[e_i]    # (q_group_dim) -> (hidden_size)
  Scale by routing weight w_i
Sum across KV groups -> final output (B, T, hidden_size)
```

#### Fully Independent vs Precompute KV -- Comparison

| Aspect | Fully Independent | Precompute KV |
|--------|-------------------|---------------|
| **Routers** | 4H (H per Q, K, V, O — each top-1) | H (one bundled QKVO per head — each top-1) |
| **Expert selection** | Different expert per Q, K, V, O | Same expert for all four |
| **Q/K norm** | Per-expert (from bank) or per-layer | Per-expert (from bank) or per-layer |
| **O pool size** | `E * q_heads_per_kv` (larger) | `E` (same as others) |
| **O input** | Per-head `head_dim` | Grouped `q_group_dim` |
| **KV computation** | Per-token: each token gets its own KV | Per-expert: KV precomputed once per expert for ALL tokens |
| **Attention** | Standard (all tokens share K/V) | Per-expert (separate attention per active expert) |
| **KV state persistence** | Yes (tokens carry K, V across depths) | No (K, V recomputed each depth from hidden states) |
| **Expressiveness** | Maximum (4H independent routing decisions) | Lower (H routing decisions, each controls all 4 projections) |
| **Efficiency** | More routing overhead | Fewer routing decisions, KV reuse across tokens |
| **Aux losses** | 4H separate load-balancing losses (H per Q, K, V, O) | H load-balancing losses (one per bundled router) |

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
| `attn_expert_mode` | `per_head_fully_independent` | `per_head_fully_independent` or `per_head_precompute_kv` |
| `branch_router_aux_loss_coef` | 0.0 | Aux loss for branch routing balance |
| `use_deepseek_routing` | False | Sigmoid + bias vs softmax routing |
| `per_layer_router` | False | Separate branch routers per depth vs shared |
| `per_layer_mlp_router` | False | Separate MLP routers per depth |
| `per_layer_attn_router` | False | Separate attention routers per depth |
| `per_layer_norm` | False | Separate RMSNorm per depth (standard transformer style) |
| `per_layer_qk_norm` | False | Shared per-depth Q/K norms instead of per-expert norms |
| `post_norm` | False | Apply RMSNorm to branch output before residual |
| `scale_branch_by_routing_weight` | True | Scale branch output by softmax probability |
| `scale_attn_by_routing_weight` | True | Scale expert projections by routing weight |
| `router_exploration_rate` | 0.0 | Random expert probability during training |
| `branch_router_exploration_rate` | None | Override exploration rate for branch router |
| `per_head_compute_mode` | `auto` | `auto`, `sparse`, or `dense` execution |
| `per_head_dense_fraction_threshold` | 0.75 | Switch to sparse when attn fraction < threshold |
| `dynamic_depth_min/max` | 1.0 | Random depth perturbation range during training |
| `routed_norm` | False | Bank of RMSNorm experts with per-token routing |
| `depthwise_attention` | False | Learned weighted combination across depths |
| `sanity_check_mode` | None | `alternating_global_moe` for deterministic debugging |

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
