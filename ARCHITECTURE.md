# Architecture Overview

This repo implements three MoE transformer architectures at increasing levels
of routing generality.  All share the same Qwen3 backbone (GQA, QK-norm, RoPE,
SwiGLU) and differ only in how expert weights are organized and routed.

---

## 1. Standard MoE

**File:** `src/models/standard_moe.py`
**Config:** `configs/standard_moe.yaml`

Vanilla Qwen3-MoE.  Each layer owns its own independent set of MLP experts.
Attention is standard (not routed).

```
┌─────────────────────────────────────────────────────────┐
│                    Standard MoE Layer                    │
│                                                         │
│   hidden ─→ RMSNorm ─→ GQA Attention ─→ + residual     │
│                          (Q,K,V,O)                      │
│                                                         │
│   hidden ─→ RMSNorm ─→ ┌──────────────┐ ─→ + residual  │
│                         │  Router       │               │
│                         │  ↓ top-k      │               │
│                         │ ┌──┬──┬──┬──┐ │               │
│                         │ │E1│E2│..│En│ │               │
│                         │ └──┴──┴──┴──┘ │               │
│                         │  SwiGLU each  │               │
│                         └──────────────┘                │
│                         Per-layer experts               │
│                         (16 experts × 16 layers = 256 total) │
└─────────────────────────────────────────────────────────┘
```

**Key details (current XS config):**
- 16 layers, each with 16 MLP experts, top-4 per token
- Attention: 16 heads, 8 KV heads (GQA 2:1), `head_dim=128`
- Expert FFN: SwiGLU with `moe_intermediate_size=768`
- Routing: DeepSeek V3 style sigmoid + expert bias + group-limited top-k
- Load balancing: batch aux disabled (`router_aux_loss_coef=0.0`), sequence-level aux enabled (`seq_aux_loss_coef=0.0001`)
- Fix vs HuggingFace: no double-softmax bug; sequence-level aux is also supported

**Parameter structure:**
```
Layer l:
  attention: Q[H,H], K[H,kv], V[H,kv], O[H,H]   (shared across tokens)
  mlp:       router[H, 16]                         (per-layer)
             16 × {gate_up[H, 2×I], down[I, H]}   (per-layer)
```

---

## 2. Global MoE

**File:** `src/models/global_moe.py`
**Config:** `configs/global_moe.yaml`

Single shared expert pool across all layers.  Each layer has its own router
but routes into the same global set of expert weights.

```
┌─────────────────────────────────────────────────────────┐
│                    Global MoE Model                     │
│                                                         │
│  ┌─── Layer 0 ────────────────────────────────────┐     │
│  │ hidden → RMSNorm → GQA Attention → + residual  │     │
│  │ hidden → RMSNorm → Router₀ ──┐    → + residual │     │
│  └──────────────────────────────┼─────────────────┘     │
│                                 │                       │
│  ┌─── Layer 1 ────────────────────────────────────┐     │
│  │ hidden → RMSNorm → GQA Attention → + residual  │     │
│  │ hidden → RMSNorm → Router₁ ──┐    → + residual │     │
│  └──────────────────────────────┼─────────────────┘     │
│                                 │                       │
│                   ...           │                       │
│                                 ▼                       │
│                  ┌──────────────────────────┐           │
│                  │   Global Expert Pool     │           │
│                  │   256 SwiGLU experts     │           │
│                  │   (= 16 layers × 16)    │           │
│                  │   gate_up_proj [256,H,2I]│           │
│                  │   down_proj   [256,I,H]  │           │
│                  └──────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

**Key details (current XS config):**
- 16 layers, 256 experts in one shared pool, top-4 per token per layer
- Same total expert parameters as Standard MoE at XS scale (`16 × 16 = 256`)
- Per-layer routers allow each layer to specialize which experts it uses
- Expert bias update (DeepSeek V3 variant): blends per-layer and global
  token counts with cosine-decaying alpha

**Parameter structure:**
```
Global:
  experts: gate_up[256, H, 2×I], down[256, I, H]   (shared)

Layer l:
  attention: Q[H,H], K[H,kv], V[H,kv], O[H,H]       (per-layer)
  mlp:       router[H, 256]                            (per-layer)
```

**Router variants:**
- `GlobalMoEForCausalLM` — softmax + top-k routing
- `DeepSeekGlobalMoEForCausalLM` — sigmoid + expert bias (aux-loss-free,
  DeepSeek V3 style with group-limited top-k)

---

## 3. Mixture-of-Everything (MoE-Everything)

**File:** `src/models/mixture_of_everything.py`
**Configs:** `configs/moe_everything_*.yaml`

Both attention AND MLP are expert-routed, with a hierarchical branch router
that decides which operation each token undergoes at each depth.
All weights are shared across depths — the forward is a plain for loop.

```
┌───────────────────────────────────────────────────────────────────────┐
│                    MoE-Everything Model                               │
│                                                                       │
│  input_ids → Embedding → init K,V projections + RoPE                 │
│                                                                       │
│  ┌─── Shared Components (reused at every depth) ─────────────────┐   │
│  │                                                                │   │
│  │  Branch Router: Linear(H, 2) → softmax → (p_attn, p_mlp)     │   │
│  │                                                                │   │
│  │  Attention Bank:                                               │   │
│  │    pre-norm(s) → router → select from E_attn expert sets of   │   │
│  │    (Q, K, V, O) projections → QK-norm → RoPE → attention      │   │
│  │                                                                │   │
│  │  MLP Bank:                                                     │   │
│  │    pre-norm → router → top-k from E_mlp SwiGLU experts        │   │
│  │                                                                │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  for depth in range(32):           ◄── same weights every iteration   │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                                                              │     │
│  │  branch_probs = BranchRouter(hidden)    → (p_attn, p_mlp)   │     │
│  │                                                              │     │
│  │  ┌─ ATTN branch ─────────────┐  ┌─ MLP branch ───────────┐  │     │
│  │  │ norm(hidden) → Q,K,V      │  │ norm(hidden) → router   │  │     │
│  │  │ blend KV with old state   │  │ → top-k SwiGLU experts  │  │     │
│  │  │ → attention → O proj      │  │ → mlp_out               │  │     │
│  │  └──────────┬────────────────┘  └──────────┬──────────────┘  │     │
│  │             │                              │                 │     │
│  │  hidden += p_attn × attn_out + p_mlp × mlp_out              │     │
│  │  KV_state = p_attn × KV_fresh + p_mlp × KV_old              │     │
│  │                                                              │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  RMSNorm → LM head → logits                                          │
└───────────────────────────────────────────────────────────────────────┘
```

**Key details:**
- 32 depth iterations, all weights shared (like a Universal Transformer)
- Token state is `(embedding, K, V)` — K,V persist across depths, refreshed
  only when the branch router selects attention
- Soft routing during training: both branches always computed, weighted by
  branch probability, so gradients flow through the router
- KV blending: `K_new = p_attn × K_fresh + p_mlp × K_old`
- Current XS configs use 16 attention expert sets and 256 shared MLP experts

**Parameter structure:**
```
Shared (one instance, reused 32×):
  branch_router: Linear(H, 2)
  attn_bank:     per-mode pre-norms, routing, E_attn × {Q,K,V,O} projections
  mlp_bank:      pre-norm, router[H, E_mlp], E_mlp × {gate_up, down}

Per-model (not per-depth):
  embed_tokens, init_k_proj, init_v_proj, init_k_norm
  rotary_emb, final_norm, lm_head
```

**Losses (current implementation):**
1. Cross-entropy (next-token prediction)
2. Optional MLP load-balancing aux from router logits, weighted by `router_aux_loss_coef`
3. Optional sequence-level aux from selected experts, weighted by `seq_aux_loss_coef`
4. Branch balance: `2 × Σ(mean_prob²)`, minimized at 50/50, weighted by `branch_router_aux_loss_coef`

---

### Attention Bank Modes

The attention bank supports 8 modes controlling how Q, K, V, O projections
are grouped for routing.  Each routing group carries its own pre-norm.

#### `bundled` — 1 norm, 1 router

All projections selected as one unit.  Simplest mode.

```
                    ┌──────────┐
  hidden ─→ norm ─→ │  Router  │ ─→ expert e
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Q[e]·x     K[e]·x     V[e]·x
              │          │          │
           q_norm[e]  k_norm[e]    │
              │          │          │
             RoPE       RoPE       │
              │          │          │
              ▼          ▼          ▼
              └──── Attention ──────┘
                       │
                    O[e]·attn_out      (same expert e)
```

#### `kv_paired` — 2 norms, 3 routers

K,V share a router (memory coherence).  Q and O routed independently.

```
  hidden ─→ kv_norm ─→ KV Router ─→ expert e₁
                         │
                    ┌────┴────┐
                    ▼         ▼
                 K[e₁]·x   V[e₁]·x
                 k_norm[e₁]
                    │         │
                   RoPE       │

  hidden ─→ q_norm ─→ Q Router ─→ expert e₂
                         │
                         ▼
                      Q[e₂]·x
                      q_norm[e₂]
                         │
                        RoPE
                         │
                         ▼
              ┌──── Attention ──────┐
              │    (Q_e₂ × K_e₁)   │
              └──────────┬──────────┘
                         │
                      O Router ─→ expert e₃
                         │
                      O[e₃]·attn_out
```

#### `qk_paired` — 2 norms, 3 routers

Q,K share a router (dot-product compatibility).  V and O routed independently.

```
  hidden ─→ qk_norm ─→ QK Router ─→ expert e₁
                          │
                     ┌────┴────┐
                     ▼         ▼
                  Q[e₁]·x   K[e₁]·x
                  q_norm[e₁] k_norm[e₁]
                     │         │
                    RoPE      RoPE

  hidden ─→ v_norm ─→ V Router ─→ expert e₂
                         │
                         ▼
                      V[e₂]·x

              ┌──── Attention ──────┐
              │  (Q_e₁ × K_e₁)·V_e₂│
              └──────────┬──────────┘
                         │
                      O Router ─→ expert e₃
                         │
                      O[e₃]·attn_out
```

#### `fully_independent` — 3 norms, 4 routers

Each projection routed independently.  Maximum flexibility.

```
  hidden ─→ q_norm ─→ Q Router → e_q → Q[e_q]·x → q_norm[e_q] → RoPE
  hidden ─→ k_norm ─→ K Router → e_k → K[e_k]·x → k_norm[e_k] → RoPE
  hidden ─→ v_norm ─→ V Router → e_v → V[e_v]·x
                                           │
                                    ┌── Attention ──┐
                                    │(Q_eq × K_ek)·V_ev│
                                    └───────┬───────┘
                                            │
                                 attn_out → O Router → e_o → O[e_o]·attn_out
```

#### `per_head_fully_independent` — 3 norms, head-wise routers

Like `fully_independent`, but routing happens separately for each head.
Each Q head, K head, V head, and O head gets its own router and expert bank.

```
  hidden ─→ q_norm ─→ Q Head 0 Router → e_q0 → Q[0,e_q0]·x → RoPE
                     Q Head 1 Router → e_q1 → Q[1,e_q1]·x → RoPE
                     ...

  hidden ─→ k_norm ─→ K Head 0 Router → e_k0 → K[0,e_k0]·x → RoPE
                     ...

  hidden ─→ v_norm ─→ V Head 0 Router → e_v0 → V[0,e_v0]·x
                     ...

  per-head attention runs as usual after GQA expansion

  attn_head_h ─→ O Head h Router → e_oh → O[h,e_oh]·attn_head_h
  final output = sum_h projected_head_h
```

#### `precompute_kv` — 1 norm, 1 router

Route first, then compute per-expert KV tables for ALL tokens.
Eliminates cross-token subspace mismatch.

```
  hidden ─→ norm ─→ Router ─→ token t assigned to expert e_t
                       │
        ┌──────────────┼──────────────────────────────────┐
        │   For each active expert e:                     │
        │     K_table = K[e] · ALL_tokens                 │
        │     V_table = V[e] · ALL_tokens                 │
        │                                                 │
        │     For tokens where e_t == e:                  │
        │       Q_t = Q[e] · x_t                          │
        │       attn_t = softmax(Q_t · K_table) · V_table │
        └─────────────────────────────────────────────────┘
                       │
                    O[e_t] · attn_out     (same expert as Q)
```

This mode ensures that when token A attends to token B, the K projection
used for B comes from the same expert as A's Q.  In other modes, A's Q
might use expert 1 while B's K uses expert 2, creating a subspace mismatch.

#### `per_head_precompute_kv` — 1 norm, per-head Q/KV/O routers

Per-head version of `precompute_kv`: each Q head, each KV head, and each O
head routes independently to pick its expert.  K and V for the same KV head
always share the same routing decision (same expert).

```
  hidden ─→ norm ─→ Q Head 0 Router → e_q0 → Q[0,e_q0]·x ─→ q_norm
                    Q Head 1 Router → e_q1 → Q[1,e_q1]·x ─→ q_norm
                    ...

             KV Head 0 Router ─→ e₀ per token  (K and V share)
             KV Head 1 Router ─→ e₁ per token
                  ...

        ┌──────────────────────────────────────────────────────┐
        │   For each KV head h, for each active expert e:      │
        │     K_table = K[h,e] · ALL_tokens   (head_dim)      │
        │     V_table = V[h,e] · ALL_tokens   (head_dim)      │
        │                                                      │
        │     For tokens where e_h == e:                       │
        │       Q heads [h×G : (h+1)×G] attend to K,V table   │
        └──────────────────────────────────────────────────────┘

  attn_head_h ─→ O Head h Router → e_oh → O[h,e_oh]·attn_head_h
  final output = sum_h projected_head_h
```

Key differences from `precompute_kv`:
- Q, K, V, O projections are per-head: `(num_heads, E, H, head_dim)` for Q/O,
  `(num_kv_heads, E, H, head_dim)` for K/V
- Each Q head, KV head, and O head has its own router
- K and V share a single router per KV head — no pair iteration needed
- Head attention output scaled by KV routing weight for gradient flow

---

## XS-Scale Comparison (Reference Configs)

All three architectures are parameter-matched at XS scale using the canonical
configs in `configs/`. Same backbone, same QKVO budget, same MLP expert budget,
same DeepSeek-style MLP routing for the shipped XS comparison.

**Reference configs:**
- `configs/standard_moe.yaml` — DeepSeek Standard MoE
- `configs/global_moe.yaml` — DeepSeek Global MoE (with bias interpolation)
- `configs/global_moe_nointerp.yaml` — DeepSeek Global MoE (global-only bias)
- `configs/moe_everything_*.yaml` — MoE-Everything (all 7 attention modes)

### Shared backbone (Qwen3-0.6B style)

| Parameter | Value |
|---|---|
| `hidden_size` | 1024 |
| `head_dim` | 128 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (GQA 2:1) |
| `moe_intermediate_size` | 768 |
| `num_experts_per_tok` | 4 (top-4) |
| `tie_word_embeddings` | true |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 1,000,000 |

### Architecture comparison

| | DeepSeek Standard | DeepSeek Global | DeepSeek Global (no interp) | MoE-Everything |
|---|---|---|---|---|
| **Config** | `standard_moe.yaml` | `global_moe.yaml` | `global_moe_nointerp.yaml` | `moe_everything_*.yaml` |
| **Model class** | `DeepSeekStandardMoEModel` | `DeepSeekGlobalMoEForCausalLM` | `DeepSeekGlobalMoEForCausalLM` | `MoEverythingForCausalLM` |
| **Layers / Depths** | 16 independent | 16 independent | 16 independent | 32 shared iterations |
| **Attention** | Standard GQA (1 per layer) | Standard GQA (1 per layer) | Standard GQA (1 per layer) | Expert bank: 16 QKVO sets |
| **MLP experts** | 16/layer (256 total) | 256 shared pool | 256 shared pool | 256 shared pool |
| **Expert sharing** | None | MLP across layers | MLP across layers | Everything across depths |
| **Branch routing** | No | No | No | Yes (attn vs MLP) |
| **KV state** | Standard KV cache | Standard KV cache | Standard KV cache | Persistent, soft-blended |
| **Pre-norms** | 2 per layer (32 total) | 2 per layer (32 total) | 2 per layer (32 total) | 1-3 attn + 1 MLP (shared) |

### Routing comparison

| | DeepSeek Standard | DeepSeek Global | MoE-Everything |
|---|---|---|---|
| **MLP router** | DeepSeek (sigmoid + bias) | DeepSeek (sigmoid + bias) | DeepSeek or Softmax |
| **Attn router** | None (fixed per layer) | None (fixed per layer) | DeepSeek or Softmax |
| **Branch router** | None | None | Softmax `Linear(H, 2)` |
| **Scaling factor** | 2.5 | 2.5 | 2.5 when `use_deepseek_routing=true` |
| **Group-limited top-k** | 8 groups, top-4 | 8 groups, top-4 | 8 groups, top-4 when `use_deepseek_routing=true` |
| **Batch aux loss** | 0 (disabled) | 0 (disabled) | 0 for MLP aux in current XS configs |
| **Seq aux loss** | 0.0001 | 0.0001 | 0.0001 in current XS configs |
| **Branch balance aux** | — | — | 0.01 in current XS configs |
| **Bias update rate** | 0.001 | 0.001 | 0.001 (when DeepSeek) |
| **Bias interpolation** | N/A | Yes (cosine decay) | N/A |

### Parameter matching

Expert pool sizes are derived from a **16-layer equivalent**:

| Component | Standard MoE | Global MoE | MoE-Everything |
|---|---|---|---|
| **Attention sets** | 16 (1 per layer) | 16 (1 per layer) | 16 (`num_attn_experts`) |
| **MLP experts** | 16 × 16 = 256 | 256 shared | 256 shared |
| **Attn QKVO shape** | `[H, q_dim]` etc. per layer | same | `[16, H, q_dim]` etc. in bank |
| **MLP expert shape** | `[16, H, 2×I]` per layer | `[256, H, 2×I]` shared | `[256, H, 2×I]` shared |

At current XS scale, the matched parameter budgets are:
- QKVO weights: `100,663,296`
- MLP expert weights: `603,979,776`

---

## Router Summary

### Router variants

- **Softmax** — standard softmax top-k (`Qwen3MoeTopKRouter`)
- **DeepSeek** — sigmoid + non-gradient expert bias + group-limited top-k (`DeepSeekRouter`)

DeepSeek routing (from DeepSeek V3):
1. FP32 gating linear → sigmoid scores (not softmax)
2. Add non-gradient `expert_bias` to scores for selection
3. Group-limited top-k: pick top groups first, then top experts within
4. Gather unbiased scores for the selected experts → normalize → scale
5. After each step: `bias += sign(avg - tokens_per_expert) × rate`

### All model classes and their routers

| Model class | Router | What it routes |
|---|---|---|
| `StandardMoEModel` | Softmax | MLP experts (per-layer) |
| `DeepSeekStandardMoEModel` | DeepSeek | MLP experts (per-layer) |
| `GlobalMoEForCausalLM` | Softmax | MLP experts (shared pool) |
| `DeepSeekGlobalMoEForCausalLM` | DeepSeek | MLP experts (shared pool) |
| `MoEverythingForCausalLM` | Softmax (default) | Branch + attn experts + MLP experts |
| `MoEverythingForCausalLM` | DeepSeek (`use_deepseek_routing=True`) | Branch (softmax) + attn experts + MLP experts (DeepSeek) |

### MoE-Everything router breakdown

| Router | What it decides | Type | Count per mode |
|---|---|---|---|
| **Branch router** | Attention vs MLP per token | Always softmax `Linear(H, 2)` | 1 (all modes) |
| **Attention router(s)** | Which QKVO expert set | Softmax or DeepSeek | bundled/precompute_kv: 1, kv/qk_paired: 3, fully_independent: 4, per_head_fully_independent: Q+K+V+O per head, per_head_precompute_kv: Q+KV+O per head |
| **MLP router** | Which SwiGLU expert | Softmax or DeepSeek | 1 (all modes) |

When `use_deepseek_routing=True`:
- Branch router stays softmax (binary soft decision, not expert selection)
- All attention bank routers become `DeepSeekRouter` (sigmoid + expert bias)
- MLP bank router becomes `DeepSeekRouter` (sigmoid + expert bias)
- All `DeepSeekRouter` instances are auto-discovered by `update_expert_biases()` in train.py

---

## Normalization Summary

All architectures use RMSNorm (no mean subtraction, just x / rms(x) × γ).

**Standard / Global MoE (per layer):**
```
input_layernorm [H]          — before attention
QK-norm [head_dim]           — per-head, after Q/K projection, before RoPE
post_attention_layernorm [H] — before MLP
```

**MoE-Everything (shared across depths):**
```
attn_bank pre-norm(s) [H]   — 1 to 3 norms depending on mode (see table below)
per-expert QK-norm [E, hd]  — per attention expert, after projection, before RoPE
mlp_bank pre-norm [H]       — before MLP routing
final norm [H]              — after all depths, before LM head
```

| Mode | Attn pre-norms | What they feed |
|---|---|---|
| `bundled` | 1: `norm` | All of Q, K, V |
| `kv_paired` | 2: `kv_norm`, `q_norm` | KV together, Q separate |
| `qk_paired` | 2: `qk_norm`, `v_norm` | QK together, V separate |
| `fully_independent` | 3: `q_pre_norm`, `k_pre_norm`, `v_pre_norm` | Each projection separate |
| `per_head_fully_independent` | 3: `q_pre_norm`, `k_pre_norm`, `v_pre_norm` | Each Q/K/V head separate; O routed per head |
| `precompute_kv` | 1: `norm` | All of Q, K, V |
| `per_head_precompute_kv` | 1: `norm` | All of Q, K, V; Q/KV/O routed per head (K,V share router) |
