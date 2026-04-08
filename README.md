# MoE Training

This repo contains three model families for pretraining routed language models built on Qwen3/Qwen3-MoE components.

For the speedrun MoE variants, the dense 12-block speedrun backbone is decomposed into
24 effective branch-router depths. Each original transformer block is split into two
sequential decision points, so the routed model performs 24 branch decisions for a
12-block baseline.

## Setup

```bash
uv sync
```

Add credentials to `.env` if you want Hugging Face and WandB access:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

## Data

Download parquet shards into `data/parquet`:

```bash
uv run python scripts/download_data.py --max_shards 64
uv run python scripts/download_data.py
```

## Training

Single-node training:

```bash
./scripts/train.sh configs/standard_moe.yaml
./scripts/train.sh configs/global_moe.yaml
./scripts/train.sh configs/moe_everything_per_head_precompute_kv.yaml
```

`scripts/train.sh` uses `torchrun` by default when multiple GPUs are visible.

- `NPROC_PER_NODE=8 ./scripts/train.sh ...` runs 8-way DDP on one node
- `LAUNCHER=accelerate ./scripts/train.sh ...` uses the older Accelerate launcher path
- `TRAIN_ENTRYPOINT=train_torch.py ./scripts/train.sh ...` uses the new raw-torch trainer
- `TRAIN_ENTRYPOINT=train_torch.py ./scripts/train.sh ... --dist-strategy fsdp` enables FSDP in the new trainer

---

# Model Families

## `standard_moe`

- `src/models/standard_moe.py`
- Standard transformer layer structure
- Dense attention in every layer
- One routed MLP expert pool per layer
- Routing only happens on the MLP branch

## `global_moe`

- `src/models/global_moe.py`
- Standard transformer layer structure
- Dense attention in every layer
- One shared global MLP expert pool across all layers
- Each layer has its own router into that shared pool

## `moe_everything`

- `src/models/mixture_of_everything.py`
- Custom depth loop instead of a standard decoder-layer stack
- A branch router chooses `attention` or `mlp` per token at each depth
- Attention experts and MLP experts live in shared banks (weights shared across all depths)
- KV state persists across depth -- tokens that take MLP keep their old KV state

## `speedrun_moe_*`

- `src/models/speedrun_moe_gpt.py`
- Speedrun-backed routed model family
- The original 12-block speedrun backbone is treated as **24 effective depths**
- At each depth, a **branch router chooses `attention` or `mlp` via argmax**
- Attention experts and MLP experts live in shared banks reused across all depths
- The two attention-bank modes are `per_head_precompute_kv` and `per_head_fully_independent`
- Expert routers are top-1 DeepSeek-style routers with expert-bias updates
- Current speedrun configs use `router_aux_loss_coef: 0.0` and `seq_aux_loss_coef: 0.01`
- The branch router has no balancing loss and no branch-bias updates

## `speedrun_moe_everything`

- `src/models/speedrun_mixture_of_everything.py`
- Separate speedrun-style variant of `moe_everything`
- Keeps the branch-routed shared-bank architecture of `moe_everything`
- Uses speedrun-style functional RMS norm instead of learned RMSNorm modules
- Does not support learned `routed_norm`

---

# MoE-Everything Attention Modes

`moe_everything` supports several attention expert modes. The current work focuses on the two per-head modes:

- **`per_head_precompute_kv`** -- QKVO bundled per expert, per-expert KV tables, one routing decision per head slot
- **`per_head_fully_independent`** -- Q, K, V each routed independently, O routed separately on attention output

## Routing Design: Per-Head-Slot Routers with Top-1

Each head position has its **own dedicated router** that picks **top-1** from the expert pool. This is NOT one router picking top-K — it is K separate routers each picking top-1.

For a model with H heads:

- **`per_head_fully_independent`**: H Q-routers + H K-routers + H V-routers + H O-routers = 4H routers total
- **`per_head_precompute_kv`**: H QKVO-routers (bundled, one decision picks Q+K+V+O together) = H routers total

Each router is a separate `nn.Linear(dim, num_experts)` → softmax → argmax. Different head slots learn to specialize on different experts independently.

## Shared Concepts

Both per-head modes share these ideas:

- **H head slots** (e.g., 6 for the speedrun model, 8 for the Qwen-based model)
- **Attention expert pool**: a bank of reusable weight sets shared across all depths
- Each head slot has its **own dedicated router** that picks **top-1** from the expert pool
- An expert is not permanently tied to any head slot -- different tokens can route different experts to the same slot
- **MLP expert pool**: a bank of reusable MLP weight sets, also shared across all depths, with its own top-1 router
- In the speedrun MoE variants, a branch router first chooses **attention or MLP** at
  each of the 24 effective depths; the chosen branch then uses the expert routing
  described below

### Per-head-slot top-1 routing

This is the key design: NOT one router picking top-K, but **K separate routers each picking top-1**.

Each router is `nn.Linear(dim, num_experts)` -> softmax -> argmax. The expert output is weighted by its softmax probability (MoE-style dispatch). Different head-slot routers learn to specialize independently.

### Expert count matching

The baseline speedrun model has 11 attention layers (layer 7 skips attention) x 6 heads = **66 unique head weight sets**. To match parameters, the expert pool should have **66 QKVO experts**. Similarly, the baseline has 12 MLP layers, so the MLP expert pool should have **12 MLP experts**.

## `per_head_precompute_kv`

**H routers (one per head slot), QKVO bundled. Each routing decision picks one expert that provides Q+K+V+O together. Per-expert KV tables ensure Q-K subspace alignment.**

```text
Pool: E experts (single shared QKVO bank)
  q_proj: [E, dim, head_dim]     (one head per expert)
  k_proj: [E, dim, head_dim]
  v_proj: [E, dim, head_dim]
  o_proj: [E, head_dim, dim]

Routers: H routers, each nn.Linear(dim, E), each picks top-1
```

### Forward at each depth

**1. Route.** Each head slot's router independently picks top-1 from the expert pool:

```text
router_0(token) -> expert 37
router_1(token) -> expert 12
router_2(token) -> expert 4
...
router_5(token) -> expert 8

head slot:    0    1    2    3    4    5
expert id:   37   12    4   55   19    8
```

**2. Project Q/K/V.** Each token uses its routed expert per head slot. The same expert provides Q, K, and V:

```text
For token t, head h, expert e = routed[t, h]:
  Q[t,h] = q_norm[e](t @ Wq[e])   -> [head_dim]
  K[t,h] = k_norm[e](t @ Wk[e])   -> [head_dim]
  V[t,h] =            t @ Wv[e]   -> [head_dim]
```

**3. Build per-expert KV tables and run attention.** For each active expert, project ALL tokens through that expert's K/V weights, then run attention and mask:

```text
for each active expert e:
  K_e = ALL_tokens @ Wk[e]         # full KV table
  V_e = ALL_tokens @ Wv[e]
  attn_e = attention(Q, K_e, V_e)  # full attention
  output += attn_e * mask(routed == e) * routing_weight
```

This ensures Q and K always come from the same learned subspace. The dense approach (full attention per expert, mask afterward) is faster on GPU than sparse/ragged alternatives.

**4. O projection.** Same expert as Q/K/V — the routing decision is shared:

```text
For token t, head h, expert e = routed[t, h]:
  O[t,h] = attn_output[t,h] @ Wo[e]    -> [dim]
  sum over heads                         -> final token output [dim]
```

### Cost

`num_active_experts * full_attention` per depth.

## `per_head_fully_independent`

**4H routers total. Q, K, V each have H routers (top-1 on input hidden state). O has H routers (top-1 on attention output). All route independently.**

```text
Q pool: E experts x [dim, head_dim]    H Q-routers, each top-1
K pool: E experts x [dim, head_dim]    H K-routers, each top-1
V pool: E experts x [dim, head_dim]    H V-routers, each top-1
O pool: E experts x [head_dim, dim]    H O-routers, each top-1 (on attn output)
```

### Forward at each depth

**1. Route Q/K/V independently.** Each component has H routers, each picking top-1. Different components can select completely different experts for the same head slot:

```text
q_router_0(hidden) -> expert 7      k_router_0(hidden) -> expert 12     v_router_0(hidden) -> expert 33
q_router_1(hidden) -> expert 42     k_router_1(hidden) -> expert 5      v_router_1(hidden) -> expert 21
q_router_2(hidden) -> expert 15     k_router_2(hidden) -> expert 42     v_router_2(hidden) -> expert 7
...
```

**2. Project Q/K/V.** Each head slot projects using its independently routed expert:

```text
Q[t,h] = q_norm[e](t @ Wq[q_expert[t,h]])   -> [head_dim]
K[t,h] = k_norm[e](t @ Wk[k_expert[t,h]])   -> [head_dim]
V[t,h] =            t @ Wv[v_expert[t,h]]   -> [head_dim]
```

**3. Standard attention.** ONE attention call. No per-expert KV tables:

```text
attn_output = attention(Q, K, V, is_causal=True)
```

Q at position t (from expert 7) may attend against K at position s (from expert 42) -- they are in different learned subspaces. This is the fundamental tradeoff vs `precompute_kv`.

**4. Route and project O.** O has its own H routers, routing on the **attention output** (not the input hidden state):

```text
o_router_0(attn_output) -> expert 52
o_router_1(attn_output) -> expert 11
...

For each head h:
  out_h = attn_h @ Wo[o_expert[h]]    -> [dim]
  sum over all heads                    -> final token output [dim]
```

### Cost

1x attention per depth -- same as standard attention. Much cheaper than `precompute_kv`, at the cost of Q-K subspace mismatch across positions.

## Comparison

| | `per_head_precompute_kv` | `per_head_fully_independent` |
|---|---|---|
| Routers per depth | H (QKVO bundled) | 4H (Q, K, V, O independent) |
| Routing | H top-1 routers | 4H top-1 routers |
| Q/K/V routing | same expert (bundled) | independent per component |
| O routing input | same as QKV (bundled) | attention output (independent) |
| Q-K subspace | always matched (per-expert KV tables) | may differ across positions |
| Expert pool | single shared QKVO bank | separate Q, K, V, O banks |
| Attention calls/depth | num_active_experts | 1 |

### Design tradeoffs

**`per_head_precompute_kv`** guarantees Q-K subspace alignment at the cost of `E_active` attention passes per depth. Attention scores are always meaningful because Q and K come from the same projection. Compute cost grows with unique experts selected across the sequence.

**`per_head_fully_independent`** gives maximum routing flexibility (each of Q/K/V/O can independently specialize) at standard attention cost. The Q-K subspace mismatch is a theoretical concern. O routing on the attention output (rather than input) lets the output projection adapt to what attention computed.

---

# Router Structure

Two router concepts in the speedrun MoE models:

- **Attention routers**: per-head-slot top-1 routers that choose attention experts
- **MLP router**: top-1 router that chooses MLP experts
- **Branch router**: top-1 router that chooses `attention` or `mlp` at each effective depth

Each head slot has its own dedicated router. In the speedrun MoE models these are DeepSeek-style sigmoid-plus-bias routers with top-1 argmax selection. The selected routing weight is multiplied into the expert output.

## Q/K Norms

In the speedrun family, Q and K follow dense speedrun GPT:

- Input pre-norm (RMS norm) on the token hidden state before routing/projection
- Functional RMS norm on projected Q and K
- No learned per-expert `q_norm_weight` or `k_norm_weight` in the speedrun variants

This differs from the original Qwen-based `moe_everything` family, which can use learned RMSNorm modules and optional routed norms.

## Load Balancing

- `router_aux_loss_coef` -- Switch-style batch aux loss for attention and MLP routers
- `seq_aux_loss_coef` -- sequence-level aux loss for attention and MLP expert routers
- `router_exploration_rate` -- per-token random expert exploration rate during training
- The speedrun branch router has no balancing loss

---

# Configs

## Depth-matched configs (param-matched across model types)

- `configs/depth_matched_fp32_no_liger/{4,8,16}_layers/`
- `configs/depth_matched/{4,8,16}_layers/`

## Root per-head configs

- `configs/moe_everything_per_head_fully_independent.yaml`
- `configs/moe_everything_per_head_precompute_kv.yaml`
- `configs/moe_everything_per_head_independent_perlayer_prenorm.yaml`
- `configs/moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml`

For `per_head_fully_independent`, `num_attn_experts` is halved relative to `per_head_precompute_kv` because Q is bundled per GQA group (each expert produces `q_group_dim = 2 * head_dim`). O uses a separate pool of `E * q_heads_per_kv` experts to match param budget.

## Baselines

- `configs/standard_moe.yaml`
- `configs/global_moe.yaml`

## Sanity config

- `configs/moe_everything_per_head_precompute_kv_sanity.yaml`
- Uses `sanity_check_mode: alternating_global_moe`
- Even depths forced to attention, odd depths forced to MLP
- Deterministic attention expert assignment, learned MLP routing
- Meant to catch implementation bugs while staying on the MoE-Everything codepath

---

# Logging

- `train/loss`, `train/ce_loss`
- `train/aux_loss`, `train/aux_loss_normalized`
- `train/seq_aux_loss`, `train/branch_aux_loss`, `train/attention_aux_loss`
- `train/tokens_per_sec`, `train/sec_per_step`, `train/tokens_seen_B`
- `eval/loss`, `eval/ce_loss`, `eval/perplexity`

## Other docs

- [status.md](status.md): current status and recent fixes
- [MULTINODE_README.md](MULTINODE_README.md): Modal multi-node usage
