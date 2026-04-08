# MoE Training

This repo contains three model families for pretraining routed language models built on Qwen3/Qwen3-MoE components.

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

---

# MoE-Everything Attention Modes

`moe_everything` supports several attention expert modes. The current work focuses on the two per-head modes:

- **`per_head_precompute_kv`** -- 1 router, Q/K/V/O bundled per expert, per-expert KV tables
- **`per_head_fully_independent`** -- 4 routers, Q/K/V/O independently routed

## Shared Concepts

Both per-head modes use GQA (Grouped Query Attention):

- **8 KV heads** (fixed positional slots, called "KV groups")
- **16 query heads** (2 query heads per KV group share the same K/V)
- **Attention expert pool**: a bank of reusable weight sets shared across all depths
- Each token's router fills the 8 KV-group slots with selected experts from the pool
- An expert is not permanently tied to any KV group -- different tokens can route different experts to the same group

### Routing is weighted by default

By default (`scale_attn_by_routing_weight: true`), attention experts are weighted by the selected routing weights, matching standard MoE dispatch. If you need the old pure-selection path for ablations, set `scale_attn_by_routing_weight: false`.

The router probabilities now serve three roles:

1. Top-k selection
2. MoE-style weighting of the selected attention experts
3. Load-balancing and exploration objectives

### Q is bundled per GQA group

In both modes, Q is "bundled" -- each Q expert produces output for **2 query heads at once** (`q_group_dim = 2 * head_dim = 256`). This ensures the 2 query heads that share the same K/V head are always coherent (come from the same learned expert). The router picks 1 Q expert per KV group (top-8 from the pool), not 1 per query head.

This was another fix discovered during development. The original `per_head_fully_independent` mode routed Q per individual head (top-16), meaning two query heads sharing a KV group could come from completely unrelated experts. Bundling ensures they stay coherent.

## `per_head_precompute_kv`

**1 router, everything bundled per expert. Each expert owns Q+K+V+O for one KV group.**

```text
Pool: 64 experts (single shared QKVO bank)
  q_proj: [64, 1024, 256]    (256 = 2 query heads x 128 head_dim)
  k_proj: [64, 1024, 128]    (1 KV head per expert)
  v_proj: [64, 1024, 128]
  o_proj: [64, 256, 1024]    (bundled: takes 2 heads, outputs hidden)
```

### Forward at each depth

**1. Route.** One router scores all 64 experts per token, picks top-8 (one per KV group). This single routing decision determines which expert provides Q, K, V, and O for each group.

```text
router(token) -> 8 expert ids
kv group:     0    1    2    3   ...    7
expert id:   37   12    4   55   ...    8
```

**2. Project Q/K/V for all tokens.** Each token uses its routed expert per group. Q is bundled (2 query heads), K/V are single heads:

```text
For token t, KV group g, expert e = routed[t, g]:
  Q[t,g] = q_norm[e](t @ Wq[e])   -> [2, 128]
  K[t,g] = k_norm[e](t @ Wk[e])   -> [128]
  V[t,g] =            t @ Wv[e]   -> [128]
```

**3. Build per-expert KV tables and run attention.** This is what makes `precompute_kv` special. Instead of building one mixed KV table (where different positions have K/V from different experts), we build a **separate KV table per active expert** across ALL tokens:

```text
Sequence: [tok_0, tok_1, tok_2, tok_3]

Routing for group 0: tok_0->exp5, tok_1->exp12, tok_2->exp5, tok_3->exp12

Expert 5 KV table (ALL tokens projected by expert 5):
  K: [k5(tok_0), k5(tok_1), k5(tok_2), k5(tok_3)]
  V: [v5(tok_0), v5(tok_1), v5(tok_2), v5(tok_3)]

Expert 12 KV table (ALL tokens projected by expert 12):
  K: [k12(tok_0), k12(tok_1), k12(tok_2), k12(tok_3)]
  V: [v12(tok_0), v12(tok_1), v12(tok_2), v12(tok_3)]
```

Why? Because tok_0's Q comes from expert 5, so it must attend against K values also from expert 5 to keep Q and K in the same learned subspace. If tok_0's Q (from exp5) attended against tok_1's K (from exp12), the dot product would be between vectors from different learned spaces -- meaningless.

For each active expert, we run full attention (all heads, all tokens) against that expert's KV table, then mask to keep only the head/token combinations that actually routed to this expert:

```text
for each active expert e:
  K_e = ALL_tokens @ Wk[e]         # full KV table
  V_e = ALL_tokens @ Wv[e]
  attn_e = attention(Q, K_e, V_e)  # full attention, all heads
  output += attn_e * mask(routed == e)  # keep only matching groups
```

The dense approach (full attention per expert, mask afterward) is intentionally kept over a sparse/ragged approach. We benchmarked both: dense is 2-9x faster on GPU because it launches one large regular kernel per expert vs many small irregular ones. The redundant computation (heads that get masked out) is cheaper than the overhead of irregular memory access patterns.

**4. O projection.** Same expert as Q/K/V, bundled per KV group (takes 2 heads' attention output, projects to hidden_size).

**5. KV state persistence.** At each depth, tokens taking the attention branch get fresh K/V stored in the KV state; tokens taking the MLP branch keep their old K/V from a previous depth. The per-expert KV tables (step 3) are rebuilt from the current hidden states at each depth -- they are not cached.

### Cost

`num_active_experts * full_attention` per depth. With 8 KV groups and diverse routing, most of the 64 experts may be active, making this significantly more expensive than standard attention. The tradeoff is mathematical correctness of the Q-K dot products.

## `per_head_fully_independent`

**4 separate routers. Q bundled per KV group. K/V per KV head. O per query head. Each component routes independently.**

```text
Q pool:  64 experts x [1024, 256]   router picks top-8 (per KV group)
K pool:  64 experts x [1024, 128]   router picks top-8 (per KV head)
V pool:  64 experts x [1024, 128]   router picks top-8 (per KV head)
O pool: 128 experts x [128, 1024]   router picks top-16 (per query head)
```

The O pool is larger (128 vs 64) because O is per query head (16 slots) while Q/K/V are per KV group/head (8 slots). The larger pool ensures O has the same total parameter count as the standard per-layer O projection.

### Forward at each depth

**1. Route Q/K/V independently.** Three separate routers, each picking top-8. Each component can select a completely different expert for the same head slot:

```text
q_router(hidden) -> 8 expert ids (per KV group)
k_router(hidden) -> 8 expert ids (per KV head)
v_router(hidden) -> 8 expert ids (per KV head)

KV group:      0     1     2     3    ...     7
Q expert:      7    42    15     3    ...    31
K expert:     12     5    42     8    ...    19
V expert:     33    21     7    50    ...     2
```

**2. Project Q (bundled per KV group).** Each Q expert produces `q_group_dim = 256` (2 query heads bundled):

```text
Q_g = q_norm[e](token @ Wq[q_expert[g]])   -> [2, 128]
```

**3. Project K/V (per KV head, independently routed).** Each expert produces one KV head. K and V can come from different experts for the same head slot:

```text
K_g = k_norm[e](token @ Wk[k_expert[g]])   -> [128]
V_g =            token @ Wv[v_expert[g]]   -> [128]
```

**4. KV state merge.** Same as precompute_kv: attention tokens get fresh K/V, MLP tokens keep old.

**5. Standard GQA attention.** ONE attention call over the blended KV state. K/V are broadcast via `repeat_kv` to match 16 query heads:

```text
attn_output = attention(Q, repeat_kv(K_blend), repeat_kv(V_blend))
```

There are **no per-expert KV tables**. The KV state is a single table where each position's K/V comes from that token's routed expert. This means Q at position t (from expert 7) may attend against K at position s (from expert 42) -- they are in different learned subspaces. This is the fundamental tradeoff vs `precompute_kv`.

**6. Route and project O (per query head).** O has its own router, routing based on the **attention output** (not the input hidden state). This means O can adapt to what attention actually computed. Picks top-16 from 128 experts, one per query head:

```text
o_router(attn_output) -> 16 expert ids (from 128 experts)

For each query head h:
  out_h = attn_h @ Wo[o_expert[h]]    -> [1024]
  sum over all heads                    -> final token output [1024]
```

### Cost

1x attention per depth -- same as standard attention. Much cheaper than `precompute_kv`, at the cost of Q-K subspace mismatch across positions.

## Comparison

| | `per_head_precompute_kv` | `per_head_fully_independent` |
|---|---|---|
| Routers | 1 | 4 (Q, K, V, O) |
| Q | bundled per KV group | bundled per KV group |
| K/V | same expert as Q | independently routed |
| O | bundled with Q/K/V (same expert) | per query head, independently routed |
| O routing input | input hidden state | attention output |
| Q-K subspace | always matched (per-expert KV tables) | may differ across positions |
| Expert pool | single shared QKVO bank | separate Q, K, V, O banks |
| Attention calls/depth | num_active_experts | 1 |
| Param budget | Q/K/V/O from same pool (64 experts) | Q/K/V: 64 experts, O: 128 experts |

### Design tradeoffs

**`per_head_precompute_kv`** guarantees Q-K subspace alignment at the cost of `E_active` attention passes per depth. It's the mathematically cleaner design -- attention scores are always meaningful because Q and K come from the same projection. But the compute cost grows with the number of unique experts selected across the sequence.

**`per_head_fully_independent`** gives maximum routing flexibility (each of Q/K/V/O can independently specialize) at standard attention cost. The Q-K subspace mismatch is a theoretical concern -- whether the model can learn useful representations despite cross-expert attention is an empirical question. O routing on the attention output (rather than input) is a unique feature that lets the output projection adapt to the attention pattern.

---

# Router Structure

Three separate router concepts in `moe_everything`:

- **Branch router**: chooses `attention` vs `mlp` per token
- **Attention routers**: choose attention experts
- **MLP router**: chooses MLP experts

Config switches:

- `per_layer_router` -- one branch router per depth
- `per_layer_attn_router` -- one attention-router set per depth
- `per_layer_mlp_router` -- one MLP router per depth
- `scale_attn_by_routing_weight` -- scale attention projections/outputs by routing weight (default: true)
- `router_exploration_rate` -- per-token random expert exploration rate during training

## Q/K Norms

Two norm layers in the per-head path:

- Input pre-norm on the token hidden state before routing/projection
- Expert-specific `q_norm_weight` and `k_norm_weight` on the projected head vectors

The expert-specific norms follow the selected expert, not the fixed head id:

```text
token A: head 0 -> expert 37 -> uses q_norm_weight[37], k_norm_weight[37]
token B: head 0 -> expert 12 -> uses q_norm_weight[12], k_norm_weight[12]
```

## Load Balancing

For the current per-head configs:

- `branch_router_aux_loss_coef: 0.001` -- Switch-style branch balance loss
- `router_aux_loss_coef: 0.001` -- Switch-style batch aux loss for MLP and attention routers
- `seq_aux_loss_coef: 0.0001` -- sequence-level aux is active
- `bias_update_rate: 0.001` -- DeepSeek-style expert-bias updates are active
- `router_exploration_rate: 0.02` -- random expert exploration is active during training

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
