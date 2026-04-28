# Routing & Expert Selection

This document covers all router implementations, expert selection mechanisms, load balancing losses, and the expert bias update system.

## Table of Contents

- [1. Router Implementations](#1-router-implementations)
- [1.5. Router Options (scoring, ordering, z-loss, group-limited top-K)](#15-router-options)
- [2. Load Balancing Losses](#2-load-balancing-losses)
- [3. Expert Bias Updates (DeepSeek V3)](#3-expert-bias-updates-deepseek-v3)
- [4. Routing Statistics & Monitoring](#4-routing-statistics--monitoring)

---

## 1. Router Implementations

Router implementations are in `src/models/routing/routers.py` (BranchRouter, BranchRouterRecorder) and `src/models/router.py` (DeepSeekRouter, ExplorationTopKRouter). Load balancing loss is in `src/models/routing/load_balancing.py`.

### DeepSeek Router (Sigmoid + Expert Bias)

**Class**: `DeepSeekRouter`
**File**: `src/models/router.py:156-251`

The primary router for production MoE models. Uses sigmoid scoring instead of softmax, combined with non-gradient expert bias updates for load balancing.

**Forward pass**:
```
logits = linear(hidden_state)          # [batch*seq, num_experts]
scores = sigmoid(logits)               # scores in (0, 1)
scores_biased = scores + expert_bias   # bias is a persistent buffer, not a parameter
top_k_weights, top_k_indices = topk(scores_biased, k)
# Optional: normalize top_k_weights to sum to 1
```

**Why sigmoid over softmax?**
- Softmax creates competition between experts (probabilities sum to 1), which can cause routing collapse
- Sigmoid allows multiple experts to have high scores independently
- Expert bias provides a non-gradient knob for load balancing that doesn't interfere with learning

**Group-limited top-K** (optional, Megatron-LM style):
1. Divide experts into `num_groups` groups
2. Score each group by its top member
3. Select `group_topk` groups
4. Within selected groups, pick top-K experts overall

This prevents all top-K experts from clustering in one region of the expert space.

### Exploration Top-K Router (Softmax + Exploration)

**Class**: `ExplorationTopKRouter`
**File**: `src/models/router.py:127-153`

Standard softmax router with random exploration:

```
logits = linear(hidden_state)
probs = softmax(logits)
top_k_weights, top_k_indices = topk(probs, k)

# During training, with exploration_rate probability:
# Replace selected experts with random experts for some tokens
if training and exploration_rate > 0:
    mask = random_mask(exploration_rate)
    top_k_indices[mask] = random_expert()
```

**When to use**: Standard MoE setups where softmax routing is preferred. Exploration prevents routing collapse by ensuring all experts receive some tokens.

### Branch Router

**Location**: `src/models/routing/routers.py` (BranchRouter), `src/models/moe_everything/attention_bank.py` (per-head attention routing)

Binary hard router: each token picks ATTENTION (0) or MLP (1).

```
logits = linear(hidden_state)  # [batch*seq, 2]
probs = softmax(logits)
choice = argmax(probs)         # hard decision
weight = probs[choice]         # soft weight for scaling

output = weight * selected_branch(hidden_state)
```

The hard decision makes this non-differentiable at the selection point, but the soft weight multiplication keeps gradients flowing to the router.

---

## 1.5. Router Options

`ExplorationTopKRouter` (the softmax-family router used by `standard_moe` / `global_moe` / `moe_everything` when `router_type != deepseek`) exposes four independent configuration knobs. `DeepSeekRouter` honours the z-loss knob; its scoring (sigmoid) and selection path (biased top-K, gather unbiased) are defining features of that router family and stay fixed. These options are parity with Megatron-LM's `TopKRouter`, and they default to behaviour-preserving values so any pre-existing config keeps its old numerics.

### High-level overview

| Option | Values | Default | Affects | Notes |
|---|---|---|---|---|
| `router_score_function` | `softmax`, `sigmoid`, `sqrtsoftplus` | `softmax` | `ExplorationTopKRouter` | Maps raw logits → per-expert scores. |
| `softmax_position` | `pre_topk`, `post_topk` | `pre_topk` | `ExplorationTopKRouter` | DEC-17 (RESOLVED → AC-1 task38): when to apply the score function relative to top-K. Deprecated alias `router_topk_ordering` ∈ `{post, pre}` is accepted with a `DeprecationWarning`; mapping is `post → pre_topk`, `pre → post_topk`. |
| `num_groups` / `group_topk` | positive ints | `None` / `None` | both routers | Group-limited top-K (Megatron / DeepSeek style). |
| `router_z_loss_coef` | float ≥ 0 | `0.0` | both routers | Per-call logit-magnitude regularizer added to aux loss. |

Source: `src/models/router.py::ExplorationTopKRouter`, `::DeepSeekRouter`; trainer integration at `src/training/routing.py::collect_router_z_loss` and `src/training/trainer.py`.

### 1.5.1 `router_score_function`

Controls how raw router logits become per-expert scores inside the softmax-family router.

| Value | Formula | Properties |
|---|---|---|
| `softmax` | `exp(x_i) / Σ_j exp(x_j)` | Probabilities, competing — all experts sum to 1 per token. |
| `sigmoid` | `1 / (1 + exp(-x))` elementwise | Scores in (0, 1) per expert, independent. Cannot saturate one expert's score without information from the others. |
| `sqrtsoftplus` | `sqrt(log(1 + exp(x)))` elementwise | Non-negative, non-competing, with a smooth floor near 0 for large negatives. Megatron offers this as a softer alternative to sigmoid when they want a monotone-non-competing score without the sigmoid saturation. |

**When to use**:
- Keep `softmax` (the default) when you want standard competing routing — the small weight you gather after top-K is already a valid probability.
- Pick `sigmoid` when you want per-expert independent scores (useful for ablations against the DeepSeek router — same scoring, no bias update).
- Pick `sqrtsoftplus` when you want a non-competing score without sigmoid's saturation (rare; include it mainly for Megatron-parity experiments).

`DeepSeekRouter` ignores this knob — its sigmoid + bias path is its defining contract.

### 1.5.2 `softmax_position`

DEC-17 (RESOLVED → AC-1 task38). Controls whether the score function
runs **before** the top-K selection (`pre_topk`, default — softmax over
all E experts → top-K → gather) or **after** (`post_topk` — top-K on
raw logits → softmax over the K selected logits).

```
# pre_topk (default; legacy alias `router_topk_ordering=post`):
scored = score_function(logits)                 # (T, E)  -- softmax BEFORE topk
topk_idx = topk(scored).indices
weights = gather(scored, topk_idx)

# post_topk (legacy alias `router_topk_ordering=pre`):
topk_idx = topk(logits).indices                 # top-K on raw logits
weights = score_function(gather(logits, topk_idx))  # softmax AFTER topk, over K only
```

The legacy field name `router_topk_ordering` ∈ `{post, pre}` is still
accepted with a `DeprecationWarning`. Mapping:

| Legacy `router_topk_ordering` | Canonical `softmax_position` |
|------------------------------|-----------------------------|
| `post` (legacy default)      | `pre_topk` (canonical default) |
| `pre`                        | `post_topk`                 |

**Top-1 guard (DEC-17)**: the runtime rejects
`softmax_position="post_topk"` with `top_k=1` at config-load time
because softmax of a single selected logit is the constant `1.0`
weight, which kills the gradient signal that would otherwise flow
through the routing weight back to the gate. `pre_topk + top_k=1` is
fine: the softmax is computed across all E logits before the top-K,
so the gathered weight is a non-constant softmax probability and
preserves gradient flow.

**Functional implication**:
- **Same indices for softmax and sigmoid** (monotonic score functions): `argmax(score(x)) == argmax(x)`, so `pre_topk` and `post_topk` pick the same K experts. Weights differ: under `post_topk` the softmax is normalized across the K selected logits (so selected weights sum to 1 when `norm_topk_prob=False`); under `pre_topk` the gathered weights are a subset of the full-softmax distribution and sum to less than 1.
- **Potentially different indices for `sqrtsoftplus`** at the ties / near-equal logit regime — because `sqrtsoftplus` is monotonic it shouldn't reorder either, but the regularized form means the ordering is the same.
- **Gradient path changes**: in `post_topk`, the score function is only evaluated on K elements, so the gradient only flows through those K logits. Under `pre_topk` the gradient flows through all E logits via the softmax normalizer. The `pre_topk` path is therefore denser per-step; `post_topk` is cheaper.

**When to use**:
- Keep `pre_topk` (default) for baseline runs — that's the behaviour configs have been training against.
- Try `post_topk` when doing Megatron-parity ablations or when you specifically want the "softmax over K" weight semantics (each selected weight is renormalized against the K peers, not against all E). Don't pair `post_topk` with `top_k=1`; the runtime will reject it.

### 1.5.3 `num_groups` / `group_topk` on the softmax router

The group-limited top-K path that the DeepSeek router already supports is now also available on the softmax-family router. Mechanics:

1. Partition the E experts into `num_groups` equal-sized groups (assumes `E % num_groups == 0`).
2. Score each group by the sum of its top `(top_k // group_topk)` expert scores.
3. Pick the top `group_topk` groups per token.
4. Run top-K over the experts inside those groups only.

**Motivation**: prevents the selected K from clustering in one region of the expert space. Useful when you have many experts and want to nudge routing toward structural diversity without adding an auxiliary loss. Megatron's own DeepSeek test configs use group-limited top-K by default.

**When to use**: skip unless you already use group-limited top-K in your DeepSeek configs and want feature parity in the softmax path.

### 1.5.4 `router_z_loss_coef`

A scalar regularizer on the **raw** router logits (before softmax / sigmoid). Per router call:

```
z = logsumexp(raw_logits, dim=-1)             # (T,)
z_loss_per_call = mean(z^2) * router_z_loss_coef
```

The router caches `z_loss_per_call` on `self._last_z_loss` (a scalar tensor with autograd); the training loop walks the model via `collect_router_z_loss(model)`, sums across routers, and adds the total to the output loss **before** `backward()`.

**What it does**: keeps the logsumexp of the router logits from drifting large, which is a proxy for "don't let any single expert's score explode under bf16 gradient scaling". The quadratic-in-logsumexp shape means the gradient pressure grows smoothly as logits grow.

**When to enable**:
- Long or unstable training runs where router logits occasionally spike into 10+ and push one expert's softmax probability to ~1.0 at the cost of the others.
- Any training setup where you want Megatron's z-loss stability lever.
- Reasonable starting coefficient: `1e-3` (small enough to not fight the main loss, large enough to notice on router logit magnitudes after a few hundred steps).

**When to leave off**:
- Short (< 1k-step) runs; the feature is an insurance policy, not a correctness fix.
- Runs already stabilized via bias updates (DeepSeek) and/or reasonable aux-loss coefs.

**No-cost default**: the trainer's z-loss integration path is a no-op when every router has `router_z_loss_coef = 0.0` (the default) — `collect_router_z_loss` returns `None` and the branch in the trainer is a single `is not None` check.

### Cross-ref: comparison to external stacks

| Stack | softmax top-K ordering | score fn | z-loss | group-limited top-K | bias-update balancing |
|---|---|---|---|---|---|
| This repo (`ExplorationTopKRouter`) | `post` / `pre` | `softmax` / `sigmoid` / `sqrtsoftplus` | yes | yes (any softmax family) | no (aux loss only) |
| This repo (`DeepSeekRouter`) | N/A — sigmoid-forced | sigmoid | yes | yes | yes (expert bias) |
| Megatron-LM `TopKRouter` | `post` / `pre` | `softmax` / `sigmoid` / `sqrtsoftplus` | yes | yes | yes (aux-free bias) + others |
| nmoe / modal-nmoe | N/A — sigmoid-only | sigmoid | (inherited from router precision) | yes | yes |

The Megatron parity gaps we still do **not** expose: aux-loss auto-scaler, `global_aux_loss` (cross-DP-group balance), `sinkhorn` routing. See `docs/research/task2-megatron-research.md` for notes on when they might become worthwhile.

---

## 2. Load Balancing Losses

All losses in `src/models/routing/load_balancing.py`.

### Batch-Level Load Balancing Loss (Switch Transformer)

**Function**: `load_balancing_loss_func()`
**File**: `src/models/routing/load_balancing.py:39-139`

```
L = num_experts * sum_i(f_i * P_i)
```

Where:
- `f_i` = fraction of tokens routed to expert i (hard assignment, from actual routing decisions)
- `P_i` = average router probability for expert i (soft, differentiable)

**At perfect balance**: `f_i = 1/E` and `P_i = 1/E`, so `L = E * E * (1/E)^2 = 1.0`

**Critical implementation detail**: `f_i` is computed locally per rank (not all-reduced). This preserves the theoretical minimum of the loss -- if you all-reduce, the minimum shifts based on world size, making the loss coefficient harder to tune.

**HuggingFace double-softmax fix**: The HF Qwen3MoE implementation applies softmax to router outputs that are already softmax probabilities. This double-softmax flattens the distribution, making it nearly uniform and hiding real imbalances. Our version uses the raw router probabilities.

### Sequence-Level Load Balancing Loss (DeepSeek V2/V3)

**Function**: `seq_load_balancing_loss_func()`
**File**: `src/models/routing/load_balancing.py:166-274`

Per-sequence balance metric from DeepSeek V2/V3 (arxiv 2412.19437, Equations 17-20):

```
For each sequence s of length T:
  f_i = (E / (K * T)) * sum_t 1[expert i in top-K at step t]   # expert frequency
  s'_it = s_it / sum_j s_jt                                     # normalize scores per token
  P_i = (1/T) * sum_t s'_it                                     # average normalized score

L_Bal = sum_i f_i * P_i
```

**Why sequence-level?** Batch-level loss averages across all sequences, which can mask per-sequence imbalances. A sequence that routes all tokens to one expert will be diluted by other sequences in the batch. Sequence-level loss catches this.

### Normalized Load Balancing Loss

**Function**: `normalized_load_balancing_loss_func()`

Diagnostic variant for sigmoid routers. Since sigmoid scores don't sum to 1 (unlike softmax), this normalizes them before computing the standard load-balancing loss. Useful for monitoring but not typically used as a training loss.

---

## 3. Expert Bias Updates (DeepSeek V3)

**Location**: `src/training/routing.py` (called from `src/training/trainer.py`)

A non-gradient mechanism for balancing expert utilization. Instead of relying solely on auxiliary losses (which can conflict with the main task loss), expert biases are updated post-step based on actual token counts.

### How it works

Each `DeepSeekRouter` has an `expert_bias` buffer (not a parameter -- no gradients):

```python
router.expert_bias = torch.zeros(num_experts)  # persistent buffer
```

After each training step:
1. **Count**: How many tokens went to each expert (tracked during forward pass)
2. **All-reduce**: Sum counts across all ranks
3. **Update**: applies the configured DEC-2 sign-update mode (default: nmoe / DeepSeek-V3 zero-sum); see "DEC-2 update modes" below.
4. **Clamp**: bias clamped to ±16 (DeepSeek-V3 scale guard).

Experts that received too many tokens get their bias decreased (making them less likely to be selected). Experts that received too few get their bias increased.

### DEC-2 update modes

`bias_update_zero_sum` (canonical block: `training:`, default `True`)
selects between two reference formulations:

| `bias_update_zero_sum` | Formula | Reference |
|-----------------------|---------|-----------|
| `True` (default) | `s = sign(load - 1/E)` ; `bias -= (s - s.mean()) * rate` | [`nmoe.Router.update_bias`](../nmoe/nmoe/model.py) (DeepSeek-V3-style). The mean-subtraction pins the cumulative bias mean at zero so `expert_bias` does not drift unboundedly under asymmetric loads. |
| `False` | `bias += sign(avg_load - load) * rate` (equivalent: `bias -= sign(load - 1/E) * rate`) | Megatron-LM `get_updated_expert_bias`. The mean is allowed to drift up to ±`rate` per step under asymmetric loads — still bounded by the clamp but otherwise unconstrained. |

Both modes target the same intuition (overloaded experts → bias down,
underloaded experts → bias up); the difference is whether the
cumulative bias mean is pinned at zero. Pick `True` (default) to match
nmoe / DeepSeek-V3; pick `False` to reproduce Megatron-LM's published
balancing recipe exactly.

### Per-layer vs Global bias updates

For **Standard MoE**: Each router's bias is updated from its own token counts independently.

For **Global MoE**: Two bias update signals are blended with an alpha schedule:

```python
alpha = 0.5 * (1 + cos(pi * step / warmup_steps))  # cosine decay 1.0 -> 0.0

per_layer_delta = sign(layer_avg - count[i]) * rate
global_delta = sign(global_avg - count[i]) * rate

bias[i] += alpha * per_layer_delta + (1 - alpha) * global_delta
```

Early training (alpha~1.0): Per-layer signal dominates, letting each layer develop its own expert preferences.
Late training (alpha~0.0): Global signal dominates, ensuring overall balance.

### Bias update parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bias_update_rate` | 0.0 (disabled) | Main bias learning rate |
| `bias_warmup_start` | 0.0 | Initial rate (ramps to `bias_update_rate`) |
| `bias_warmup_steps` | 0 | Steps to ramp bias rate |
| `bias_update_zero_sum` | True | DEC-2 mode selector — see "DEC-2 update modes" above. |
| `bias_interpolation` | False | Enable alpha interpolation (global MoE) |
| `bias_interpolation_warmup_steps` | 5000 | Alpha schedule duration |

---

## 4. Routing Statistics & Monitoring

Three distinct components collaborate here; conflating them has caused repeated doc drift.

### 4a. Per-forward-pass records

**File**: `src/models/routing/stats.py` — the `RoutingStats` dataclass.

Accumulated inside a single forward pass and cleared after each step. Carries:

| Field | Description |
|-------|-------------|
| `aux_loss`, `seq_aux_loss`, `branch_aux_loss`, `attention_aux_loss` | Running auxiliary-loss scalars for the forward pass |
| `router_records` | Per-layer MLP/MoE router decisions (selected experts, routing weights, exploration masks) |
| `branch_records` | Per-depth branch router decisions (attn vs MLP) for MoE-Everything |
| `attn_records` | Per-depth attention routing records for MoE-Everything `per_head_*` modes |

These records are the source of truth for branch probabilities, per-head attention routing, and expert-bias health — they are populated by the model's forward pass, not by any training-loop aggregator.

### 4b. Aggregated count / load / margin summaries

**File**: `src/utils/routing_stats.py` — decoupled utility.

Computes time-aggregated summaries from router probabilities / counts that accumulate across many forward passes:

| Metric | Description |
|--------|-------------|
| Per-expert token counts | How many tokens each expert received (per layer, over the reporting window) |
| Load imbalance | Max/min expert utilization ratio |
| Active expert count | Number of experts that received > 0 tokens |
| Router margins | Gap between K-th and (K+1)-th router scores |
| Entropy | Routing distribution entropy (lower = more decisive) |

This module does NOT carry branch records or expert-bias values itself. Those live in §4a (records) and in the router modules (`src/models/routing/bias.py` for the global bias buffers).

### 4c. Plotting / WandB bridge

**File**: `src/training/logging.py`

Pulls the `RoutingStats` object off the model (`model._routing_stats_obj`), converts the records into heatmaps / curves via `src/utils/routing_plots.py`, and logs to WandB. This is the glue that binds §4a records to §4b aggregates in a single WandB panel.

### Visualization helpers

**File**: `src/utils/routing_plots.py`

- Routing heatmaps (expert utilization across layers)
- Expert bias charts (bias value distribution over training; reads from `src/models/routing/bias.py` buffers)
- Per-head attention routing curves (MoE-Everything)

## 5. Future Work / Outstanding Open Questions

### 5.1 Quantile-balancing implementation (planned, not yet shipped)

The `quantile` load-balancing method (`load_balancing_method: quantile`, with `quantile_eta` and `quantile_target_q` knobs) is specified in `docs/plan.md` (DEC-4, DEC-5, DEC-11, DEC-13, DEC-19) but **not yet implemented**. When task15-task19 in the plan land, this section MUST be expanded with:

- The `update_bias_from_quantile` algorithm (cross-microbatch accumulation, fp32 boundaries, `all_gather`-based exact global quantile per DEC-4, post-update buffer clearing).
- The bank-level state ownership (DEC-19) for `global_moe` and `moe_everything` — the `expert_bias`, `local_quantile_scores` accumulator, and `quantile_ema` buffers live on the bank; per-layer routers are stateless consumers.
- The default values: `quantile_eta = 0.05`, `quantile_target_q = 1 - effective_top_k / effective_num_experts` per router class (plain top-k, group-limited, per-head top-1 each documented).

### 5.2 Open question — per-family tuning of `quantile_eta` / `quantile_target_q`

**Status: deferred (`TODO.md` "Per-family tuning of `quantile_eta` / `quantile_target_q`").**

The defaults above come from the original draft (`quantile_eta=0.05`) and from the analytic balanced-routing target (`target_q=1-effective_top_k/effective_num_experts`). They are reasonable starting points but have NOT been tuned per family (`standard_moe`, `global_moe`, `moe_everything`). Whether these defaults remain optimal across all three families is an open question that requires training-to-convergence comparisons to answer reliably.

The deferred work — sweep `quantile_eta ∈ {0.01, 0.02, 0.05, 0.1, 0.2}` × 3 families × multiple seeds, then sweep `quantile_target_q` deviations from the analytic default — is captured in `TODO.md` and will be authored as a follow-up plan once the §8 comparison sweep (`docs/plan.md` DEC-9) is run.

If you find yourself reaching for `quantile_eta` to tune a particular run, **first check whether the AC-10 sanity test ("skewed input → uniform load within K steps where K ≤ ½ deepseek_bias") still holds at your chosen value**. If it does, the default is probably fine and any remaining gap is in another part of the pipeline. If it doesn't, document the failure and add the case to `TODO.md` so the eventual tuning sweep covers it.
