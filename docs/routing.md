# Routing & Expert Selection

This document covers all router implementations, expert selection mechanisms, load balancing losses, and the expert bias update system.

## Table of Contents

- [1. Router Implementations](#1-router-implementations)
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
3. **Update**: `bias[i] -= sign(count[i] - mean_count) * bias_rate`

Experts that received too many tokens get their bias decreased (making them less likely to be selected). Experts that received too few get their bias increased.

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
| `bias_interpolation` | False | Enable alpha interpolation (global MoE) |
| `bias_interpolation_warmup_steps` | 5000 | Alpha schedule duration |

---

## 4. Routing Statistics & Monitoring

### Collected Metrics

**File**: `src/utils/routing_stats.py`

| Metric | Description |
|--------|-------------|
| Per-expert token counts | How many tokens each expert received per layer |
| Load imbalance | Max/min expert utilization ratio |
| Active expert count | Number of experts that received > 0 tokens |
| Router margins | Gap between K-th and (K+1)-th router scores (higher = more decisive routing) |
| Branch probabilities | Attention vs MLP selection rates per depth |
| Expert bias stats | Mean, std, min, max of bias values |

### Visualization

**File**: `src/utils/routing_plots.py`

- **Routing heatmaps**: Expert utilization across layers (logged to WandB)
- **Expert bias charts**: Bias value distribution over training
- **Per-head utilization histograms**: For MoE-Everything attention experts
