# Mixture-of-Everything

We want to design an architecture that dynamically allocates both **weights** and **computation graph** at the **token level**.

We start with a new view about transformer and token states.

---

## 1. A new view of token state as $(E, K, V)$

We represent each token $i$ at depth $\ell$ as a triple:

$$
s_i^{(\ell)} = \big(E_i^{(\ell)},\; K_i^{(\ell)},\; V_i^{(\ell)}\big)
$$

- $E_i^{(\ell)} \in \mathbb{R}^{d}$ — the hidden states
- $K_i^{(\ell)} \in \mathbb{R}^{d_k}$ — Key
- $V_i^{(\ell)} \in \mathbb{R}^{d_v}$ — Value

Based on this separation, we will show what the model updates the token state for attention and MLP separately.

The full sequence state at depth $\ell$ is

$$
S^{(\ell)} = \{(E_i^{(\ell)}, K_i^{(\ell)}, V_i^{(\ell)})\}_{i=0}^{n-1}.
$$

**Initialization.** At depth 0, we set:

$$
E_i^{(0)} = \mathrm{Embed}(x_i)
$$

$$
K_i^{(0)} = \mathrm{RoPE}\!\left(E_i^{(0)} W_K^{(0)},\; i\right), \qquad V_i^{(0)} = E_i^{(0)} W_V^{(0)}
$$

This creates an initial KV cache from the embedding before any attention is run. We use **RoPE** as the only source of positional information — no learned or additive positional embeddings.

Another way to initialize this initial KV cache is to create a big matrix like the token embedding. This could create more memory and might be hard to optimize.

---

## 2. Decomposing the Transformer

A Transformer block is just two operations applied to the token state. They differ in what they update:

| Operation | Updates $E$ | Updates $(K, V)$ | Reads other tokens |
| --- | --- | --- | --- |
| **Attention** | yes | yes | yes |
| **MLP** | yes | no | no |

### 2.1 Attention

Given token $i$ at depth $ell$:

$$
\tilde{E}_i = \mathrm{RMSNorm}(E_i^{(\ell)})
$$

First, compute Q, K, V from the current state:

$$
Q_i = \mathrm{RoPE}(\tilde{E}_i \, W_Q, \; i), \qquad K_i^{(\ell)} = \mathrm{RoPE}(\tilde{E}_i \, W_K, \; i), \qquad V_i^{(\ell)} = \tilde{E}_i \, W_V
$$

Then, attend over all visible positions using the updated KV table:

$$
\alpha_{ij} = \mathrm{softmax}_j\!\left(\frac{Q_i^\top K_j^{(\ell)}}{\sqrt{d_k}}\right), \qquad j \leq i
$$

$$
\mathrm{AttnOut}_i = \left(\sum_{j \leq i} \alpha_{ij} \, V_j^{(\ell)}\right) W_O
$$

State update — only $E$ changes after attention:

$$
E_i^{(\ell+1)} = E_i^{(\ell)} + \mathrm{AttnOut}_i
$$

$$
K_i^{(\ell+1)} = K_i^{(\ell)}, \qquad V_i^{(\ell+1)} = V_i^{(\ell)}
$$

```python
# Attention step for token i at depth l
E_norm = rmsnorm(E_i)
Q_i = rope(E_norm @ W_Q, position=i)
K_i = rope(E_norm @ W_K, position=i)      # refresh KV before attention
V_i = E_norm @ W_V

scores = (Q_i @ K_all.T) / sqrt(d_k)
scores = causal_mask(scores, i)
alpha = softmax(scores)
attn_out = (alpha @ V_all) @ W_O

E_i = E_i + attn_out                       # only E updates after attention
```

### 2.2 MLP

Each MLP uses a **SiLU-gated** feedforward network. Given token $i$ at depth $ell$:

$$
\tilde{E}_i = \mathrm{RMSNorm}(E_i^{(\ell)})
$$

$$
\mathrm{MLPOut}_i = \big(\mathrm{SiLU}(\tilde{E}_i \, W_{\text{gate}}) \odot \tilde{E}_i \, W_{\text{up}}\big) \, W_{\text{down}}
$$

State update — MLP updates only $E$:

$$
E_i^{(\ell+1)} = E_i^{(\ell)} + \mathrm{MLPOut}_i
$$

$$
K_i^{(\ell+1)} = K_i^{(\ell)}, \qquad V_i^{(\ell+1)} = V_i^{(\ell)}
$$

```python
# MLP step for token i at depth l
E_norm = rmsnorm(E_i)

gate = silu(E_norm @ W_gate)
up   = E_norm @ W_up
mlp_out = (gate * up) @ W_down

E_i = E_i + mlp_out
# K_i, V_i unchanged
```

---

## 4. Main architectural design

### 4.1 Mixture of Everything

Each token at each depth uses a hierarchical router:

1. choose **ATTENTION** or **MLP**
2. then choose weights/operators inside that branch

Formally,

$$
r_i^{(\ell)} \in \{\mathrm{ATTN}, \mathrm{MLP}\}.
$$

**MLP bank.** A set of $N_{\text{MLP}}$ experts:

$$
\mathcal{B}_{\text{MLP}} = \{f_1, f_2, \dots, f_{N_{\text{MLP}}}\}
$$

The router selects top-$K$ experts and produces a weighted combination:

$$
\mathrm{MLPOut}_i = \sum_{m \in \mathrm{TopK}(i,\ell)} w_{i,m}^{(\ell)} \; f_m\!\left(\tilde{E}_i\right)
$$

**Attention bank.** A set of $N_{\text{ATTN}}$ attention weight sets. What goes into each set is a design choice — the options below vary in how much is shared vs. independent:

**Fully independent selection.** Each matrix is chosen independently per token:

$$
W_Q \sim \mathcal{B}_Q,\qquad
W_K \sim \mathcal{B}_K,\qquad
W_V \sim \mathcal{B}_V,\qquad
W_O \sim \mathcal{B}_O
$$

Maximum flexibility, but query and key may come from unrelated expert spaces.

$Q,O$ **paired,** $(K,V)$ **paired.** $K$ and $V$ are always chosen together:

$$
(W_Q, W_O) \sim \mathcal{B}_{QO},\qquad
(W_K, W_V) \sim \mathcal{B}_{KV}
$$

$K$ and $V$ stay coherent per token, but cross-token $Q$-$K$ mismatch remains.

$(Q,K,V)$ **bundled.** Everything selected as one unit:

$$
(W_Q, W_K, W_V, W_O) \sim \mathcal{B}_{\mathrm{ATTN}}
$$

This removes internal mismatch within a bundle, but because $(K,V)$ persist across depth, cross-token compatibility is still not guaranteed once different tokens carry memories written by different bundles at different times.

The central question is whether attention scores remain meaningful when different tokens publish $(K,V)$ using different expert-specific projections.

$$
K_i = \phi_K^{(e_i)}(E_i),\qquad
V_i = \phi_V^{(e_i)}(E_i).
$$

Then a single query must compare against keys generated by many different expert families inside one softmax, which might not be ideal, as they live in different subspaces.

### 4.1.2

Compute all KV heads for every single token first, and let the router choose $W_Q$, $W_O$, and then choose the corresponding $K$, $V$ with $Q$ and $O$.

---

### 4.2 Mixture of Depthwise Recurrent MLPs, fixed attention

Attention is standard and always applied at fixed block positions. Routing is used only for MLP experts and optionally for the number of MLP rounds between attention steps.

A token path may therefore look like

$$
\mathrm{Attn}_{\ell_1}
\rightarrow
\mathrm{MLP}
\rightarrow
\mathrm{MLP}
\rightarrow
\mathrm{MLP}
\rightarrow
\mathrm{Attn}_{\ell_2}.
$$

A block is:

```
1. Standard attention
   E <- E + Attn(RMSNorm(E))
   K, V refreshed from E using standard shared projections

2. MLP routing for R rounds
   E <- E + MoE-MLP(RMSNorm(E))
   K, V unchanged during MLP rounds
```

This avoids public (K,V) subspace mismatch because all attention uses the same shared projections.

---

### 4.3 Mixture of MLPs + recursion + mixture of depth

Attention uses shared weights. MLP uses routed experts. On top of this, the model can **loop** — repeating the same weights multiple times before moving on. A per-token router decides at each step:

$$
r_i^{(\ell)} \in \{\mathrm{NEXT}, \mathrm{RECURSE}\}
$$

**RECURSE** re-applies the current block with the same weights. **NEXT** advances to the next block. Different tokens can recurse a different number of times, so recursion depth is token-dependent.

**What gets looped.** Let $B$ be a recurrent block consisting of one or more layers $L_1, dots, L_k$. A block with $k=1$ is a single attention+MLP layer; $k>1$ groups multiple layers as one recurrent unit. The model is:

$$
\underbrace{B_1, \dots, B_P}_{\text{non-recurrent}} \;\rightarrow\; \underbrace{B_{P+1}, \dots, B_{P+R}}_{\text{recurrent}} \;\rightarrow\; \underbrace{B_{P+R+1}, \dots, B_{P+R+C}}_{\text{non-recurrent}}
$$

Each recurrent block $B_j$ repeats $r_i^{(j)}$ times for token $i$, with all iterations sharing the same weights. At each recurrent step $t$:

$$
E_i^{(t)} = B_j(E_i^{(t-1)})
$$

Since the same block weights are reused across iterations, all KV projections stay in the same subspace.

```python
# General recurrent execution for token i
E_i = embed(x_i)

for block in all_blocks:
    if block.recurrent:
        for t in range(r_i[block]):         # r_i per block, token-dependent
            E_i = block(E_i)                # same weights every iteration
    else:
        E_i = block(E_i)                    # run once
```

Within between each attention block, we could still do as many MLP forwards as we want.

**References:**

- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** — [https://arxiv.org/abs/2502.05171](https://arxiv.org/abs/2502.05171)
- **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** — [https://arxiv.org/abs/2507.10524](https://arxiv.org/abs/2507.10524)
- **Scaling Latent Reasoning via Looped Language Models** — [https://arxiv.org/abs/2510.25741](https://arxiv.org/abs/2510.25741)

---

## 5. Depth-wise attention and attention residuals

If the model includes recursion or looping, one may additionally introduce attention over depth/history rather than only over sequence positions.

Let the history of token $i$ up to depth $\ell$ be

$$
\mathcal{H}_i^{(\ell)} = \{E_i^{(0)}, E_i^{(1)}, \dots, E_i^{(\ell)}\}.
$$

A generic depth-attention update is

$$
\widetilde{E}_i^{(\ell+1)}
=
E_i^{(\ell)} + \mathrm{DepthAttn}\!\left(E_i^{(\ell)}, \mathcal{H}_i^{(\ell)}\right).
$$

A practical implementation in the style of **Attention Residuals** is to maintain a running residual memory across recursive steps and allow the current step to attend to that history.

### 5.1 Attention Residuals-style pseudocode

```python
# token i at recurrent step t inside layer l
# states from previous recurrent steps are stored in depth memory

depth_keys   = []
depth_values = []
residual     = E_i

for t in range(T_l_i):   # T_l_i can be token-dependent if routed
    x = rmsnorm(residual)

    # standard sequence attention at this recurrent step
    q_seq = rope(x @ W_Q_seq[l], position=i)
    k_seq = K_all_seq[l][t]          # current sequence KV table
    v_seq = V_all_seq[l][t]
    seq_out = attention(q_seq, k_seq, v_seq, causal=True) @ W_O_seq[l]

    # depth attention over previous recurrent states of the same token
    if len(depth_keys) > 0:
        q_depth = x @ W_Q_depth[l]
        depth_out = attention(
            q_depth,
            stack(depth_keys),
            stack(depth_values),
            causal=False
        ) @ W_O_depth[l]
    else:
        depth_out = 0.0

    # combine sequence attention and depth residual attention
    residual = residual + seq_out + depth_out

    # publish current recurrent state into depth memory
    z = rmsnorm(residual)
    depth_keys.append(z @ W_K_depth[l])
    depth_values.append(z @ W_V_depth[l])

# final recurrent output of this layer
E_i_next = residual
```

This is the key idea: recurrent steps do not only overwrite the state; they can also **attend back over prior intermediate states** within the same layer or loop.

**References:**

- **Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves** — [https://arxiv.org/abs/2601.21582](https://arxiv.org/abs/2601.21582)
- **Attention Residuals** — [https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)
- **DeepCrossAttention: Supercharging Transformer Residual Connections** — [https://arxiv.org/abs/2502.06785](https://arxiv.org/abs/2502.06785)

Without depth-history access, recursion repeatedly overwrites the same token state. With depth attention / attention residuals, the model can instead:

- preserve intermediate reasoning states
- retrieve earlier partial computations
- avoid forcing all information through a single overwritten latent state

This is especially relevant if recursion depth becomes large.

---

## Comparison

1. Comparing to standard MoE with the same amount of total parameters.
2. How to control active parameters?

## Synthetic tasks

$s_0, s_1, \dots, s_{100}, i, s_i$ — only train on $s_i$. $q$ can be computed at $i$.

Can also train a router to decide whether we need to discard some KV as we don't need them.