"""
Fixed load-balancing loss for MoE training.

Two variants:
  1. load_balancing_loss_func — standard batch-level Switch Transformer loss
  2. seq_load_balancing_loss_func — DeepSeek V2/V3 sequence-level loss

Fixes vs the HuggingFace transformers implementation:
  1. No double softmax — router already returns softmax probabilities,
     the HF loss applies softmax again which flattens the distribution
     and makes the loss blind to imbalance.
  2. f_i is kept local per rank (no all_reduce) to preserve the
     theoretical minimum of the load-balancing loss.
"""
import torch
import torch.nn.functional as F


def normalize_router_scores(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
) -> torch.Tensor | tuple[torch.Tensor] | None:
    """Normalize router scores to sum-to-1 per token.

    For softmax routers this is effectively a no-op. For DeepSeek sigmoid
    routers this produces a probability-like diagnostic view that is easier
    to interpret for logging and comparisons.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return gate_logits

    normalized = []
    for layer_gate in gate_logits:
        scores = layer_gate.float()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        normalized.append(scores)
    return tuple(normalized)


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    attention_mask: torch.Tensor | None = None,
    token_masks: tuple[torch.Tensor] | None = None,
) -> torch.Tensor | int:
    """
    Computes auxiliary load balancing loss (Switch Transformer).

    Args:
        gate_logits: Tuple of [T, E] softmax-probability tensors, one per layer.
                     These are ALREADY softmax probabilities from the router.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        attention_mask: Optional [batch_size, seq_len] mask.

    Returns:
        Scalar load-balancing loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device

    if token_masks is not None:
        filtered = []
        for layer_idx, layer_gate in enumerate(gate_logits):
            if layer_idx >= len(token_masks) or token_masks[layer_idx] is None:
                filtered.append(layer_gate.to(compute_device))
                continue
            layer_mask = token_masks[layer_idx].reshape(-1).bool().to(layer_gate.device)
            if layer_mask.any():
                filtered.append(layer_gate[layer_mask].to(compute_device))
        if not filtered:
            return gate_logits[0].new_zeros(())
        concatenated_gate_logits = torch.cat(filtered, dim=0)
    else:
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    # gate_logits are already softmax probabilities from the router — use directly.
    routing_weights = concatenated_gate_logits

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # f_i: fraction of tokens routed to each expert (hard assignment)
        # Kept local per rank — no all_reduce — to preserve the theoretical
        # minimum of the load-balancing loss.
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # p_i: average router probability per expert (soft, differentiable)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def normalized_load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    attention_mask: torch.Tensor | None = None,
    token_masks: tuple[torch.Tensor] | None = None,
) -> torch.Tensor | int:
    """Diagnostic batch aux loss using normalized router scores.

    This is mainly useful for DeepSeek sigmoid routers, where the raw
    batch aux can be large because scores are not normalized to sum to 1.
    """
    normalized = normalize_router_scores(gate_logits)
    return load_balancing_loss_func(
        normalized,
        num_experts=num_experts,
        top_k=top_k,
        attention_mask=attention_mask,
        token_masks=token_masks,
    )


def seq_load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    batch_size: int = 1,
    selected_experts: tuple[torch.Tensor] | None = None,
    token_masks: tuple[torch.Tensor] | None = None,
) -> torch.Tensor | int:
    """
    Sequence-level load balancing loss (DeepSeek V3, arxiv 2412.19437 Eqs. 17-20).

    Matches the paper and Megatron-LM:
      Eq. 19: s'_{i,t} = s_{i,t} / Σ_j s_{j,t}    (normalize scores to sum-to-1)
      Eq. 18: f_i = (E / (K * T)) * Σ_t 1[expert i in topK]
      Eq. 20: P_i = (1/T) * Σ_t s'_{i,t}
      Eq. 17: L_Bal = Σ_i f_i * P_i

    For softmax routers the normalization is a no-op (already sums to 1).
    For sigmoid routers it converts raw scores to a proper distribution.
    At uniform routing the loss equals 1.0.

    Args:
        gate_logits: Tuple of [T, E] score tensors, one per layer.
                     T = batch_size * seq_len.  May be softmax probs
                     (sum-to-1) or raw sigmoid scores (will be normalized).
        num_experts: Total number of experts (E).
        top_k: Number of experts selected per token (K).
        batch_size: Batch size (needed to reshape T → seq_len).
        selected_experts: Optional tuple of actual routed expert indices from
                          router forward, one tensor per layer of shape [T, K].
                          If provided, f_i is computed from these assignments
                          instead of top-k(scores).

    Returns:
        Scalar load-balancing loss (without coefficient — multiply by alpha outside).
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    total_loss = gate_logits[0].new_zeros(())
    num_active_layers = 0

    for layer_idx, layer_gate in enumerate(gate_logits):
        T, E = layer_gate.shape
        seq_len = T // batch_size

        # Eq. 19: normalize scores to sum-to-1 per token
        # (no-op for softmax routers, necessary for sigmoid routers)
        scores = layer_gate.float()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)

        # Reshape: (B*S, E) → (B, S, E) — per-sequence view
        scores = scores.reshape(batch_size, seq_len, E)               # (B, S, E)

        if selected_experts is not None and layer_idx < len(selected_experts):
            selected = selected_experts[layer_idx]
            if selected is not None and selected.shape == (T, top_k):
                selected = selected.reshape(batch_size, seq_len, top_k).to(scores.device)
            else:
                selected = None
        else:
            selected = None

        if token_masks is not None and layer_idx < len(token_masks):
            token_mask = token_masks[layer_idx]
            if token_mask is not None:
                token_mask = token_mask.reshape(batch_size, seq_len).to(scores.device).float()
            else:
                token_mask = None
        else:
            token_mask = None

        # Fall back to score top-k if actual routing indices aren't available.
        if selected is None:
            _, selected = torch.topk(scores, top_k, dim=-1)          # (B, S, K)
        expert_mask = F.one_hot(selected, num_experts)                # (B, S, K, E)

        if token_mask is None:
            # Eq. 18: f_i = (E / (K * T)) * Σ_t 1[expert i selected]
            expert_counts = expert_mask.float().sum(dim=(1, 2))           # (B, E)
            f_i = (num_experts / (top_k * seq_len)) * expert_counts       # (B, E)

            # Eq. 20: P_i = (1/T) * Σ_t s'_{i,t}
            P_i = scores.mean(dim=1)                                      # (B, E)

            # Eq. 17: L_Bal = Σ_i f_i * P_i
            per_seq_loss = (f_i * P_i).sum(dim=-1)                        # (B,)
            total_loss = total_loss + per_seq_loss.mean()
            num_active_layers += 1
            continue

        token_count = token_mask.sum(dim=1)                               # (B,)
        valid = token_count > 0
        if not valid.any():
            continue

        expert_mask = expert_mask.float() * token_mask.unsqueeze(-1).unsqueeze(-1)
        expert_counts = expert_mask.sum(dim=(1, 2))                       # (B, E)
        f_i = (num_experts / (top_k * token_count.clamp(min=1.0).unsqueeze(-1))) * expert_counts
        P_i = (scores * token_mask.unsqueeze(-1)).sum(dim=1) / token_count.clamp(min=1.0).unsqueeze(-1)

        # Eq. 17: L_Bal = Σ_i f_i * P_i
        per_seq_loss = (f_i * P_i).sum(dim=-1)                        # (B,)
        total_loss = total_loss + per_seq_loss[valid].mean()
        num_active_layers += 1

    if num_active_layers == 0:
        return gate_logits[0].new_zeros(())
    return total_loss / num_active_layers
