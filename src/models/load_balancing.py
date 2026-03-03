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


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    attention_mask: torch.Tensor | None = None,
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


def seq_load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    batch_size: int = 1,
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

    Returns:
        Scalar load-balancing loss (without coefficient — multiply by alpha outside).
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    total_loss = gate_logits[0].new_zeros(())

    for layer_gate in gate_logits:
        T, E = layer_gate.shape
        seq_len = T // batch_size

        # Eq. 19: normalize scores to sum-to-1 per token
        # (no-op for softmax routers, necessary for sigmoid routers)
        scores = layer_gate.float()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)

        # Reshape: (B*S, E) → (B, S, E) — per-sequence view
        scores = scores.reshape(batch_size, seq_len, E)               # (B, S, E)

        # Per-sequence topk selection (normalization is monotonic → same topk)
        _, selected = torch.topk(scores, top_k, dim=-1)              # (B, S, K)
        expert_mask = F.one_hot(selected, num_experts)                # (B, S, K, E)

        # Eq. 18: f_i = (E / (K * T)) * Σ_t 1[expert i selected]
        expert_counts = expert_mask.float().sum(dim=(1, 2))           # (B, E)
        f_i = (num_experts / (top_k * seq_len)) * expert_counts       # (B, E)

        # Eq. 20: P_i = (1/T) * Σ_t s'_{i,t}
        P_i = scores.mean(dim=1)                                      # (B, E)

        # Eq. 17: L_Bal = Σ_i f_i * P_i
        per_seq_loss = (f_i * P_i).sum(dim=-1)                        # (B,)

        total_loss = total_loss + per_seq_loss.mean()

    return total_loss / len(gate_logits)
