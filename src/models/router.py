"""
DeepSeek V3 aux-loss-free router with expert bias.

Replaces softmax routing with sigmoid + non-gradient expert bias:
  1. FP32 gating linear — logits = fp32(W) @ fp32(h)
  2. Sigmoid scoring — scores = sigmoid(logits)
  3. Group-limited top-k — select group_topk groups, then topk experts within
  4. Biased selection — topk(scores + expert_bias) with unbiased weight gather
  5. Normalization + scaling — normalize to sum-to-1, then multiply by scaling_factor
  6. Bias update after each step — bias += sign(avg - tokens_per_expert) * rate

The bias is updated externally by update_expert_biases() in train.py,
not through gradient descent — this prevents the router from gaming the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_groups: int,
    group_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform top-k routing on a subset of expert groups.

    Matches Megatron-LM implementation:
      1. Divide experts into num_groups equal groups
      2. Score each group by sum of top-(topk // group_topk) expert scores within it
      3. Select group_topk groups per token
      4. Pick topk experts from those groups only

    Args:
        scores: (T, E) routing scores.
        topk: Number of experts to select per token.
        num_groups: Number of expert groups.
        group_topk: Number of groups to select per token.

    Returns:
        (top_scores, top_indices) each of shape (T, topk).
    """
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // num_groups

    # Score each group by sum of top-(topk // group_topk) within that group
    group_scores = (
        scores.view(num_tokens, num_groups, experts_per_group)
        .topk(topk // group_topk, dim=-1)[0]
        .sum(dim=-1)
    )  # (T, num_groups)

    # Select top groups
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]  # (T, group_topk)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)  # (T, num_groups)

    # Expand group mask to expert mask
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, experts_per_group)
        .reshape(num_tokens, -1)
    )  # (T, E)

    # Mask out experts in non-selected groups, then topk
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    top_scores, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return top_scores, top_indices


class DeepSeekRouter(Qwen3MoeTopKRouter):
    """
    Sigmoid + expert-bias router (DeepSeek V3 style).

    Subclasses Qwen3MoeTopKRouter so that OutputRecorder hooks
    (used by HF for collecting router_logits) still fire.
    """

    def __init__(self, config):
        super().__init__(config)
        # DeepSeek V3 routing parameters
        self.scaling_factor = getattr(config, "topk_scaling_factor", None)
        self.num_groups = getattr(config, "num_groups", None)
        self.group_topk = getattr(config, "group_topk", None)

        # Persistent buffer: survives checkpointing
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )
        # Non-persistent buffer: reset each step, not saved in state_dict
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        # Last forward's selected experts (T, K). Used by seq aux loss
        # so f_i matches actual biased routing assignments.
        self._last_top_k_idx = None

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # 1. FP32 gating linear — match Megatron-LM moe_router_dtype=fp32
        router_logits = F.linear(
            hidden_states.float(), self.weight.float()
        )  # (T, E) in fp32

        # 2. Sigmoid scoring in fp32
        scores = torch.sigmoid(router_logits)  # (T, E) in (0, 1), fp32

        # 3. Biased top-k selection (with optional group-limited routing)
        bias = self.expert_bias.float()
        biased_scores = scores + bias.unsqueeze(0)  # (T, E)

        if self.num_groups is not None and self.group_topk is not None:
            # Group-limited top-k: select from top groups only
            _, top_k_idx = group_limited_topk(
                biased_scores,
                topk=self.top_k,
                num_groups=self.num_groups,
                group_topk=self.group_topk,
            )
        else:
            _, top_k_idx = torch.topk(biased_scores, self.top_k, dim=-1)  # (T, K)

        # 4. Gather unbiased scores for selected experts
        router_top_value = scores.gather(1, top_k_idx)  # (T, K)

        # 5. Normalize + scaling factor
        if self.norm_topk_prob:
            router_top_value = router_top_value / (
                router_top_value.sum(dim=-1, keepdim=True) + 1e-20
            )
        if self.scaling_factor is not None:
            router_top_value = router_top_value * self.scaling_factor

        router_top_value = router_top_value.to(hidden_states.dtype)

        # 6. Accumulate token counts (only during real forward, not recompute)
        if torch.is_grad_enabled():
            with torch.no_grad():
                counts = torch.bincount(
                    top_k_idx.reshape(-1),
                    minlength=self.num_experts,
                ).float()
                self.local_tokens_per_expert += counts
                self._last_top_k_idx = top_k_idx.detach()
        else:
            self._last_top_k_idx = top_k_idx.detach()

        # Return sigmoid probs as router_logits (values in (0,1), don't sum to 1)
        return scores, router_top_value, top_k_idx
