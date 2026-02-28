"""
DeepSeek V3 aux-loss-free router with expert bias.

Replaces softmax routing with sigmoid + non-gradient expert bias:
  1. Sigmoid scoring — scores = sigmoid(W @ h)
  2. Biased top-k selection — topk(scores + expert_bias)
  3. Unbiased weights — gather original scores for selected experts, normalize
  4. Bias update after each step — bias += sign(avg - tokens_per_expert) * rate

The bias is updated externally by update_expert_biases() in train.py,
not through gradient descent — this prevents the router from gaming the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter


class DeepSeekRouter(Qwen3MoeTopKRouter):
    """
    Sigmoid + expert-bias router (DeepSeek V3 style).

    Subclasses Qwen3MoeTopKRouter so that OutputRecorder hooks
    (used by HF for collecting router_logits) still fire.
    """

    def __init__(self, config):
        super().__init__(config)
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

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # 1. Sigmoid scoring (not softmax)
        router_logits = F.linear(hidden_states, self.weight)       # (T, E)
        scores = torch.sigmoid(router_logits.float())              # (T, E) in (0, 1)

        # 2. Biased top-k selection
        # Ensure bias is float32 even if mixed precision recast the buffer
        bias = self.expert_bias.float()
        biased_scores = scores + bias.unsqueeze(0)                 # (T, E)
        _, top_k_idx = torch.topk(biased_scores, self.top_k, dim=-1)  # (T, K)

        # 3. Gather unbiased scores for selected experts, normalize to sum-to-1
        router_top_value = scores.gather(1, top_k_idx)             # (T, K)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(hidden_states.dtype)

        # 4. Accumulate token counts (only during real forward, not recompute)
        if torch.is_grad_enabled():
            with torch.no_grad():
                counts = torch.bincount(
                    top_k_idx.reshape(-1),
                    minlength=self.num_experts,
                ).float()
                self.local_tokens_per_expert += counts

        # Return sigmoid probs as router_logits (values in (0,1), don't sum to 1)
        return scores, router_top_value, top_k_idx
