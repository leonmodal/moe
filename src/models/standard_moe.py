"""
Standard MoE — Qwen3MoE with fixed load-balancing loss.

Uses our local load_balancing_loss_func instead of the HF one
(which has a double-softmax bug).
"""
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
    MoeCausalLMOutputWithPast,
)

from .load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from .router import DeepSeekRouter

StandardMoEConfig = Qwen3MoeConfig


class StandardMoEModel(Qwen3MoeForCausalLM):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)

        # Recompute aux loss with our fixed loss function
        if output.router_logits is not None and output.aux_loss is not None:
            old_aux = output.aux_loss
            new_aux = load_balancing_loss_func(
                output.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
            )
            if output.loss is not None:
                output.loss = output.loss - self.router_aux_loss_coef * old_aux + self.router_aux_loss_coef * new_aux
            output.aux_loss = new_aux

        # Sequence-level aux loss (DeepSeek V2/V3)
        seq_coef = getattr(self, "_seq_aux_loss_coef", 0.0)
        if seq_coef > 0 and output.router_logits is not None and output.loss is not None:
            input_ids = kwargs.get("input_ids")
            bsz = input_ids.shape[0] if input_ids is not None else 1
            seq_aux = seq_load_balancing_loss_func(
                output.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                batch_size=bsz,
            )
            output.loss = output.loss + seq_coef * seq_aux

        return output


class DeepSeekStandardMoEModel(StandardMoEModel):
    """Standard MoE with DeepSeek V3 sigmoid + expert-bias routing."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        # Replace each MoE layer's softmax router with DeepSeekRouter
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                layer.mlp.gate = DeepSeekRouter(config)
        self.post_init()
