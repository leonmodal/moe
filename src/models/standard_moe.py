"""
Standard MoE — Qwen3MoE with fixed load-balancing loss.

Uses our local load_balancing_loss_func instead of the HF one
(which has a double-softmax bug).
"""
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    MoeCausalLMOutputWithPast,
)

from .load_balancing import load_balancing_loss_func

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

        return output
