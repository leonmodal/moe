"""
Standard MoE — Qwen3MoE with fixed load-balancing loss.

Uses our local load_balancing_loss_func instead of the HF one
(which has a double-softmax bug).
"""
import torch

from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)

from .load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from .router import DeepSeekRouter, ExplorationTopKRouter, collect_router_topk_indices

StandardMoEConfig = Qwen3MoeConfig


class StandardMoEModel(Qwen3MoeForCausalLM):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self._seq_aux_loss_coef = getattr(config, "seq_aux_loss_coef", 0.0)
        for layer in self.model.layers:
            if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                continue
            old_gate = getattr(layer.mlp, "gate", None)
            if isinstance(old_gate, (DeepSeekRouter, ExplorationTopKRouter)) or old_gate is None:
                continue
            new_gate = ExplorationTopKRouter(config)
            new_gate.weight.data.copy_(old_gate.weight.data)
            layer.mlp.gate = new_gate

    def _collect_selected_experts(self) -> tuple[object, ...] | None:
        routers = [
            getattr(getattr(layer, "mlp", None), "gate", None)
            for layer in self.model.layers
            if isinstance(getattr(layer, "mlp", None), Qwen3MoeSparseMoeBlock)
        ]
        return collect_router_topk_indices(router for router in routers if router is not None)

    def get_all_balancing_owners(self):
        """Yield (owner_module, label) for every load-balancing owner.

        Standard MoE has per-layer expert pools, so each layer's `gate` owns its
        own `expert_bias` / `local_tokens_per_expert` buffers (DEC-19 rule:
        per-layer pools keep per-router state).
        """
        for layer in self.model.layers:
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is not None and hasattr(gate, "expert_bias") and hasattr(gate, "local_tokens_per_expert"):
                yield gate, "mlp"

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        selected_experts = self._collect_selected_experts()
        if selected_experts is not None:
            output.selected_experts = selected_experts

        # AC-1: gate aux / seq-aux additions by the resolved
        # `load_balancing_method`. `None` (no method set) keeps legacy
        # coefficient-driven behavior; explicit methods restrict to the
        # method's active loss term. `normalize_balancing_config` already
        # auto-zeros conflicting coefficients, but the explicit gate here
        # is belt-and-suspenders against any future default that leaks a
        # non-zero coefficient into a non-active method.
        method = getattr(self, "_load_balancing_method", None)
        aux_active = method is None or method == "aux_loss"
        seq_aux_active = method is None or method == "seq_aux_loss"

        # Recompute aux loss with our fixed loss function. The base class
        # already added `self.router_aux_loss_coef * old_aux` to the loss; we
        # subtract that off in BOTH branches so non-aux methods don't pay for
        # the base class's broken double-softmax aux contribution. For aux
        # methods we then add `coef * new_aux` (graph-bearing) so the
        # corrected aux loss term can backprop through router_logits. For
        # non-aux methods (DEC-15 DETACH-ONLY) we:
        #   - subtract `coef * old_aux.detach()` (no gradient flow through
        #     the base class's old_aux), and
        #   - compute `new_aux` under `torch.no_grad()` purely as detached
        #     telemetry on `output.aux_loss` so the model output never
        #     carries an autograd graph reference through aux tensors.
        if output.router_logits is not None and output.aux_loss is not None:
            old_aux = output.aux_loss
            if aux_active:
                new_aux = load_balancing_loss_func(
                    output.router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    selected_experts=selected_experts,
                )
                if output.loss is not None:
                    output.loss = output.loss - self.router_aux_loss_coef * old_aux
                    output.loss = output.loss + self.router_aux_loss_coef * new_aux
                output.aux_loss = new_aux
            else:
                if output.loss is not None:
                    output.loss = output.loss - self.router_aux_loss_coef * old_aux.detach()
                with torch.no_grad():
                    new_aux = load_balancing_loss_func(
                        output.router_logits,
                        self.num_experts,
                        self.num_experts_per_tok,
                        selected_experts=selected_experts,
                    )
                output.aux_loss = new_aux  # already detached via no_grad context

        # Sequence-level aux loss (DeepSeek V2/V3) — gated by method.
        seq_coef = getattr(self, "_seq_aux_loss_coef", 0.0)
        if (
            seq_aux_active
            and seq_coef > 0
            and output.router_logits is not None
            and output.loss is not None
        ):
            input_ids = kwargs.get("input_ids")
            bsz = input_ids.shape[0] if input_ids is not None else 1
            seq_aux = seq_load_balancing_loss_func(
                output.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                batch_size=bsz,
                selected_experts=selected_experts,
            )
            output.loss = output.loss + seq_coef * seq_aux

        # DEC-15 DETACH-ONLY: for non-aux methods, the model output's
        # `router_logits` must be detached even if the caller forced
        # `output_router_logits=True`. Telemetry is preserved (the values are
        # available) but the autograd graph is not retained — so a downstream
        # consumer can't accidentally backprop through them.
        if not (aux_active or seq_aux_active) and output.router_logits is not None:
            output.router_logits = tuple(t.detach() for t in output.router_logits)

        return output


class DeepSeekStandardMoEModel(StandardMoEModel):
    """Standard MoE with DeepSeek V3 sigmoid + expert-bias routing."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        # Replace each MoE layer's softmax router with DeepSeekRouter
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                old_gate = layer.mlp.gate
                new_gate = DeepSeekRouter(config)
                new_gate.weight.data.copy_(old_gate.weight.data)
                layer.mlp.gate = new_gate
