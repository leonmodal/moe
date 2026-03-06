"""
Global MoE — minimal subclasses on top of Qwen3MoE.

The only changes vs standard Qwen3MoE:
  1. GlobalMoEConfig: num_experts=2048 (the global pool size)
  2. GlobalSparseMoeBlock: router only — no expert weights stored here
  3. GlobalMoEDecoderLayer: uses GlobalSparseMoeBlock; forward accepts global_experts kwarg
  4. GlobalMoEModel: owns one shared Qwen3MoeExperts(2048); injects it into every layer call
  5. GlobalMoEForCausalLM: wraps GlobalMoEModel, uses fixed load-balancing loss

Everything else (GQA, QK-norm, RoPE, RMSNorm, KV cache, flash-attn dispatch,
gradient checkpointing, generation) is inherited from Qwen3.

Memory note: GlobalMoEDecoderLayer.__init__ passes a dummy config with
mlp_only_layers set to avoid allocating a per-layer 2048-expert pool that
would immediately be thrown away.
"""
import copy

import torch.nn as nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeTopKRouter,
)

from .load_balancing import load_balancing_loss_func, seq_load_balancing_loss_func
from .router import DeepSeekRouter


class GlobalMoEConfig(Qwen3MoeConfig):
    """
    Same as Qwen3MoeConfig but num_experts refers to the global shared pool
    (2048 by default, matching 16 layers × 128 per-layer experts).
    """
    model_type = "global_moe"

    def __init__(self, num_experts: int = 2048, **kwargs):
        super().__init__(num_experts=num_experts, **kwargs)


class GlobalSparseMoeBlock(nn.Module):
    """
    Router-only MoE block.  No expert weights live here.
    The global expert pool is passed in at forward() time from GlobalMoEModel.
    This ensures Qwen3MoeExperts is registered exactly once in the module tree.
    """

    def __init__(self, config: GlobalMoEConfig, router_class=None):
        super().__init__()
        if router_class is None:
            router_class = Qwen3MoeTopKRouter
        self.gate = router_class(config)

    def forward(self, hidden_states, global_experts: Qwen3MoeExperts):
        B, T, H = hidden_states.shape
        flat = hidden_states.view(-1, H)
        # gate returns (router_logits, routing_weights, selected_experts)
        # OutputRecorder hook on Qwen3MoeTopKRouter captures router_logits[0]
        # automatically for the aux loss — nothing extra needed here.
        _, routing_weights, selected_experts = self.gate(flat)
        out = global_experts(flat, selected_experts, routing_weights)
        return out.reshape(B, T, H)


class GlobalMoEDecoderLayer(Qwen3MoeDecoderLayer):
    """
    Decoder layer that routes into the global expert pool instead of its own.

    Inherits everything from Qwen3MoeDecoderLayer (attention, norms,
    gradient-checkpointing support) and only changes the mlp.
    """

    def __init__(self, config: GlobalMoEConfig, layer_idx: int, router_class=None):
        # Trick: tell the parent to build a cheap dense MLP so it doesn't
        # allocate a 2048-expert Qwen3MoeExperts that we'd immediately discard.
        init_cfg = copy.deepcopy(config)
        init_cfg.mlp_only_layers = list(range(config.num_hidden_layers))
        super().__init__(init_cfg, layer_idx)

        # Now replace the dense MLP with our router-only block.
        self.mlp = GlobalSparseMoeBlock(config, router_class=router_class)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        global_experts=None,   # injected by GlobalMoEModel
        **kwargs,
    ):
        # Attention sub-layer — identical to Qwen3MoeDecoderLayer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE FFN sub-layer — route into the globally shared expert pool
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, global_experts)
        hidden_states = residual + hidden_states

        return hidden_states


class GlobalMoEModel(Qwen3MoeModel):
    """
    Qwen3MoeModel with a single shared expert pool across all layers.
    """

    def __init__(self, config: GlobalMoEConfig, router_class=None):
        # Build parent with all-dense layers to avoid creating 16 × 2048-expert
        # pools that would be immediately replaced (and would OOM during init).
        init_cfg = copy.deepcopy(config)
        init_cfg.mlp_only_layers = list(range(config.num_hidden_layers))
        super().__init__(init_cfg)
        self.config = config  # restore original config

        # Single shared expert pool — registered once here at the model root.
        # DDP/FSDP sees these parameters exactly once.
        self.global_experts = Qwen3MoeExperts(config)

        # Replace per-layer MoE blocks with router-only versions.
        self.layers = nn.ModuleList([
            GlobalMoEDecoderLayer(config, i, router_class=router_class)
            for i in range(config.num_hidden_layers)
        ])
        self.post_init()

    def forward(self, *args, **kwargs):
        # Inject global_experts into kwargs; Qwen3MoeModel.forward passes
        # **kwargs through to every decoder layer call.
        kwargs["global_experts"] = self.global_experts
        return super().forward(*args, **kwargs)


class GlobalMoEForCausalLM(Qwen3MoeForCausalLM):
    """
    Causal LM head wrapping GlobalMoEModel.
    Uses fixed load-balancing loss (no double softmax, global-batch f_i).
    """

    _router_class = None  # Override in subclasses

    def __init__(self, config: GlobalMoEConfig):
        super().__init__(config)
        self.model = GlobalMoEModel(config, router_class=self._router_class)
        self.num_experts = config.num_experts
        self.post_init()

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
            # For DeepSeekRouter, use actual routed expert assignments if available
            # so f_i matches biased routing (scores + expert_bias).
            selected_experts = None
            try:
                selected = []
                for layer in self.model.layers:
                    gate = getattr(getattr(layer, "mlp", None), "gate", None)
                    idx = getattr(gate, "_last_top_k_idx", None)
                    if idx is None:
                        selected = None
                        break
                    selected.append(idx)
                if selected is not None and len(selected) == len(output.router_logits):
                    selected_experts = tuple(selected)
            except Exception:
                selected_experts = None
            seq_aux = seq_load_balancing_loss_func(
                output.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                batch_size=bsz,
                selected_experts=selected_experts,
            )
            output.loss = output.loss + seq_coef * seq_aux

        return output


class DeepSeekGlobalMoEForCausalLM(GlobalMoEForCausalLM):
    """Global MoE with DeepSeek V3 sigmoid + expert-bias routing."""

    _router_class = DeepSeekRouter
