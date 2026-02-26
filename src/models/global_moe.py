"""
Global MoE — minimal subclasses on top of Qwen3MoE.

The only changes vs standard Qwen3MoE:
  1. GlobalMoEConfig: num_experts=2048 (the global pool size)
  2. GlobalSparseMoeBlock: router only — no expert weights stored here
  3. GlobalMoEDecoderLayer: uses GlobalSparseMoeBlock; forward accepts global_experts kwarg
  4. GlobalMoEModel: owns one shared Qwen3MoeExperts(2048); injects it into every layer call
  5. GlobalMoEForCausalLM: wraps GlobalMoEModel

Everything else (GQA, QK-norm, RoPE, RMSNorm, KV cache, flash-attn dispatch,
gradient checkpointing, load-balancing loss, generation) is inherited from Qwen3.

Memory note: GlobalMoEDecoderLayer.__init__ passes a dummy config with
mlp_only_layers set to avoid allocating a per-layer 2048-expert pool that
would immediately be thrown away.
"""
import copy

import torch
import torch.nn as nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeTopKRouter,
    load_balancing_loss_func,
)


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

    def __init__(self, config: GlobalMoEConfig):
        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)  # same router as Qwen3

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

    def __init__(self, config: GlobalMoEConfig, layer_idx: int):
        # Trick: tell the parent to build a cheap dense MLP so it doesn't
        # allocate a 2048-expert Qwen3MoeExperts that we'd immediately discard.
        init_cfg = copy.deepcopy(config)
        init_cfg.mlp_only_layers = list(range(config.num_hidden_layers))
        super().__init__(init_cfg, layer_idx)

        # Now replace the dense MLP with our router-only block.
        self.mlp = GlobalSparseMoeBlock(config)

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

    def __init__(self, config: GlobalMoEConfig):
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
            GlobalMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)
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
    Inherits loss computation, aux-loss, generation, etc. from Qwen3MoeForCausalLM.
    """

    def __init__(self, config: GlobalMoEConfig):
        super().__init__(config)
        self.model = GlobalMoEModel(config)
        self.num_experts = config.num_experts
        self.post_init()
