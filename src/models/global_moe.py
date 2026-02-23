"""
Global MoE: a single shared pool of 2048 experts used by ALL layers.

Architecture is identical to StandardMoEModel except:
  - One top-level `GlobalExpertPool` (2048 experts) instead of per-layer pools.
  - Each layer's router selects top-8 from the global 2048, not its own 128.
  - `GlobalMoELayer` receives `global_experts` at forward-time (not as a submodule)
    so FSDP/DDP wraps the expert pool exactly once at the model root.

Gradient flow:
  - Same expert receives gradients from ALL 16 layers simultaneously → effective
    per-expert batch size is ~16× larger than in standard MoE → expert updates are
    much more frequent, potentially accelerating specialization.

Same total parameter count as StandardMoEModel (2048 experts either way).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_components import (
    Qwen3MoEConfig,
    RMSNorm,
    RotaryEmbedding,
    GQAttention,
    DenseMLP,
    ExpertPool,
    TopKRouter,
    load_balancing_loss,
)
from dataclasses import dataclass


@dataclass
class GlobalMoEConfig(Qwen3MoEConfig):
    """
    Drop-in extension of Qwen3MoEConfig for Global MoE.
    `num_experts` is reinterpreted as the **global pool size** (2048 by default).
    Each layer routes top-`num_experts_per_tok` from this shared pool.
    """
    num_experts: int = 2048          # global pool (= 16 layers × 128 per-layer)
    num_experts_per_tok: int = 8


class GlobalMoEBlock(nn.Module):
    """
    MoE FFN block that routes into a globally shared expert pool.

    The expert pool is NOT stored here — it is passed into forward() by the
    top-level model.  This ensures there is exactly one registration of the
    expert weights in the module tree (at GlobalMoEModel.global_experts),
    which is required for correct FSDP parameter sharding.
    """

    def __init__(self, config: GlobalMoEConfig):
        super().__init__()
        self.router = TopKRouter(config.hidden_size, config.num_experts, config)

    def forward(
        self,
        x: torch.Tensor,             # [B, T, H]
        global_experts: ExpertPool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        flat = x.view(B * T, H)
        router_logits, scores, indices = self.router(flat)
        out = global_experts(flat, indices, scores)
        return out.view(B, T, H), router_logits


class GlobalMoELayer(nn.Module):
    """Pre-norm transformer block that routes into the global expert pool."""

    def __init__(self, config: GlobalMoEConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm          = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn                = GQAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp                      = GlobalMoEBlock(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        global_experts: ExpertPool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        normed = self.post_attention_layernorm(x)
        ffn_out, router_logits = self.mlp(normed, global_experts)
        x = x + ffn_out
        return x, router_logits


class GlobalMoEModel(nn.Module):
    """
    Global MoE pretraining model (Qwen3 architecture).

    2048 experts shared across all 16 layers.  Each layer routes top-8
    from the global pool independently — different layers may select
    different sets of experts for the same token.
    """

    def __init__(self, config: GlobalMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb   = RotaryEmbedding(config)

        # ── Single top-level expert pool (registered once) ──────────────
        self.global_experts = ExpertPool(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )

        # ── Per-layer attention + router (no per-layer experts) ─────────
        self.layers = nn.ModuleList([
            GlobalMoELayer(config, i) for i in range(config.num_hidden_layers)
        ])

        self.norm    = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        nn.init.normal_(self.embed_tokens.weight, std=std)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=std)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x, position_ids)

        all_router_logits = []
        for layer in self.layers:
            # global_experts passed at call time — not a submodule of layer
            x, router_logits = layer(x, cos, sin, self.global_experts)
            all_router_logits.append(router_logits)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss, metrics = None, {}
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            aux_loss = load_balancing_loss(
                tuple(all_router_logits),
                self.config.num_experts,
                self.config.num_experts_per_tok,
            )
            loss = ce_loss + self.config.router_aux_loss_coef * aux_loss
            metrics = {
                "ce_loss":    ce_loss.detach(),
                "aux_loss":   aux_loss.detach(),
                "total_loss": loss.detach(),
            }

        return logits, loss, metrics
