"""Recurrent 4-8-4 MoE models built from normal Qwen3-MoE blocks.

The recurrent core reuses a stack of decoder blocks for multiple passes.  This
module intentionally does not use MoE-Everything branch routing; every block is
a normal transformer block whose FFN is a routed MoE.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from src.models.global_moe import GlobalSparseMoeBlock
from src.models.load_balancing import load_balancing_loss_func
from src.models.mask_compat import call_mask_function
from src.models.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeDecoderLayer,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)
from src.models.router import (
    DeepSeekRouter,
    ExplorationTopKRouter,
    checkpoint_recompute_context,
)


class RecurrentMoEConfig(Qwen3MoeConfig):
    """Config for recurrent standard/global MoE block stacks."""

    model_type = "recurrent_moe"

    def __init__(
        self,
        *,
        prelude_layers: int = 4,
        recurrent_layers: int = 8,
        coda_layers: int = 4,
        recurrent_expert_pool: str = "per_layer",
        core_attention: bool = True,
        core_attention_layers: list[int] | tuple[int, ...] | None = None,
        recurrent_loop: str = "flat",
        hrm_l_layers: int = 4,
        hrm_h_layers: int = 4,
        hrm_history_attention: bool = False,
        recurrent_sandwich_norm: bool = False,
        recurrent_history_attention: bool = False,
        recurrent_adapter: bool = True,
        recurrent_adapter_type: str = "mlp",
        recurrent_adapter_intermediate_size: int | None = None,
        recurrent_adapter_activation: str = "gelu",
        recurrent_adapter_bias: bool = False,
        eval_recurrence: int = 32,
        mean_backprop_depth: int = 8,
        boundary_num_experts: int | None = None,
        boundary_num_experts_per_tok: int | None = None,
        boundary_moe_intermediate_size: int | None = None,
        global_router_update: bool = False,
        **kwargs,
    ):
        total_layers = int(prelude_layers) + int(recurrent_layers) + int(coda_layers)
        kwargs["num_hidden_layers"] = int(kwargs.get("num_hidden_layers", total_layers))
        super().__init__(**kwargs)
        self.prelude_layers = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers = int(coda_layers)
        if self.prelude_layers < 0 or self.recurrent_layers <= 0 or self.coda_layers < 0:
            raise ValueError(
                "prelude_layers/coda_layers must be >= 0 and recurrent_layers must be > 0"
            )
        if recurrent_expert_pool not in {"per_layer", "global"}:
            raise ValueError(
                "recurrent_expert_pool must be 'per_layer' or 'global', "
                f"got {recurrent_expert_pool!r}"
            )
        self.recurrent_expert_pool = recurrent_expert_pool
        self.core_attention = bool(core_attention)
        if core_attention_layers is None:
            self.core_attention_layers = None
        else:
            layers = tuple(int(idx) for idx in core_attention_layers)
            invalid = [idx for idx in layers if idx < 0 or idx >= self.recurrent_layers]
            if invalid:
                raise ValueError(
                    "core_attention_layers entries must be recurrent block indices "
                    f"in [0, {self.recurrent_layers - 1}], got {invalid}"
                )
            self.core_attention_layers = layers
        if recurrent_loop not in {"flat", "hrm"}:
            raise ValueError(
                "recurrent_loop must be 'flat' or 'hrm', "
                f"got {recurrent_loop!r}"
            )
        self.recurrent_loop = str(recurrent_loop)
        self.hrm_l_layers = int(hrm_l_layers)
        self.hrm_h_layers = int(hrm_h_layers)
        if self.recurrent_loop == "hrm":
            if self.hrm_l_layers <= 0 or self.hrm_h_layers <= 0:
                raise ValueError("HRM requires positive hrm_l_layers and hrm_h_layers")
            if self.hrm_l_layers + self.hrm_h_layers != self.recurrent_layers:
                raise ValueError(
                    "HRM expects hrm_l_layers + hrm_h_layers == recurrent_layers, "
                    f"got {self.hrm_l_layers} + {self.hrm_h_layers} != {self.recurrent_layers}"
                )
        self.hrm_history_attention = bool(hrm_history_attention)
        self.recurrent_sandwich_norm = bool(recurrent_sandwich_norm)
        self.recurrent_history_attention = bool(recurrent_history_attention)
        self.recurrent_adapter = bool(recurrent_adapter)
        if recurrent_adapter_type not in {"linear", "mlp"}:
            raise ValueError(
                "recurrent_adapter_type must be 'linear' or 'mlp', "
                f"got {recurrent_adapter_type!r}"
            )
        self.recurrent_adapter_type = recurrent_adapter_type
        self.recurrent_adapter_intermediate_size = (
            int(recurrent_adapter_intermediate_size)
            if recurrent_adapter_intermediate_size is not None
            else int(self.intermediate_size)
        )
        self.recurrent_adapter_activation = str(recurrent_adapter_activation)
        self.recurrent_adapter_bias = bool(recurrent_adapter_bias)
        self.eval_recurrence = int(eval_recurrence)
        self.mean_backprop_depth = int(mean_backprop_depth)
        self.boundary_num_experts = (
            int(boundary_num_experts) if boundary_num_experts is not None else None
        )
        self.boundary_num_experts_per_tok = (
            int(boundary_num_experts_per_tok)
            if boundary_num_experts_per_tok is not None
            else None
        )
        self.boundary_moe_intermediate_size = (
            int(boundary_moe_intermediate_size)
            if boundary_moe_intermediate_size is not None
            else None
        )
        self.global_router_update = bool(global_router_update)


class SharedDeepSeekBiasOwner(nn.Module):
    """Bank-level DeepSeek bias/counter owner for a shared expert pool."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = int(num_experts)
        self.balancing = "deepseek_bias"
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("quantile_ema", torch.zeros(self.num_experts, dtype=torch.float32))
        self.local_quantile_scores: list[torch.Tensor] = []


class RecurrentGlobalDecoderLayer(Qwen3MoeDecoderLayer):
    """Decoder layer whose MoE routes into a shared global expert pool."""

    def __init__(self, config: RecurrentMoEConfig, layer_idx: int, router_class):
        init_cfg = copy.deepcopy(config)
        init_cfg.mlp_only_layers = list(range(max(1, config.num_hidden_layers)))
        super().__init__(init_cfg, layer_idx)
        self.mlp = GlobalSparseMoeBlock(config, router_class=router_class)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        global_experts=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, global_experts)
        hidden_states = residual + hidden_states
        return hidden_states


def _clone_config_with_experts(
    config: RecurrentMoEConfig,
    *,
    num_experts: int | None,
    num_experts_per_tok: int | None,
    moe_intermediate_size: int | None,
) -> RecurrentMoEConfig:
    cloned = copy.deepcopy(config)
    if num_experts is not None:
        cloned.num_experts = int(num_experts)
    if num_experts_per_tok is not None:
        cloned.num_experts_per_tok = int(num_experts_per_tok)
    if moe_intermediate_size is not None:
        cloned.moe_intermediate_size = int(moe_intermediate_size)
    return cloned


def _replace_gate(layer: nn.Module, config: Qwen3MoeConfig, *, router_class) -> None:
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, Qwen3MoeSparseMoeBlock):
        return
    old_gate = mlp.gate
    new_gate = router_class(config)
    new_gate.weight.data.copy_(old_gate.weight.data)
    mlp.gate = new_gate


def _make_standard_layer(config: Qwen3MoeConfig, layer_idx: int, *, router_class) -> Qwen3MoeDecoderLayer:
    layer = Qwen3MoeDecoderLayer(config, layer_idx)
    _replace_gate(layer, config, router_class=router_class)
    return layer


def _attach_shared_bias(router: nn.Module, owner: SharedDeepSeekBiasOwner) -> None:
    """Make a DeepSeekRouter consume and update bank-level bias state."""

    if not hasattr(router, "_buffers"):
        return
    router._buffers["expert_bias"] = owner.expert_bias
    router._buffers["local_tokens_per_expert"] = owner.local_tokens_per_expert
    router._buffers["quantile_ema"] = owner.quantile_ema
    if hasattr(router, "local_quantile_scores"):
        router.local_quantile_scores = owner.local_quantile_scores


def _gate_for(layer: nn.Module) -> nn.Module | None:
    return getattr(getattr(layer, "mlp", None), "gate", None)


class RecurrentAdapterMLP(nn.Module):
    """Adapter MLP from concat recurrent state back to model hidden size."""

    def __init__(self, config: RecurrentMoEConfig):
        super().__init__()
        input_size = config.hidden_size * 2
        intermediate_size = int(getattr(config, "recurrent_adapter_intermediate_size", config.intermediate_size))
        bias = bool(getattr(config, "recurrent_adapter_bias", False))
        activation = str(getattr(config, "recurrent_adapter_activation", "gelu"))
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=bias)
        self.act_fn = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class HHistoryAttention(nn.Module):
    """Content-only attention over stored H states before the coda."""

    def __init__(self, config: RecurrentMoEConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(
                "H-history attention requires hidden_size divisible by "
                f"num_attention_heads, got {hidden_size} and {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_norm = Qwen3MoeRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.kv_norm = Qwen3MoeRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        h_history: list[torch.Tensor],
    ) -> torch.Tensor:
        if not h_history:
            return hidden_states

        B, T, C = hidden_states.shape
        history = torch.stack(
            [
                state.to(device=hidden_states.device, dtype=hidden_states.dtype)
                for state in h_history
            ],
            dim=2,
        )
        num_states = history.shape[2]

        query = self.q_proj(self.q_norm(hidden_states))
        kv = self.kv_norm(history)
        key = self.k_proj(kv)
        value = self.v_proj(history)

        query = query.view(B * T, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B * T, num_states, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B * T, num_states, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = torch.softmax(attn.float(), dim=-1).to(dtype=query.dtype)
        context = torch.matmul(weights, value).squeeze(2)
        context = context.contiguous().view(B, T, C)
        return hidden_states + self.o_proj(context)


class RecurrentMoEModel(nn.Module):
    """Backbone for recurrent standard/global MoE models."""

    def __init__(self, config: RecurrentMoEConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if bool(getattr(config, "recurrent_adapter", True)):
            adapter_type = getattr(config, "recurrent_adapter_type", "mlp")
            self.recurrent_adapter = (
                RecurrentAdapterMLP(config)
                if adapter_type == "mlp"
                else nn.Linear(
                    config.hidden_size * 2,
                    config.hidden_size,
                    bias=bool(getattr(config, "recurrent_adapter_bias", False)),
                )
            )
        else:
            self.recurrent_adapter = None
        self.h_history_attention = (
            HHistoryAttention(config)
            if (
                getattr(config, "recurrent_loop", "flat") == "hrm"
                and bool(getattr(config, "hrm_history_attention", False))
            )
            else None
        )
        self.flat_history_attention = (
            HHistoryAttention(config)
            if (
                getattr(config, "recurrent_loop", "flat") == "flat"
                and bool(getattr(config, "recurrent_history_attention", False))
            )
            else None
        )
        if bool(getattr(config, "recurrent_sandwich_norm", False)):
            self.recurrent_post_attention_norms = nn.ModuleList(
                [
                    Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(config.recurrent_layers)
                ]
            )
            self.recurrent_post_mlp_norms = nn.ModuleList(
                [
                    Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(config.recurrent_layers)
                ]
            )
        else:
            self.recurrent_post_attention_norms = nn.ModuleList()
            self.recurrent_post_mlp_norms = nn.ModuleList()
        self.gradient_checkpointing = False

        router_class = DeepSeekRouter if getattr(config, "use_deepseek_routing", False) else ExplorationTopKRouter
        boundary_config = _clone_config_with_experts(
            config,
            num_experts=config.boundary_num_experts,
            num_experts_per_tok=config.boundary_num_experts_per_tok,
            moe_intermediate_size=config.boundary_moe_intermediate_size,
        )

        self.prelude_blocks = nn.ModuleList(
            _make_standard_layer(boundary_config, i, router_class=router_class)
            for i in range(config.prelude_layers)
        )

        self.global_experts = None
        self.shared_recurrent_bias = None
        if config.recurrent_expert_pool == "global":
            from src.models.modeling_qwen3_moe import Qwen3MoeExperts

            self.global_experts = Qwen3MoeExperts(config)
            self.shared_recurrent_bias = SharedDeepSeekBiasOwner(config.num_experts)
            self.recurrent_blocks = nn.ModuleList(
                RecurrentGlobalDecoderLayer(config, config.prelude_layers + i, router_class)
                for i in range(config.recurrent_layers)
            )
            for block in self.recurrent_blocks:
                gate = _gate_for(block)
                if gate is not None and self.shared_recurrent_bias is not None:
                    _attach_shared_bias(gate, self.shared_recurrent_bias)
        else:
            self.recurrent_blocks = nn.ModuleList(
                _make_standard_layer(
                    config,
                    config.prelude_layers + i,
                    router_class=router_class,
                )
                for i in range(config.recurrent_layers)
            )

        coda_offset = config.prelude_layers + config.recurrent_layers
        self.coda_blocks = nn.ModuleList(
            _make_standard_layer(boundary_config, coda_offset + i, router_class=router_class)
            for i in range(config.coda_layers)
        )

        self._all_mlp_router_logits: list[torch.Tensor] = []
        self._all_mlp_selected_experts: list[torch.Tensor] = []
        self._all_mlp_token_masks: list[None] = []
        self._all_mlp_router_scopes: list[dict[str, Any]] = []
        self._last_num_steps_no_grad = 0
        self._last_num_steps_with_grad = 0
        self._last_hrm_h_cycles = 0
        self._last_hrm_l_cycles = 0
        self._recurrent_loop_diagnostics: list[dict[str, Any]] = []
        self._recurrent_layer_diagnostics: list[dict[str, Any]] = []

    def _reattach_shared_recurrent_bias(self) -> None:
        if self.shared_recurrent_bias is None:
            return
        for block in self.recurrent_blocks:
            gate = _gate_for(block)
            if gate is not None:
                _attach_shared_bias(gate, self.shared_recurrent_bias)

    def _apply(self, fn):
        result = super()._apply(fn)
        # Module-wide moves/casts clone buffers independently and can break
        # the shared owner <-> recurrent gate aliases. Restore the aliases so
        # training updates the same bias/counter tensor that routing uses.
        self._reattach_shared_recurrent_bias()
        return result

    @staticmethod
    def _checkpoint_context_fn():
        return checkpoint_recompute_context(False), checkpoint_recompute_context(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None):
        import functools
        from torch.utils.checkpoint import checkpoint

        kwargs = {"use_reentrant": False, "context_fn": self._checkpoint_context_fn}
        if gradient_checkpointing_kwargs:
            kwargs.update(gradient_checkpointing_kwargs)
        checkpointing_func = functools.partial(checkpoint, **kwargs)
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = checkpointing_func
                module.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

    def _resolve_num_steps(self, num_steps: torch.Tensor | tuple[int, int] | list[int] | int | None) -> tuple[int, int]:
        if num_steps is None:
            if self.training:
                return 0, max(1, int(getattr(self.config, "eval_recurrence", 1)))
            return int(getattr(self.config, "eval_recurrence", 1)), 0
        if isinstance(num_steps, torch.Tensor):
            flat = num_steps.detach().view(-1).to("cpu")
            if flat.numel() == 1:
                return 0, int(flat[0].item())
            return int(flat[0].item()), int(flat[1].item())
        if isinstance(num_steps, int):
            return 0, int(num_steps)
        if len(num_steps) == 1:
            return 0, int(num_steps[0])
        return int(num_steps[0]), int(num_steps[1])

    def _resolve_hrm_steps(
        self,
        num_steps: torch.Tensor | tuple[int, ...] | list[int] | int | None,
    ) -> tuple[int, int, tuple[int, ...]]:
        if num_steps is None:
            h_no_grad, h_with_grad = self._resolve_num_steps(num_steps)
            h_total = max(0, h_no_grad + h_with_grad)
            return h_no_grad, h_with_grad, tuple(1 for _ in range(h_total))
        if isinstance(num_steps, torch.Tensor):
            values = [int(v) for v in num_steps.detach().view(-1).to("cpu").tolist()]
        elif isinstance(num_steps, int):
            values = [0, int(num_steps)]
        else:
            values = [int(v) for v in num_steps]

        if len(values) == 0:
            return 0, max(1, int(getattr(self.config, "eval_recurrence", 1))), tuple()
        if len(values) == 1:
            h_no_grad, h_with_grad = 0, max(0, int(values[0]))
            return h_no_grad, h_with_grad, tuple(1 for _ in range(h_with_grad))

        h_no_grad = max(0, int(values[0]))
        h_with_grad = max(0, int(values[1]))
        h_total = h_no_grad + h_with_grad
        l_counts = [max(1, int(v)) for v in values[2:2 + h_total]]
        if len(l_counts) < h_total:
            l_counts.extend([1] * (h_total - len(l_counts)))
        return h_no_grad, h_with_grad, tuple(l_counts[:h_total])

    def _record_router(
        self,
        layer: nn.Module,
        *,
        collect_logits: bool,
        pool: str,
        block_index: int | None,
        scope_extra: dict[str, Any] | None = None,
    ) -> None:
        gate = _gate_for(layer)
        if gate is None:
            return
        selected = getattr(gate, "_last_top_k_idx", None)
        if selected is not None:
            self._all_mlp_selected_experts.append(selected.detach())
            self._all_mlp_token_masks.append(None)
            scope = {
                "pool": pool,
                "block_index": block_index,
                "num_experts": int(getattr(gate, "num_experts", 0) or 0),
            }
            if scope_extra:
                scope.update(scope_extra)
            self._all_mlp_router_scopes.append(scope)
        if collect_logits:
            scores = getattr(gate, "_last_router_scores_detached", None)
            if scores is not None:
                # Aux methods need graph-bearing router scores. The new
                # recurrent configs use deepseek_bias, so detached telemetry is
                # sufficient for the current study.
                self._all_mlp_router_logits.append(scores.detach())

    def _accumulate_no_grad_counts(self, layer: nn.Module) -> None:
        if not self.training:
            return
        gate = _gate_for(layer)
        if gate is None or not hasattr(gate, "local_tokens_per_expert"):
            return
        selected = getattr(gate, "_last_top_k_idx", None)
        if selected is None:
            return
        with torch.no_grad():
            counts = torch.bincount(
                selected.reshape(-1),
                minlength=int(getattr(gate, "num_experts", selected.max().item() + 1)),
            ).to(device=gate.local_tokens_per_expert.device, dtype=torch.float32)
            gate.local_tokens_per_expert += counts

    def _inject_recurrent_input(
        self,
        hidden_states: torch.Tensor,
        recurrent_input: torch.Tensor,
    ) -> torch.Tensor:
        if self.recurrent_adapter is None:
            return hidden_states
        return self.recurrent_adapter(
            torch.cat(
                [hidden_states, recurrent_input.to(device=hidden_states.device)],
                dim=-1,
            )
        )

    def _recurrent_block_uses_attention(self, block_idx: int) -> bool:
        layers = getattr(self.config, "core_attention_layers", None)
        if layers is not None:
            return int(block_idx) in layers
        return bool(self.config.core_attention)

    def _recurrent_sandwich_index(self, *, pool: str, block_index: int | None) -> int | None:
        if (
            not bool(getattr(self.config, "recurrent_sandwich_norm", False))
            or pool != "mlp_recurrent"
            or block_index is None
        ):
            return None
        local_idx = int(block_index) - int(self.config.prelude_layers)
        if local_idx < 0 or local_idx >= len(self.recurrent_post_mlp_norms):
            return None
        return local_idx

    def _run_sandwich_recurrent_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        local_idx: int,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        use_attention: bool,
    ) -> torch.Tensor:
        if use_attention:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = self.recurrent_post_attention_norms[local_idx](
                residual + hidden_states
            )

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        if isinstance(layer, RecurrentGlobalDecoderLayer):
            hidden_states = layer.mlp(hidden_states, self.global_experts)
        else:
            hidden_states = layer.mlp(hidden_states)
        return self.recurrent_post_mlp_norms[local_idx](residual + hidden_states)

    def _recurrent_residual_metrics(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            prev = before.detach().float()
            curr = after.detach().float()
            delta = curr - prev
            eps = 1e-8
            residual_rms = delta.pow(2).mean().sqrt()
            state_rms = prev.pow(2).mean().sqrt().clamp_min(eps)
            cosine = F.cosine_similarity(
                prev.reshape(-1, prev.shape[-1]),
                curr.reshape(-1, curr.shape[-1]),
                dim=-1,
                eps=eps,
            ).mean()

            prev_last = prev[:, -1, :]
            curr_last = curr[:, -1, :]
            delta_last = curr_last - prev_last
            last_residual_rms = delta_last.pow(2).mean().sqrt()
            last_state_rms = prev_last.pow(2).mean().sqrt().clamp_min(eps)
            last_cosine = F.cosine_similarity(
                prev_last,
                curr_last,
                dim=-1,
                eps=eps,
            ).mean()

        return {
            "residual_rms": residual_rms.detach(),
            "relative_residual_rms": (residual_rms / state_rms).detach(),
            "cosine": cosine.detach(),
            "last_token_residual_rms": last_residual_rms.detach(),
            "last_token_relative_residual_rms": (last_residual_rms / last_state_rms).detach(),
            "last_token_cosine": last_cosine.detach(),
        }

    def _record_recurrent_loop_diagnostics(
        self,
        *,
        loop_index: int,
        phase: str,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> None:
        row = {
            "loop": int(loop_index),
            "phase": str(phase),
        }
        row.update(self._recurrent_residual_metrics(before, after))
        self._recurrent_loop_diagnostics.append(row)

    def _record_recurrent_layer_diagnostics(
        self,
        *,
        phase: str,
        before: torch.Tensor,
        after: torch.Tensor,
        scope: dict[str, Any],
    ) -> None:
        row = {
            "computation_index": len(self._recurrent_layer_diagnostics),
            "phase": str(phase),
            **scope,
        }
        row.update(self._recurrent_residual_metrics(before, after))
        self._recurrent_layer_diagnostics.append(row)

    def _run_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        use_attention: bool,
        collect_logits: bool,
        pool: str,
        block_index: int | None,
        manual_count_no_grad: bool = False,
        scope_extra: dict[str, Any] | None = None,
        collect_recurrence_diagnostics: bool = False,
    ) -> torch.Tensor:
        diagnostic_scope = {
            "pool": pool,
            "block_index": block_index,
        }
        if scope_extra:
            diagnostic_scope.update(scope_extra)
        diagnostic_before = (
            hidden_states.detach()
            if collect_recurrence_diagnostics and pool == "mlp_recurrent"
            else None
        )
        sandwich_idx = self._recurrent_sandwich_index(
            pool=pool,
            block_index=block_index,
        )
        if sandwich_idx is not None:
            def sandwich_forward(x: torch.Tensor) -> torch.Tensor:
                return self._run_sandwich_recurrent_layer(
                    layer,
                    x,
                    local_idx=sandwich_idx,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    use_attention=use_attention,
                )

            if (
                getattr(layer, "gradient_checkpointing", False)
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden_states = layer._gradient_checkpointing_func(
                    sandwich_forward,
                    hidden_states,
                )
            else:
                hidden_states = sandwich_forward(hidden_states)
        elif use_attention:
            if isinstance(layer, RecurrentGlobalDecoderLayer):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    global_experts=self.global_experts,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
        else:
            def no_attention_forward(x: torch.Tensor) -> torch.Tensor:
                residual = x
                y = layer.post_attention_layernorm(x)
                if isinstance(layer, RecurrentGlobalDecoderLayer):
                    y = layer.mlp(y, self.global_experts)
                else:
                    y = layer.mlp(y)
                return residual + y

            if (
                getattr(layer, "gradient_checkpointing", False)
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden_states = layer._gradient_checkpointing_func(
                    no_attention_forward,
                    hidden_states,
                )
            else:
                hidden_states = no_attention_forward(hidden_states)

        if diagnostic_before is not None:
            self._record_recurrent_layer_diagnostics(
                phase="no_grad" if manual_count_no_grad else "grad",
                before=diagnostic_before,
                after=hidden_states,
                scope=diagnostic_scope,
            )
        if manual_count_no_grad:
            self._accumulate_no_grad_counts(layer)
        self._record_router(
            layer,
            collect_logits=collect_logits,
            pool=pool,
            block_index=block_index,
            scope_extra=scope_extra,
        )
        return hidden_states

    def _run_hrm_module(
        self,
        blocks: list[tuple[int, nn.Module]],
        hidden_states: torch.Tensor,
        *,
        module_name: str,
        cycle_index: int,
        l_index: int | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        collect_logits: bool,
        manual_count_no_grad: bool = False,
        collect_recurrence_diagnostics: bool = False,
    ) -> torch.Tensor:
        for local_idx, (block_idx, block) in enumerate(blocks):
            hidden_states = self._run_layer(
                block,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_attention=self._recurrent_block_uses_attention(block_idx),
                collect_logits=collect_logits,
                pool="mlp_recurrent",
                block_index=self.config.prelude_layers + block_idx,
                manual_count_no_grad=manual_count_no_grad,
                collect_recurrence_diagnostics=collect_recurrence_diagnostics,
                scope_extra={
                    "hrm_module": module_name,
                    "hrm_cycle": int(cycle_index),
                    "hrm_l_index": "" if l_index is None else int(l_index),
                    "hrm_local_block": int(local_idx),
                },
            )
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        num_steps: torch.Tensor | tuple[int, int] | list[int] | int | None = None,
        output_router_logits: bool = False,
        collect_recurrence_diagnostics: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, T = inputs_embeds.shape[:2]
        cache_position = torch.arange(T, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = (
            create_causal_mask
            if self.config.sliding_window is None
            else create_sliding_window_causal_mask
        )
        causal_mask = call_mask_function(
            mask_function,
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        self._all_mlp_router_logits = []
        self._all_mlp_selected_experts = []
        self._all_mlp_token_masks = []
        self._all_mlp_router_scopes = []
        self._recurrent_loop_diagnostics = []
        self._recurrent_layer_diagnostics = []

        hidden_states = inputs_embeds
        for block_idx, block in enumerate(self.prelude_blocks):
            hidden_states = self._run_layer(
                block,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_attention=True,
                collect_logits=output_router_logits,
                pool="mlp_boundary",
                block_index=block_idx,
            )

        recurrent_input = hidden_states
        n_no_grad, n_with_grad = self._resolve_num_steps(num_steps)
        self._last_num_steps_no_grad = n_no_grad
        self._last_num_steps_with_grad = n_with_grad
        self._last_hrm_h_cycles = 0
        self._last_hrm_l_cycles = 0
        loop_index = 0

        if getattr(self.config, "recurrent_loop", "flat") == "hrm":
            h_no_grad, h_with_grad, l_counts = self._resolve_hrm_steps(num_steps)
            self._last_num_steps_no_grad = h_no_grad
            self._last_num_steps_with_grad = h_with_grad
            self._last_hrm_h_cycles = h_no_grad + h_with_grad
            self._last_hrm_l_cycles = int(sum(l_counts))
            l_span = int(getattr(self.config, "hrm_l_layers", 0))
            l_blocks = list(enumerate(self.recurrent_blocks[:l_span]))
            h_blocks = list(enumerate(self.recurrent_blocks[l_span:], start=l_span))
            h_history: list[torch.Tensor] = []

            def run_hrm_cycle(
                *,
                cycle_idx: int,
                l_count: int,
                manual_count_no_grad: bool,
            ) -> torch.Tensor:
                nonlocal hidden_states
                before_cycle = hidden_states.detach() if collect_recurrence_diagnostics else None
                hidden_states = self._inject_recurrent_input(hidden_states, recurrent_input)
                for l_idx in range(max(1, int(l_count))):
                    hidden_states = self._run_hrm_module(
                        l_blocks,
                        hidden_states,
                        module_name="L",
                        cycle_index=cycle_idx,
                        l_index=l_idx,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        collect_logits=output_router_logits,
                        manual_count_no_grad=manual_count_no_grad,
                        collect_recurrence_diagnostics=collect_recurrence_diagnostics,
                    )
                hidden_states = self._run_hrm_module(
                    h_blocks,
                    hidden_states,
                    module_name="H",
                    cycle_index=cycle_idx,
                    l_index=None,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    collect_logits=output_router_logits,
                    manual_count_no_grad=manual_count_no_grad,
                    collect_recurrence_diagnostics=collect_recurrence_diagnostics,
                )
                h_history.append(
                    hidden_states.detach() if not torch.is_grad_enabled() else hidden_states
                )
                if before_cycle is not None:
                    self._record_recurrent_loop_diagnostics(
                        loop_index=cycle_idx,
                        phase="no_grad" if manual_count_no_grad else "grad",
                        before=before_cycle,
                        after=hidden_states,
                    )
                return hidden_states

            if h_no_grad > 0:
                with torch.no_grad():
                    for cycle_offset in range(h_no_grad):
                        run_hrm_cycle(
                            cycle_idx=cycle_offset,
                            l_count=l_counts[cycle_offset],
                            manual_count_no_grad=True,
                        )
                hidden_states = hidden_states.detach()
                h_history = [state.detach() for state in h_history]

            for grad_offset in range(h_with_grad):
                cycle_idx = h_no_grad + grad_offset
                run_hrm_cycle(
                    cycle_idx=cycle_idx,
                    l_count=l_counts[cycle_idx],
                    manual_count_no_grad=False,
                )

            if self.h_history_attention is not None and h_history:
                hidden_states = self.h_history_attention(hidden_states, h_history)

            coda_offset = self.config.prelude_layers + self.config.recurrent_layers
            for coda_idx, block in enumerate(self.coda_blocks):
                hidden_states = self._run_layer(
                    block,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    use_attention=True,
                    collect_logits=output_router_logits,
                    pool="mlp_boundary",
                    block_index=coda_offset + coda_idx,
                )

            hidden_states = self.norm(hidden_states)
            router_logits = tuple(self._all_mlp_router_logits) if output_router_logits else None
            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
                router_logits=router_logits,
            )

        flat_history: list[torch.Tensor] = []
        if n_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_no_grad):
                    before_loop = hidden_states.detach() if collect_recurrence_diagnostics else None
                    hidden_states = self._inject_recurrent_input(hidden_states, recurrent_input)
                    for block_idx, block in enumerate(self.recurrent_blocks):
                        hidden_states = self._run_layer(
                            block,
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            use_attention=self._recurrent_block_uses_attention(block_idx),
                            collect_logits=output_router_logits,
                            pool="mlp_recurrent",
                            block_index=self.config.prelude_layers + block_idx,
                            manual_count_no_grad=True,
                            collect_recurrence_diagnostics=collect_recurrence_diagnostics,
                            scope_extra={"loop": int(loop_index)},
                        )
                    if before_loop is not None:
                        self._record_recurrent_loop_diagnostics(
                            loop_index=loop_index,
                            phase="no_grad",
                            before=before_loop,
                            after=hidden_states,
                        )
                    if self.flat_history_attention is not None:
                        flat_history.append(hidden_states.detach())
                    loop_index += 1
            hidden_states = hidden_states.detach()

        for _ in range(n_with_grad):
            before_loop = hidden_states.detach() if collect_recurrence_diagnostics else None
            hidden_states = self._inject_recurrent_input(hidden_states, recurrent_input)
            for block_idx, block in enumerate(self.recurrent_blocks):
                hidden_states = self._run_layer(
                    block,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    use_attention=self._recurrent_block_uses_attention(block_idx),
                    collect_logits=output_router_logits,
                    pool="mlp_recurrent",
                    block_index=self.config.prelude_layers + block_idx,
                    collect_recurrence_diagnostics=collect_recurrence_diagnostics,
                    scope_extra={"loop": int(loop_index)},
                )
            if before_loop is not None:
                self._record_recurrent_loop_diagnostics(
                    loop_index=loop_index,
                    phase="grad",
                    before=before_loop,
                    after=hidden_states,
                )
            if self.flat_history_attention is not None:
                flat_history.append(hidden_states)
            loop_index += 1

        if self.flat_history_attention is not None and flat_history:
            hidden_states = self.flat_history_attention(hidden_states, flat_history)

        coda_offset = self.config.prelude_layers + self.config.recurrent_layers
        for coda_idx, block in enumerate(self.coda_blocks):
            hidden_states = self._run_layer(
                block,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_attention=True,
                collect_logits=output_router_logits,
                pool="mlp_boundary",
                block_index=coda_offset + coda_idx,
            )

        hidden_states = self.norm(hidden_states)
        router_logits = tuple(self._all_mlp_router_logits) if output_router_logits else None
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            router_logits=router_logits,
        )


class RecurrentMoEForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: RecurrentMoEConfig):
        super().__init__(config)
        self.model = RecurrentMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self._seq_aux_loss_coef = getattr(config, "seq_aux_loss_coef", 0.0)
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def get_all_balancing_owners(self):
        inner = self.model
        for blocks in (inner.prelude_blocks, inner.coda_blocks):
            for block in blocks:
                gate = _gate_for(block)
                if gate is not None and hasattr(gate, "expert_bias") and hasattr(gate, "local_tokens_per_expert"):
                    yield gate, "mlp"
        if inner.shared_recurrent_bias is not None:
            yield inner.shared_recurrent_bias, "mlp"
        else:
            for block in inner.recurrent_blocks:
                gate = _gate_for(block)
                if gate is not None and hasattr(gate, "expert_bias") and hasattr(gate, "local_tokens_per_expert"):
                    yield gate, "mlp"

    def _compute_lm_ce_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        if not getattr(self.config, "use_fused_linear_ce", True):
            logits = self.lm_head(hidden_states)
            return self.loss_function(logits, labels, self.vocab_size, **kwargs)

        ignore_index = int(kwargs.get("ignore_index", -100))
        label_smoothing = float(kwargs.get("label_smoothing", 0.0) or 0.0)
        if hidden_states.is_cuda:
            try:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            except ImportError as exc:
                raise RuntimeError(
                    "Recurrent MoE uses Liger fused linear CE by default on CUDA. "
                    "Install liger-kernel in the training image."
                ) from exc
            flat_hidden = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
            shifted_labels = F.pad(labels, (0, 1), value=ignore_index)
            shifted_labels = shifted_labels[..., 1:].contiguous().view(-1)
            loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
                reduction="mean",
            )
            return loss_fn(self.lm_head.weight, flat_hidden, shifted_labels)

        logits = self.lm_head(hidden_states)
        return self.loss_function(logits, labels, self.vocab_size, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        num_steps: torch.Tensor | tuple[int, int] | list[int] | int | None = None,
        return_logits: bool | None = None,
        collect_recurrence_diagnostics: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        del past_key_values, use_cache, cache_position
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else bool(getattr(self.config, "output_router_logits", False))
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            num_steps=num_steps,
            output_router_logits=output_router_logits,
            collect_recurrence_diagnostics=collect_recurrence_diagnostics,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        use_fused_linear_ce = bool(getattr(self.config, "use_fused_linear_ce", True))
        if return_logits is None:
            return_logits = labels is None or not use_fused_linear_ce
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) if return_logits else None

        loss = None
        ce_loss = None
        if labels is not None:
            if use_fused_linear_ce:
                ce_loss = self._compute_lm_ce_loss(hidden_states, labels, **kwargs)
            else:
                ce_loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
            loss = ce_loss

        aux_loss = None
        if output_router_logits and outputs.router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
            )
            method = getattr(self, "_load_balancing_method", None)
            mlp_method = getattr(self.config, "mlp_router_balancing", None)
            aux_active = (mlp_method == "aux_loss") if mlp_method is not None else method in (None, "aux_loss")
            if aux_active and loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
            elif aux_loss is not None:
                aux_loss = aux_loss.detach()

        output = MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            router_logits=outputs.router_logits,
        )
        output.ce_loss = ce_loss
        output.selected_experts = tuple(self.model._all_mlp_selected_experts) or None
        output.router_token_masks = tuple(self.model._all_mlp_token_masks) or None
        output.num_steps_no_grad = self.model._last_num_steps_no_grad
        output.num_steps_with_grad = self.model._last_num_steps_with_grad
        output.hrm_h_cycles = self.model._last_hrm_h_cycles
        output.hrm_l_cycles = self.model._last_hrm_l_cycles
        return output


__all__ = [
    "RecurrentMoEConfig",
    "RecurrentMoEModel",
    "RecurrentMoEForCausalLM",
    "SharedDeepSeekBiasOwner",
]
