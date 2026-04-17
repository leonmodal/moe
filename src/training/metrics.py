"""Output metrics computation shared between training and eval."""

from __future__ import annotations

import logging

import torch

from src.models.load_balancing import (
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
)

logger = logging.getLogger(__name__)

# Track whether we've already warned about a structural lookup miss, so the
# telemetry drift surfaces once rather than drowning stdout on every step.
_WARNED_MISSING_EXPERT_INDICES = False


def get_selected_experts_for_seq_aux(model) -> tuple[torch.Tensor, ...] | None:
    """Extract per-layer selected expert indices from model internals.

    Falls back through two layouts:
      1. `model.model.layers[*].mlp.gate._last_top_k_idx` (standard HF stack).
      2. `model.model._all_mlp_selected_experts` (MoE-Everything aggregate).

    Returns `None` when neither path yields indices — typically a dense model
    or a model whose routers haven't run yet. Only `AttributeError` /
    `TypeError` are caught: a broader `except Exception` would silently mask
    genuine bugs (dtype mismatches, CUDA errors, router refactors) that
    should bubble up. Exception-drop logs once so a refactor-induced
    regression doesn't zero-out `seq_aux_loss` telemetry without notice.
    """
    global _WARNED_MISSING_EXPERT_INDICES
    try:
        inner_model = getattr(model, "model", None)
        layers = getattr(inner_model, "layers", None)
        if layers is not None:
            selected = []
            for layer in layers:
                gate = getattr(getattr(layer, "mlp", None), "gate", None)
                idx = getattr(gate, "_last_top_k_idx", None)
                if idx is None:
                    return None
                selected.append(idx)
            return tuple(selected) if selected else None

        selected = getattr(inner_model, "_all_mlp_selected_experts", None)
        if selected:
            return tuple(selected)
        return None
    except (AttributeError, TypeError) as exc:
        if not _WARNED_MISSING_EXPERT_INDICES:
            logger.warning(
                "get_selected_experts_for_seq_aux: expected structure missing "
                "(%s). `seq_aux_loss` will fall back to recomputing from "
                "router_logits; refactors that move `layer.mlp.gate` or "
                "`_all_mlp_selected_experts` will silently degrade telemetry "
                "until this is fixed.",
                exc,
            )
            _WARNED_MISSING_EXPERT_INDICES = True
        return None


def get_output_selected_experts(output, model) -> tuple[torch.Tensor, ...] | None:
    selected = getattr(output, "selected_experts", None)
    if selected:
        return tuple(selected)
    return get_selected_experts_for_seq_aux(model)


def get_output_router_token_masks(output) -> tuple[torch.Tensor | None, ...] | None:
    masks = getattr(output, "router_token_masks", None)
    if masks:
        return tuple(masks)
    return None


def compute_output_metrics(
    output,
    raw_model,
    model_cfg,
    input_ids: torch.Tensor,
    *,
    seq_aux_loss_coef: float,
) -> tuple[dict[str, float], tuple[torch.Tensor, ...] | None, tuple[torch.Tensor | None, ...] | None]:
    """Compute training/eval metrics from model output.

    Returns (metrics_dict, selected_experts, router_token_masks).
    """
    router_token_masks = get_output_router_token_masks(output)
    selected_experts = get_output_selected_experts(output, raw_model)

    aux = getattr(output, "aux_loss", None)
    aux_normalized = None
    if getattr(output, "router_logits", None) is not None:
        aux_normalized = normalized_load_balancing_loss_func(
            output.router_logits,
            model_cfg.num_experts,
            model_cfg.num_experts_per_tok,
            token_masks=router_token_masks,
            selected_experts=selected_experts,
        )
    ce_tensor = getattr(output, "ce_loss", None)
    seq_aux = getattr(output, "seq_aux_loss", None)
    if seq_aux is None and seq_aux_loss_coef > 0 and getattr(output, "router_logits", None) is not None:
        seq_aux = seq_load_balancing_loss_func(
            output.router_logits,
            model_cfg.num_experts,
            model_cfg.num_experts_per_tok,
            batch_size=input_ids.shape[0],
            selected_experts=selected_experts,
            token_masks=router_token_masks,
        )
    branch_aux = getattr(output, "branch_aux_loss", None)
    attention_aux = getattr(output, "attention_aux_loss", None)

    # Batch every tensor we need to read onto the host into a single
    # stacked tensor. Previously each `.item()` call was its own device→host
    # sync; combining them collapses 6–7 per-micro-batch syncs into one
    # `.tolist()`. Non-tensor inputs (None / python floats) resolve
    # without going through the device.
    _tensors: list[torch.Tensor] = []
    _positions: dict[str, int] = {}

    def _maybe_enqueue(name: str, value) -> None:
        if isinstance(value, torch.Tensor):
            _positions[name] = len(_tensors)
            _tensors.append(value.detach().float().reshape(()))

    _maybe_enqueue("total", output.loss)
    _maybe_enqueue("aux", aux)
    _maybe_enqueue("aux_normalized", aux_normalized)
    _maybe_enqueue("ce", ce_tensor)
    _maybe_enqueue("seq_aux", seq_aux)
    _maybe_enqueue("branch_aux", branch_aux)
    _maybe_enqueue("attention_aux", attention_aux)
    if _tensors:
        _stacked = torch.stack(_tensors).tolist()
    else:
        _stacked = []

    def _resolve(name: str, raw, default: float = 0.0) -> float:
        idx = _positions.get(name)
        if idx is not None:
            return _stacked[idx]
        if raw is None:
            return default
        return float(raw)

    total_value = _resolve("total", output.loss)
    aux_value = _resolve("aux", aux)
    aux_normalized_value = _resolve("aux_normalized", aux_normalized)
    seq_aux_value = _resolve("seq_aux", seq_aux)
    branch_aux_value = _resolve("branch_aux", branch_aux)
    attention_aux_value = _resolve("attention_aux", attention_aux)

    if ce_tensor is None:
        # Reconstruct CE from total - Σ(coef · aux). Same contract as before:
        # when the model doesn't expose a direct `ce_loss`, subtract the
        # weighted aux terms off the total loss.
        ce_value = total_value - getattr(raw_model, "router_aux_loss_coef", 0.0) * aux_value
        ce_value -= seq_aux_loss_coef * seq_aux_value
        ce_value -= getattr(raw_model, "branch_router_aux_loss_coef", 0.0) * branch_aux_value
    else:
        ce_value = _resolve("ce", ce_tensor)

    metrics = {
        "loss": total_value,
        "ce_loss": ce_value,
        "aux_loss": aux_value,
        "aux_loss_normalized": aux_normalized_value,
        "seq_aux_loss": seq_aux_value,
        "branch_aux_loss": branch_aux_value,
        "attention_aux_loss": attention_aux_value,
    }
    return metrics, selected_experts, router_token_masks
