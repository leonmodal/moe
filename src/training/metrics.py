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


def get_output_router_token_masks(output, model=None) -> tuple[torch.Tensor | None, ...] | None:
    masks = getattr(output, "router_token_masks", None)
    if masks:
        return tuple(masks)
    # the DETACH-ONLY policy fallback: when the model's forward skipped writing router_token_masks
    # to the output (non-aux methods), the moe_everything inner model still
    # accumulates per-depth token masks under `_all_mlp_token_masks`. Use them
    # so non-aux MoE-Everything metrics weight expert load by active tokens
    # rather than over the zero-padded full sequence.
    if model is not None:
        inner = getattr(model, "model", None)
        inner_masks = getattr(inner, "_all_mlp_token_masks", None)
        if inner_masks:
            return tuple(inner_masks)
    return None


def _collect_detached_router_scores(model) -> tuple[torch.Tensor, ...] | None:
    """DETACH-ONLY telemetry consumer (MLP-only).

    When the model's `forward` skips returning gradient-bearing
    `router_logits` (non-aux methods), telemetry consumers can still
    observe per-expert scores by reading internal model state. This
    helper returns a tuple of per-depth MLP router-score tensors with
    the same shape `[total_tokens, num_experts]` that
    `output.router_logits` would have had for aux methods.

    Resolution order (first hit wins):

    1. `model.model._all_mlp_router_logits` (`moe_everything` inner
       accumulator) — densified per-depth tensors that match the
       full-sequence shape expected by `normalized_load_balancing_loss_func`.
       These are grad-bearing internally; we detach them here.
    2. `get_all_balancing_owners()` walker filtered to MLP routers
       (`standard_moe` / `global_moe`) — per-layer gate's
       `_last_router_scores_detached` snapshot.
    3. Class-based scan over `DeepSeekRouter` / `ExplorationTopKRouter`
       (older or dense models without the canonical interfaces).

    Attention routers (`q`/`k`/`v`/`o`) and branch router(s) are
    intentionally excluded — their expert dimensions differ from MLP
    `num_experts` and would crash `normalized_load_balancing_loss_func`
    if mixed into a single tuple. Attention diagnostics, when needed,
    must be computed separately with their own `num_attn_experts` /
    per-head top-k.

    Returns `None` when no MLP router has produced a snapshot.
    """
    # Path 1: moe_everything's inner accumulator. The mlp_bank stores the
    # densified per-depth router logits in `last_router_logits` and the
    # outer `MoEverythingModel` accumulates them into
    # `_all_mlp_router_logits`. These have shape `[total_tokens,
    # num_experts]` — exactly the shape `compute_output_metrics` expects.
    inner = getattr(model, "model", None)
    inner_logits = getattr(inner, "_all_mlp_router_logits", None)
    if inner_logits:
        return tuple(t.detach() for t in inner_logits)

    # Path 2: walker-based collection for `standard_moe` / `global_moe`.
    # Each layer's gate populates `_last_router_scores_detached` on every
    # forward; for these families every token routes through the gate so
    # the snapshot shape is `[total_tokens, num_experts]`.
    walker = getattr(model, "get_all_balancing_owners", None)
    snapshots: list[torch.Tensor] = []
    if walker is not None:
        for owner, label in walker():
            if label != "mlp":
                continue
            snap = getattr(owner, "_last_router_scores_detached", None)
            if snap is not None:
                snapshots.append(snap)
        if snapshots:
            return tuple(snapshots)

    # Path 3: legacy class-based fallback.
    from src.models.router import DeepSeekRouter, ExplorationTopKRouter
    for module in model.modules():
        if isinstance(module, (DeepSeekRouter, ExplorationTopKRouter)):
            snap = getattr(module, "_last_router_scores_detached", None)
            if snap is not None:
                snapshots.append(snap)
    return tuple(snapshots) if snapshots else None


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
    router_token_masks = get_output_router_token_masks(output, raw_model)
    selected_experts = get_output_selected_experts(output, raw_model)

    aux = getattr(output, "aux_loss", None)
    aux_normalized = None
    # DETACH-ONLY: when the model output's `router_logits` is None
    # (non-aux method that skips the gradient-bearing path), fall back to
    # the per-router `_last_router_scores_detached` snapshot. Telemetry
    # remains available without retaining the autograd graph.
    router_logits_for_metrics = getattr(output, "router_logits", None)
    if router_logits_for_metrics is None:
        router_logits_for_metrics = _collect_detached_router_scores(raw_model)
    if router_logits_for_metrics is not None:
        aux_normalized = normalized_load_balancing_loss_func(
            router_logits_for_metrics,
            model_cfg.num_experts,
            model_cfg.num_experts_per_tok,
            token_masks=router_token_masks,
            selected_experts=selected_experts,
        )
    ce_tensor = getattr(output, "ce_loss", None)
    seq_aux = getattr(output, "seq_aux_loss", None)
    if seq_aux is None and seq_aux_loss_coef > 0 and router_logits_for_metrics is not None:
        # the DETACH-ONLY policy: same detached fallback for seq aux telemetry.
        seq_aux = seq_load_balancing_loss_func(
            router_logits_for_metrics,
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
