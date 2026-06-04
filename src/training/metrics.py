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


def output_get(output, key: str, default=None):
    if isinstance(output, dict):
        return output.get(key, default)
    return getattr(output, key, default)


def output_set(output, key: str, value) -> None:
    if isinstance(output, dict):
        output[key] = value
    else:
        setattr(output, key, value)


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
    selected = output_get(output, "selected_experts", None)
    if selected:
        return tuple(selected)
    return get_selected_experts_for_seq_aux(model)


def get_output_router_token_masks(output, model=None) -> tuple[torch.Tensor | None, ...] | None:
    masks = output_get(output, "router_token_masks", None)
    if masks:
        return tuple(masks)
    # Detach-only fallback: when the model's forward skipped writing router_token_masks
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


def _top_k_for_router_width(model_cfg, num_experts: int) -> int:
    if int(getattr(model_cfg, "boundary_num_experts", -1) or -1) == int(num_experts):
        return int(
            getattr(
                model_cfg,
                "boundary_num_experts_per_tok",
                getattr(model_cfg, "num_experts_per_tok", 2),
            )
        )
    return int(getattr(model_cfg, "num_experts_per_tok", 2))


def _group_router_metric_inputs(
    router_logits: tuple[torch.Tensor, ...] | None,
    selected_experts: tuple[torch.Tensor, ...] | None,
    router_token_masks: tuple[torch.Tensor | None, ...] | None,
    model_cfg,
) -> list[dict[str, object]]:
    """Group router telemetry by expert-pool shape.

    Recurrent global MoE mixes boundary routers (small per-layer pools) with
    recurrent routers (larger shared pool). The load-balancing helpers assume
    a homogeneous expert dimension per call, so metrics split mixed telemetry
    into homogeneous groups and average the resulting per-router losses.
    """
    if router_logits is None or not isinstance(router_logits, tuple):
        return []

    use_selected = selected_experts is not None and len(selected_experts) == len(router_logits)
    use_masks = router_token_masks is not None and len(router_token_masks) == len(router_logits)
    grouped: dict[tuple[int, int], dict[str, object]] = {}

    for idx, scores in enumerate(router_logits):
        if scores is None or scores.ndim != 2 or scores.shape[0] == 0:
            continue
        num_experts = int(scores.shape[-1])
        selected = None
        if use_selected:
            candidate = selected_experts[idx]
            if candidate is not None and candidate.shape[0] == scores.shape[0]:
                selected = candidate

        token_mask = None
        if use_masks:
            candidate_mask = router_token_masks[idx]
            if candidate_mask is not None and candidate_mask.numel() == scores.shape[0]:
                token_mask = candidate_mask

        top_k = int(selected.shape[-1]) if selected is not None else _top_k_for_router_width(model_cfg, num_experts)
        key = (num_experts, top_k)
        group = grouped.setdefault(key, {"logits": [], "selected": [], "masks": []})
        group["logits"].append(scores)
        group["selected"].append(selected)
        group["masks"].append(token_mask)

    return list(grouped.values())


def _weighted_router_metric(groups: list[dict[str, object]], fn, **kwargs):
    total = None
    weight_sum = 0
    for group in groups:
        logits = tuple(group["logits"])
        if not logits:
            continue
        num_experts = int(logits[0].shape[-1])
        selected = tuple(group["selected"])
        masks = tuple(group["masks"])
        top_k = int(selected[0].shape[-1]) if selected and selected[0] is not None else _top_k_for_router_width(
            kwargs["model_cfg"],
            num_experts,
        )
        loss = fn(
            logits,
            num_experts,
            top_k,
            selected_experts=selected,
            token_masks=masks,
            **{k: v for k, v in kwargs.items() if k != "model_cfg"},
        )
        weight = len(logits)
        total = loss * weight if total is None else total + loss * weight
        weight_sum += weight
    if total is None or weight_sum == 0:
        return None
    return total / weight_sum


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

    aux = output_get(output, "aux_loss", None)
    aux_normalized = None
    # DETACH-ONLY: when the model output's `router_logits` is None
    # (non-aux method that skips the gradient-bearing path), fall back to
    # the per-router `_last_router_scores_detached` snapshot. Telemetry
    # remains available without retaining the autograd graph.
    router_logits_for_metrics = output_get(output, "router_logits", None)
    if router_logits_for_metrics is None:
        router_logits_for_metrics = _collect_detached_router_scores(raw_model)
    if router_logits_for_metrics is not None:
        router_metric_groups = _group_router_metric_inputs(
            router_logits_for_metrics,
            selected_experts,
            router_token_masks,
            model_cfg,
        )
        aux_normalized = _weighted_router_metric(
            router_metric_groups,
            normalized_load_balancing_loss_func,
            model_cfg=model_cfg,
        )
    ce_tensor = output_get(output, "ce_loss", None)
    seq_aux = output_get(output, "seq_aux_loss", None)
    if seq_aux is None and seq_aux_loss_coef > 0 and router_logits_for_metrics is not None:
        # Detach-only telemetry: same detached fallback for seq aux telemetry.
        seq_aux = _weighted_router_metric(
            router_metric_groups,
            seq_load_balancing_loss_func,
            model_cfg=model_cfg,
            batch_size=input_ids.shape[0],
        )
    branch_aux = output_get(output, "branch_aux_loss", None)
    branch_entropy = output_get(output, "branch_entropy_loss", None)
    attention_aux = output_get(output, "attention_aux_loss", None)

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

    output_loss = output_get(output, "loss")
    _maybe_enqueue("total", output_loss)
    _maybe_enqueue("aux", aux)
    _maybe_enqueue("aux_normalized", aux_normalized)
    _maybe_enqueue("ce", ce_tensor)
    _maybe_enqueue("seq_aux", seq_aux)
    _maybe_enqueue("branch_aux", branch_aux)
    _maybe_enqueue("branch_entropy", branch_entropy)
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

    total_value = _resolve("total", output_loss)
    aux_value = _resolve("aux", aux)
    aux_normalized_value = _resolve("aux_normalized", aux_normalized)
    seq_aux_value = _resolve("seq_aux", seq_aux)
    branch_aux_value = _resolve("branch_aux", branch_aux)
    branch_entropy_value = _resolve("branch_entropy", branch_entropy)
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
        "branch_entropy_loss": branch_entropy_value,
        "attention_aux_loss": attention_aux_value,
    }
    return metrics, selected_experts, router_token_masks
