"""Output metrics computation shared between training and eval."""

from __future__ import annotations

import torch

from src.models.load_balancing import (
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
)


def get_selected_experts_for_seq_aux(model) -> tuple[torch.Tensor, ...] | None:
    """Extract selected expert indices from model internals."""
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
    except Exception:
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
    aux_value = aux.detach().float().item() if isinstance(aux, torch.Tensor) else float(aux or 0.0)

    aux_normalized = None
    if getattr(output, "router_logits", None) is not None:
        aux_normalized = normalized_load_balancing_loss_func(
            output.router_logits,
            model_cfg.num_experts,
            model_cfg.num_experts_per_tok,
            token_masks=router_token_masks,
            selected_experts=selected_experts,
        )
    if isinstance(aux_normalized, torch.Tensor):
        aux_normalized_value = aux_normalized.detach().float().item()
    elif aux_normalized is not None:
        aux_normalized_value = float(aux_normalized)
    else:
        aux_normalized_value = 0.0

    total_value = output.loss.detach().float().item()
    ce_tensor = getattr(output, "ce_loss", None)
    if isinstance(ce_tensor, torch.Tensor):
        ce_value = ce_tensor.detach().float().item()
    elif ce_tensor is not None:
        ce_value = float(ce_tensor)
    else:
        ce_value = total_value - getattr(raw_model, "router_aux_loss_coef", 0.0) * aux_value

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
    if isinstance(seq_aux, torch.Tensor):
        seq_aux_value = seq_aux.detach().float().item()
    elif seq_aux is not None:
        seq_aux_value = float(seq_aux)
    else:
        seq_aux_value = 0.0

    branch_aux = getattr(output, "branch_aux_loss", None)
    branch_aux_value = branch_aux.detach().float().item() if isinstance(branch_aux, torch.Tensor) else float(branch_aux or 0.0)
    attention_aux = getattr(output, "attention_aux_loss", None)
    attention_aux_value = attention_aux.detach().float().item() if isinstance(attention_aux, torch.Tensor) else float(attention_aux or 0.0)

    if ce_tensor is None:
        ce_value -= seq_aux_loss_coef * seq_aux_value
        ce_value -= getattr(raw_model, "branch_router_aux_loss_coef", 0.0) * branch_aux_value

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
