"""Validation loop with loss sanity checks."""

from __future__ import annotations

import torch

from .distributed import reduce_scalar, unwrap_model
from .metrics import compute_output_metrics


@torch.no_grad()
def run_validation(
    *,
    model,
    model_cfg,
    eval_dataloader,
    max_batches: int,
    is_dense: bool,
    seq_aux_loss_coef: float,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return metrics dict with 'eval/' prefix."""
    if eval_dataloader is None:
        return {}

    was_training = model.training
    model.eval()
    raw_model = unwrap_model(model)
    totals = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "aux_loss": 0.0,
        "aux_loss_normalized": 0.0,
        "seq_aux_loss": 0.0,
        "branch_aux_loss": 0.0,
        "attention_aux_loss": 0.0,
    }
    batches = 0

    for batch_idx, batch in enumerate(eval_dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        input_ids = batch["input_ids"]
        labels = input_ids
        output = model(
            input_ids=input_ids,
            labels=labels,
            **({} if is_dense else {"output_router_logits": True}),
        )
        metrics, _, _ = compute_output_metrics(
            output,
            raw_model,
            model_cfg,
            input_ids,
            seq_aux_loss_coef=seq_aux_loss_coef,
        )
        for key in totals:
            totals[key] += metrics[key]
        batches += 1

    if was_training:
        model.train()
    if batches == 0:
        return {}

    return {
        f"eval/{key}": reduce_scalar(value / batches, device=device)
        for key, value in totals.items()
    }
