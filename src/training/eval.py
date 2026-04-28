"""Validation loop with loss sanity checks."""

from __future__ import annotations

import logging

import torch

from .distributed import reduce_scalar, unwrap_model, is_main_process
from .metrics import compute_output_metrics

logger = logging.getLogger(__name__)

# Reference loss thresholds. Loss stuck above these values after significant
# training may indicate a bug (e.g., the shifted-logits CE label bug fixed
# in commit 740f306 caused loss to stall at ~4.0).
LOSS_SANITY_THRESHOLD = 4.0
LOSS_SANITY_MIN_STEPS = 1000


def check_loss_sanity(
    ce_loss: float,
    step: int,
    threshold: float = LOSS_SANITY_THRESHOLD,
    min_steps: int = LOSS_SANITY_MIN_STEPS,
) -> bool:
    """Check if loss is at a sensible level after sufficient training.

    Returns True if loss is healthy, False if it appears stuck.
    Logs a warning when loss appears stuck above the threshold.
    """
    if step < min_steps:
        return True
    if ce_loss > threshold:
        if is_main_process():
            logger.warning(
                f"Loss sanity check: CE loss {ce_loss:.4f} is above threshold "
                f"{threshold:.1f} after {step} steps. This may indicate a bug "
                f"(e.g., label shifting, loss computation, or data pipeline issue). "
                f"Reference: FineWeb GPT-2 should reach ~3.28 by 1695 steps."
            )
        return False
    return True


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
    step: int = 0,
    loss_sanity_threshold: float = LOSS_SANITY_THRESHOLD,
) -> dict[str, float]:
    """Run validation and return metrics dict with 'eval/' prefix.

    If step >= LOSS_SANITY_MIN_STEPS and CE loss exceeds the threshold,
    a warning is logged to help detect training regressions early.
    """
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

    try:
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
            # DEC-15: respect the method-aware `output_router_logits` policy
            # the model was built with. `getattr` falls back to True for the
            # legacy back-compat path (no method stamped → preserve pre-DEC-15
            # behavior).
            orl = getattr(raw_model, "config", None)
            orl_value = getattr(orl, "output_router_logits", True) if orl is not None else True
            output = model(
                input_ids=input_ids,
                labels=labels,
                **({} if is_dense else {"output_router_logits": orl_value}),
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
    finally:
        # Restore training mode even if the eval loop raises (e.g. OOM on an
        # outlier batch). Leaving the model in .eval() would silently disable
        # dropout for the rest of training.
        if was_training:
            model.train()

    if batches == 0:
        return {}

    result = {
        f"eval/{key}": reduce_scalar(value / batches, device=device)
        for key, value in totals.items()
    }

    # Loss sanity check
    if step > 0 and "eval/ce_loss" in result:
        check_loss_sanity(result["eval/ce_loss"], step, loss_sanity_threshold)

    return result
