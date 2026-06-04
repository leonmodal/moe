"""Validation loop with loss sanity checks."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch

from .distributed import reduce_scalar, reduce_scalar_dict, unwrap_model, is_main_process
from .metrics import compute_output_metrics

logger = logging.getLogger(__name__)

# Reference loss thresholds. Loss stuck above these values after significant
# training may indicate a bug (e.g., the shifted-logits CE label bug fixed
# in commit 740f306 caused loss to stall at ~4.0).
LOSS_SANITY_THRESHOLD = 4.0
LOSS_SANITY_MIN_STEPS = 1000


def _normalize_recurrence_sweep(value: Any) -> list[int]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        value = [value]
    steps: list[int] = []
    for item in value:
        step = int(item)
        if step > 0 and step not in steps:
            steps.append(step)
    return steps


def _metric_label(value: Any) -> str:
    return str(value).replace(":", "_").replace("/", "_")


def _recurrence_prefix(num_steps: int) -> str:
    return f"eval/recurrence_{int(num_steps):03d}"


def _default_eval_recurrence(raw_model: Any) -> int:
    config = getattr(raw_model, "config", None)
    return max(1, int(getattr(config, "eval_recurrence", 1) or 1))


def _collect_eval_batches(eval_dataloader, max_batches: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(eval_dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batches.append({
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        })
    return batches


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _is_recurrent_diagnostics_model(raw_model: Any) -> bool:
    config = getattr(raw_model, "config", None)
    return str(getattr(config, "model_type", "")) in {
        "recurrent_standard_moe",
        "recurrent_global_moe",
        "hrm_recurrent_standard_moe",
    }


def _load_balance_eval_metrics(raw_model: Any, *, step: int, prefix: str) -> dict[str, float]:
    from src.utils.load_balance_artifacts import build_load_balance_artifacts

    payload = build_load_balance_artifacts(raw_model, step=step)
    if payload is None:
        return {}
    metrics: dict[str, float] = {}
    for row in payload.get("summary_rows", []):
        if row.get("depth", "") != "":
            continue
        pool = _metric_label(row.get("pool", "unknown"))
        for key in (
            "active_expert_fraction",
            "max_over_ideal",
            "cv",
            "normalized_entropy",
        ):
            value = row.get(key)
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                metrics[f"{prefix}/load_balance/{pool}/{key}"] = float(value)
    return metrics


def _recurrent_eval_metrics(
    raw_model: Any,
    *,
    step: int,
    prefix: str,
    input_ids: torch.Tensor | None,
) -> dict[str, float]:
    from src.utils.recurrent_diagnostics import build_recurrent_diagnostics

    payload = build_recurrent_diagnostics(raw_model, step=step, input_ids=input_ids)
    if payload is None:
        return {}
    summary = payload.get("summary") or {}
    metrics: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            metrics[f"{prefix}/recurrent/{key}"] = float(value)
    return metrics


def _save_eval_artifacts(
    raw_model: Any,
    *,
    step_dir: Path,
    step: int,
    input_ids: torch.Tensor,
    tokenizer,
    include_recurrent_diagnostics: bool,
) -> None:
    from src.utils.branch_patterns import save_branch_route_artifacts
    from src.utils.load_balance_artifacts import save_load_balance_artifacts

    save_branch_route_artifacts(
        raw_model,
        step_dir=str(step_dir),
        step=step,
        input_ids=input_ids,
        tokenizer=tokenizer,
    )
    save_load_balance_artifacts(
        raw_model,
        step_dir=str(step_dir),
        step=step,
    )
    if include_recurrent_diagnostics:
        from src.utils.recurrent_diagnostics import save_recurrent_diagnostics_artifacts

        save_recurrent_diagnostics_artifacts(
            raw_model,
            step_dir=str(step_dir),
            step=step,
            input_ids=input_ids,
        )


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
    output_dir: str | None = None,
    tokenizer=None,
    recurrence_sweep: Any = None,
    attention_eval_targets: Any = None,
    attention_eval_save_heatmaps: bool = False,
) -> dict[str, float]:
    """Run validation and return metrics dict with 'eval/' prefix.

    If step >= LOSS_SANITY_MIN_STEPS and CE loss exceeds the threshold,
    a warning is logged to help detect training regressions early.
    """
    if eval_dataloader is None:
        return {}

    eval_batches = _collect_eval_batches(eval_dataloader, max_batches)
    if not eval_batches:
        return {}

    was_training = model.training
    model.eval()
    raw_model = unwrap_model(model)
    model_config = getattr(raw_model, "config", None)
    model_type = str(getattr(model_config, "model_type", ""))
    orl_value = (
        getattr(model_config, "output_router_logits", True)
        if model_config is not None
        else True
    )
    recurrence_steps = _normalize_recurrence_sweep(recurrence_sweep)
    default_recurrence = _default_eval_recurrence(raw_model)
    pass_steps: list[int | None]
    if recurrence_steps:
        pass_steps = sorted(set(recurrence_steps + [default_recurrence]))
    else:
        pass_steps = [None]

    result: dict[str, float] = {}
    include_recurrent_diagnostics = _is_recurrent_diagnostics_model(raw_model)

    try:
        for recurrence in pass_steps:
            prefix = (
                _recurrence_prefix(recurrence)
                if recurrence is not None
                else "eval"
            )
            totals = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "aux_loss": 0.0,
                "aux_loss_normalized": 0.0,
                "seq_aux_loss": 0.0,
                "branch_aux_loss": 0.0,
                "branch_entropy_loss": 0.0,
                "attention_aux_loss": 0.0,
            }
            batches = 0
            last_input_ids = None

            for batch_idx, batch_cpu in enumerate(eval_batches):
                batch = _to_device_batch(batch_cpu, device)
                input_ids = batch["input_ids"]
                last_input_ids = input_ids.detach()
                # Honor dataset-emitted labels when present (synthetic tasks
                # use -100 to mask the unpredictable initial random state);
                # parquet eval falls back to input_ids.
                labels = batch.get("labels", input_ids)
                model_kwargs = {} if is_dense else {"output_router_logits": orl_value}
                if model_type in {"moe_everything", "recurrent_moe_everything"}:
                    model_kwargs["return_logits"] = False
                if recurrence is not None:
                    model_kwargs["num_steps"] = torch.tensor(
                        [0, int(recurrence)],
                        device=device,
                        dtype=torch.long,
                    )
                    if (
                        include_recurrent_diagnostics
                        and batch_idx == len(eval_batches) - 1
                    ):
                        model_kwargs["collect_recurrence_diagnostics"] = True
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    **model_kwargs,
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

            if batches == 0:
                continue

            pass_result = {
                f"{prefix}/{key}": reduce_scalar(value / batches, device=device)
                for key, value in totals.items()
            }
            ce_value = pass_result.get(f"{prefix}/ce_loss")
            if ce_value is not None:
                pass_result[f"{prefix}/perplexity"] = math.exp(min(20.0, ce_value))

            if last_input_ids is not None:
                load_balance_metrics = _load_balance_eval_metrics(
                    raw_model,
                    step=step,
                    prefix=prefix,
                )
                recurrent_metrics = (
                    _recurrent_eval_metrics(
                        raw_model,
                        step=step,
                        prefix=prefix,
                        input_ids=last_input_ids,
                    )
                    if include_recurrent_diagnostics
                    else {}
                )
                aux_metrics = {**load_balance_metrics, **recurrent_metrics}
                if aux_metrics:
                    aux_metrics = reduce_scalar_dict(aux_metrics, device=device)
                    pass_result.update(aux_metrics)

            result.update(pass_result)

            if (
                recurrence is not None
                and recurrence == default_recurrence
                and recurrence_steps
            ):
                # Preserve the historical `eval/*` metric keys using the
                # default recurrence pass instead of doing a duplicate forward.
                default_prefix = _recurrence_prefix(recurrence)
                for key, value in list(pass_result.items()):
                    if key.startswith(default_prefix + "/"):
                        result["eval/" + key[len(default_prefix) + 1:]] = value

            if (
                output_dir is not None
                and step > 0
                and last_input_ids is not None
                and is_main_process()
            ):
                eval_step_dir = Path(output_dir) / "eval_logs" / f"step_{step:08d}"
                if recurrence is None:
                    artifact_dirs = [eval_step_dir]
                else:
                    artifact_dirs = [
                        eval_step_dir / f"recurrence_{int(recurrence):03d}"
                    ]
                    if recurrence == default_recurrence:
                        artifact_dirs.append(eval_step_dir)
                for artifact_dir in artifact_dirs:
                    _save_eval_artifacts(
                        raw_model,
                        step_dir=artifact_dir,
                        step=step,
                        input_ids=last_input_ids,
                        tokenizer=tokenizer,
                        include_recurrent_diagnostics=(
                            include_recurrent_diagnostics and recurrence is not None
                        ),
                    )
    finally:
        # Restore training mode even if the eval loop raises (e.g. OOM on an
        # outlier batch). Leaving the model in .eval() would silently disable
        # dropout for the rest of training.
        if was_training:
            model.train()

    # Attention evaluation against a known ground-truth pattern (synthetic
    # tasks only). Runs one extra forward over the last eval batch with the
    # attention bank's capture flag set; logs per-depth IoU / KL / entropy
    # against the dataset's `ground_truth_A`. Heatmap PNGs are saved at the
    # caller's discretion (typically `attention_eval_save_heatmaps=True` at
    # checkpoint cadence).
    if attention_eval_targets is not None and eval_batches:
        from .attention_eval import evaluate_attention_against_ground_truth

        try:
            targets, row_mask = attention_eval_targets
            attn_metrics = evaluate_attention_against_ground_truth(
                model=model,
                eval_batch=eval_batches[-1],
                targets=targets,
                row_mask=row_mask,
                device=device,
                output_dir=output_dir,
                step=step,
                save_heatmaps=attention_eval_save_heatmaps,
            )
            if attn_metrics:
                # Reduce scalars across ranks so DDP runs converge to the same
                # logged value (each rank computes attention on a different
                # batch shard; mean is the right reduction).
                attn_metrics = reduce_scalar_dict(attn_metrics, device=device)
                result.update(attn_metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attention eval failed at step %d: %s", step, exc)

    if not result:
        return {}

    # Loss sanity check
    if step > 0 and "eval/ce_loss" in result:
        check_loss_sanity(result["eval/ce_loss"], step, loss_sanity_threshold)

    return result
