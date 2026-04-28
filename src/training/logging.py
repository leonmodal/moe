"""WandB logging, console output, and routing graph support."""

from __future__ import annotations

import math
import os

from .distributed import is_main_process


def setup_wandb(
    *,
    project: str | None,
    run_name: str | None,
    config: dict,
    output_dir: str,
    resume_run_id: str | None = None,
):
    """Initialize WandB logging if project is set and this is the main process."""
    if not project or not is_main_process():
        return None

    import wandb

    kwargs = dict(
        project=project,
        name=run_name,
        config=config,
        dir=output_dir,
    )
    if resume_run_id:
        kwargs["id"] = resume_run_id
        kwargs["resume"] = "allow"

    return wandb.init(**kwargs)


def log_training_step(
    wandb_run,
    *,
    step: int,
    metrics: dict[str, float],
    grad_norm: float,
    lr: float,
    tok_per_s: float,
    tokens_seen: float,
    elapsed: float,
    log_every: int,
) -> None:
    """Log training step to console and WandB."""
    if step % log_every != 0 or not is_main_process():
        return

    print(
        f"step {step:6d}  "
        f"loss={metrics['loss']:.4f}  "
        f"ce={metrics['ce_loss']:.4f}  "
        f"aux={metrics['aux_loss']:.4f}  "
        f"aux_n={metrics['aux_loss_normalized']:.4f}  "
        f"seq_aux={metrics['seq_aux_loss']:.4f}  "
        f"branch_aux={metrics['branch_aux_loss']:.4f}  "
        f"attn_aux={metrics['attention_aux_loss']:.4f}  "
        f"lr={lr:.2e}  "
        f"tok/s={tok_per_s/1e3:.1f}k  "
        f"sec/step={elapsed:.3f}  "
        f"|g|={grad_norm:.3f}",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/ce": metrics["ce_loss"],
                "train/aux": metrics["aux_loss"],
                "train/aux_n": metrics["aux_loss_normalized"],
                "train/seq_aux": metrics["seq_aux_loss"],
                "train/branch_aux": metrics["branch_aux_loss"],
                "train/attn_aux": metrics["attention_aux_loss"],
                "train/grad_norm": grad_norm,
                "train/tok_per_s": tok_per_s,
                "train/lr": lr,
                "train/tokens_seen": tokens_seen,
            },
            step=step,
        )


def log_eval_metrics(
    wandb_run,
    *,
    step: int,
    eval_metrics: dict[str, float],
) -> None:
    """Log eval metrics to console and WandB."""
    if not eval_metrics or not is_main_process():
        return

    ce = eval_metrics.get("eval/ce_loss", 0.0)
    ppl = math.exp(min(20.0, ce))
    eval_metrics["eval/perplexity"] = ppl
    print(
        f"eval {step:6d}  "
        f"ce={ce:.4f}  "
        f"ppl={ppl:.2f}  "
        f"aux={eval_metrics.get('eval/aux_loss', 0.0):.4f}",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.log(eval_metrics, step=step)


def save_routing_plots(
    model,
    *,
    output_dir: str,
    step: int,
) -> None:
    """Save routing heatmaps and snapshot plots for MoE models."""
    if not is_main_process():
        return

    raw_model = model
    if hasattr(model, 'module'):
        raw_model = model.module
    if hasattr(model, '_fsdp_wrapped_module'):
        raw_model = model._fsdp_wrapped_module

    if not hasattr(raw_model, '_routing_stats_obj') or raw_model._routing_stats_obj is None:
        return

    from src.utils.routing_plots import plot_expert_heatmaps, plot_routing_snapshot

    stats_obj = raw_model._routing_stats_obj
    heatmap_dir = os.path.join(output_dir, "routing_logs", f"step_{step:08d}")
    os.makedirs(heatmap_dir, exist_ok=True)

    heatmap_data = stats_obj.expert_heatmap_data()
    plot_expert_heatmaps(heatmap_data, heatmap_dir, step)

    if hasattr(stats_obj, 'routing_snapshot'):
        snapshot = stats_obj.routing_snapshot()
        bias_data = {}
        if hasattr(raw_model, '_global_q_bias'):
            bias_data["expert_biases"] = {
                "Q": raw_model._global_q_bias.detach().cpu().numpy(),
                "K": raw_model._global_k_bias.detach().cpu().numpy(),
                "V": raw_model._global_v_bias.detach().cpu().numpy(),
                "O": raw_model._global_o_bias.detach().cpu().numpy(),
                "MLP": raw_model._global_mlp_bias.detach().cpu().numpy(),
            }
        # Branch-router biases (the shared `expert_bias`/`local_tokens_per_expert` buffer interface: buffer name unified to `expert_bias`).
        # Singular and plural attribute names both exist depending on
        # `per_layer_router`; iterate both.
        attn_b, mlp_b = [], []
        for br in (
            list(getattr(raw_model, "branch_routers", []) or [])
            + ([getattr(raw_model, "branch_router", None)]
               if getattr(raw_model, "branch_router", None) is not None else [])
        ):
            bias = getattr(br, "expert_bias", None)
            if bias is not None and bias.numel() == 2:
                b = bias.detach().cpu().numpy()
                attn_b.append(float(b[0]))
                mlp_b.append(float(b[1]))
        if attn_b:
            # Payload key kept as `branch_bias` so the plotting consumer
            # (`src/utils/routing_plots.py`) doesn't need to change.
            bias_data["branch_bias"] = {"attn": attn_b, "mlp": mlp_b}
        plot_routing_snapshot(snapshot, heatmap_dir, step, bias_data=bias_data)
