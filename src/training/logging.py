"""WandB logging, console output, and routing graph support."""

from __future__ import annotations

import json
import math
import os

from .distributed import is_main_process, unwrap_model


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
    log_dense_until: int = 0,
    branch_explore_rate: float | None = None,
    branch_attn_fraction: float | None = None,
    branch_attn_per_depth: list[float] | None = None,
    branch_explore_mask_fraction: float | None = None,
    recurrence_mean: int | None = None,
    recurrence_backprop_depth: int | None = None,
    recurrence_num_steps_no_grad: float | None = None,
    recurrence_num_steps_with_grad: float | None = None,
    hrm_h_cycles: float | None = None,
    hrm_l_cycles: float | None = None,
    recurrence_diagnostics: dict[str, float] | None = None,
) -> None:
    """Log training step to console and WandB.

    Branch routing telemetry (only emitted when active on this model):
      * `branch_explore_rate`: the `p_explore(step)` rate applied
        BEFORE this step's forward. This is the rate every microbatch
        in the step actually saw, not the post-step or
        pre-next-step rate.
      * `branch_attn_fraction`: the global-mean fraction of branch
        tokens that chose ATTN (selected_experts == 0) across all
        active branch routers on this step's forward.
      * `branch_attn_per_depth`: per-router ATTN fractions for
        `per_layer_router=True` builds; logged under
        `train/branch_attn_fraction/depth_<i>` so per-depth
        divergence is visible in W&B.
      * `branch_explore_mask_fraction`: diagnostic-only fraction of
        branch tokens routed via the random-override mask (NOT the
        same as `% ATTN`). Logged under a separate key so the two
        cannot be confused.
    """
    # Dense-early window: when step < log_dense_until, log every step.
    # Past that window, fall back to the sparse `log_every` cadence.
    in_dense_window = step < int(log_dense_until)
    if not in_dense_window and step % log_every != 0:
        return
    if not is_main_process():
        return

    extra = ""
    if branch_explore_rate is not None:
        extra += f"  br_p_explore={branch_explore_rate:.4f}"
    if branch_attn_fraction is not None:
        extra += f"  br_attn={branch_attn_fraction:.4f}"
    if branch_explore_mask_fraction is not None:
        extra += f"  br_explore_mask={branch_explore_mask_fraction:.4f}"
    if recurrence_mean is not None:
        extra += f"  mean_rec={recurrence_mean}"
    if recurrence_backprop_depth is not None:
        extra += f"  bptt={recurrence_backprop_depth}"
    if recurrence_num_steps_no_grad is not None and recurrence_num_steps_with_grad is not None:
        extra += (
            f"  rec=({recurrence_num_steps_no_grad:.1f},"
            f"{recurrence_num_steps_with_grad:.1f})"
        )
    if hrm_h_cycles is not None and hrm_l_cycles is not None:
        extra += f"  hrm=(H{hrm_h_cycles:.1f},L{hrm_l_cycles:.1f})"
    if recurrence_diagnostics is not None:
        rel = recurrence_diagnostics.get("relative_residual_rms_last")
        route = recurrence_diagnostics.get("routing_last_token_jaccard_distance_mean")
        if rel is not None:
            extra += f"  rec_res={rel:.4f}"
        if route is not None:
            extra += f"  route_d={route:.4f}"
    print(
        f"step {step:6d}  "
        f"loss={metrics['loss']:.4f}  "
        f"ce={metrics['ce_loss']:.4f}  "
        f"aux={metrics['aux_loss']:.4f}  "
        f"aux_n={metrics['aux_loss_normalized']:.4f}  "
        f"seq_aux={metrics['seq_aux_loss']:.4f}  "
        f"branch_aux={metrics['branch_aux_loss']:.4f}  "
        f"branch_ent={metrics.get('branch_entropy_loss', 0.0):.4f}  "
        f"attn_aux={metrics['attention_aux_loss']:.4f}  "
        f"lr={lr:.2e}  "
        f"tok/s={tok_per_s/1e3:.1f}k  "
        f"sec/step={elapsed:.3f}  "
        f"|g|={grad_norm:.3f}"
        f"{extra}",
        flush=True,
    )

    if wandb_run is not None:
        payload = {
            "train/loss": metrics["loss"],
            "train/ce": metrics["ce_loss"],
            "train/aux": metrics["aux_loss"],
            "train/aux_n": metrics["aux_loss_normalized"],
            "train/seq_aux": metrics["seq_aux_loss"],
            "train/branch_aux": metrics["branch_aux_loss"],
            "train/branch_entropy": metrics.get("branch_entropy_loss", 0.0),
            "train/attn_aux": metrics["attention_aux_loss"],
            "train/grad_norm": grad_norm,
            "train/tok_per_s": tok_per_s,
            "train/lr": lr,
            "train/tokens_seen": tokens_seen,
        }
        if branch_explore_rate is not None:
            payload["train/branch_explore_rate"] = branch_explore_rate
        if branch_attn_fraction is not None:
            payload["train/branch_attn_fraction"] = branch_attn_fraction
        if branch_attn_per_depth is not None and len(branch_attn_per_depth) > 1:
            for idx, value in enumerate(branch_attn_per_depth):
                payload[f"train/branch_attn_fraction/depth_{idx}"] = value
        if branch_explore_mask_fraction is not None:
            payload["train/branch_explore_mask_fraction"] = branch_explore_mask_fraction
        if recurrence_mean is not None:
            payload["train/mean_recurrence"] = recurrence_mean
        if recurrence_backprop_depth is not None:
            payload["train/mean_backprop_depth"] = recurrence_backprop_depth
        if recurrence_num_steps_no_grad is not None:
            payload["train/num_steps_no_grad"] = recurrence_num_steps_no_grad
        if recurrence_num_steps_with_grad is not None:
            payload["train/num_steps_with_grad"] = recurrence_num_steps_with_grad
        if hrm_h_cycles is not None:
            payload["train/hrm_h_cycles"] = hrm_h_cycles
        if hrm_l_cycles is not None:
            payload["train/hrm_l_cycles"] = hrm_l_cycles
        if hrm_h_cycles is not None and hrm_h_cycles > 0 and hrm_l_cycles is not None:
            payload["train/hrm_l_per_h"] = hrm_l_cycles / hrm_h_cycles
        if recurrence_diagnostics is not None:
            for key, value in recurrence_diagnostics.items():
                if isinstance(value, (float, int)) and math.isfinite(float(value)):
                    payload[f"train/recurrent/{key}"] = float(value)
        wandb_run.log(payload, step=step)


def log_eval_metrics(
    wandb_run,
    *,
    step: int,
    eval_metrics: dict[str, float],
    output_dir: str | None = None,
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
    recurrence_prefixes = sorted({
        key.rsplit("/", 1)[0]
        for key in eval_metrics
        if key.startswith("eval/recurrence_") and key.endswith("/ce_loss")
    })
    for prefix in recurrence_prefixes:
        label = prefix.rsplit("/", 1)[-1].replace("recurrence_", "")
        rec_extra = ""
        rec_res = eval_metrics.get(
            f"{prefix}/recurrent/relative_residual_rms_last"
        )
        route_d = eval_metrics.get(
            f"{prefix}/recurrent/routing_last_token_jaccard_distance_mean"
        )
        mlp_cv = eval_metrics.get(f"{prefix}/load_balance/mlp_recurrent/cv")
        mlp_entropy = eval_metrics.get(
            f"{prefix}/load_balance/mlp_recurrent/normalized_entropy"
        )
        if rec_res is not None:
            rec_extra += f"  rec_res={rec_res:.4f}"
        if route_d is not None:
            rec_extra += f"  route_d={route_d:.4f}"
        if mlp_cv is not None:
            rec_extra += f"  mlp_cv={mlp_cv:.4f}"
        if mlp_entropy is not None:
            rec_extra += f"  mlp_ent={mlp_entropy:.4f}"
        print(
            f"eval_rec {step:6d}  loops={int(label)}  "
            f"ce={eval_metrics[f'{prefix}/ce_loss']:.4f}  "
            f"ppl={eval_metrics.get(f'{prefix}/perplexity', 0.0):.2f}"
            f"{rec_extra}",
            flush=True,
        )

    if wandb_run is not None:
        wandb_run.log(eval_metrics, step=step)

    if output_dir is not None:
        eval_dir = os.path.join(output_dir, "eval_logs")
        step_dir = os.path.join(eval_dir, f"step_{step:08d}")
        os.makedirs(step_dir, exist_ok=True)
        payload = {"step": step, **eval_metrics}
        metrics_path = os.path.join(step_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        history_path = os.path.join(eval_dir, "eval_metrics.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def save_routing_plots(
    model,
    *,
    output_dir: str,
    step: int,
    input_ids=None,
    tokenizer=None,
) -> None:
    """Save routing heatmaps and snapshot plots for MoE models."""
    if not is_main_process():
        return

    raw_model = unwrap_model(model)

    stats_obj = getattr(raw_model, '_routing_stats_obj', None)
    heatmap_dir = os.path.join(output_dir, "routing_logs", f"step_{step:08d}")
    wrote_any = False

    if stats_obj is not None:
        from src.utils.routing_plots import plot_expert_heatmaps, plot_routing_snapshot

        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_data = stats_obj.expert_heatmap_data()
        plot_expert_heatmaps(heatmap_data, heatmap_dir, step)
        wrote_any = True

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

    from src.utils.branch_patterns import save_branch_route_artifacts
    from src.utils.load_balance_artifacts import save_load_balance_artifacts
    from src.utils.recurrent_diagnostics import save_recurrent_diagnostics_artifacts

    wrote_branch = save_branch_route_artifacts(
        raw_model,
        step_dir=heatmap_dir,
        step=step,
        input_ids=input_ids,
        tokenizer=tokenizer,
    )
    wrote_any = wrote_any or wrote_branch
    wrote_load = save_load_balance_artifacts(
        raw_model,
        step_dir=heatmap_dir,
        step=step,
    )
    wrote_any = wrote_any or wrote_load
    wrote_recurrent = save_recurrent_diagnostics_artifacts(
        raw_model,
        step_dir=heatmap_dir,
        step=step,
        input_ids=input_ids,
    )
    wrote_any = wrote_any or wrote_recurrent
    if not wrote_any and os.path.isdir(heatmap_dir) and not os.listdir(heatmap_dir):
        os.rmdir(heatmap_dir)
