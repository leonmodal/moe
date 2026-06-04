"""Generate presentation plots for the synthetic-task sweep.

Pulls per-step metrics from W&B (project `moe-synthetic`) and writes:
  - `summary_ce_bar.png` — final CE per config, grouped by task
  - `trajectory_ce_1L_linearmap.png` — CE vs step for linear-map configs
  - `trajectory_ce_4L_cellular.png` — CE vs step for cellular configs
  - `trajectory_attn_iou_4L_cellular.png` — attn IoU vs step for CA configs
  - `summary.csv` — final metrics per run
  - `findings.md` — written interpretation

Usage:
  WANDB_API_KEY=<key> python scripts/make_synthetic_plots.py \
      --output-dir presentations/synthetic_results
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Run name -> (display label, task, family, color)
RUN_REGISTRY = {
    # 1L linear-map
    "baseline_dense_1L_linearmap_30k": ("Dense (active-matched, 30k)", "linear_map_1L", "dense", "#1f77b4"),
    "baseline_standardmoe_1L_linearmap_30k": ("Standard MoE (30k)", "linear_map_1L", "standard_moe", "#2ca02c"),
    "1L_linearmap_branch_top1_explore_decay_30k": ("MoE-E branch=top1_explore_decay (30k)", "linear_map_1L", "moe_everything_top1", "#d62728"),
    "1L_linearmap_branch_fixed_alternating_30k": ("MoE-E branch=fixed_alternating (30k)", "linear_map_1L", "moe_everything_fixed_alt", "#ff7f0e"),
    "1L_linearmap_branch_sampling_entropy_30k": ("MoE-E branch=sampling_entropy (30k)", "linear_map_1L", "moe_everything_sampling", "#9467bd"),
    "1L_linearmap_branch_fixed_alternating_localbank_30k": ("MoE-E LOCAL bank, fixed_alt (30k)", "linear_map_1L", "moe_everything_localbank", "#8c564b"),
    # 4L cellular
    "baseline_dense_4L_cellular": ("Dense (active-matched, 10k)", "cellular_4L", "dense", "#1f77b4"),
    "baseline_standardmoe_4L_cellular": ("Standard MoE (10k)", "cellular_4L", "standard_moe", "#2ca02c"),
    "4L_cellular_branch_top1_explore_decay": ("MoE-E branch=top1_explore_decay (10k)", "cellular_4L", "moe_everything_top1", "#d62728"),
    "4L_cellular_branch_fixed_alternating": ("MoE-E branch=fixed_alternating (10k)", "cellular_4L", "moe_everything_fixed_alt", "#ff7f0e"),
    "4L_cellular_branch_sampling_entropy": ("MoE-E branch=sampling_entropy (10k)", "cellular_4L", "moe_everything_sampling", "#9467bd"),
    "4L_cellular_branch_fixed_alternating_localbank": ("MoE-E LOCAL bank, fixed_alt (10k)", "cellular_4L", "moe_everything_localbank", "#8c564b"),
}


# Task-level baselines / floors.
TASK_INFO = {
    "linear_map_1L": {
        "title": "Linear Map task (S=16, s=3, T=2, C=2)",
        "chance_ce": math.log(2),
        # Paper's irreducible noise floor: (S-1)/(ST) * ln C
        "paper_floor": (15.0 / 32.0) * math.log(2),
    },
    "cellular_4L": {
        "title": "Cellular Automata task (S=16, T=16, C=4, k=1, N=256)",
        "chance_ce": math.log(4),
        # Paper floor: (S-1)/(ST) * ln C
        "paper_floor": (15.0 / 256.0) * math.log(4),
    },
}


def fetch_run_history(api, project: str, run_name: str, keys: list[str]):
    runs = list(api.runs(project, filters={"display_name": run_name}))
    if not runs:
        return None
    # Prefer a finished run when multiple share the same display name (e.g.,
    # when a re-launch reused the experiment_name after a crash). Falls back
    # to whichever has the highest _step if no finished version exists.
    finished = [r for r in runs if r.state == "finished"]
    if finished:
        r = max(finished, key=lambda x: int(x.summary.get("_step", 0) or 0))
    else:
        r = max(runs, key=lambda x: int(x.summary.get("_step", 0) or 0))
    return r, r.history(keys=["_step"] + keys, samples=500)


def plot_ce_summary_bar(api, project: str, out_path: Path):
    """Final CE per config, grouped by task. Adds chance and paper-floor lines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, task_key in zip(axes, ["linear_map_1L", "cellular_4L"]):
        names = [n for n, (_, t, _, _) in RUN_REGISTRY.items() if t == task_key]
        labels, vals, colors = [], [], []
        for name in names:
            res = fetch_run_history(api, project, name, ["eval/ce_loss"])
            if res is None:
                continue
            r, hist = res
            final_ce = r.summary.get("eval/ce_loss")
            if final_ce is None or (isinstance(final_ce, float) and math.isnan(final_ce)):
                continue
            label, _, _, color = RUN_REGISTRY[name]
            labels.append(label)
            vals.append(final_ce)
            colors.append(color)

        ypos = np.arange(len(labels))
        ax.barh(ypos, vals, color=colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("final eval CE loss (lower = better)")

        info = TASK_INFO[task_key]
        ax.axvline(info["chance_ce"], color="black", ls="--", lw=1, label=f"chance ln(C)={info['chance_ce']:.3f}")
        ax.axvline(info["paper_floor"], color="green", ls=":", lw=1, label=f"paper floor={info['paper_floor']:.3f}")
        ax.set_title(info["title"], fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        # Annotate values
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle("Synthetic-task final CE — active-param-matched comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_trajectory(api, project: str, task_key: str, metric: str, ylabel: str,
                    out_path: Path, log_y: bool = False):
    """Trajectory plot for a metric across all runs in a task."""
    fig, ax = plt.subplots(figsize=(10, 6))
    info = TASK_INFO[task_key]
    names = [n for n, (_, t, _, _) in RUN_REGISTRY.items() if t == task_key]

    for name in names:
        res = fetch_run_history(api, project, name, [metric])
        if res is None:
            continue
        r, hist = res
        if hist is None or hist.empty:
            continue
        label, _, _, color = RUN_REGISTRY[name]
        steps = hist["_step"].to_numpy()
        vals = hist[metric].to_numpy()
        mask = ~np.isnan(vals)
        ax.plot(steps[mask], vals[mask], label=label, color=color, alpha=0.85, lw=1.5)

    if metric == "eval/ce_loss":
        ax.axhline(info["chance_ce"], color="black", ls="--", lw=1, label=f"chance ln(C)={info['chance_ce']:.3f}")
        ax.axhline(info["paper_floor"], color="green", ls=":", lw=1, label=f"paper floor={info['paper_floor']:.3f}")

    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(f"{info['title']} — {ylabel}", fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary_csv(api, project: str, out_path: Path):
    rows = ["run_name,task,family,final_ce,attn_iou,attn_kl_agg,attn_kl_best,final_step"]
    for name, (label, task, family, _) in RUN_REGISTRY.items():
        runs = list(api.runs(project, filters={"display_name": name}))
        if not runs:
            continue
        finished = [r for r in runs if r.state == "finished"]
        chosen = (
            max(finished, key=lambda x: int(x.summary.get("_step", 0) or 0))
            if finished
            else max(runs, key=lambda x: int(x.summary.get("_step", 0) or 0))
        )
        s = chosen.summary
        ce = s.get("eval/ce_loss", "")
        iou = s.get("attn_eval/overall/iou_agg_mean", "")
        kl_agg = s.get("attn_eval/overall/kl_agg_mean", "")
        kl_best = s.get("attn_eval/overall/kl_best_mean", "")
        step = s.get("_step", "")
        def fmt(v):
            if isinstance(v, float):
                if math.isnan(v):
                    return ""
                return f"{v:.4f}"
            return str(v)
        rows.append(f"{name},{task},{family},{fmt(ce)},{fmt(iou)},{fmt(kl_agg)},{fmt(kl_best)},{step}")
    out_path.write_text("\n".join(rows) + "\n")
    print(f"  wrote {out_path}")


def write_findings(out_path: Path):
    findings = """# Synthetic-task sweep — findings

## Setup
Two synthetic attention-pattern tasks from Zhao et al. NeurIPS 2026 §3.1:

- **Linear map** (1L configs): predict `x_1 = A·x_0 mod 2` for sparse binary A. Sequence = `[x_0, x_1]` (32 tokens, vocab 2). Ground-truth attention pattern = the nonzero positions of each row of A. Single-block architecture suffices in principle.
- **Cellular automata** (4L configs): apply a randomly-sampled local-3-window rule for T=16 steps. Sequence = 256 tokens, vocab 4. Ground-truth attention pattern = local 3-window on the previous state.

## Architectures compared (all active-param-matched)

Active per token = 15.7M (1L) / 62.8M (4L). All three families use the same hidden size (1024), heads (16), head_dim (128), and active MLP capacity (≈9.4M).

| Family | Attn | MLP | Routing |
|---|---|---|---|
| Dense | Q+K+V+O (dense) | dense (`intermediate=3072`) | none |
| Standard MoE | Q+K+V+O (dense) | top-4 of 16 (1L) / 64 (4L) experts, `moe_intermediate=768` | MLP only |
| MoE-Everything | top-1 of 8 / 32 attn experts (qkvo bundle) | same MoE as above | + branch routing (attn vs MLP per depth) |

MoE-Everything is tested with 3 branching strategies: `top1_explore_decay`, `fixed_alternating`, `sampling_entropy`.

## Headline results (final CE)

| | Dense | Standard MoE | MoE-E (best variant) |
|---|---|---|---|
| Linear map (chance=0.693, paper floor=0.325) | **0.384** ✓ | **0.363** ✓ | 0.693 ❌ (all 3 variants stuck at chance) |
| Cellular automata (chance=1.386, paper floor=0.081) | **0.251** ✓ | **0.251** ✓ | 0.553 (`sampling_entropy`) |

## Key takeaway

**MoE-Everything's attention-routing hurts attention-pattern learning** on these tasks. With matched active parameters and matched compute budget, both the dense and the standard-MoE baseline emerge cleanly on both tasks. All three MoE-Everything branching strategies fail to emerge on linear map, and only partially emerge on cellular automata. This matches the paper's thesis that *learning the right attention pattern* is the bottleneck — and adding per-token routing on top of attention appears to make the search harder rather than easier on tasks where the optimal attention is sparse and structured.

## Caveats

- These tasks are *deliberately* attention-pattern-stressful. They are not LM-pretraining proxies. The result is "the routing hurts when the task forces sparse-attention learning"; it does not generalize to "MoE-Everything is worse for LM."
- Cellular-automata baselines were trained on 10k steps; the 1L linear-map runs that emerged needed 30k (the 10k runs were all still on the plateau).
- The labels-mask in the synthetic dataset was being clobbered by the trainer until commit "fix trainer labels-mask" — without that fix even dense+standard_moe stayed at chance. After the fix, dense and standard_moe emerged; MoE-Everything still didn't.
- Attention IoU at emergence is high for the dense/standard_moe runs (visible in the heatmap PNGs on the `moe-checkpoints` volume at `/synthetic/<exp>/attn_eval/step_<N>/depth_<d>.png`).
"""
    out_path.write_text(findings)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="leon-modal-modal/moe-synthetic")
    parser.add_argument("--output-dir", default="presentations/synthetic_results", type=Path)
    args = parser.parse_args()

    import wandb
    api = wandb.Api()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating summary CSV...")
    write_summary_csv(api, args.project, args.output_dir / "summary.csv")

    print("Generating CE bar chart...")
    plot_ce_summary_bar(api, args.project, args.output_dir / "summary_ce_bar.png")

    print("Generating 1L linear-map CE trajectory...")
    plot_trajectory(api, args.project, "linear_map_1L", "eval/ce_loss",
                    "eval CE loss", args.output_dir / "trajectory_ce_1L_linearmap.png")

    print("Generating 4L cellular CE trajectory...")
    plot_trajectory(api, args.project, "cellular_4L", "eval/ce_loss",
                    "eval CE loss", args.output_dir / "trajectory_ce_4L_cellular.png")

    print("Generating 4L cellular attn IoU trajectory...")
    plot_trajectory(api, args.project, "cellular_4L", "attn_eval/overall/iou_agg_mean",
                    "attention IoU (vs ground truth)", args.output_dir / "trajectory_attn_iou_4L_cellular.png")

    print("Generating 1L linear-map attn IoU trajectory...")
    plot_trajectory(api, args.project, "linear_map_1L", "attn_eval/overall/iou_agg_mean",
                    "attention IoU (vs ground truth)", args.output_dir / "trajectory_attn_iou_1L_linearmap.png")

    print("Writing findings.md...")
    write_findings(args.output_dir / "findings.md")

    print(f"\nDone. All outputs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
