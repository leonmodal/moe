"""Comprehensive attention-IoU comparison across all 10 finished runs.

Two views:
  1. Final overall IoU per architecture (bar chart per task)
  2. Per-depth IoU per architecture (heatmap per task)

Combines W&B IoU values for MoE-Everything runs with metrics.json IoU values
saved on the Modal volume for the post-hoc dense/standard-MoE runs.

Outputs:
  presentations/synthetic_results/iou_summary_bar.png
  presentations/synthetic_results/iou_per_depth_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ATTN_COLOR = "#3B82F6"
MLP_COLOR = "#F59E0B"
DENSE_COLOR = "#0F766E"      # teal
STDMOE_COLOR = "#16A34A"     # green
TOP1_COLOR = "#DC2626"       # red
FIXED_COLOR = "#F59E0B"      # amber
SAMPLING_COLOR = "#7C3AED"   # violet
CHANCE_COLOR = "#9CA3AF"

# Hand-collected IoU values. Overall values from W&B summary for MoE-E and
# from metrics.json on the Modal volume for dense/standard_moe (regen).
# Per-depth values: MoE-E from W&B summary keys
# `attn_eval/depth_<d>/iou_agg_mean`; dense/standard_moe from regenerated
# metrics.json.
IOU_DATA = {
    "1L_linearmap": {
        "title": "1L linear-map (vocab 2, seq_len 32) — attention IoU vs ground-truth A",
        "rows": [
            {"label": "Dense", "color": DENSE_COLOR, "overall": 0.150, "per_depth": [0.150]},
            {"label": "Standard MoE", "color": STDMOE_COLOR, "overall": 0.131, "per_depth": [0.131]},
            {"label": "MoE-E\ntop1_explore_decay", "color": TOP1_COLOR, "overall": 0.084, "per_depth": [None, None]},
            {"label": "MoE-E\nfixed_alternating", "color": FIXED_COLOR, "overall": 0.050, "per_depth": [None, None]},
            {"label": "MoE-E\nsampling_entropy", "color": SAMPLING_COLOR, "overall": 0.013, "per_depth": [None, None]},
        ],
        "max_depths": 2,
        "chance_iou": 0.094,  # approx: random top-3 of 32 with target nnz=3
    },
    "4L_cellular": {
        "title": "4L cellular automata (vocab 4, seq_len 256) — attention IoU vs ground-truth window",
        "rows": [
            {"label": "Dense", "color": DENSE_COLOR, "overall": 0.293, "per_depth": [0.026, 0.633, 0.080, 0.435]},
            {"label": "Standard MoE", "color": STDMOE_COLOR, "overall": 0.311, "per_depth": [0.485, 0.529, 0.218, 0.013]},
            {"label": "MoE-E\ntop1_explore_decay", "color": TOP1_COLOR, "overall": 0.284, "per_depth": [None] * 8},
            {"label": "MoE-E\nfixed_alternating", "color": FIXED_COLOR, "overall": 0.617, "per_depth": [None] * 8},
            {"label": "MoE-E\nsampling_entropy", "color": SAMPLING_COLOR, "overall": 0.496, "per_depth": [None] * 8},
        ],
        "max_depths": 8,
        "chance_iou": 0.012,  # approx: random top-3 of 256 with nnz=3
    },
}


def pull_moe_per_depth_iou():
    """Fetch per-depth IoU values from W&B for MoE-Everything runs and fill
    them into IOU_DATA in-place."""
    import wandb
    api = wandb.Api()
    moe_run_map = {
        "1L_linearmap": {
            "MoE-E\ntop1_explore_decay": ("1L_linearmap_branch_top1_explore_decay_30k", 2),
            "MoE-E\nfixed_alternating": ("1L_linearmap_branch_fixed_alternating_30k", 2),
            "MoE-E\nsampling_entropy": ("1L_linearmap_branch_sampling_entropy_30k", 2),
        },
        "4L_cellular": {
            "MoE-E\ntop1_explore_decay": ("4L_cellular_branch_top1_explore_decay", 8),
            "MoE-E\nfixed_alternating": ("4L_cellular_branch_fixed_alternating", 8),
            "MoE-E\nsampling_entropy": ("4L_cellular_branch_sampling_entropy", 8),
        },
    }
    for task, mapping in moe_run_map.items():
        for label, (run_name, n_depths) in mapping.items():
            rs = list(api.runs("leon-modal-modal/moe-synthetic", filters={"display_name": run_name}))
            if not rs:
                continue
            s = rs[0].summary
            depth_vals = []
            for d in range(n_depths):
                v = s.get(f"attn_eval/depth_{d}/iou_agg_mean")
                depth_vals.append(float(v) if v is not None and not math.isnan(v) else None)
            for row in IOU_DATA[task]["rows"]:
                if row["label"] == label:
                    row["per_depth"] = depth_vals
                    break


def plot_overall_bar(out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, task in zip(axes, ["1L_linearmap", "4L_cellular"]):
        spec = IOU_DATA[task]
        rows = spec["rows"]
        labels = [r["label"] for r in rows]
        vals = [r["overall"] for r in rows]
        colors = [r["color"] for r in rows]
        ypos = np.arange(len(labels))
        ax.barh(ypos, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(spec["chance_iou"], color=CHANCE_COLOR, ls="--", lw=1.5,
                   label=f"chance ≈ {spec['chance_iou']:.3f}")
        ax.set_xlim(0, max(0.7, max(vals) * 1.15))
        ax.set_xlabel("attention IoU (vs ground truth, higher = better)", fontsize=10)
        ax.set_title(spec["title"], fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        for i, v in enumerate(vals):
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Attention IoU — which architecture learned the ground-truth attention pattern?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_depth_heatmap(out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [1, 2.5]})
    for ax, task in zip(axes, ["1L_linearmap", "4L_cellular"]):
        spec = IOU_DATA[task]
        rows = spec["rows"]
        max_d = spec["max_depths"]
        matrix = []
        for row in rows:
            depths = row["per_depth"]
            line = []
            for d in range(max_d):
                if d < len(depths) and depths[d] is not None:
                    line.append(depths[d])
                else:
                    line.append(np.nan)
            matrix.append(line)
        mat = np.array(matrix)
        # imshow with NaN → gray; use a perceptually-uniform sequential cmap.
        masked = np.ma.masked_invalid(mat)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="#E5E7EB")
        im = ax.imshow(masked, aspect="auto", vmin=0, vmax=0.7, cmap=cmap)
        ax.set_xticks(np.arange(max_d))
        ax.set_xticklabels([f"d{d}" for d in range(max_d)], fontsize=9)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
        ax.set_xlabel("depth", fontsize=10)
        ax.set_title(spec["title"].split(" — ")[0], fontsize=10)
        # Annotate cells
        for ri, row_vals in enumerate(mat):
            for ci, v in enumerate(row_vals):
                if np.isnan(v):
                    ax.text(ci, ri, "—", ha="center", va="center",
                            color="#9CA3AF", fontsize=10)
                else:
                    txt_color = "white" if v > 0.4 else "#111827"
                    ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                            color=txt_color, fontsize=9, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("IoU vs ground truth", fontsize=8)

    fig.suptitle("Per-depth attention IoU — where in the network did each architecture get attention right?",
                 fontsize=13, fontweight="bold")
    # Note about missing per-depth values
    fig.text(0.5, -0.02,
             "Note: per-depth IoU for MoE-Everything 1L runs was not logged at the final step (only overall IoU is shown in the bar chart above).",
             ha="center", fontsize=8, color="#6B7280", style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="presentations/synthetic_results", type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Pulling per-depth IoU from W&B for MoE-E runs...")
    pull_moe_per_depth_iou()

    plot_overall_bar(args.output_dir / "iou_summary_bar.png")
    plot_per_depth_heatmap(args.output_dir / "iou_per_depth_heatmap.png")


if __name__ == "__main__":
    main()
