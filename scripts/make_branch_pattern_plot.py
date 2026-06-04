"""Visualize the final per-depth branch routing pattern across MoE-Everything runs.

For each depth, `train/branch_attn_fraction/depth_<d>` is the batch-averaged
fraction of tokens that went through the attention branch at that depth.
0.5 = even split; 1.0 = all-attention; 0.0 = all-MLP. Under hard branching
each token picks one modality, so the per-depth fraction is the share of
tokens routed to attention at that depth (e.g., 0.94 means most tokens used
attention there).

Outputs:
  presentations/synthetic_results/branch_routing_patterns.png  ← heatmap grid
  presentations/synthetic_results/branch_routing_patterns.md   ← text strings
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Runs grouped by task (label, num_depths).
RUNS = {
    "1L linear-map (2 depths)": [
        ("MoE-E top1_explore_decay (30k)", "1L_linearmap_branch_top1_explore_decay_30k", 2),
        ("MoE-E fixed_alternating (30k)", "1L_linearmap_branch_fixed_alternating_30k", 2),
        ("MoE-E sampling_entropy (30k)", "1L_linearmap_branch_sampling_entropy_30k", 2),
    ],
    "4L cellular (8 depths)": [
        ("MoE-E top1_explore_decay", "4L_cellular_branch_top1_explore_decay", 8),
        ("MoE-E fixed_alternating", "4L_cellular_branch_fixed_alternating", 8),
        ("MoE-E sampling_entropy", "4L_cellular_branch_sampling_entropy", 8),
    ],
}


def pattern_string(fractions: list[float]) -> str:
    """Render the per-depth pattern as a string of A/M (or 'a'/'m' for soft)."""
    out = []
    for f in fractions:
        if f >= 0.85:
            out.append("A")
        elif f <= 0.15:
            out.append("M")
        elif f >= 0.60:
            out.append("a")  # attention-leaning
        elif f <= 0.40:
            out.append("m")  # MLP-leaning
        else:
            out.append(".")  # mixed
    return "".join(out)


def fetch_fractions(api, project: str, run_name: str, num_depths: int):
    runs = list(api.runs(project, filters={"display_name": run_name}))
    if not runs:
        return None
    r = runs[0]
    fracs = []
    for d in range(num_depths):
        key = f"train/branch_attn_fraction/depth_{d}"
        val = r.summary.get(key)
        if val is None:
            return None
        fracs.append(float(val))
    return fracs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="leon-modal-modal/moe-synthetic")
    ap.add_argument(
        "--output-dir",
        default="presentations/synthetic_results",
        type=Path,
    )
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build figure with one subplot per task.
    n_tasks = len(RUNS)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(11, 2.4 * n_tasks))
    if n_tasks == 1:
        axes = [axes]

    md_lines = ["# Final per-depth branch routing patterns", ""]
    md_lines.append(
        "Each depth in the MoE-Everything stack picks between attention and MLP per token "
        "via a learned branch router. The numbers below are the batch-averaged fraction of "
        "tokens that went through the **attention** branch at each depth (1.0 = all-attn, "
        "0.0 = all-mlp). The compact pattern string uses **A** if ≥0.85, **M** if ≤0.15, "
        "**a/m** for moderate lean (0.6/0.4 thresholds), and **.** for ~50/50."
    )
    md_lines.append("")
    md_lines.append("Convention: depth 0 is the first depth (closest to input), depth N-1 is the last.")
    md_lines.append("")

    for ax, (task_label, runs) in zip(axes, RUNS.items()):
        labels = []
        matrix = []
        for run_label, run_name, num_depths in runs:
            fracs = fetch_fractions(api, args.project, run_name, num_depths)
            if fracs is None:
                continue
            labels.append(run_label)
            matrix.append(fracs)
            md_lines.append(f"## {task_label}")
            md_lines.append("")
            md_lines.append(f"- **{run_label}**:")
            md_lines.append(f"  - pattern: `{pattern_string(fracs)}`")
            md_lines.append(
                "  - per-depth attn fraction: ["
                + ", ".join(f"{f:.3f}" for f in fracs)
                + "]"
            )
            md_lines.append("")

        if not matrix:
            ax.set_visible(False)
            continue

        mat = np.array(matrix)
        im = ax.imshow(
            mat,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels([f"d{d}" for d in range(mat.shape[1])])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(task_label, fontsize=11)
        ax.set_xlabel("depth (0 = nearest input)")
        # Annotate cells with letter + fraction
        for r_idx, row in enumerate(mat):
            for c_idx, frac in enumerate(row):
                letter = pattern_string([frac])
                txt_color = "white" if (frac > 0.7 or frac < 0.3) else "black"
                ax.text(
                    c_idx,
                    r_idx,
                    f"{letter}\n{frac:.2f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=8,
                )
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("attn fraction (red=attn, blue=MLP)", fontsize=8)

    fig.suptitle("MoE-Everything: per-depth branch routing patterns at final step", fontsize=12)
    fig.tight_layout()
    out_png = args.output_dir / "branch_routing_patterns.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    (args.output_dir / "branch_routing_patterns.md").write_text("\n".join(md_lines))
    print(f"wrote {args.output_dir / 'branch_routing_patterns.md'}")


if __name__ == "__main__":
    main()
