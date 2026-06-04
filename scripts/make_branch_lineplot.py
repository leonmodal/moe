"""Per-depth attn-vs-MLP stacked bar chart for every MoE-Everything run.

One figure, 3 rows × 2 cols:
  rows = branching strategy (top1_explore_decay, fixed_alternating, sampling_entropy)
  cols = task (1L linear-map, 4L cellular automata)

Each cell shows stacked bars per depth: blue (attention) on top of yellow (MLP),
summing to 100% at every depth. Each segment is annotated with its percentage.
Uses the moe_viz palette so it matches `moe_viz/components/branch_view.py`.

Outputs:
  presentations/synthetic_results/branch_patterns.png

Pass `--single-run NAME` to render just one run instead of the whole grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# moe_viz palette
ATTN_COLOR = "#3B82F6"  # blue
MLP_COLOR = "#F59E0B"   # yellow / amber

# Display order. Each entry: (run_name, num_depths, label)
GRID = {
    "1L linear-map (2 depths, step 30000)": [
        ("1L_linearmap_branch_top1_explore_decay_30k", 2, "top1_explore_decay (CE 0.69)"),
        ("1L_linearmap_branch_fixed_alternating_30k", 2, "fixed_alternating (CE 0.69)"),
        ("1L_linearmap_branch_sampling_entropy_30k", 2, "sampling_entropy (CE 0.69)"),
        ("1L_linearmap_branch_fixed_alternating_localbank_30k", 2, "fixed_alt + LOCAL bank (CE 0.66)"),
    ],
    "4L cellular automata (8 depths, step 10000)": [
        ("4L_cellular_branch_top1_explore_decay", 8, "top1_explore_decay (CE 1.31)"),
        ("4L_cellular_branch_fixed_alternating", 8, "fixed_alternating (CE 1.07)"),
        ("4L_cellular_branch_sampling_entropy", 8, "sampling_entropy (CE 0.55)"),
        ("4L_cellular_branch_fixed_alternating_localbank", 8, "fixed_alt + LOCAL bank (CE 0.49)"),
    ],
}


def fetch_attn(api, project: str, run_name: str, num_depths: int) -> list[float] | None:
    runs = list(api.runs(project, filters={"display_name": run_name}))
    if not runs:
        return None
    r = runs[0]
    out = []
    for d in range(num_depths):
        v = r.summary.get(f"train/branch_attn_fraction/depth_{d}")
        if v is None:
            return None
        out.append(float(v))
    return out


def draw_stacked(ax, depths: np.ndarray, attn_frac: list[float], title: str,
                 *, show_xticks: bool = True, show_ylabel: bool = True):
    """Stacked bars with attn on bottom (blue), mlp on top (yellow), labels in % on each segment."""
    attn_arr = np.array(attn_frac)
    mlp_arr = 1.0 - attn_arr
    # Bars
    ax.bar(depths, attn_arr, color=ATTN_COLOR, edgecolor="white",
           linewidth=1.2, label="attention")
    ax.bar(depths, mlp_arr, bottom=attn_arr, color=MLP_COLOR, edgecolor="white",
           linewidth=1.2, label="MLP")
    # Per-segment percentage labels (only annotate the visible side).
    for d, (a, m) in enumerate(zip(attn_arr, mlp_arr)):
        if a >= 0.06:
            ax.text(
                d, a / 2, f"{a*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
            )
        if m >= 0.06:
            ax.text(
                d, a + m / 2, f"{m*100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
            )
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)], fontsize=8)
    if show_xticks:
        ax.set_xticks(depths)
        ax.set_xticklabels([f"d{d}" for d in depths], fontsize=8)
    else:
        ax.set_xticks([])
    if show_ylabel:
        ax.set_ylabel("share of tokens", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="leon-modal-modal/moe-synthetic")
    ap.add_argument(
        "--output",
        default="presentations/synthetic_results/branch_patterns.png",
        type=Path,
    )
    ap.add_argument("--single-run", default=None,
                    help="If given, render just one run as a single figure.")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()

    if args.single_run:
        # Find num_depths from any matching grid entry.
        target = None
        for entries in GRID.values():
            for name, n, label in entries:
                if name == args.single_run:
                    target = (name, n, label)
                    break
            if target:
                break
        if target is None:
            raise SystemExit(f"Unknown run: {args.single_run}")
        attn = fetch_attn(api, args.project, target[0], target[1])
        if attn is None:
            raise SystemExit(f"No data for {target[0]}")
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_stacked(ax, np.arange(target[1]), attn, target[2])
        ax.set_xlabel("depth (0 = nearest input)")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {args.output}")
        return

    # Grid mode: rows = max branching variants per task, cols = tasks (1L, 4L).
    # Use width_ratios so the 1L column (2 depths) is narrower than the 4L
    # column (8 depths) — keeps bar widths visually consistent.
    task_titles = list(GRID.keys())
    n_rows = max(len(GRID[t]) for t in task_titles)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14, 3 * n_rows),
        gridspec_kw={"width_ratios": [1, 4]},
        squeeze=False,
    )

    # Column titles
    axes[0][0].annotate(task_titles[0], xy=(0.5, 1.18), xycoords="axes fraction",
                        ha="center", fontsize=11, fontweight="bold")
    axes[0][1].annotate(task_titles[1], xy=(0.5, 1.18), xycoords="axes fraction",
                        ha="center", fontsize=11, fontweight="bold")

    last_row = n_rows - 1
    for row_idx in range(n_rows):
        for col_idx, task in enumerate(task_titles):
            entries = GRID[task]
            if row_idx >= len(entries):
                axes[row_idx][col_idx].set_axis_off()
                continue
            run_name, num_depths, label = entries[row_idx]
            attn = fetch_attn(api, args.project, run_name, num_depths)
            ax = axes[row_idx][col_idx]
            if attn is None:
                ax.text(0.5, 0.5, f"no data\n{run_name}", ha="center", va="center")
                ax.set_axis_off()
                continue
            depths = np.arange(num_depths)
            draw_stacked(
                ax, depths, attn, label,
                show_xticks=(row_idx == last_row),
                show_ylabel=(col_idx == 0),
            )
            if row_idx == last_row:
                ax.set_xlabel("depth (0 = nearest input)", fontsize=9)

    # Single shared legend on the figure.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ATTN_COLOR, label="attention"),
        plt.Rectangle((0, 0), 1, 1, color=MLP_COLOR, label="MLP"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, frameon=True,
               bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(
        "MoE-Everything: per-depth attention vs MLP routing share — final step",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
