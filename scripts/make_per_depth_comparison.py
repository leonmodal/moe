"""Clean per-depth attention-pattern comparison grid.

One figure per task. Each row = one architecture. Each column = one depth.
Final column = ground-truth attention pattern as reference.

Aesthetic priorities:
  - Clean subplot grid with consistent square cells
  - Crop the heatmap PNGs tightly to just the imshow data area (no per-cell
    axis labels or titles — they appear once at the row/column headers)
  - For MoE-Everything cells, a thin colored stripe above the cell encodes
    the branch routing decision (blue = attention-dominant, yellow = MLP-
    dominant) using the moe_viz palette
  - Empty cells (architecture has fewer depths than max) are shown as a
    light gray placeholder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


# moe_viz palette
ATTN_COLOR = "#3B82F6"
MLP_COLOR = "#F59E0B"
NEUTRAL = "#9CA3AF"
ROW_BG = "#FFFFFF"
EMPTY_BG = "#F3F4F6"
GT_BG = "#FAFAFA"

# Per-depth attention fractions (from W&B `train/branch_attn_fraction/depth_<d>`
# at the final training step). 1.0 = all-attention, 0.0 = all-MLP.
BRANCH_FRACTIONS = {
    "4L_cellular_branch_top1_explore_decay":      [0.76, 0.24, 0.29, 0.23, 0.33, 0.50, 0.99, 0.72],
    "4L_cellular_branch_fixed_alternating":       [1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00],
    "4L_cellular_branch_sampling_entropy":        [0.63, 0.06, 1.00, 0.47, 0.27, 1.00, 1.00, 1.00],
    "4L_cellular_branch_fixed_alternating_localbank": [1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00],
    "1L_linearmap_branch_top1_explore_decay_30k": [0.99, 0.01],
    "1L_linearmap_branch_fixed_alternating_30k":  [1.00, 0.00],
    "1L_linearmap_branch_sampling_entropy_30k":   [0.77, 0.99],
    "1L_linearmap_branch_fixed_alternating_localbank_30k": [1.00, 0.00],
}

# Per-depth attention IoU values (vs ground-truth pattern). Lower = uniform,
# higher = pattern matches. MoE-E values pulled from W&B `attn_eval/depth_<d>/iou_agg_mean`
# at final step; dense/standard_moe values pulled from regen metrics.json.
# Key format: (run_id, num_depths_listed).
IOU_VALUES = {
    # 4L cellular (8 depths for MoE-E, 4 for dense/standard_moe)
    "4L_dense":               [0.026, 0.633, 0.080, 0.435],
    "4L_standardmoe":         [0.485, 0.529, 0.218, 0.013],
    "4L_top1_explore_decay":  [0.00, 0.12, 0.07, 0.30, 0.50, 0.50, 0.35, 0.43],
    "4L_fixed_alternating":   [0.93, 0.93, 0.94, 0.94, 0.49, 0.49, 0.11, 0.11],
    "4L_sampling_entropy":    [0.21, 0.00, 0.93, 0.94, 0.94, 0.49, 0.39, 0.08],
    # Local-bank variant: only attn-active depths produce real IoU (None at
    # MLP-only depths, which we render as "MLP only" cells anyway).
    "4L_fixed_alternating_localbank": [0.722, None, 0.586, None, 0.938, None, 0.009, None],
    # 1L linear-map
    "1L_dense":               [0.150],
    "1L_standardmoe":         [0.131],
    "1L_top1_explore_decay":  [0.108, 0.058],
    "1L_fixed_alternating":   [0.050, 0.050],
    "1L_sampling_entropy":    [0.013, 0.013],
    "1L_fixed_alternating_localbank": [0.144, None],
}

# Overall IoU shown in row labels. For MoE-E fixed_alternating we report the
# mean over the ATTN-active depths only (the MLP-only depths' IoU measures
# phantom attention that the model discards). For the soft-routing variants
# and the baselines we use the mean across all depths.
IOU_OVERALL = {
    "4L_dense": 0.293, "4L_standardmoe": 0.311,
    "4L_top1_explore_decay": 0.284,
    # fixed_alternating: attn-active depths are 0, 2, 4, 6 → mean(0.93, 0.94, 0.49, 0.11) = 0.617
    "4L_fixed_alternating": 0.617,
    "4L_sampling_entropy": 0.496,
    # Local-bank: 4 attn-active depths → mean(0.722, 0.586, 0.938, 0.009)
    "4L_fixed_alternating_localbank": 0.564,
    "1L_dense": 0.150, "1L_standardmoe": 0.131,
    "1L_top1_explore_decay": 0.084,
    # fixed_alternating 1L: attn-active depth is 0 → IoU(d0) = 0.050
    "1L_fixed_alternating": 0.050,
    "1L_sampling_entropy": 0.013,
    "1L_fixed_alternating_localbank": 0.144,
}

# Pixel coords inside the 3-panel heatmap PNGs (1440x480) for cropping just
# the data area of each subplot. Tuned empirically against matplotlib output.
PANEL_WIDTH = 480
# Crop only the imshow region (excludes title, axis labels, ticks).
DATA_X = (96, 458)   # within each panel: start, end
DATA_Y = (82, 425)   # vertical bounds of the data area


def crop_panel(png_path: Path, panel_idx: int) -> np.ndarray | None:
    """Extract the data area of one of the 3 panels (agg=0, best=1, gt=2)."""
    if not png_path.exists():
        return None
    img = np.asarray(Image.open(png_path))
    x_off = panel_idx * PANEL_WIDTH
    x0, x1 = DATA_X
    y0, y1 = DATA_Y
    return img[y0:y1, x_off + x0 : x_off + x1]


def routing_bar(ax, fraction: float, fig) -> None:
    """Draw a proportional blue/yellow routing bar above the cell.

    The bar is `fraction` wide blue (attn) followed by `1-fraction` yellow
    (MLP). Drawn in figure coords just above the axes so it doesn't compete
    with the heatmap content.
    """
    bbox = ax.get_position()
    stripe_h = 0.012  # figure-fraction height
    stripe_y = bbox.y1 + 0.003
    attn_w = bbox.width * fraction
    mlp_w = bbox.width * (1.0 - fraction)
    if attn_w > 0:
        fig.add_artist(
            Rectangle(
                (bbox.x0, stripe_y),
                attn_w, stripe_h,
                facecolor=ATTN_COLOR, edgecolor="none",
                transform=fig.transFigure,
            )
        )
    if mlp_w > 0:
        fig.add_artist(
            Rectangle(
                (bbox.x0 + attn_w, stripe_y),
                mlp_w, stripe_h,
                facecolor=MLP_COLOR, edgecolor="none",
                transform=fig.transFigure,
            )
        )


# group_key -> dict describing rows and source heatmap PNGs
ROOT = Path("presentations/synthetic_results")

GROUPS = {
    "1L_linearmap": {
        "title": "Per-depth attention patterns — 1L linear-map (step 30000)",
        "max_depths": 2,
        "rows": [
            {
                "label": "Dense",
                "sublabel": "CE 0.38",
                "branch_key": None,
                "iou_key": "1L_dense",
                "panels": ["raw_heatmaps/1L_dense_30k_step30000_depth00.png"],
            },
            {
                "label": "Standard MoE",
                "sublabel": "CE 0.36",
                "branch_key": None,
                "iou_key": "1L_standardmoe",
                "panels": ["raw_heatmaps/1L_standardmoe_30k_step30000_depth00.png"],
            },
            {
                "label": "MoE-Everything",
                "sublabel": "top1_explore_decay  ·  CE 0.69",
                "branch_key": "1L_linearmap_branch_top1_explore_decay_30k",
                "iou_key": "1L_top1_explore_decay",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/1L_linearmap_branch_top1_explore_decay_30k/step_00030000/depth_{d:02d}.png"
                    for d in range(2)
                ],
            },
            {
                "label": "MoE-Everything",
                "sublabel": "fixed_alternating  ·  CE 0.69",
                "branch_key": "1L_linearmap_branch_fixed_alternating_30k",
                "iou_key": "1L_fixed_alternating",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/1L_linearmap_branch_fixed_alternating_30k/step_00030000/depth_{d:02d}.png"
                    for d in range(2)
                ],
            },
            {
                "label": "MoE-Everything",
                "sublabel": "sampling_entropy  ·  CE 0.69",
                "branch_key": "1L_linearmap_branch_sampling_entropy_30k",
                "iou_key": "1L_sampling_entropy",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/1L_linearmap_branch_sampling_entropy_30k/step_00030000/depth_{d:02d}.png"
                    for d in range(2)
                ],
            },
            {
                "label": "MoE-E LOCAL bank",
                "sublabel": "fixed_alternating  ·  CE 0.66  ← emerged",
                "branch_key": "1L_linearmap_branch_fixed_alternating_localbank_30k",
                "iou_key": "1L_fixed_alternating_localbank",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/1L_linearmap_branch_fixed_alternating_localbank_30k/step_00030000/depth_{d:02d}.png"
                    for d in range(2)
                ],
            },
        ],
    },
    "4L_cellular": {
        "title": "Per-depth attention patterns — 4L cellular automata (step 10000)",
        "max_depths": 8,
        "rows": [
            {
                "label": "Dense",
                "sublabel": "CE 0.25",
                "branch_key": None,
                "iou_key": "4L_dense",
                "panels": [
                    f"raw_heatmaps/4L_dense_step10000_depth{d:02d}.png" for d in range(4)
                ],
            },
            {
                "label": "Standard MoE",
                "sublabel": "CE 0.25",
                "branch_key": None,
                "iou_key": "4L_standardmoe",
                "panels": [
                    f"raw_heatmaps/4L_standardmoe_step10000_depth{d:02d}.png" for d in range(4)
                ],
            },
            {
                "label": "MoE-Everything",
                "sublabel": "top1_explore_decay  ·  CE 1.31",
                "branch_key": "4L_cellular_branch_top1_explore_decay",
                "iou_key": "4L_top1_explore_decay",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/4L_cellular_branch_top1_explore_decay/step_00010000/depth_{d:02d}.png"
                    for d in range(8)
                ],
            },
            {
                "label": "MoE-Everything",
                "sublabel": "fixed_alternating  ·  CE 1.07",
                "branch_key": "4L_cellular_branch_fixed_alternating",
                "iou_key": "4L_fixed_alternating",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/4L_cellular_branch_fixed_alternating/step_00010000/depth_{d:02d}.png"
                    for d in range(8)
                ],
            },
            {
                "label": "MoE-Everything",
                "sublabel": "sampling_entropy  ·  CE 0.55",
                "branch_key": "4L_cellular_branch_sampling_entropy",
                "iou_key": "4L_sampling_entropy",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/4L_cellular_branch_sampling_entropy/step_00010000/depth_{d:02d}.png"
                    for d in range(8)
                ],
            },
            {
                "label": "MoE-E LOCAL bank",
                "sublabel": "fixed_alternating  ·  CE 0.49  ← best MoE-E",
                "branch_key": "4L_cellular_branch_fixed_alternating_localbank",
                "iou_key": "4L_fixed_alternating_localbank",
                "panels": [
                    f"raw_heatmaps/_moe_pulls/4L_cellular_branch_fixed_alternating_localbank/step_00010000/depth_{d:02d}.png"
                    for d in range(8)
                ],
            },
        ],
    },
}


def render(group_key: str) -> None:
    spec = GROUPS[group_key]
    rows = spec["rows"]
    max_d = spec["max_depths"]
    n_rows = len(rows)
    n_cols = max_d + 1  # +1 for GT column

    # Per-cell aspect ratio. The 4L grid has 8+1 columns so we'd get a very
    # wide figure if each cell were 1.7 in; tighten it slightly for that case.
    cell_w = 1.6 if max_d <= 4 else 1.4
    cell_h = cell_w
    left_margin = 2.0
    top_margin = 1.3
    right_margin = 0.4
    # Extra room at the bottom for the inline IoU labels under each cell.
    bottom_margin = 0.55
    fig_w = left_margin + cell_w * n_cols + right_margin
    fig_h = top_margin + cell_h * n_rows + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=left_margin / fig_w,
        right=1 - right_margin / fig_w,
        top=1 - top_margin / fig_h,
        bottom=bottom_margin / fig_h,
        # Bigger hspace so the inline IoU label under one row doesn't crowd
        # the next row's routing stripe.
        hspace=0.45,
        wspace=0.08,
    )

    for r, row in enumerate(rows):
        fractions = BRANCH_FRACTIONS.get(row["branch_key"], []) if row["branch_key"] else []
        iou_key = row.get("iou_key")
        ious = IOU_VALUES.get(iou_key, []) if iou_key else []
        overall_iou = IOU_OVERALL.get(iou_key) if iou_key else None

        # Render each depth column for this row.
        for d in range(max_d):
            ax = fig.add_subplot(gs[r, d])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor(ROW_BG)
            # Set frame styling
            for spine in ax.spines.values():
                spine.set_edgecolor("#E5E7EB")
                spine.set_linewidth(0.8)

            # Determine whether this cell is "MLP only" under hard branching
            # (i.e., the branch router routes ≈0% of tokens to attention at
            # this depth, so the captured attention computation is discarded
            # by the model). Threshold at 0.02 to catch true MLP-only depths
            # without grouping soft "MLP-leaning" depths with them.
            is_mlp_only = (
                fractions
                and d < len(fractions)
                and fractions[d] < 0.02
            )

            if d >= len(row["panels"]):
                # Architecture has fewer depths than max_d → light gray.
                ax.set_facecolor(EMPTY_BG)
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        color=NEUTRAL, fontsize=14, transform=ax.transAxes)
            elif is_mlp_only:
                # MLP-only depth under hard branching. The captured attention
                # is "phantom" (computed then multiplied by 0), so we suppress
                # both the heatmap and the IoU label below.
                ax.set_facecolor(EMPTY_BG)
                ax.text(
                    0.5, 0.55, "MLP only",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="#6B7280",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.5, 0.35, "(attention discarded)",
                    ha="center", va="center", fontsize=8,
                    color="#9CA3AF", style="italic",
                    transform=ax.transAxes,
                )
            else:
                img = crop_panel(ROOT / row["panels"][d], panel_idx=0)
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            color=NEUTRAL, fontsize=9, transform=ax.transAxes)

            # Proportional routing bar above each MoE-E cell.
            if fractions and d < len(fractions):
                routing_bar(ax, fractions[d], fig)

            # Inline IoU label below each cell (only where we have a value).
            # Suppress at MLP-only cells: the IoU there is measuring phantom
            # attention that the model discards.
            if ious and d < len(ious) and ious[d] is not None and not is_mlp_only:
                bbox = ax.get_position()
                label_x = bbox.x0 + bbox.width / 2
                label_y = bbox.y0 - 0.012
                iou_v = ious[d]
                # Color-grade the label: green for high IoU, gray for low.
                if iou_v >= 0.5:
                    color = "#15803D"  # green
                elif iou_v >= 0.25:
                    color = "#CA8A04"  # amber
                else:
                    color = "#6B7280"  # gray
                fig.text(
                    label_x, label_y,
                    f"IoU {iou_v:.2f}",
                    ha="center", va="top", fontsize=8.5,
                    color=color, fontweight="bold",
                )

            # Column header on top row.
            if r == 0:
                ax.set_title(f"depth {d}", fontsize=10, pad=10, color="#374151")

        # Ground-truth column (rightmost). Pull the GT panel from any
        # available source PNG for this row.
        ax_gt = fig.add_subplot(gs[r, n_cols - 1])
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        ax_gt.set_facecolor(GT_BG)
        for spine in ax_gt.spines.values():
            spine.set_edgecolor("#D1D5DB")
            spine.set_linewidth(1.0)
        gt_img = None
        for p in row["panels"]:
            cand = crop_panel(ROOT / p, panel_idx=2)
            if cand is not None:
                gt_img = cand
                break
        if gt_img is not None:
            ax_gt.imshow(gt_img)
        if r == 0:
            ax_gt.set_title("ground truth", fontsize=10, pad=10, color="#374151")

        # Row label on the LEFT side of the row (architecture name +
        # sublabel/CE). Positioned in figure coords so it doesn't collide
        # with the first axes.
        # Use the first axes' bbox to align.
        first_ax = fig.axes[r * n_cols]
        bbox = first_ax.get_position()
        y_center = bbox.y0 + bbox.height / 2
        fig.text(
            bbox.x0 - 0.015, y_center + 0.012,
            row["label"],
            ha="right", va="center",
            fontsize=11, fontweight="bold", color="#111827",
        )
        sublabel = row["sublabel"]
        if overall_iou is not None:
            sublabel = f"{sublabel}\noverall IoU {overall_iou:.2f}"
        fig.text(
            bbox.x0 - 0.015, y_center - 0.022,
            sublabel,
            ha="right", va="center",
            fontsize=9, color="#6B7280",
        )

    # Figure title + subtitle/legend block centered up top.
    title_y = 1.0 - 0.30 / fig_h
    fig.text(0.5, title_y, spec["title"],
             ha="center", va="top", fontsize=14, fontweight="bold", color="#111827")
    # Two-line caption: routing-stripe legend on line 1, MLP-only explanation on line 2.
    line1_y = title_y - 0.30 / fig_h
    line2_y = line1_y - 0.22 / fig_h
    cap_x = 0.5
    fig.text(
        cap_x - 0.18, line1_y,
        "stripe above each cell shows depth's branch-routing split:",
        ha="right", va="center", fontsize=9, color="#4B5563",
    )
    fig.add_artist(
        Rectangle(
            (cap_x - 0.17, line1_y - 0.008),
            0.07, 0.016, facecolor=ATTN_COLOR, edgecolor="none",
            transform=fig.transFigure,
        )
    )
    fig.add_artist(
        Rectangle(
            (cap_x - 0.10, line1_y - 0.008),
            0.04, 0.016, facecolor=MLP_COLOR, edgecolor="none",
            transform=fig.transFigure,
        )
    )
    fig.text(
        cap_x - 0.05, line1_y,
        "= 70% attention / 30% MLP",
        ha="left", va="center", fontsize=9, color="#4B5563",
    )
    fig.text(
        cap_x, line2_y,
        "gray “MLP only” cells: branch router routes 100% of tokens to MLP at that depth, "
        "so the attention there is discarded by the model.",
        ha="center", va="center", fontsize=8, color="#6B7280", style="italic",
    )

    out_path = ROOT / f"per_depth_grid_{group_key}.png"
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["all", "1L_linearmap", "4L_cellular"])
    args = ap.parse_args()
    targets = list(GROUPS.keys()) if args.group == "all" else [args.group]
    for g in targets:
        render(g)


if __name__ == "__main__":
    main()
