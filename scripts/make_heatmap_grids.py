"""Compose comparison-grid figures from the per-run heatmap PNGs.

Generates two figures:
  - heatmap_grid_1L_linearmap.png   — all 5 1L runs side by side
  - heatmap_grid_4L_cellular.png    — all 5 4L runs side by side (one depth each)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


GROUPS = {
    "1L_linearmap": [
        ("Dense (CE 0.38)", "1L_dense_30k_step30000_depth00.png"),
        ("Standard MoE (CE 0.36)", "1L_standardmoe_30k_step30000_depth00.png"),
        ("MoE-E sampling_entropy (CE 0.69)", "1L_moe_sampling_30k_step30000.png"),
        # The two other MoE-E 1L variants stuck at chance produce essentially
        # identical heatmaps; omit them for clarity. They can be viewed
        # individually under heatmaps/ if needed.
    ],
    "4L_cellular": [
        ("Dense depth-1 (CE 0.25)", "4L_dense_step10000_depth01.png"),
        ("Standard MoE depth-1 (CE 0.25)", "4L_standardmoe_step10000_depth01.png"),
        ("MoE-E top1_explore_decay (CE 1.31)", "4L_top1_explore_decay_step10000.png"),
        ("MoE-E fixed_alternating (CE 1.07)", "4L_fixed_alternating_step10000.png"),
        ("MoE-E sampling_entropy (CE 0.55)", "4L_sampling_entropy_step10000.png"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap-dir", default="presentations/synthetic_results/heatmaps", type=Path)
    ap.add_argument("--output-dir", default="presentations/synthetic_results", type=Path)
    args = ap.parse_args()

    for group_name, entries in GROUPS.items():
        n = len(entries)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]
        for ax, (title, fname) in zip(axes, entries):
            path = args.heatmap_dir / fname
            if not path.exists():
                ax.text(0.5, 0.5, f"missing: {fname}", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            img = np.asarray(Image.open(path))
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

        task_label = "1L linear-map (step 30000)" if "1L" in group_name else "4L cellular automata (step 10000)"
        fig.suptitle(
            f"{task_label} — attention patterns per architecture\n"
            "Each row: agg routing-weighted | best-match per row | ground truth",
            fontsize=12,
        )
        fig.tight_layout()
        out_path = args.output_dir / f"heatmap_grid_{group_name}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
