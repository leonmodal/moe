"""
Generate routing analysis plots from a snapshot dict.

Called during training right after saving the JSON. Produces PNGs in the same
directory as the JSON file.

Plots generated:
  1. Global token fraction bar chart (global MoE only)
  2. Per-layer token fraction heatmap (layers x experts)
  3. Layer usage vs token fraction scatter (global MoE only)
"""
import os

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np


def plot_routing_snapshot(snapshot: dict, output_dir: str, step: int) -> None:
    """Generate all plots for one routing snapshot.

    Args:
        snapshot: The dict that was saved as JSON (step, layers, global_pool).
        output_dir: Directory to save PNGs (same as the routing_logs dir).
        step: Training step (used in filenames and titles).
    """
    prefix = os.path.join(output_dir, f"step_{step:08d}")

    layers = snapshot["layers"]
    global_pool = snapshot["global_pool"]

    # --- Per-layer token fraction heatmap ---
    _plot_layer_heatmap(layers, prefix, step)

    # --- Global MoE plots ---
    if global_pool is not None:
        _plot_global_fracs(global_pool, prefix, step)
        _plot_usage_vs_tokens(global_pool, prefix, step)

    plt.close("all")


def _plot_layer_heatmap(layers: dict, prefix: str, step: int) -> None:
    """Heatmap: layers (y) x experts (x), color = token fraction."""
    sorted_idxs = sorted(layers.keys(), key=int)
    if not sorted_idxs:
        return

    num_experts = len(layers[sorted_idxs[0]]["token_fracs"])
    num_layers = len(sorted_idxs)
    data = np.array([layers[i]["token_fracs"] for i in sorted_idxs])  # (L, E)
    ideal = 1.0 / num_experts

    fig_width = max(8, num_experts * 0.12)
    fig_height = max(4, num_layers * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Layer")
    ax.set_title(f"Per-Layer Expert Token Fraction (step {step}, ideal={ideal:.4f})")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(i) for i in sorted_idxs], fontsize=max(6, 10 - num_layers // 10))

    # Sparse x-ticks for many experts
    xtick_step = max(1, num_experts // 20)
    ax.set_xticks(range(0, num_experts, xtick_step))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Token Fraction")

    fig.tight_layout()
    fig.savefig(f"{prefix}_layer_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_global_fracs(global_pool: dict, prefix: str, step: int) -> None:
    """Bar chart: per-expert token fraction across the shared pool."""
    fracs = np.array(global_pool["token_fracs"])
    num_experts = len(fracs)
    ideal = 1.0 / num_experts

    fig_width = max(8, num_experts * 0.08)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    colors = ["#e74c3c" if f > ideal * 2 else "#3498db" for f in fracs]
    ax.bar(range(num_experts), fracs, color=colors, width=0.8)
    ax.axhline(ideal, color="black", linestyle="--", linewidth=1, label=f"ideal={ideal:.4f}")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Token Fraction")
    ax.set_title(f"Global Pool Token Fraction (step {step})")
    ax.legend()

    xtick_step = max(1, num_experts // 20)
    ax.set_xticks(range(0, num_experts, xtick_step))

    fig.tight_layout()
    fig.savefig(f"{prefix}_global_fracs.png", dpi=120)
    plt.close(fig)


def _plot_usage_vs_tokens(global_pool: dict, prefix: str, step: int) -> None:
    """Scatter: x = number of layers using this expert, y = token fraction."""
    fracs = np.array(global_pool["token_fracs"])
    usage = np.array(global_pool["layer_usage_count"])
    num_layers = global_pool["num_layers"]
    num_experts = len(fracs)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(usage, fracs, alpha=0.6, s=20, c="#2c3e50")

    # Trend line
    if len(usage) > 2 and np.std(usage) > 0:
        z = np.polyfit(usage, fracs, 1)
        x_line = np.linspace(usage.min(), usage.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5,
                label=f"trend (slope={z[0]:.2e})")
        ax.legend()

    ideal = 1.0 / num_experts
    ax.axhline(ideal, color="gray", linestyle=":", linewidth=1, label=f"ideal frac={ideal:.4f}")

    ax.set_xlabel(f"Layers Using Expert (out of {num_layers})")
    ax.set_ylabel("Token Fraction (global pool)")
    ax.set_title(f"Layer Usage vs Token Share (step {step}, {num_experts} experts)")

    fig.tight_layout()
    fig.savefig(f"{prefix}_usage_vs_tokens.png", dpi=120)
    plt.close(fig)
