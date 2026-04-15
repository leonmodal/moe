"""
Generate routing analysis plots from a snapshot dict.

Folder structure:
  routing_logs/
    step_00000050/
      snapshot.json
      branch_ratios.png
      branch_histogram.png
      mlp/
        expert_histogram.png
        per_layer_expert_histograms.png
      q/
        expert_histogram.png
        per_layer_expert_histograms.png
      k/
        expert_histogram.png
        per_layer_expert_histograms.png
      ...
      attn_norm/
        expert_histogram.png
      mlp_norm/
        expert_histogram.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_routing_snapshot(snapshot: dict, step_dir: str, step: int, bias_data: dict | None = None) -> None:
    """Generate all plots for one routing snapshot into step_dir."""

    # Branch plots (top-level in step folder)
    branch = snapshot.get("branch")
    branch_bias = bias_data.get("branch_bias") if bias_data else None
    if branch:
        _plot_branch_routing(branch, step_dir, step, branch_bias=branch_bias)
        _plot_branch_histogram(branch, step_dir, step)

    # Global expert biases
    expert_biases = bias_data.get("expert_biases") if bias_data else None
    if expert_biases:
        _plot_global_expert_biases(expert_biases, step_dir, step)

    # Global per-projection-type histograms (q, k, v, o, mlp aggregated across all heads)
    global_proj = snapshot.get("global_projection_pools", {})
    if global_proj:
        _plot_global_projection_histograms(global_proj, step_dir, step)

    # MLP expert histogram
    mlp_layers = snapshot.get("layers", {})
    global_pool = snapshot.get("global_pool")
    if global_pool is not None or mlp_layers:
        sub = os.path.join(step_dir, "mlp")
        os.makedirs(sub, exist_ok=True)
        if global_pool is not None:
            _plot_expert_histogram(global_pool, sub, step, name="MLP")
        if mlp_layers:
            _plot_per_layer_expert_histograms(mlp_layers, sub, step, name="MLP")

    # Attention expert histograms (q, k, v, o, or attn)
    for router_name, router_snapshot in sorted(snapshot.get("attention", {}).items()):
        router_layers = router_snapshot.get("layers", {})
        router_pool = router_snapshot.get("global_pool")
        if router_pool is not None or router_layers:
            # Structure: projection/head/ (e.g. k/h0/) or just the name
            parts = router_name.split("_", 1)
            if len(parts) == 2:
                sub = os.path.join(step_dir, parts[0], parts[1])
            else:
                sub = os.path.join(step_dir, router_name)
            os.makedirs(sub, exist_ok=True)
            if router_pool is not None:
                _plot_expert_histogram(router_pool, sub, step, name=router_name.upper())
            if router_layers:
                _plot_per_layer_expert_histograms(router_layers, sub, step, name=router_name.upper())

    # Norm expert histograms (attn_norm, mlp_norm)
    for norm_name, norm_data in sorted(snapshot.get("norms", {}).items()):
        sub = os.path.join(step_dir, norm_name)
        os.makedirs(sub, exist_ok=True)
        _plot_expert_histogram(norm_data, sub, step, name=norm_name)

    plt.close("all")


def _plot_branch_routing(branch: dict, step_dir: str, step: int, branch_bias: dict | None = None) -> None:
    """Combined branch routing plot: fraction, ratio, sigmoid weights, and bias."""
    layers = branch.get("layers", {})
    if not layers:
        return

    sorted_keys = sorted(layers.keys(), key=int)
    depth_idxs = [int(k) for k in sorted_keys]
    attn = np.array([layers[k]["attn_frac"] for k in sorted_keys])
    mlp = np.array([layers[k]["mlp_frac"] for k in sorted_keys])
    ratio = np.array([layers[k]["attn_to_mlp_ratio"] for k in sorted_keys])
    total = branch.get("total", {})

    has_weights = "mean_attn_weight" in layers.get(sorted_keys[0], {})
    has_bias = branch_bias is not None and "attn" in branch_bias

    n_rows = 2
    if has_weights:
        n_rows += 1
    if has_bias:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.2 * n_rows), sharex=True)
    row = 0

    # Row 1: Token fraction
    axes[row].plot(depth_idxs, attn, marker="o", label="Attention", color="#1f77b4", markersize=4)
    axes[row].plot(depth_idxs, mlp, marker="o", label="MLP", color="#ff7f0e", markersize=4)
    axes[row].set_ylabel("Token Fraction")
    axes[row].set_title(f"Branch Routing by Depth (step {step})")
    axes[row].set_ylim(0.0, 1.0)
    axes[row].legend(fontsize=8)
    axes[row].grid(alpha=0.2)
    row += 1

    # Row 2: Ratio
    axes[row].plot(depth_idxs, ratio, marker="o", color="#2c3e50", markersize=4)
    axes[row].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[row].set_ylabel("Attn / MLP")
    axes[row].set_title(
        f"Total: attn={total.get('attn_frac', 0.0):.3f}, "
        f"mlp={total.get('mlp_frac', 0.0):.3f}, "
        f"attn/mlp={total.get('attn_to_mlp_ratio', 0.0):.3f}"
    )
    axes[row].grid(alpha=0.2)
    row += 1

    # Row 3: Sigmoid weights (if available)
    if has_weights:
        attn_w = np.array([layers[k].get("mean_attn_weight", 0) for k in sorted_keys])
        mlp_w = np.array([layers[k].get("mean_mlp_weight", 0) for k in sorted_keys])
        axes[row].plot(depth_idxs, attn_w, marker="o", label="Attn weight", color="#1f77b4", markersize=4)
        axes[row].plot(depth_idxs, mlp_w, marker="o", label="MLP weight", color="#ff7f0e", markersize=4)
        axes[row].set_ylabel("Sigmoid Weight")
        axes[row].set_title(
            f"Mean sigmoid weight (attn={total.get('mean_attn_weight', 0):.3f}, "
            f"mlp={total.get('mean_mlp_weight', 0):.3f})"
        )
        axes[row].set_ylim(0.0, 1.0)
        axes[row].legend(fontsize=8)
        axes[row].grid(alpha=0.2)
        row += 1

    # Row 4: Branch bias (if available)
    if has_bias:
        attn_b = np.array(branch_bias["attn"])
        mlp_b = np.array(branch_bias["mlp"])
        axes[row].plot(depth_idxs, attn_b, marker="o", label="Attn bias", color="#1f77b4", markersize=4)
        axes[row].plot(depth_idxs, mlp_b, marker="o", label="MLP bias", color="#ff7f0e", markersize=4)
        axes[row].axhline(0, color="black", linewidth=0.5, linestyle="--")
        axes[row].set_ylabel("Bias Value")
        axes[row].set_title("Branch Router Bias by Depth (per-depth DeepSeek bias)")
        axes[row].legend(fontsize=8)
        axes[row].grid(alpha=0.2)
        row += 1

    axes[-1].set_xlabel("Depth")
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "branch_routing.png"), dpi=100)
    plt.close(fig)


def _plot_branch_histogram(branch: dict, step_dir: str, step: int) -> None:
    """Histogram of per-depth attention fractions."""
    layers = branch.get("layers", {})
    if not layers:
        return

    depth_idxs = sorted(layers.keys(), key=int)
    attn_fracs = [layers[i]["attn_frac"] for i in depth_idxs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(attn_fracs, bins=20, range=(0, 1), color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Attention Probability")
    ax.set_ylabel("Count (depths)")
    ax.set_title(f"Branch Probability Distribution (step {step})")
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "branch_histogram.png"), dpi=100)
    plt.close(fig)


def _plot_expert_histogram(pool: dict, sub_dir: str, step: int, name: str) -> None:
    """Bar chart of expert token fractions for a single pool."""
    fracs = np.array(pool["token_fracs"])
    num_experts = len(fracs)
    if num_experts == 0:
        return
    ideal = 1.0 / num_experts
    active = int(np.sum(fracs > 0))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(num_experts), fracs, color="#3498db", width=0.8)
    ax.axhline(ideal, color="black", linestyle="--", linewidth=1, label=f"ideal={ideal:.4f}")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Token Fraction")
    ax.set_title(f"{name} Expert Usage (step {step}, {active}/{num_experts} active)")
    ax.legend()

    xtick_step = max(1, num_experts // 20)
    ax.set_xticks(range(0, num_experts, xtick_step))

    fig.tight_layout()
    fig.savefig(os.path.join(sub_dir, "expert_histogram.png"), dpi=100)
    plt.close(fig)


def _plot_per_layer_expert_histograms(layers: dict, sub_dir: str, step: int, name: str) -> None:
    """Grid of per-layer expert usage histograms for one pool family."""
    if not layers:
        return

    layer_keys = sorted(layers.keys(), key=int)
    num_layers = len(layer_keys)
    cols = min(4, num_layers)
    rows = int(np.ceil(num_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for plot_idx, layer_key in enumerate(layer_keys):
        ax = axes.flat[plot_idx]
        ax.set_visible(True)

        layer = layers[layer_key]
        fracs = np.array(layer["token_fracs"], dtype=float)
        num_experts = len(fracs)
        if num_experts == 0:
            ax.set_title(f"Layer {int(layer_key):02d}")
            ax.text(0.5, 0.5, "No experts", ha="center", va="center", transform=ax.transAxes)
            continue

        ideal = 1.0 / num_experts
        active = int(np.sum(fracs > 0))
        ax.bar(range(num_experts), fracs, color="#5dade2", width=0.8)
        ax.axhline(ideal, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Layer {int(layer_key):02d} ({active}/{num_experts} active)")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Frac")
        xtick_step = max(1, num_experts // 8)
        ax.set_xticks(range(0, num_experts, xtick_step))

    fig.suptitle(f"{name} Per-Layer Expert Usage (step {step})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(sub_dir, "per_layer_expert_histograms.png"), dpi=100)
    plt.close(fig)


def plot_expert_heatmaps(
    heatmap_data: dict[str, list[list[float]]],
    step_dir: str,
    step: int,
) -> None:
    """Plot expert-activation heatmaps from ``RoutingStats.expert_heatmap_data()``.

    For each router family (e.g. ``mlp``, ``qkvo_h0``), produces a heatmap where
    the x-axis is the expert index and y-axis is the layer/depth index.  The cell
    colour encodes the fraction of tokens routed to that expert at that depth.
    """
    if not heatmap_data:
        return

    for family, rows in heatmap_data.items():
        mat = np.array(rows, dtype=float)  # (num_depths, num_experts)
        if mat.size == 0:
            continue

        num_depths, num_experts = mat.shape
        fig_w = max(6, num_experts * 0.25 + 2)
        fig_h = max(4, num_depths * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Token Fraction")

        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Depth")
        ax.set_title(f"{family} Expert Activation Heatmap (step {step})")

        # Tick labels
        xtick_step = max(1, num_experts // 20)
        ax.set_xticks(range(0, num_experts, xtick_step))
        ax.set_yticks(range(num_depths))

        fig.tight_layout()
        # Structure: projection/head/ (e.g. k/h0/) or just mlp/
        parts = family.split("_", 1)
        if len(parts) == 2:
            family_dir = os.path.join(step_dir, parts[0], parts[1])
        else:
            family_dir = os.path.join(step_dir, family)
        os.makedirs(family_dir, exist_ok=True)
        fig.savefig(os.path.join(family_dir, "expert_heatmap.png"), dpi=100)
        plt.close(fig)


def _plot_global_expert_biases(expert_biases: dict, step_dir: str, step: int) -> None:
    """Plot global expert bias bar charts for each projection type."""
    proj_names = sorted(expert_biases.keys())
    if not proj_names:
        return

    n = len(proj_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, name in enumerate(proj_names):
        ax = axes[0, i]
        bias = expert_biases[name]
        num_experts = len(bias)
        colors = ["#e74c3c" if b < 0 else "#2ecc71" for b in bias]
        ax.bar(range(num_experts), bias, color=colors, width=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Bias Value")
        ax.set_title(f"{name} Bias ({num_experts} experts)")
        xtick_step = max(1, num_experts // 10)
        ax.set_xticks(range(0, num_experts, xtick_step))

    fig.suptitle(f"Global Expert Biases (step {step})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(step_dir, "global_expert_biases.png"), dpi=100)
    plt.close(fig)


def _plot_global_projection_histograms(
    global_proj: dict[str, dict],
    step_dir: str,
    step: int,
) -> None:
    """Plot one global expert usage histogram per projection type (q, k, v, o, mlp).

    Each bar chart shows expert token fractions aggregated across ALL heads and depths.
    All subplots in one figure for easy comparison.
    """
    proj_names = sorted(global_proj.keys())
    if not proj_names:
        return

    n = len(proj_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, name in enumerate(proj_names):
        ax = axes[0, i]
        fracs = np.array(global_proj[name]["token_fracs"])
        num_experts = len(fracs)
        if num_experts == 0:
            continue
        ideal = 1.0 / num_experts
        active = int(np.sum(fracs > 0))
        ax.bar(range(num_experts), fracs, color="#3498db", width=0.8)
        ax.axhline(ideal, color="black", linestyle="--", linewidth=1, label=f"ideal={ideal:.4f}")
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Token Fraction")
        ax.set_title(f"{name.upper()} Global ({active}/{num_experts} active)")
        ax.legend(fontsize=8)
        xtick_step = max(1, num_experts // 10)
        ax.set_xticks(range(0, num_experts, xtick_step))

    fig.suptitle(f"Global Expert Usage by Projection Type (step {step})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(step_dir, "global_projection_histograms.png"), dpi=100)
    plt.close(fig)
