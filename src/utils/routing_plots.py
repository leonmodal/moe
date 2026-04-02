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


def plot_routing_snapshot(snapshot: dict, step_dir: str, step: int) -> None:
    """Generate all plots for one routing snapshot into step_dir."""

    # Branch plots (top-level in step folder)
    branch = snapshot.get("branch")
    if branch:
        _plot_branch_ratios(branch, step_dir, step)
        _plot_branch_histogram(branch, step_dir, step)

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


def _plot_branch_ratios(branch: dict, step_dir: str, step: int) -> None:
    """Plot attention-vs-MLP branch ratios by depth and in aggregate."""
    layers = branch.get("layers", {})
    if not layers:
        return

    depth_idxs = sorted(layers.keys(), key=int)
    attn = np.array([layers[i]["attn_frac"] for i in depth_idxs])
    mlp = np.array([layers[i]["mlp_frac"] for i in depth_idxs])
    ratio = np.array([layers[i]["attn_to_mlp_ratio"] for i in depth_idxs])
    total = branch.get("total", {})

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(depth_idxs, attn, marker="o", label="Attention", color="#1f77b4")
    axes[0].plot(depth_idxs, mlp, marker="o", label="MLP", color="#ff7f0e")
    axes[0].set_ylabel("Mean Branch Probability")
    axes[0].set_title(f"Branch Mix by Depth (step {step})")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(depth_idxs, ratio, marker="o", color="#2c3e50")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Depth")
    axes[1].set_ylabel("Attention / MLP")
    axes[1].set_title(
        "Total: "
        f"attn={total.get('attn_frac', 0.0):.3f}, mlp={total.get('mlp_frac', 0.0):.3f}, "
        f"attn/mlp={total.get('attn_to_mlp_ratio', 0.0):.3f}"
    )
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "branch_ratios.png"), dpi=100)
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
