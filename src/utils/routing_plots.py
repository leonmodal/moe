"""
Generate routing analysis plots from a snapshot dict.

Called during training right after saving the JSON. Produces PNGs in the same
directory as the JSON file.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_routing_snapshot(snapshot: dict, output_dir: str, step: int) -> None:
    """Generate all plots for one routing snapshot."""
    prefix = os.path.join(output_dir, f"step_{step:08d}")

    layers = snapshot.get("layers", {})
    global_pool = snapshot.get("global_pool")
    if layers:
        _plot_layer_heatmap(layers, prefix, step, title_prefix="Per-Layer Expert Token Fraction")
    if global_pool is not None:
        _plot_global_fracs(global_pool, prefix, step, title_prefix="Global Pool Token Fraction")
        _plot_usage_vs_tokens(global_pool, prefix, step, title_prefix="Layer Usage vs Token Share")

    for router_name, router_snapshot in snapshot.get("attention", {}).items():
        router_prefix = f"{prefix}_attention_{router_name}"
        router_layers = router_snapshot.get("layers", {})
        router_pool = router_snapshot.get("global_pool")
        title = f"Attention Router {router_name}"
        if router_layers:
            _plot_layer_heatmap(router_layers, router_prefix, step, title_prefix=title)
        if router_pool is not None:
            _plot_global_fracs(router_pool, router_prefix, step, title_prefix=f"{title} Global Pool")
            _plot_usage_vs_tokens(router_pool, router_prefix, step, title_prefix=f"{title} Usage vs Token Share")

    branch = snapshot.get("branch")
    if branch:
        _plot_branch_ratios(branch, prefix, step)

    plt.close("all")


def _plot_layer_heatmap(layers: dict, prefix: str, step: int, title_prefix: str) -> None:
    """Heatmap: layers (y) x experts (x), color = token fraction."""
    sorted_idxs = sorted(layers.keys(), key=int)
    if not sorted_idxs:
        return

    num_experts = len(layers[sorted_idxs[0]]["token_fracs"])
    num_layers = len(sorted_idxs)
    data = np.array([layers[i]["token_fracs"] for i in sorted_idxs])
    ideal = 1.0 / num_experts if num_experts > 0 else 0.0

    fig_width = max(8, num_experts * 0.12)
    fig_height = max(4, num_layers * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title_prefix} (step {step}, ideal={ideal:.4f})")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(i) for i in sorted_idxs], fontsize=max(6, 10 - num_layers // 10))

    xtick_step = max(1, num_experts // 20) if num_experts > 0 else 1
    ax.set_xticks(range(0, num_experts, xtick_step))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Token Fraction")

    fig.tight_layout()
    fig.savefig(f"{prefix}_layer_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_global_fracs(global_pool: dict, prefix: str, step: int, title_prefix: str) -> None:
    """Bar chart: per-expert token fraction across the shared pool."""
    fracs = np.array(global_pool["token_fracs"])
    num_experts = len(fracs)
    ideal = 1.0 / num_experts if num_experts > 0 else 0.0

    fig_width = max(8, num_experts * 0.08 if num_experts > 0 else 8)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    colors = ["#e74c3c" if ideal > 0 and f > ideal * 2 else "#3498db" for f in fracs]
    ax.bar(range(num_experts), fracs, color=colors, width=0.8)
    ax.axhline(ideal, color="black", linestyle="--", linewidth=1, label=f"ideal={ideal:.4f}")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Token Fraction")
    ax.set_title(f"{title_prefix} (step {step})")
    ax.legend()

    xtick_step = max(1, num_experts // 20) if num_experts > 0 else 1
    ax.set_xticks(range(0, num_experts, xtick_step))

    fig.tight_layout()
    fig.savefig(f"{prefix}_global_fracs.png", dpi=120)
    plt.close(fig)


def _plot_usage_vs_tokens(global_pool: dict, prefix: str, step: int, title_prefix: str) -> None:
    """Scatter: x = number of layers using this expert, y = token fraction."""
    fracs = np.array(global_pool["token_fracs"])
    usage = np.array(global_pool["layer_usage_count"])
    num_layers = global_pool["num_layers"]
    num_experts = len(fracs)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(usage, fracs, alpha=0.6, s=20, c="#2c3e50")

    if len(usage) > 2 and np.std(usage) > 0:
        z = np.polyfit(usage, fracs, 1)
        x_line = np.linspace(usage.min(), usage.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5, label=f"trend (slope={z[0]:.2e})")
        ax.legend()

    ideal = 1.0 / num_experts if num_experts > 0 else 0.0
    ax.axhline(ideal, color="gray", linestyle=":", linewidth=1, label=f"ideal frac={ideal:.4f}")

    ax.set_xlabel(f"Layers Using Expert (out of {num_layers})")
    ax.set_ylabel("Token Fraction (global pool)")
    ax.set_title(f"{title_prefix} (step {step}, {num_experts} experts)")

    fig.tight_layout()
    fig.savefig(f"{prefix}_usage_vs_tokens.png", dpi=120)
    plt.close(fig)


def _plot_branch_ratios(branch: dict, prefix: str, step: int) -> None:
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
        "Total ratio: "
        f"attn={total.get('attn_frac', 0.0):.3f}, mlp={total.get('mlp_frac', 0.0):.3f}, "
        f"attn/mlp={total.get('attn_to_mlp_ratio', 0.0):.3f}"
    )
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{prefix}_branch_ratios.png", dpi=120)
    plt.close(fig)
