"""Print parameter breakdowns for the active 16-layer config matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.training.model_factory import build_model


def categorize_param(name: str) -> str:
    if "embed" in name or "lm_head" in name:
        return "Embedding"
    if any(k in name for k in ("gate_up_proj", "down_proj", "gate_proj", "up_proj")):
        return "MLP"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "attn")):
        return "Attention"
    if "router" in name or "gate" in name or "branch" in name:
        return "Router"
    if "norm" in name:
        return "Norm"
    return "Other"


def analyze_model(config_path: Path) -> tuple[dict[str, int], int]:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    with torch.no_grad():
        model, _ = build_model(cfg)

    categories: dict[str, int] = {}
    for name, param in model.named_parameters():
        categories[categorize_param(name)] = categories.get(categorize_param(name), 0) + param.numel()
    total = sum(categories.values())
    del model
    return categories, total


def main() -> None:
    config_paths = sorted((REPO / "configs" / "16_layers").glob("*.yaml"))
    rows: list[tuple[str, dict[str, int], int]] = []
    for path in config_paths:
        try:
            categories, total = analyze_model(path)
            rows.append((path.name, categories, total))
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")

    print(f"{'config':70s} {'total':>14s} {'attn':>14s} {'mlp':>14s} {'router':>14s}")
    for name, categories, total in rows:
        print(
            f"{name:70s} {total:14,d} "
            f"{categories.get('Attention', 0):14,d} "
            f"{categories.get('MLP', 0):14,d} "
            f"{categories.get('Router', 0):14,d}"
        )


if __name__ == "__main__":
    main()
