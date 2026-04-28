"""Drift detection for nested-schema yamls under `configs/nested/`.

Every yaml in `configs/nested/` MUST be on the nested schema:
* `model.{mlp,attn,branch}_router` blocks present.
* No top-level `training.load_balancing_method` (nested-only).

A drift test guards the matrix from regressions: if an operator
manually edits a nested yaml back to flat shape, the drift test
fails immediately so the matrix's "nested" promise is enforced.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _load_validator():
    if "src.training" not in sys.modules:
        import types
        pkg = types.ModuleType("src.training")
        pkg.__path__ = [str(REPO / "src" / "training")]
        sys.modules["src.training"] = pkg
    spec = importlib.util.spec_from_file_location(
        "src.training.balancing_fields",
        str(REPO / "src" / "training" / "balancing_fields.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.training.balancing_fields"] = mod
    spec.loader.exec_module(mod)
    return mod


def _nested_yamls() -> list[Path]:
    """All nested-schema yamls in the repo. After the in-place
    migration, the original `configs/{4,8,16}_layers/*.yaml` are
    nested too, plus the additional `configs/nested/*.yaml`
    examples. Both directories are walked here so the drift check
    catches a regression anywhere in the active config tree."""
    paths: list[Path] = []
    for sub in ("configs/4_layers", "configs/8_layers",
                "configs/16_layers", "configs/nested"):
        paths.extend(sorted((REPO / sub).rglob("*.yaml")))
    return paths


def test_nested_directory_has_yamls():
    """Sanity: the nested config directory exists and has at least
    one yaml. The full 39-config matrix is the AC-18 target; this
    test asserts the directory is on the right shape."""
    paths = _nested_yamls()
    assert paths, (
        "configs/nested/ has no yamls; the nested-schema matrix is "
        "the AC-18 deliverable. Author yamls under configs/nested/."
    )


def test_every_nested_yaml_has_per_class_blocks():
    """Every MoE yaml in the active config tree MUST set at least
    one of the per-class router blocks (`mlp_router` / `attn_router`
    / `branch_router`). Dense yamls have no MoE routers so they're
    exempt from the per-class requirement."""
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("type") == "dense":
            continue  # dense has no MoE routers
        has_any = any(
            isinstance(mcfg.get(name), dict) and mcfg[name]
            for name in ("mlp_router", "attn_router", "branch_router")
        )
        if not has_any:
            deficits.append((str(p), "no per-class router blocks"))
    assert not deficits, (
        f"MoE yamls without per-class blocks: {deficits}"
    )


def test_no_nested_yaml_has_top_level_method():
    """nested-schema yamls must NOT carry `training.load_balancing_method`
    or `model.load_balancing_method` — the per-class blocks are
    authoritative. This is the AC-13 nested-only contract.
    """
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        for block in ("training", "model"):
            if cfg.get(block, {}).get("load_balancing_method") is not None:
                deficits.append((str(p), f"{block}.load_balancing_method present"))
    assert not deficits, (
        f"nested yamls still carrying top-level load_balancing_method: "
        f"{deficits}"
    )


def test_every_nested_yaml_validates():
    """Every nested yaml must pass the canonical validator
    (`validate_branch_router_config`) without exception."""
    bf = _load_validator()
    failures: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        try:
            bf.validate_branch_router_config(cfg)
        except ValueError as exc:
            failures.append((str(p), str(exc)))
    assert not failures, (
        f"nested yamls failing validation: {failures}"
    )


if __name__ == "__main__":
    print("Run via pytest")
