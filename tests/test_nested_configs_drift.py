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
    """All matrix yamls in the active config tree. After
    Round 30's reorganization, the matrix lives ONLY under
    `configs/{4,8,16}_layers/`; non-matrix examples (sanity,
    perlayer-prenorm variants, seq_aux experiments) live under
    `configs/extras/` and are explicitly excluded.
    """
    paths: list[Path] = []
    for sub in ("configs/4_layers", "configs/8_layers", "configs/16_layers"):
        paths.extend(sorted((REPO / sub).rglob("*.yaml")))
    return paths


# AC-18 matrix specification: 13 yamls per depth × 3 depths = 39.
# 1 dense + (4 MoE families × 3 methods) = 13 per depth.
_MATRIX_DEPTHS = ("4_layers", "8_layers", "16_layers")
_MATRIX_FAMILIES = (
    "standard_moe",
    "global_moe",
    "moe_everything_per_head_fully_independent",
    "moe_everything_per_head_precompute_kv",
)
_MATRIX_METHODS = ("aux_loss", "deepseek_bias", "quantile")


def _expected_matrix() -> set[tuple[str, str, str]]:
    expected: set[tuple[str, str, str]] = set()
    for depth in _MATRIX_DEPTHS:
        expected.add((depth, "dense", "none"))
        for family in _MATRIX_FAMILIES:
            for method in _MATRIX_METHODS:
                expected.add((depth, family, method))
    return expected


def _classify(path: Path) -> tuple[str, str, str]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg.get("model", {}) or {}
    depth = path.parent.name  # "4_layers" / "8_layers" / "16_layers"
    family = mcfg.get("type", "unknown")
    if family == "moe_everything":
        attn_mode = mcfg.get("attn_expert_mode", "")
        family = (
            "moe_everything_per_head_precompute_kv"
            if "precompute_kv" in attn_mode
            else "moe_everything_per_head_fully_independent"
        )
    if family == "dense":
        method = "none"
    else:
        nested_mlp = mcfg.get("mlp_router", {}) or {}
        method = nested_mlp.get("balancing", "unknown")
    return depth, family, method


def test_matrix_has_exactly_39_yamls_in_13x3_layout():
    """The AC-18 matrix is exactly 13 yamls per depth × 3 depths
    = 39 total. Round 30 restructured the tree to enforce this
    layout; non-matrix examples were moved to `configs/extras/`."""
    paths = _nested_yamls()
    assert len(paths) == 39, (
        f"matrix should have exactly 39 yamls; found {len(paths)}"
    )
    by_depth: dict[str, list[Path]] = {}
    for p in paths:
        by_depth.setdefault(p.parent.name, []).append(p)
    for depth, files in by_depth.items():
        assert len(files) == 13, (
            f"{depth} has {len(files)} yamls; expected exactly 13"
        )


def test_matrix_covers_every_required_cell():
    """Every (depth, family, method) tuple in the spec must have
    AT LEAST one yaml, and there must be NO yamls outside the
    spec. Round 28's review demanded an explicit matrix-spec test."""
    expected = _expected_matrix()
    found: set[tuple[str, str, str]] = set()
    extras: list[tuple[Path, tuple[str, str, str]]] = []
    for p in _nested_yamls():
        cls = _classify(p)
        if cls in expected:
            found.add(cls)
        else:
            extras.append((p, cls))
    missing = expected - found
    assert not missing, (
        f"matrix is missing {len(missing)} required cell(s): "
        f"{sorted(missing)}"
    )
    assert not extras, (
        f"yamls present that don't fit the spec: {extras}"
    )


def test_matrix_has_quantile_rows():
    """Codex Round 29 review explicitly flagged the absence of
    quantile rows. Pin this contract: every depth must have
    quantile yamls for each MoE family."""
    quantile_paths = [
        p for p in _nested_yamls()
        if _classify(p)[2] == "quantile"
    ]
    assert len(quantile_paths) == len(_MATRIX_DEPTHS) * len(_MATRIX_FAMILIES), (
        f"expected {len(_MATRIX_DEPTHS) * len(_MATRIX_FAMILIES)} "
        f"quantile yamls (4 families × 3 depths); got {len(quantile_paths)}"
    )


def test_nested_directory_has_yamls():
    """Sanity: the matrix directories exist and have yamls."""
    paths = _nested_yamls()
    assert paths, (
        "matrix directories empty; expected 39 yamls across "
        "configs/{4,8,16}_layers/"
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


def test_migrator_is_idempotent_on_active_config_tree():
    """Round 29 review Finding 1: the migrator's dry run on the
    active config tree (configs/) must report ZERO changes. If the
    migrator finds a change, the repo carries either a redundant
    top-level coefficient or a flat field that should already have
    been migrated. Catches regression where new yamls are authored
    without going through the migrator first.
    """
    spec = importlib.util.spec_from_file_location(
        "migrate_configs", str(REPO / "scripts" / "migrate_configs.py"),
    )
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)

    drift: list[tuple[str, list[str]]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        changes = mc.migrate_config(cfg)
        if changes:
            drift.append((str(p), changes))
    assert not drift, (
        f"migrator found changes on the active config tree — should be "
        f"idempotent: {drift}"
    )


if __name__ == "__main__":
    print("Run via pytest")
