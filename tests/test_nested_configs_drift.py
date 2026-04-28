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


_TOP_LEVEL_BALANCING_FIELDS = (
    "router_aux_loss_coef",
    "seq_aux_loss_coef",
    "bias_update_rate",
    "bias_update_zero_sum",
    "bias_warmup_start",
    "bias_warmup_steps",
    "load_balancing_method",
)


def test_matrix_has_no_top_level_balancing_fields():
    """Round 30 review Finding 1: every active matrix yaml must
    have ZERO top-level balancing coefficients/methods. The
    per-class blocks (`model.{mlp,attn,branch}_router`) are
    authoritative; top-level fields are off-axis pollution that
    can keep the matrix in a mixed nested-plus-flat state.
    """
    leaks: list[tuple[str, str, object]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        tcfg = cfg.get("training", {}) or {}
        for key in _TOP_LEVEL_BALANCING_FIELDS:
            if key in tcfg:
                leaks.append((str(p), f"training.{key}", tcfg[key]))
        mcfg = cfg.get("model", {}) or {}
        if "load_balancing_method" in mcfg:
            leaks.append((str(p), "model.load_balancing_method", mcfg["load_balancing_method"]))
    assert not leaks, (
        f"matrix yamls still carry top-level balancing fields "
        f"(off-axis pollution): {leaks}"
    )


def test_matrix_aux_loss_rows_have_only_aux_coef_in_per_class():
    """Per-method axis rule for aux_loss rows: the per-class
    `mlp_router` block has `balancing: aux_loss` and may carry
    `router_aux_loss_coef`. It must NOT carry seq_aux_loss_coef,
    bias_update_*, or quantile_* knobs (those would silently mix
    methods)."""
    incompatible_for_aux = (
        "seq_aux_loss_coef",
        "bias_update_rate", "bias_update_zero_sum",
        "bias_warmup_start", "bias_warmup_steps",
        "quantile_eta", "quantile_target_q", "quantile_global_state",
    )
    leaks = []
    for p in _nested_yamls():
        if not str(p).endswith("_aux_loss.yaml"):
            continue
        with p.open() as f:
            cfg = yaml.safe_load(f)
        for group in ("mlp_router", "attn_router", "branch_router"):
            block = cfg.get("model", {}).get(group, {}) or {}
            if block.get("balancing") != "aux_loss":
                continue
            for key in incompatible_for_aux:
                if key in block and block[key]:
                    leaks.append((str(p), f"{group}.{key}", block[key]))
    assert not leaks, (
        f"aux_loss matrix rows carry off-axis per-class knobs: {leaks}"
    )


# Required active knob values for each method, per the AC-18 matrix
# spec. Each matrix row's per-class block at `balancing == method`
# MUST carry the corresponding key at this exact value.
_REQUIRED_ACTIVE_KNOBS = {
    "aux_loss": {"router_aux_loss_coef": 0.001},
    "seq_aux_loss": {"seq_aux_loss_coef": 0.0001},
    "deepseek_bias": {"bias_update_rate": 0.001},
    "quantile": {
        "quantile_eta": 0.005,
        "quantile_target_q": 0.5,
        "quantile_global_state": True,
    },
}


# Branch-router contract for `per_head_precompute_kv` matrix rows.
# Per the matrix spec, every yaml whose `attn_expert_mode` is
# `per_head_precompute_kv` must enable branch exploration_only with
# the documented decay schedule.
_PRECOMPUTE_KV_BRANCH_SPEC = {
    "balancing": "exploration_only",
    "exploration_rate": 0.1,
    "exploration_decay": "cosine",
    "exploration_min": 0.01,
    "exploration_warmup_steps": 1000,
}


def test_cli_validator_rejects_missing_active_knob_in_active_matrix(tmp_path):
    """Round 32 review Finding 2: scripts/validate_configs.py
    must fail on a matrix yaml that omits the active method's
    required knob (e.g. `balancing: aux_loss` without
    `router_aux_loss_coef: 0.001`). Build a temporary matrix-path
    yaml with the missing knob and assert the CLI reports the issue.
    """
    import os
    import subprocess
    # Place the temporary yaml under a fake "active matrix" path
    # so the strict-mode gate fires.
    matrix_dir = tmp_path / "configs" / "4_layers"
    matrix_dir.mkdir(parents=True)
    bad_yaml = matrix_dir / "standard_moe_aux_loss.yaml"
    bad_yaml.write_text(
        """experiment_name: validator_negative_probe
model:
  type: standard_moe
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 1
  num_experts: 4
  num_experts_per_tok: 1
  moe_intermediate_size: 32
  mlp_router:
    balancing: aux_loss
training:
  learning_rate: 1e-3
  weight_decay: 0
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 0
  max_steps: 1
  batch_size: 1
  gradient_accumulation: 1
  mixed_precision: ""
  output_dir: /tmp
"""
    )
    cmd = [
        sys.executable, "scripts/validate_configs.py",
        str(bad_yaml),
    ]
    env = {"PYTHONPATH": str(REPO), **os.environ}
    result = subprocess.run(
        cmd, cwd=str(REPO), env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0, (
        f"CLI did not detect missing aux active knob:\n"
        f"STDOUT={result.stdout}\nSTDERR={result.stderr}"
    )
    assert "router_aux_loss_coef" in result.stdout, (
        f"CLI output did not mention the missing knob:\n{result.stdout}"
    )


def test_precompute_kv_rows_set_dec6_branch_exploration():
    """Every matrix row whose `attn_expert_mode == per_head_precompute_kv`
    must set branch_router to the documented exploration_only spec
    (rate=0.1, decay=cosine, min=0.01, warmup=1000). The KV
    precompute path is the architectural variant that benefits most
    from the random branch curriculum, so the matrix locks the
    branch contract for those rows."""
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("attn_expert_mode") != "per_head_precompute_kv":
            continue
        branch = mcfg.get("branch_router", {}) or {}
        for key, expected in _PRECOMPUTE_KV_BRANCH_SPEC.items():
            actual = branch.get(key)
            if actual != expected:
                deficits.append((str(p), f"branch_router.{key} = {actual!r} (expected {expected!r})"))
    assert not deficits, (
        f"precompute_kv matrix rows missing the branch exploration_only spec: "
        f"{deficits}"
    )


def test_matrix_rows_carry_required_active_knobs_per_method():
    """Round 31 review Finding 1: every matrix row's per-class
    block must explicitly carry the active knobs for its method.
    Aux rows MUST have `router_aux_loss_coef: 0.001`; deepseek
    rows MUST have `bias_update_rate: 0.001`; quantile rows MUST
    have the full quantile knob set. Implicit defaults are not
    acceptable — the per-class block is the source of truth.
    """
    deficits: list[tuple[str, str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("type") == "dense":
            continue
        for group in ("mlp_router", "attn_router", "branch_router"):
            block = mcfg.get(group, {}) or {}
            method = block.get("balancing")
            if method in (None, "none", "exploration_only"):
                continue
            required = _REQUIRED_ACTIVE_KNOBS.get(method, {})
            for key, expected in required.items():
                actual = block.get(key)
                if actual != expected:
                    deficits.append((str(p), f"{group}.{key}", f"expected {expected!r}, got {actual!r}"))
    assert not deficits, (
        f"matrix rows missing required active knobs: {deficits}"
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
