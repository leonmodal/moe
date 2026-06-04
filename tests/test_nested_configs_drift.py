"""Drift detection for the active 16-layer config matrix."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _load_balancing_validator():
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
    return sorted((REPO / "configs" / "16_layers").glob("*.yaml"))


_MATRIX_DEPTHS = ("16_layers",)
_EXPECTED_CONFIG_FILES = (
    "standard_moe_deepseek_bias.yaml",
    "moe_everything_per_head_recompute_k_qk_v_o_deepseek_bias.yaml",
    "moe_everything_per_head_recompute_k_qk_v_o_ema_qk_v_deepseek_bias.yaml",
    "moe_everything_per_head_recompute_kv_qk_v_o_deepseek_bias.yaml",
    "moe_everything_per_head_recompute_kv_qk_v_o_ema_qk_v_deepseek_bias.yaml",
)


def _expected_matrix() -> set[tuple[str, str, str]]:
    return {
        ("16_layers", "standard_moe", "deepseek_bias"),
        ("16_layers", "moe_everything:per_head_recompute_k:qk_v_o:none", "deepseek_bias"),
        ("16_layers", "moe_everything:per_head_recompute_k:qk_v_o:ema_qk_v", "deepseek_bias"),
        ("16_layers", "moe_everything:per_head_recompute_kv:qk_v_o:none", "deepseek_bias"),
        ("16_layers", "moe_everything:per_head_recompute_kv:qk_v_o:ema_qk_v", "deepseek_bias"),
    }


def _classify(path: Path) -> tuple[str, str, str]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg.get("model", {}) or {}
    depth = path.parent.name
    family = mcfg.get("type", "unknown")
    if family == "moe_everything":
        family = (
            f"moe_everything:{mcfg.get('attn_expert_mode', '')}:"
            f"{mcfg.get('attn_routing_bundle', '')}:"
            f"{mcfg.get('attn_router_context', 'none')}"
        )
    if family == "dense":
        method = "none"
    else:
        method = (mcfg.get("mlp_router", {}) or {}).get("balancing", "unknown")
    return depth, family, method


def test_matrix_has_exactly_five_yamls_in_16_layer_layout():
    paths = _nested_yamls()
    assert len(paths) == 5, f"16-layer launch set should have 5 yamls; found {len(paths)}"
    assert {p.parent.name for p in paths} == {"16_layers"}
    assert {p.name for p in paths} == set(_EXPECTED_CONFIG_FILES)


def test_matrix_covers_every_required_cell():
    expected = _expected_matrix()
    found: set[tuple[str, str, str]] = set()
    extras: list[tuple[Path, tuple[str, str, str]]] = []
    for p in _nested_yamls():
        cls = _classify(p)
        if cls in expected:
            found.add(cls)
        else:
            extras.append((p, cls))
    assert not (expected - found), f"matrix is missing cells: {sorted(expected - found)}"
    assert not extras, f"yamls present that do not fit the spec: {extras}"


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
    assert not leaks, f"matrix yamls carry top-level balancing fields: {leaks}"


def test_matrix_aux_loss_rows_have_only_aux_coef_in_per_class():
    incompatible_for_aux = (
        "seq_aux_loss_coef",
        "bias_update_rate",
        "bias_update_zero_sum",
        "bias_warmup_start",
        "bias_warmup_steps",
        "quantile_eta",
        "quantile_target_q",
        "quantile_global_state",
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
    assert not leaks, f"aux_loss matrix rows carry off-axis knobs: {leaks}"


_REQUIRED_ACTIVE_KNOBS = {
    "aux_loss": {"router_aux_loss_coef": 0.001},
    "seq_aux_loss": {"seq_aux_loss_coef": 0.0001},
    "deepseek_bias": {"bias_update_rate": 0.001},
    "quantile": {
        "quantile_eta": 0.005,
        "quantile_target_q": 0.5,
        "quantile_global_state": True,
    },
    "sampling_entropy": {
        "entropy_coef": 0.01,
        "entropy_decay": "cosine",
        "entropy_min": 0.0,
        "entropy_decay_steps": 1000,
    },
    "exploration_only": {
        "exploration_rate": 0.10,
        "exploration_decay": "cosine",
        "exploration_min": 0.0,
        "exploration_warmup_steps": 1000,
    },
}


_RECOMPUTE_BRANCH_SPECS = (
    {
        "balancing": "sampling_entropy",
        "entropy_coef": 0.01,
        "entropy_decay": "cosine",
        "entropy_min": 0.0,
        "entropy_decay_steps": 1000,
    },
    {
        "balancing": "exploration_only",
        "exploration_rate": 0.10,
        "exploration_decay": "cosine",
        "exploration_min": 0.0,
        "exploration_warmup_steps": 1000,
    },
    {
        "balancing": "fixed_alternating",
    },
)


_BAD_AUX_YAML_TEXT = """experiment_name: validator_negative_probe
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


def test_cli_validator_rejects_missing_active_knob_absolute_path(tmp_path):
    import os
    import subprocess

    matrix_dir = tmp_path / "configs" / "16_layers"
    matrix_dir.mkdir(parents=True)
    bad_yaml = matrix_dir / "bad_standard_moe.yaml"
    bad_yaml.write_text(_BAD_AUX_YAML_TEXT)
    cmd = [sys.executable, "scripts/validate_configs.py", str(bad_yaml)]
    env = {"PYTHONPATH": str(REPO), **os.environ}
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "router_aux_loss_coef" in result.stdout


def test_cli_validator_rejects_missing_active_knob_relative_path():
    import os
    import subprocess

    tmp_name = "_TMP_validator_relative_probe.yaml"
    target = REPO / "configs" / "16_layers" / tmp_name
    target.write_text(_BAD_AUX_YAML_TEXT)
    try:
        cmd = [sys.executable, "scripts/validate_configs.py", f"configs/16_layers/{tmp_name}"]
        env = {"PYTHONPATH": str(REPO), **os.environ}
        result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert "router_aux_loss_coef" in result.stdout
    finally:
        if target.exists():
            target.unlink()


def test_cli_validator_rejects_recompute_branch_drift_relative():
    import os
    import subprocess

    rel = "configs/16_layers/moe_everything_per_head_recompute_kv_qk_v_o_deepseek_bias.yaml"
    target = REPO / rel
    original = target.read_text()
    try:
        bad_text = original.replace(
            (
                "  branch_router:\n"
                "    balancing: sampling_entropy\n"
                "    entropy_coef: 0.01\n"
                "    entropy_decay: cosine\n"
                "    entropy_min: 0.0\n"
                "    entropy_decay_steps: 1000\n"
            ),
            "  branch_router:\n    balancing: none\n",
        )
        target.write_text(bad_text)
        cmd = [sys.executable, "scripts/validate_configs.py", rel]
        env = {"PYTHONPATH": str(REPO), **os.environ}
        result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert "recompute attention row" in result.stdout
    finally:
        target.write_text(original)


def test_cli_validator_accepts_qkvo_branch_ablation_configs():
    import os
    import subprocess

    rels = [
        "configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_sampling_entropy.yaml",
        "configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_top1_explore_decay.yaml",
        "configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_fixed_alternating.yaml",
    ]
    cmd = [sys.executable, "scripts/validate_configs.py", *rels]
    env = {"PYTHONPATH": str(REPO), **os.environ}
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_recompute_rows_set_branch_exploration():
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("attn_expert_mode") not in {"per_head_recompute_k", "per_head_recompute_kv"}:
            continue
        branch = mcfg.get("branch_router", {}) or {}
        if not any(
            all(branch.get(key) == expected for key, expected in spec.items())
            for spec in _RECOMPUTE_BRANCH_SPECS
        ):
            deficits.append((str(p), f"branch_router={branch!r}"))
    assert not deficits, f"recompute matrix rows missing a supported branch spec: {deficits}"


def test_moe_everything_rows_are_fully_per_layer():
    required = (
        "per_layer_router",
        "per_layer_mlp_router",
        "per_layer_attn_router",
        "per_layer_norm",
        "per_layer_qk_norm",
    )
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("type") != "moe_everything":
            continue
        for key in required:
            if mcfg.get(key) is not True:
                deficits.append((str(p), f"{key}={mcfg.get(key)!r}, expected True"))
    assert not deficits, f"MoE-Everything launch rows must be fully per-layer: {deficits}"


def test_no_recompute_rows_keep_branch_router_none():
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("attn_expert_mode") != "per_head_no_recompute":
            continue
        balancing = (mcfg.get("branch_router", {}) or {}).get("balancing")
        if balancing != "none":
            deficits.append((str(p), f"branch_router.balancing={balancing!r}"))
    assert not deficits


def test_matrix_rows_carry_required_active_knobs_per_method():
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
            if method in (None, "none", "fixed_alternating"):
                continue
            for key, expected in _REQUIRED_ACTIVE_KNOBS.get(method, {}).items():
                actual = block.get(key)
                if actual != expected:
                    deficits.append((str(p), f"{group}.{key}", f"expected {expected!r}, got {actual!r}"))
    assert not deficits, f"matrix rows missing required active knobs: {deficits}"


def test_matrix_has_only_deepseek_bias_rows():
    non_deepseek = [p for p in _nested_yamls() if _classify(p)[2] != "deepseek_bias"]
    assert not non_deepseek, f"launch set should contain only deepseek_bias rows: {non_deepseek}"


def test_every_moe_yaml_has_per_class_blocks():
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg.get("model", {}) or {}
        if mcfg.get("type") == "dense":
            continue
        has_any = any(
            isinstance(mcfg.get(name), dict) and mcfg[name]
            for name in ("mlp_router", "attn_router", "branch_router")
        )
        if not has_any:
            deficits.append((str(p), "no per-class router blocks"))
    assert not deficits, f"MoE yamls without per-class blocks: {deficits}"


def test_no_yaml_has_top_level_method():
    deficits: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        for block in ("training", "model"):
            if cfg.get(block, {}).get("load_balancing_method") is not None:
                deficits.append((str(p), f"{block}.load_balancing_method present"))
    assert not deficits, f"yamls still carry top-level load_balancing_method: {deficits}"


def test_every_yaml_validates():
    bf = _load_balancing_validator()
    failures: list[tuple[str, str]] = []
    for p in _nested_yamls():
        with p.open() as f:
            cfg = yaml.safe_load(f)
        try:
            bf.validate_branch_router_config(cfg)
        except ValueError as exc:
            failures.append((str(p), str(exc)))
    assert not failures, f"yamls failing validation: {failures}"


def test_migrator_is_idempotent_on_active_config_tree():
    spec = importlib.util.spec_from_file_location(
        "migrate_configs", str(REPO / "scripts" / "migrate_configs.py")
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
    assert not drift, f"migrator found changes on active configs: {drift}"
