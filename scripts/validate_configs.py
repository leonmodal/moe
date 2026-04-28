#!/usr/bin/env python3
"""Config validation/linter for active training configs.

Usage:
    python scripts/validate_configs.py                    # validate all active configs
    python scripts/validate_configs.py configs/standard_moe.yaml  # validate specific config

Validates:
- Required fields are present
- Model type uses normalized taxonomy
- No deprecated model types
- Router type and attention mode enums are valid
- Data format is parquet (token-bin removed)
- Training config fields are valid
- Recursively scans all config subdirectories
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


SUPPORTED_MODEL_TYPES = {"dense", "standard_moe", "global_moe", "moe_everything"}

DEPRECATED_MODEL_TYPES = {
    "deepseek_standard_moe": "standard_moe (with router_type: deepseek)",
    "deepseek_global_moe": "global_moe (with router_type: deepseek)",
    "gpt2_dense": "dense",
    "speedrun_gpt": "archived to legacy/",
    "speedrun_moe_fully_independent": "archived to legacy/",
    "speedrun_moe_precompute_kv": "archived to legacy/",
    "speedrun_moe_everything": "archived to legacy/",
}

VALID_ROUTER_TYPES = {"softmax", "deepseek"}
# Must match src/models/moe_everything/attention_bank.py runtime enum. The old
# "bundled" mode (1 router per projection, top-K) has been replaced by the
# H-routers-per-projection-top-1 designs (see docs/routing.md) and is rejected
# at runtime with ValueError; the validator must agree.
VALID_ATTN_EXPERT_MODES = {
    "per_head_fully_independent", "per_head_precompute_kv",
}
VALID_LR_SCHEDULERS = {"cosine", "linear", "constant", "stable_decay"}
VALID_OPTIMIZERS = {"adamw", "muon"}
VALID_MIXED_PRECISION = {"bf16", "fp16", "fp32", ""}

REQUIRED_MODEL_FIELDS = {"type", "vocab_size", "hidden_size", "num_hidden_layers"}
REQUIRED_MoE_FIELDS = {"num_experts", "num_experts_per_tok", "moe_intermediate_size"}
REQUIRED_TRAINING_FIELDS = {
    "learning_rate", "weight_decay", "max_grad_norm", "lr_scheduler",
    "warmup_steps", "max_steps", "batch_size", "gradient_accumulation",
    "mixed_precision", "output_dir",
}


def validate_config(path: Path) -> list[str]:
    """Validate a single config file. Returns list of issues."""
    issues = []

    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return [f"Failed to parse YAML: {e}"]

    if not isinstance(cfg, dict):
        return ["Config must be a YAML mapping"]

    # Model section
    model = cfg.get("model", {})
    if not model:
        issues.append("Missing 'model' section")
    else:
        mtype = model.get("type")
        if not mtype:
            issues.append("Missing model.type")
        elif mtype in DEPRECATED_MODEL_TYPES:
            issues.append(
                f"Deprecated model type '{mtype}' — use {DEPRECATED_MODEL_TYPES[mtype]}"
            )
        elif mtype not in SUPPORTED_MODEL_TYPES:
            issues.append(f"Unknown model type '{mtype}'")

        missing_model = REQUIRED_MODEL_FIELDS - set(model.keys())
        if missing_model:
            issues.append(f"Missing required model fields: {missing_model}")

        if mtype in ("standard_moe", "global_moe", "moe_everything"):
            missing_moe = REQUIRED_MoE_FIELDS - set(model.keys())
            if missing_moe:
                issues.append(f"Missing required MoE fields: {missing_moe}")

        # Validate enum fields
        router_type = model.get("router_type")
        if router_type is not None and router_type not in VALID_ROUTER_TYPES:
            issues.append(f"Invalid router_type '{router_type}' — must be one of {VALID_ROUTER_TYPES}")

        attn_mode = model.get("attn_expert_mode")
        if attn_mode is not None and attn_mode not in VALID_ATTN_EXPERT_MODES:
            issues.append(f"Invalid attn_expert_mode '{attn_mode}' — must be one of {VALID_ATTN_EXPERT_MODES}")

    # Training section
    training = cfg.get("training", {})
    if not training:
        issues.append("Missing 'training' section")
    else:
        missing_training = REQUIRED_TRAINING_FIELDS - set(training.keys())
        if missing_training:
            issues.append(f"Missing required training fields: {missing_training}")

        lr_sched = training.get("lr_scheduler")
        if lr_sched is not None and lr_sched not in VALID_LR_SCHEDULERS:
            issues.append(f"Invalid lr_scheduler '{lr_sched}' — must be one of {VALID_LR_SCHEDULERS}")

        optimizer = training.get("optimizer")
        if optimizer is not None and optimizer not in VALID_OPTIMIZERS:
            issues.append(f"Invalid optimizer '{optimizer}' — must be one of {VALID_OPTIMIZERS}")

        mp = training.get("mixed_precision", "")
        if mp and mp not in VALID_MIXED_PRECISION:
            issues.append(f"Invalid mixed_precision '{mp}' — must be one of {VALID_MIXED_PRECISION}")

    # Data section
    data = cfg.get("data", {})
    if data:
        if data.get("format") == "token_bin":
            issues.append("Token-bin data format is no longer supported — use parquet")
        if "files_glob" in data:
            issues.append("'files_glob' field indicates token-bin format — use parquet with 'data_dir'")

    # Nested-schema validator: catches typos / illegal values under
    # `model.branch_router`, `model.mlp_router`, and `model.attn_router`.
    # Imported lazily to keep this script's startup fast and avoid
    # transitively pulling in the data pipeline (which would bring
    # pandas in via the package init chain).
    try:
        import importlib.util
        import types as _types
        repo = Path(__file__).resolve().parent.parent
        if "src.training" not in sys.modules:
            pkg = _types.ModuleType("src.training")
            pkg.__path__ = [str(repo / "src" / "training")]
            sys.modules["src.training"] = pkg
        spec = importlib.util.spec_from_file_location(
            "src.training.balancing_fields",
            str(repo / "src" / "training" / "balancing_fields.py"),
        )
        bf = importlib.util.module_from_spec(spec)
        sys.modules["src.training.balancing_fields"] = bf
        spec.loader.exec_module(bf)
        try:
            bf.validate_branch_router_config(cfg)
        except ValueError as exc:
            issues.append(f"nested-schema validator: {exc}")
    except Exception as exc:
        issues.append(f"nested-schema validator unavailable: {exc}")

    # Strict mode for the active matrix: yamls under
    # `configs/{4,8,16}_layers/` must not carry top-level
    # balancing fields. The per-class blocks
    # (`model.{mlp,attn,branch}_router`) are authoritative; a
    # top-level coefficient is off-axis pollution that violates
    # the matrix method-axis contract. Yamls under `configs/extras/`
    # are exempt as legacy / non-matrix fixtures.
    path_str = str(path)
    in_active_matrix = (
        "/configs/4_layers/" in path_str
        or "/configs/8_layers/" in path_str
        or "/configs/16_layers/" in path_str
    )
    if in_active_matrix:
        forbidden = (
            "router_aux_loss_coef", "seq_aux_loss_coef",
            "bias_update_rate", "bias_update_zero_sum",
            "bias_warmup_start", "bias_warmup_steps",
            "load_balancing_method",
        )
        tcfg = cfg.get("training", {}) or {}
        for key in forbidden:
            if key in tcfg:
                issues.append(
                    f"active matrix yaml carries top-level "
                    f"training.{key}={tcfg[key]!r}; per-class "
                    f"`model.{{mlp,attn,branch}}_router` blocks "
                    f"are authoritative."
                )
        mcfg_local = cfg.get("model", {}) or {}
        if "load_balancing_method" in mcfg_local:
            issues.append(
                f"active matrix yaml carries top-level "
                f"model.load_balancing_method={mcfg_local['load_balancing_method']!r}; "
                f"the nested per-class blocks are authoritative."
            )

        # Active per-class knob enforcement: each method-active
        # block must explicitly carry the matrix's required active
        # knob values. Implicit defaults are not acceptable on the
        # active matrix because the per-class block is the source
        # of truth.
        required_active = {
            "aux_loss": {"router_aux_loss_coef": 0.001},
            "seq_aux_loss": {"seq_aux_loss_coef": 0.0001},
            "deepseek_bias": {"bias_update_rate": 0.001},
            "quantile": {
                "quantile_eta": 0.005,
                "quantile_target_q": 0.5,
                "quantile_global_state": True,
            },
        }
        for group in ("mlp_router", "attn_router", "branch_router"):
            block = mcfg_local.get(group, {}) or {}
            method = block.get("balancing")
            if method not in required_active:
                continue
            for knob, expected in required_active[method].items():
                actual = block.get(knob)
                if actual != expected:
                    issues.append(
                        f"active matrix yaml: model.{group}.{knob}="
                        f"{actual!r}, expected {expected!r} for "
                        f"balancing={method!r}"
                    )

    return issues


def _find_all_configs(config_dir: Path) -> list[Path]:
    """Recursively find all YAML configs, excluding legacy/ directories."""
    configs = []
    for path in sorted(config_dir.rglob("*.yaml")):
        # Skip any path component named 'legacy' or 'speedrun' within config dir
        parts = path.relative_to(config_dir).parts
        if "speedrun" in parts:
            continue
        configs.append(path)
    return configs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate MoE training configs")
    parser.add_argument("configs", nargs="*", help="Config files to validate (default: all in configs/)")
    args = parser.parse_args()

    if args.configs:
        config_files = [Path(c) for c in args.configs]
    else:
        config_dir = Path("configs")
        config_files = _find_all_configs(config_dir)

    total_issues = 0
    for config_file in config_files:
        issues = validate_config(config_file)
        if issues:
            print(f"\n{config_file}:")
            for issue in issues:
                print(f"  - {issue}")
            total_issues += len(issues)

    if total_issues == 0:
        print(f"All {len(config_files)} configs passed validation.")
    else:
        print(f"\n{total_issues} issues found across {len(config_files)} configs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
