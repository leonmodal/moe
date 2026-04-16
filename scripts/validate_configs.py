#!/usr/bin/env python3
"""Config validation/linter for active training configs.

Usage:
    python scripts/validate_configs.py                    # validate all active configs
    python scripts/validate_configs.py configs/standard_moe.yaml  # validate specific config

Validates:
- Required fields are present
- Model type uses normalized taxonomy
- No deprecated model types
- Data format is parquet (token-bin removed)
- Training config fields are valid
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

        if mtype in ("standard_moe", "global_moe", "moe_everything",
                      "deepseek_standard_moe", "deepseek_global_moe"):
            missing_moe = REQUIRED_MoE_FIELDS - set(model.keys())
            if missing_moe:
                issues.append(f"Missing required MoE fields: {missing_moe}")

    # Training section
    training = cfg.get("training", {})
    if not training:
        issues.append("Missing 'training' section")
    else:
        missing_training = REQUIRED_TRAINING_FIELDS - set(training.keys())
        if missing_training:
            issues.append(f"Missing required training fields: {missing_training}")

    # Data section
    data = cfg.get("data", {})
    if data:
        if data.get("format") == "token_bin":
            issues.append("Token-bin data format is no longer supported — use parquet")
        if "files_glob" in data:
            issues.append("'files_glob' field indicates token-bin format — use parquet with 'data_dir'")

    return issues


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate MoE training configs")
    parser.add_argument("configs", nargs="*", help="Config files to validate (default: all in configs/)")
    args = parser.parse_args()

    if args.configs:
        config_files = [Path(c) for c in args.configs]
    else:
        config_dir = Path("configs")
        config_files = sorted(config_dir.glob("*.yaml"))
        # Also check subdirectories (except speedrun which is archived)
        for subdir in config_dir.iterdir():
            if subdir.is_dir() and subdir.name not in ("speedrun",):
                config_files.extend(sorted(subdir.glob("*.yaml")))

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
