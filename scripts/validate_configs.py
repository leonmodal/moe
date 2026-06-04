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


SUPPORTED_MODEL_TYPES = {
    "dense",
    "standard_moe",
    "global_moe",
    "moe_everything",
    "recurrent_moe_everything",
    "recurrent_standard_moe",
    "recurrent_global_moe",
    "hrm_recurrent_standard_moe",
}

DEPRECATED_MODEL_TYPES = {
    "deepseek_standard_moe": "standard_moe (with router_type: deepseek)",
    "deepseek_global_moe": "global_moe (with router_type: deepseek)",
    "gpt2_dense": "dense",
    "speedrun_gpt": "archived to legacy/",
    "speedrun_moe_everything": "archived to legacy/",
}

VALID_ROUTER_TYPES = {"softmax", "deepseek"}
VALID_ATTN_EXPERT_MODES = {
    "per_head_no_recompute",
    "per_head_recompute_k",
    "per_head_recompute_kv",
}
VALID_ATTN_ROUTING_BUNDLES = {
    "q_k_v_o",
    "qk_v_o",
    "qk_vo",
    "qkv_o",
    "qkvo",
}
VALID_ATTN_ROUTER_CONTEXTS = {"none", "ema_qk_v"}
VALID_LR_SCHEDULERS = {"cosine", "linear", "constant", "stable_decay"}
VALID_OPTIMIZERS = {"adamw", "muon"}
VALID_MIXED_PRECISION = {"bf16", "fp16", "fp32", ""}
VALID_FSDP_SHARDING_STRATEGIES = {
    "auto", "full_shard", "shard_grad_op", "no_shard", "hybrid_shard",
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

        if mtype in (
            "standard_moe",
            "global_moe",
            "moe_everything",
            "recurrent_moe_everything",
            "recurrent_standard_moe",
            "recurrent_global_moe",
            "hrm_recurrent_standard_moe",
        ):
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
        attn_bundle = model.get("attn_routing_bundle")
        if attn_bundle is not None and attn_bundle not in VALID_ATTN_ROUTING_BUNDLES:
            issues.append(
                f"Invalid attn_routing_bundle '{attn_bundle}' — must be one of "
                f"{VALID_ATTN_ROUTING_BUNDLES}"
            )
        if attn_mode == "per_head_no_recompute" and attn_bundle not in (None, "q_k_v_o"):
            issues.append(
                "per_head_no_recompute currently supports only "
                "attn_routing_bundle='q_k_v_o'"
            )
        if attn_mode in {"per_head_recompute_k", "per_head_recompute_kv"}:
            resolved_bundle = attn_bundle or "qkvo"
            if not resolved_bundle.startswith("qk"):
                issues.append(
                    f"{attn_mode} requires Q/K to share a route; got "
                    f"attn_routing_bundle={resolved_bundle!r}"
                )
        attn_router_context = model.get("attn_router_context", "none")
        if attn_router_context not in VALID_ATTN_ROUTER_CONTEXTS:
            issues.append(
                f"Invalid attn_router_context '{attn_router_context}' — "
                f"must be one of {VALID_ATTN_ROUTER_CONTEXTS}"
            )
        if attn_router_context == "ema_qk_v":
            decay = model.get("attn_router_context_decay", 0.95)
            if not isinstance(decay, (int, float)) or not (0.0 <= float(decay) < 1.0):
                issues.append(
                    "attn_router_context_decay must be a number in [0, 1) "
                    "when attn_router_context='ema_qk_v'"
                )

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

        fsdp_sharding = training.get("fsdp_sharding_strategy")
        if fsdp_sharding is not None and fsdp_sharding not in VALID_FSDP_SHARDING_STRATEGIES:
            issues.append(
                f"Invalid fsdp_sharding_strategy '{fsdp_sharding}' — must be one of "
                f"{VALID_FSDP_SHARDING_STRATEGIES}"
            )

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
    # `configs/16_layers/` must not carry top-level
    # balancing fields. The per-class blocks
    # (`model.{mlp,attn,branch}_router`) are authoritative; a
    # top-level coefficient is off-axis pollution that violates
    # the matrix method-axis contract. Yamls under `configs/extras/`
    # are exempt as legacy / non-matrix fixtures.
    #
    # Path classification: resolve to absolute, then check whether
    # the path's components include `configs/<depth>_layers/`. Was
    # previously substring-matched on `/configs/<depth>_layers/`,
    # which silently skipped relative paths like
    # `configs/16_layers/foo.yaml` (no leading slash) — the default
    # CLI scan returns relative paths.
    in_active_matrix = False
    try:
        resolved = Path(path).resolve()
        parts = resolved.parts
        for i, part in enumerate(parts):
            if part == "configs" and i + 1 < len(parts):
                if parts[i + 1] == "16_layers":
                    in_active_matrix = True
                    break
    except (OSError, ValueError):
        in_active_matrix = False
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

        # Active matrix also forbids the flat `branch_*` bridge
        # fields. The nested `model.branch_router` block is the
        # only source of truth on the active matrix; flat fields
        # are reserved for legacy / external configs handled
        # outside the matrix.
        flat_branch_keys = (
            "branch_balancing",
            "branch_router_aux_loss_coef",
            "branch_seq_aux_loss_coef",
            "branch_bias_update_rate",
            "branch_bias_update_zero_sum",
            "branch_bias_warmup_start",
            "branch_bias_warmup_steps",
            "branch_quantile_eta",
            "branch_quantile_target_q",
            "branch_quantile_global_state",
            "branch_exploration_rate",
            "branch_exploration_decay",
            "branch_exploration_min",
            "branch_exploration_warmup_steps",
            "branch_entropy_coef",
            "branch_entropy_decay",
            "branch_entropy_min",
            "branch_entropy_decay_steps",
        )
        for key in flat_branch_keys:
            if key in mcfg_local:
                issues.append(
                    f"active matrix yaml carries flat-bridge "
                    f"model.{key}={mcfg_local[key]!r}; the nested "
                    f"`model.branch_router` block is authoritative."
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

        # Recompute branch contract: active recompute rows must use
        # one of the documented branch policies. Drift here silently
        # changes the branch ablation axis.
        recompute_branch_specs = (
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
            {
                # `weighted_sum`: soft binary mixer — both branches always
                # compute and the outputs are blended by the softmax probs.
                # No additional knobs (no decay, no entropy bonus).
                "balancing": "weighted_sum",
            },
        )
        if mcfg_local.get("attn_expert_mode") in {"per_head_recompute_k", "per_head_recompute_kv"}:
            branch_block = mcfg_local.get("branch_router", {}) or {}
            if not any(
                all(branch_block.get(knob) == expected for knob, expected in spec.items())
                for spec in recompute_branch_specs
            ):
                issues.append(
                    "active matrix yaml: recompute attention row must use one "
                    "of the documented branch_router specs: sampling_entropy, "
                    "exploration_only, fixed_alternating, or weighted_sum."
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
