"""One-shot yaml migrator: move balancing fields from `model:` to `training:`.

Per the canonical-block resolver design, every yaml under
`configs/` must place the following balancing-related fields in the `training:`
block, NOT the `model:` block:

  bias_update_rate
  bias_warmup_start
  bias_warmup_steps
  seq_aux_loss_coef
  router_aux_loss_coef
  load_balancing_method
  bias_rate_q / bias_rate_k / bias_rate_v / bias_rate_o / bias_rate_mlp / bias_rate_branch

Today's yamls put `bias_update_rate` (and friends) under `model:`, but
`TrainingConfig` reads them from `cfg["training"]`. The result: the trainer
silently runs with `bias_update_rate=0.0` no matter what the yaml says — the
deepseek bias-update path is never exercised in the production training path.
This migrator fixes that by literally moving the fields between blocks.

Usage:
  python scripts/migrate_balancing_fields_to_training.py [--dry-run] [<yaml-glob>]

Without arguments, runs in-place on every yaml under `configs/`.

Idempotent: re-running on an already-migrated yaml is a no-op.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# Fields that MUST live under the `training:` block (canonical-block resolver).
_BALANCING_FIELDS = {
    "bias_update_rate",
    "bias_warmup_start",
    "bias_warmup_steps",
    "seq_aux_loss_coef",
    "router_aux_loss_coef",
    "load_balancing_method",
    "bias_rate_q",
    "bias_rate_k",
    "bias_rate_v",
    "bias_rate_o",
    "bias_rate_mlp",
    "bias_rate_branch",
}


def migrate_yaml_text(text: str) -> tuple[str, list[str]]:
    """Migrate a single yaml file's text. Returns (new_text, moved_fields)."""
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        return text, []

    model_cfg = cfg.get("model")
    training_cfg = cfg.get("training")
    if not isinstance(model_cfg, dict) or not isinstance(training_cfg, dict):
        return text, []

    moved: list[str] = []
    for field in list(model_cfg.keys()):
        if field not in _BALANCING_FIELDS:
            continue
        # If the field is also already in `training:` and the values match,
        # the migration was already applied; just delete the model: copy.
        # If they disagree, prefer the `training:` value (it's what the
        # current trainer was reading, even if silently 0.0); record the
        # mismatch.
        model_value = model_cfg.pop(field)
        if field in training_cfg and training_cfg[field] != model_value:
            print(
                f"  WARN: {field} disagrees between blocks "
                f"(model: {model_value!r}, training: {training_cfg[field]!r}); "
                f"keeping training: value (canonical block authoritative).",
                file=sys.stderr,
            )
        elif field not in training_cfg:
            training_cfg[field] = model_value
        moved.append(field)

    if not moved:
        return text, []

    # Reorder so `training:` appears after `model:` (preserves the existing
    # convention in our yamls). yaml.safe_dump writes keys in dict insertion
    # order in Python 3.7+.
    new_cfg = {}
    for k in cfg:
        if k == "training":
            new_cfg[k] = training_cfg
        elif k == "model":
            new_cfg[k] = model_cfg
        else:
            new_cfg[k] = cfg[k]
    new_text = yaml.safe_dump(new_cfg, sort_keys=False, indent=2, default_flow_style=False)
    return new_text, moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument(
        "paths", nargs="*", default=None,
        help="yaml files or directories to migrate (default: configs/)"
    )
    args = parser.parse_args()

    paths = args.paths or ["configs"]
    yaml_files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            yaml_files.extend(sorted(path.rglob("*.yaml")))
        elif path.is_file():
            yaml_files.append(path)
        else:
            print(f"path not found: {p}", file=sys.stderr)
            return 2

    total_changed = 0
    for yaml_file in yaml_files:
        text = yaml_file.read_text()
        new_text, moved = migrate_yaml_text(text)
        if not moved:
            continue
        action = "would migrate" if args.dry_run else "migrated"
        print(f"{action} {yaml_file}: moved {moved} from model: -> training:")
        if not args.dry_run:
            yaml_file.write_text(new_text)
        total_changed += 1

    print(f"\nDone. {total_changed} yaml(s) changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
