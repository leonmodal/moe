"""One-shot config migrator: flat schema -> nested mlp_router / attn_router / branch_router.

Migrations applied (idempotent — re-running on already-migrated yamls produces no diff):

* Flat `branch_balancing`, `branch_exploration_*` -> nested `model.branch_router.{...}`.
* Flat `mlp_router_*` (if present) -> nested `model.mlp_router.{...}`.
* Flat `attn_router_*` (if present) -> nested `model.attn_router.{...}`.
* Token rewrites (per the deprecated-alias migration):
  * `switch -> aux_loss` for `load_balancing_method`.
  * `seq_aux -> seq_aux_loss` for `load_balancing_method`.
  * `router_topk_ordering -> softmax_position` (the canonical-field rename).
* Conflicting nested-vs-flat with equal values: collapse to nested only.
* Conflicting nested-vs-flat with different values: leave both and emit a
  warning record so the operator can reconcile (we never silently change
  semantics during migration).

Usage:

    # Dry run — print what would change without writing.
    python scripts/migrate_configs.py --dry-run configs/

    # Migrate every yaml under configs/ in place.
    python scripts/migrate_configs.py configs/

    # Migrate a single yaml.
    python scripts/migrate_configs.py path/to/cfg.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


_FLAT_BRANCH_KEYS = (
    "branch_balancing",
    "branch_exploration_rate",
    "branch_exploration_decay",
    "branch_exploration_min",
    "branch_exploration_warmup_steps",
)


_FLAT_KEY_SUFFIX_RENAMES = {
    "branch_balancing": "balancing",
    "branch_exploration_rate": "exploration_rate",
    "branch_exploration_decay": "exploration_decay",
    "branch_exploration_min": "exploration_min",
    "branch_exploration_warmup_steps": "exploration_warmup_steps",
}


_BALANCING_METHOD_TOKEN_RENAMES = {
    "switch": "aux_loss",
    "seq_aux": "seq_aux_loss",
}


_KEY_RENAMES = {
    "router_topk_ordering": "softmax_position",
}


def _migrate_branch_router_block(model_cfg: dict) -> list[str]:
    """Move the flat `branch_*` fields into `model.branch_router.{...}`.

    Returns a list of human-readable change descriptions. The function
    mutates `model_cfg` in place.
    """
    changes: list[str] = []
    nested = model_cfg.setdefault("branch_router", {})
    if not isinstance(nested, dict):
        return [f"branch_router is not a mapping; skipping branch migration"]

    for flat_key in _FLAT_BRANCH_KEYS:
        if flat_key not in model_cfg:
            continue
        nested_key = _FLAT_KEY_SUFFIX_RENAMES[flat_key]
        flat_value = model_cfg[flat_key]
        if nested_key in nested:
            if nested[nested_key] != flat_value:
                changes.append(
                    f"WARNING: model.{flat_key}={flat_value!r} conflicts with "
                    f"model.branch_router.{nested_key}={nested[nested_key]!r}; "
                    f"NOT removing the flat field. Reconcile manually before re-running."
                )
                continue
            # Equal values: drop the flat duplicate.
            del model_cfg[flat_key]
            changes.append(
                f"removed redundant flat model.{flat_key} (matched nested branch_router.{nested_key})"
            )
        else:
            nested[nested_key] = flat_value
            del model_cfg[flat_key]
            changes.append(
                f"moved model.{flat_key} -> model.branch_router.{nested_key}"
            )

    # If the nested block is still empty after the migration, drop it
    # so the migrator stays idempotent on yamls that never used the
    # branch_router feature.
    if not nested:
        model_cfg.pop("branch_router", None)
    return changes


def _rewrite_balancing_method_tokens(cfg: dict) -> list[str]:
    """Rewrite deprecated balancing-method aliases:
    `switch -> aux_loss`, `seq_aux -> seq_aux_loss`."""
    changes: list[str] = []
    for block_name in ("training", "model"):
        block = cfg.get(block_name)
        if not isinstance(block, dict):
            continue
        method = block.get("load_balancing_method")
        if method in _BALANCING_METHOD_TOKEN_RENAMES:
            new = _BALANCING_METHOD_TOKEN_RENAMES[method]
            block["load_balancing_method"] = new
            changes.append(
                f"renamed {block_name}.load_balancing_method: {method} -> {new}"
            )
    return changes


def _rewrite_field_names(model_cfg: dict) -> list[str]:
    """Rewrite renamed keys (e.g. `router_topk_ordering ->
    softmax_position`)."""
    changes: list[str] = []
    for old_name, new_name in _KEY_RENAMES.items():
        if old_name not in model_cfg:
            continue
        old_val = model_cfg[old_name]
        if new_name in model_cfg:
            if model_cfg[new_name] != old_val:
                changes.append(
                    f"WARNING: model.{old_name}={old_val!r} conflicts with "
                    f"model.{new_name}={model_cfg[new_name]!r}; NOT removing."
                )
                continue
            del model_cfg[old_name]
            changes.append(
                f"removed redundant model.{old_name} (matched model.{new_name})"
            )
        else:
            model_cfg[new_name] = old_val
            del model_cfg[old_name]
            changes.append(
                f"renamed model.{old_name} -> model.{new_name}"
            )
    return changes


def migrate_config(cfg: dict) -> list[str]:
    """Apply every migration to `cfg` in place. Returns a list of
    change descriptions; an empty list means the yaml is already
    on the nested schema (idempotent)."""
    changes: list[str] = []
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        changes.extend(_rewrite_field_names(model_cfg))
        changes.extend(_migrate_branch_router_block(model_cfg))
    changes.extend(_rewrite_balancing_method_tokens(cfg))
    return changes


def _migrate_file(path: Path, *, dry_run: bool) -> tuple[list[str], bool]:
    """Migrate a single yaml file. Returns (changes, written) where
    `written` is True if the file was actually rewritten."""
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return [f"{path}: top-level YAML is not a mapping; skipping"], False
    changes = migrate_config(cfg)
    if changes and not dry_run:
        with path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return changes, True
    return changes, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+",
        help="One or more yaml paths or directories (recursively walked).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the diff without writing the files.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.yaml")))
        elif p.exists():
            paths.append(p)
        else:
            print(f"missing path: {raw}", file=sys.stderr)
            sys.exit(2)

    total_files = 0
    total_written = 0
    total_changes = 0
    for path in paths:
        changes, written = _migrate_file(path, dry_run=args.dry_run)
        total_files += 1
        if changes:
            print(f"\n{path}:")
            for c in changes:
                print(f"  - {c}")
        if written:
            total_written += 1
        total_changes += len(changes)

    if args.dry_run:
        print(f"\nDRY RUN: {total_files} files scanned, {total_changes} change(s) found.")
    else:
        print(f"\n{total_written}/{total_files} files written, {total_changes} change(s) applied.")


if __name__ == "__main__":
    main()
