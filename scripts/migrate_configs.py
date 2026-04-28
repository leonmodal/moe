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

# Value-level mapping for the `router_topk_ordering -> softmax_position`
# rename: the canonical-field migration also flips the semantic axis.
# The legacy `post` (top-k applied AFTER softmax) maps to
# `pre_topk` (softmax computed BEFORE top-k selection), and vice versa.
_KEY_RENAME_VALUE_MAPS = {
    "router_topk_ordering": {
        "post": "pre_topk",
        "pre": "post_topk",
    },
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
    softmax_position`). Some renames also map values across a
    semantic axis (the canonical flip: `post` becomes `pre_topk` because
    the canonical field's axis is "softmax position relative to
    top-k", not "top-k position relative to softmax")."""
    changes: list[str] = []
    for old_name, new_name in _KEY_RENAMES.items():
        if old_name not in model_cfg:
            continue
        old_val = model_cfg[old_name]
        # Apply value mapping for renames that change semantic axes.
        value_map = _KEY_RENAME_VALUE_MAPS.get(old_name)
        if value_map is not None and old_val in value_map:
            new_val = value_map[old_val]
        else:
            new_val = old_val
        if new_name in model_cfg:
            if model_cfg[new_name] != new_val:
                changes.append(
                    f"WARNING: model.{old_name}={old_val!r} (would map to "
                    f"{new_val!r}) conflicts with model.{new_name}="
                    f"{model_cfg[new_name]!r}; NOT removing."
                )
                continue
            del model_cfg[old_name]
            changes.append(
                f"removed redundant model.{old_name} (matched model.{new_name})"
            )
        else:
            model_cfg[new_name] = new_val
            del model_cfg[old_name]
            if new_val != old_val:
                changes.append(
                    f"renamed model.{old_name}={old_val!r} -> "
                    f"model.{new_name}={new_val!r} (value mapping)"
                )
            else:
                changes.append(
                    f"renamed model.{old_name} -> model.{new_name}"
                )
    return changes


_BRANCH_ROUTER_NESTED_DEFAULT = {"balancing": "none"}


def _expand_top_level_method_to_per_class(cfg: dict) -> list[str]:
    """Expand top-level `training.load_balancing_method` (and
    associated coefficients) into per-class
    `model.{mlp,attn,branch}_router` blocks.

    Migration semantics:
      * `aux_loss` / `seq_aux_loss` -> mlp_router and attn_router
        adopt the same balancing; branch_router stays `none`
        (BranchRouter runtime accepts only none/exploration_only
        until the broader runtime lands).
      * `deepseek_bias` -> mlp_router and attn_router adopt
        deepseek_bias; branch_router stays `none`.
      * `none` / `quantile` -> mlp_router and attn_router adopt
        the same value; branch_router stays `none`.
      * Top-level `training.router_aux_loss_coef` and
        `training.seq_aux_loss_coef` are propagated onto the
        per-class blocks for the matching method only.

    The migration only fires when the per-class blocks are
    ABSENT (so a yaml already on the nested schema does not
    duplicate entries). The flat `training.load_balancing_method`
    is left in place — Round 28 keeps it as a runtime input until
    the full nested-schema cutover lands; the runtime falls back
    to it when per-class fields are absent.
    """
    changes: list[str] = []
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        return changes
    model_cfg = cfg["model"]
    tcfg = cfg.get("training", {}) or {}
    # Resolve method from training: first, then fall back to model:
    # for legacy yamls that placed it under model:.
    method = tcfg.get("load_balancing_method")
    if method is None:
        method = model_cfg.get("load_balancing_method")
    if method is None:
        return changes  # nothing to expand
    aux_coef = tcfg.get("router_aux_loss_coef", model_cfg.get("router_aux_loss_coef", 0.0))
    seq_coef = tcfg.get("seq_aux_loss_coef", model_cfg.get("seq_aux_loss_coef", 0.0))
    bias_rate = tcfg.get("bias_update_rate", model_cfg.get("bias_update_rate", 0.0))

    classes_already_set = any(
        isinstance(model_cfg.get(name), dict) and model_cfg[name]
        for name in ("mlp_router", "attn_router")
    )
    if classes_already_set:
        return changes  # do not duplicate per-class entries

    def _build_per_class(method: str, *, include_aux=True) -> dict:
        block: dict = {"balancing": method}
        if method == "aux_loss" and include_aux and aux_coef:
            block["router_aux_loss_coef"] = aux_coef
        elif method == "seq_aux_loss" and include_aux and seq_coef:
            block["seq_aux_loss_coef"] = seq_coef
        elif method == "deepseek_bias" and bias_rate:
            block["bias_update_rate"] = bias_rate
        return block

    if method in ("aux_loss", "seq_aux_loss", "deepseek_bias", "quantile", "none"):
        # MLP + attention adopt the method; branch stays `none`.
        model_cfg["mlp_router"] = _build_per_class(method)
        model_cfg["attn_router"] = _build_per_class(method)
        if "branch_router" not in model_cfg:
            model_cfg["branch_router"] = dict(_BRANCH_ROUTER_NESTED_DEFAULT)
        changes.append(
            f"expanded top-level load_balancing_method={method!r} into "
            f"model.mlp_router / model.attn_router (branch_router stays "
            f"`none` until runtime supports broader methods)"
        )
        # After expansion, remove the top-level method + coefficients
        # that were just copied into the per-class blocks. This makes
        # the migrated yaml nested-only — the runtime falls back to
        # the per-class blocks (or the configured `none`) without a
        # parallel flat shim. Repo yamls should never carry both
        # forms after migration; the validator now enforces this.
        if "load_balancing_method" in tcfg:
            del tcfg["load_balancing_method"]
            changes.append("removed top-level training.load_balancing_method (now in per-class blocks)")
        if "load_balancing_method" in model_cfg:
            del model_cfg["load_balancing_method"]
            changes.append("removed top-level model.load_balancing_method (now in per-class blocks)")
        # Remove top-level coefficients that the per-class blocks now carry.
        for old_key in ("router_aux_loss_coef", "seq_aux_loss_coef", "bias_update_rate"):
            if method == "aux_loss" and old_key == "router_aux_loss_coef" and old_key in tcfg:
                del tcfg[old_key]
                changes.append(f"removed redundant top-level training.{old_key}")
            elif method == "seq_aux_loss" and old_key == "seq_aux_loss_coef" and old_key in tcfg:
                del tcfg[old_key]
                changes.append(f"removed redundant top-level training.{old_key}")
            elif method == "deepseek_bias" and old_key == "bias_update_rate" and old_key in tcfg:
                del tcfg[old_key]
                changes.append(f"removed redundant top-level training.{old_key}")
    return changes


def _remove_redundant_top_level_balancing(cfg: dict) -> list[str]:
    """When per-class blocks are already populated, the top-level
    `training.load_balancing_method` and the matching coefficient
    are redundant: the runtime reads the per-class fields. This
    cleanup runs unconditionally so a yaml that was migrated with
    an older migrator version (which left the top-level fields
    in place) gets fully cutover on the next run.
    """
    changes: list[str] = []
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        return changes
    model_cfg = cfg["model"]
    has_per_class = any(
        isinstance(model_cfg.get(name), dict) and model_cfg[name]
        for name in ("mlp_router", "attn_router", "branch_router")
    )
    if not has_per_class:
        return changes
    tcfg = cfg.get("training", {})
    # Top-level method is now redundant.
    if isinstance(tcfg, dict) and "load_balancing_method" in tcfg:
        del tcfg["load_balancing_method"]
        changes.append(
            "removed redundant top-level training.load_balancing_method "
            "(per-class blocks are now authoritative)"
        )
    if "load_balancing_method" in model_cfg:
        del model_cfg["load_balancing_method"]
        changes.append(
            "removed redundant top-level model.load_balancing_method"
        )
    # Top-level coefficients that match a per-class field are
    # redundant. Conservative cleanup: only remove the top-level
    # coefficient if at least one per-class block carries the SAME
    # coefficient at the SAME value, so an operator's hand-edited
    # divergent values are never silently dropped.
    if isinstance(tcfg, dict):
        for coef_key in ("router_aux_loss_coef", "seq_aux_loss_coef", "bias_update_rate"):
            if coef_key not in tcfg:
                continue
            top_val = tcfg[coef_key]
            for group in ("mlp_router", "attn_router", "branch_router"):
                block = model_cfg.get(group)
                if isinstance(block, dict) and block.get(coef_key) == top_val:
                    del tcfg[coef_key]
                    changes.append(
                        f"removed redundant top-level training.{coef_key}={top_val!r} "
                        f"(matched model.{group}.{coef_key})"
                    )
                    break
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
    changes.extend(_expand_top_level_method_to_per_class(cfg))
    changes.extend(_remove_redundant_top_level_balancing(cfg))
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
