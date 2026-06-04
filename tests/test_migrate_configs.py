"""Tests for `scripts/migrate_configs.py`. Covers the migration
contract end-to-end: flat fields move to nested groups, deprecated
tokens are rewritten, conflicts are handled, and the migrator is
idempotent across re-runs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def _load_migrator():
    spec = importlib.util.spec_from_file_location(
        "migrate_configs", str(REPO / "scripts" / "migrate_configs.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migrate_flat_branch_fields_to_nested():
    mc = _load_migrator()
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_balancing": "exploration_only",
            "branch_exploration_rate": 0.7,
            "branch_exploration_decay": "cosine",
            "branch_exploration_min": 0.05,
            "branch_exploration_warmup_steps": 1000,
        }
    }
    changes = mc.migrate_config(cfg)
    assert any("branch_router" in c for c in changes)
    assert "branch_balancing" not in cfg["model"]
    assert "branch_exploration_rate" not in cfg["model"]
    nested = cfg["model"]["branch_router"]
    assert nested["balancing"] == "exploration_only"
    assert nested["exploration_rate"] == 0.7
    assert nested["exploration_decay"] == "cosine"
    assert nested["exploration_min"] == 0.05
    assert nested["exploration_warmup_steps"] == 1000


def test_migrate_idempotent():
    """Running the migrator twice on the same yaml leaves the
    second run with zero changes."""
    mc = _load_migrator()
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_balancing": "exploration_only",
            "branch_exploration_rate": 0.7,
        }
    }
    first_changes = mc.migrate_config(cfg)
    assert first_changes
    second_changes = mc.migrate_config(cfg)
    assert second_changes == [], (
        f"migrator is not idempotent: second run produced {second_changes}"
    )


def test_migrate_renames_load_balancing_method_aliases():
    mc = _load_migrator()
    cfg = {"training": {"load_balancing_method": "switch"}}
    changes = mc.migrate_config(cfg)
    assert any("aux_loss" in c for c in changes)
    assert cfg["training"]["load_balancing_method"] == "aux_loss"

    cfg = {"training": {"load_balancing_method": "seq_aux"}}
    mc.migrate_config(cfg)
    assert cfg["training"]["load_balancing_method"] == "seq_aux_loss"


def test_migrate_renames_router_topk_ordering_to_softmax_position():
    mc = _load_migrator()
    cfg = {"model": {"type": "moe_everything", "router_topk_ordering": "post_softmax"}}
    changes = mc.migrate_config(cfg)
    assert any("softmax_position" in c for c in changes)
    assert "router_topk_ordering" not in cfg["model"]
    assert cfg["model"]["softmax_position"] == "post_softmax"


def test_migrate_dec17_value_mapping_post_to_pre_topk():
    """Round 27 review Finding 1 fix: the rename from
    router_topk_ordering -> softmax_position also flips the
    semantic axis. `router_topk_ordering: post` (top-k AFTER
    softmax) maps to `softmax_position: pre_topk` (softmax BEFORE
    top-k), and vice versa.
    """
    mc = _load_migrator()
    cfg = {"model": {"router_topk_ordering": "post"}}
    mc.migrate_config(cfg)
    assert cfg["model"]["softmax_position"] == "pre_topk", (
        f"DEC-17 value mapping post -> pre_topk not applied; got "
        f"{cfg['model']['softmax_position']!r}"
    )
    cfg = {"model": {"router_topk_ordering": "pre"}}
    mc.migrate_config(cfg)
    assert cfg["model"]["softmax_position"] == "post_topk"


def test_migrate_expand_top_level_aux_loss_into_per_class_blocks():
    """Top-level load_balancing_method expands into per-class blocks.

    The migrator leaves branch routing as `none` by default so old global
    method choices do not accidentally change the attention-vs-MLP schedule.
    Branch methods are selected explicitly with `model.branch_router`.
    """
    mc = _load_migrator()
    cfg = {
        "training": {
            "load_balancing_method": "aux_loss",
            "router_aux_loss_coef": 0.001,
        },
        "model": {"type": "moe_everything"},
    }
    changes = mc.migrate_config(cfg)
    assert any("mlp_router" in c for c in changes)
    assert cfg["model"]["mlp_router"]["balancing"] == "aux_loss"
    assert cfg["model"]["mlp_router"]["router_aux_loss_coef"] == 0.001
    assert cfg["model"]["attn_router"]["balancing"] == "aux_loss"
    assert cfg["model"]["branch_router"]["balancing"] == "none"


def test_migrate_expand_top_level_seq_aux_into_per_class():
    mc = _load_migrator()
    cfg = {
        "training": {
            "load_balancing_method": "seq_aux_loss",
            "seq_aux_loss_coef": 0.0001,
        },
        "model": {"type": "moe_everything"},
    }
    mc.migrate_config(cfg)
    assert cfg["model"]["mlp_router"]["balancing"] == "seq_aux_loss"
    assert cfg["model"]["mlp_router"]["seq_aux_loss_coef"] == 0.0001
    assert cfg["model"]["attn_router"]["balancing"] == "seq_aux_loss"


def test_migrate_expand_top_level_deepseek_bias_into_per_class():
    mc = _load_migrator()
    cfg = {
        "training": {
            "load_balancing_method": "deepseek_bias",
            "bias_update_rate": 0.001,
        },
        "model": {"type": "moe_everything"},
    }
    mc.migrate_config(cfg)
    assert cfg["model"]["mlp_router"]["balancing"] == "deepseek_bias"
    assert cfg["model"]["mlp_router"]["bias_update_rate"] == 0.001
    assert cfg["model"]["attn_router"]["balancing"] == "deepseek_bias"
    # Branch stays `none` unless explicitly configured via model.branch_router.
    assert cfg["model"]["branch_router"]["balancing"] == "none"


def test_migrate_does_not_duplicate_per_class_when_already_present():
    """When mlp_router or attn_router is already set, the
    expansion step is a no-op — the operator's nested block is
    authoritative."""
    mc = _load_migrator()
    cfg = {
        "training": {"load_balancing_method": "aux_loss",
                     "router_aux_loss_coef": 0.5},
        "model": {
            "type": "moe_everything",
            "mlp_router": {"balancing": "deepseek_bias", "bias_update_rate": 0.001},
        }
    }
    mc.migrate_config(cfg)
    # Migration must NOT overwrite the operator's mlp_router.
    assert cfg["model"]["mlp_router"]["balancing"] == "deepseek_bias"
    assert "attn_router" not in cfg["model"]


def test_migrate_warns_on_conflicting_nested_and_flat():
    """When the same field appears in both flat and nested form
    with DIFFERENT values, the migrator does NOT silently drop
    one — it emits a WARNING change record so the operator
    reconciles manually."""
    mc = _load_migrator()
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_balancing": "none",
            "branch_router": {"balancing": "exploration_only"},
        }
    }
    changes = mc.migrate_config(cfg)
    assert any("WARNING" in c for c in changes), (
        f"expected a WARNING change record on conflicting values; got {changes}"
    )
    # Both forms should still be present so the operator can resolve.
    assert "branch_balancing" in cfg["model"]
    assert cfg["model"]["branch_router"]["balancing"] == "exploration_only"


def test_migrate_collapses_redundant_equal_values():
    mc = _load_migrator()
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_balancing": "exploration_only",
            "branch_router": {"balancing": "exploration_only"},
        }
    }
    changes = mc.migrate_config(cfg)
    assert any("redundant" in c for c in changes)
    assert "branch_balancing" not in cfg["model"]
    assert cfg["model"]["branch_router"]["balancing"] == "exploration_only"


def test_migrate_noop_on_already_nested():
    mc = _load_migrator()
    cfg = {
        "model": {
            "type": "moe_everything",
            "branch_router": {
                "balancing": "exploration_only",
                "exploration_rate": 0.5,
            },
        }
    }
    assert mc.migrate_config(cfg) == []


def test_migrate_noop_on_unrelated_yaml():
    mc = _load_migrator()
    cfg = {
        "training": {"learning_rate": 1e-3, "batch_size": 8},
        "model": {"type": "dense", "vocab_size": 32},
    }
    assert mc.migrate_config(cfg) == []


def test_existing_repo_yamls_migrate_without_warnings(tmp_path):
    """Sanity over every shipped yaml: the migrator should produce
    EITHER zero changes (already nested or never used branch fields)
    OR only NON-WARNING changes (clean upgrade). Any WARNING change
    record on a shipped yaml indicates an internal inconsistency
    we need to fix in the source yaml first.
    """
    mc = _load_migrator()
    paths = sorted(REPO.glob("configs/**/*.yaml"))
    assert len(paths) > 0
    warnings: list[tuple[str, list[str]]] = []
    for p in paths:
        with p.open() as f:
            cfg = yaml.safe_load(f)
        changes = mc.migrate_config(cfg)
        warning_changes = [c for c in changes if "WARNING" in c]
        if warning_changes:
            warnings.append((str(p), warning_changes))
    assert not warnings, (
        f"migrator emitted WARNING records on shipped yamls; "
        f"reconcile manually: {warnings}"
    )


def test_migrate_file_writes_back_when_changes_apply(tmp_path):
    mc = _load_migrator()
    p = tmp_path / "before.yaml"
    p.write_text(yaml.safe_dump({
        "model": {
            "type": "moe_everything",
            "branch_balancing": "exploration_only",
            "branch_exploration_rate": 0.5,
        }
    }))
    changes, written = mc._migrate_file(p, dry_run=False)
    assert changes
    assert written
    with p.open() as f:
        out = yaml.safe_load(f)
    assert "branch_balancing" not in out["model"]
    assert out["model"]["branch_router"]["balancing"] == "exploration_only"


def test_migrate_file_dry_run_does_not_write(tmp_path):
    mc = _load_migrator()
    p = tmp_path / "before.yaml"
    original_text = yaml.safe_dump({
        "model": {
            "type": "moe_everything",
            "branch_balancing": "none",
        }
    })
    p.write_text(original_text)
    changes, written = mc._migrate_file(p, dry_run=True)
    assert changes
    assert not written
    assert p.read_text() == original_text


if __name__ == "__main__":
    print("Run via pytest")
