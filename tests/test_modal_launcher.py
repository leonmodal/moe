"""AC-9 unit test: pin the `modal_train.py` torchrun argv shape.

`modal_train.train()` is a Modal decorator stack and cannot be invoked in a
plain pytest process. We instead expose the argv assembly as a pure helper
(`build_train_script_args`) and pin its output here so a future refactor
cannot silently drop one of the required flags (`--config`, `--auto_resume`,
`--data_dir`, `--output_dir`, `--max_checkpoints`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

modal_train = pytest.importorskip("modal_train")


def test_build_train_script_args_pins_required_flags():
    args = modal_train.build_train_script_args(
        "/root/moe/configs/scaling/m_standard.yaml",
        data_dir="/data/parquet",
        output_dir="/checkpoints/m_standard",
        max_checkpoints=3,
    )
    assert args == [
        "--config", "/root/moe/configs/scaling/m_standard.yaml",
        "--auto_resume",
        "--data_dir", "/data/parquet",
        "--output_dir", "/checkpoints/m_standard",
        "--max_checkpoints", "3",
    ]


def test_build_train_script_args_can_disable_auto_resume():
    args = modal_train.build_train_script_args(
        "/cfg.yaml", data_dir="/data", output_dir="/ckpt",
        max_checkpoints=0, auto_resume=False,
    )
    assert "--auto_resume" not in args
    assert args[:2] == ["--config", "/cfg.yaml"]
    assert args[-6:] == [
        "--data_dir", "/data",
        "--output_dir", "/ckpt",
        "--max_checkpoints", "0",
    ]


def test_resolve_output_dir_uses_experiment_name():
    assert modal_train.resolve_output_dir({"experiment_name": "my_exp"}) == "/checkpoints/my_exp"


def test_resolve_output_dir_defaults_when_experiment_name_missing():
    assert modal_train.resolve_output_dir({}) == "/checkpoints/default"


def test_resolve_output_dir_honors_custom_root():
    assert modal_train.resolve_output_dir(
        {"experiment_name": "x"}, checkpoint_root="/tmp/ckpts"
    ) == "/tmp/ckpts/x"


def test_training_script_constant_points_at_unified_entrypoint():
    # Pin the constant so a rename of scripts/train.py surfaces here.
    assert modal_train.TRAINING_SCRIPT == "/root/moe/scripts/train.py"
