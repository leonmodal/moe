"""Regression tests for `src/training/data.build_dataset_from_config`.

Round 10 code review found that the `prefetch_files` field added to
`DataConfig` in Round 3 was never copied through the trainer-side builder,
so `data.prefetch_files: 0` in a training config was silently ignored.
Tests below pin the plumbing so future additions to `DataConfig` surface
here if the builder forgets to copy them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data import build_dataset_from_config


def _write_fixture(root: Path, num_files: int = 2, rows_per_file: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for f in range(num_files):
        rows = [f"file{f} row{r}" for r in range(rows_per_file)]
        pd.DataFrame({"text": rows}).to_parquet(root / f"shard_{f:04d}.parquet")
    return root


class _StubTokenizer:
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (hash(tok) & 0xFD) for tok in text.split()] + [3]


@pytest.mark.parametrize("prefetch_files", [0, 1, 2])
def test_build_dataset_from_config_plumbs_prefetch_files(tmp_path, prefetch_files):
    data_dir = _write_fixture(tmp_path / "data")
    ds = build_dataset_from_config(
        {"data_dir": str(data_dir), "prefetch_files": prefetch_files},
        rank=0, world_size=1, seed=0, tokenizer=_StubTokenizer(),
    )
    assert ds.config.prefetch_files == prefetch_files, (
        f"`data.prefetch_files: {prefetch_files}` must propagate into "
        f"DataConfig; got {ds.config.prefetch_files}"
    )


def test_build_dataset_from_config_prefetch_defaults_to_one(tmp_path):
    # When the YAML omits the field, the default must match `DataConfig`
    # (read one file ahead).
    data_dir = _write_fixture(tmp_path / "data")
    ds = build_dataset_from_config(
        {"data_dir": str(data_dir)},
        rank=0, world_size=1, seed=0, tokenizer=_StubTokenizer(),
    )
    assert ds.config.prefetch_files == 1
