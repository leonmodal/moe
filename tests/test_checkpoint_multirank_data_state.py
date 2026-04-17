"""Pin the per-rank data_state save/load contract.

Before this fix only rank 0 wrote `data_state.pt`, which under multi-rank
runs meant every non-zero rank silently re-loaded rank 0's state and
jumped to a file in rank 0's parquet shard. That caused data duplication
and cross-rank desync on resume.

After the fix:
- every rank writes `data_state_rank{r}.pt`;
- rank 0 also writes the legacy unsuffixed `data_state.pt` so old
  single-process checkpoints still load;
- `_load_data_state(resume_from)` prefers the per-rank file and falls
  back to the shared one only if the per-rank file is absent.

These tests exercise the helper directly (single-process, `dist_rank()`
returns 0) because the real multi-rank path requires a live distributed
group; the invariant is "per-rank file wins over shared file when both
are present", which is provable on a single process by writing both
files with distinct payloads and asserting the loader returns the
per-rank one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import _load_data_state


def test_load_data_state_prefers_per_rank_file(tmp_path):
    ckpt = tmp_path / "checkpoint-42"
    ckpt.mkdir()
    per_rank = {"file_idx": 7, "text_idx": 3, "buffer": [1, 2]}
    shared = {"file_idx": 0, "text_idx": 0, "buffer": []}
    torch.save(per_rank, ckpt / "data_state_rank0.pt")
    torch.save(shared, ckpt / "data_state.pt")
    loaded = _load_data_state(str(ckpt))
    assert loaded == per_rank, (
        "when both `data_state_rank{r}.pt` and legacy `data_state.pt` are "
        "present, the per-rank file must win so each rank resumes at its "
        "own parquet shard position"
    )


def test_load_data_state_falls_back_to_shared(tmp_path):
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir()
    shared = {"file_idx": 3, "text_idx": 5, "buffer": [9]}
    torch.save(shared, ckpt / "data_state.pt")
    loaded = _load_data_state(str(ckpt))
    assert loaded == shared, (
        "when only the legacy shared `data_state.pt` is present (pre-fix "
        "checkpoint), it must be loaded as the fallback"
    )


def test_load_data_state_returns_none_when_absent(tmp_path):
    ckpt = tmp_path / "checkpoint-0"
    ckpt.mkdir()
    assert _load_data_state(str(ckpt)) is None
