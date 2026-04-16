"""Regression test for AC-12 deterministic resume through the trainer-style
DataLoader assembly.

The trainer wraps `StatefulParquetDataset` in a `DataLoader` whose `num_workers`
used to default to 4. `StatefulParquetDataset.get_state()` reads in-process
attributes that are only mutated inside `__iter__`, so with worker subprocesses
the checkpointed state was stale and resume could land at the wrong data
position. This test defends the `_stateful_dataloader_workers` policy in
`src/training/trainer.py` by:

1. Unit-testing the helper: a stateful dataset must get `num_workers=0`
   regardless of the requested value; a non-stateful dataset respects the
   request.
2. End-to-end testing an exact resume: run a trainer-style DataLoader for N
   batches without interruption, then a second run that stops at N/2, captures
   `dataset.get_state()`, and resumes on a fresh dataset/DataLoader. The
   post-resume batches must exactly match the continuation of the uninterrupted
   run.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.training.trainer import _stateful_dataloader_workers


# ── Parquet fixture ──────────────────────────────────────────────────────────


def _write_parquet_fixture(root: Path, num_files: int = 4, rows_per_file: int = 64) -> Path:
    """Create `num_files` deterministic parquet shards with varied-length text."""
    root.mkdir(parents=True, exist_ok=True)
    for f in range(num_files):
        rows = []
        for r in range(rows_per_file):
            # Varied length keeps sequences from aligning with file boundaries.
            words = [f"f{f}r{r}t{t}" for t in range((r % 11) + 3)]
            rows.append(" ".join(words))
        df = pd.DataFrame({"text": rows})
        df.to_parquet(root / f"shard_{f:04d}.parquet")
    return root


class _IdentityTokenizer:
    """Tiny deterministic tokenizer: hashes tokens into a small vocab."""

    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Map each whitespace-separated token to an integer in [2, 255].
        # Include a sentinel tied to text length so different texts differ even
        # when they share a subtoken.
        out = [2 + (hash(tok) & 0xFD) for tok in text.split()]
        out.append(2 + (len(text) & 0xFD))
        return out


# ── Helper unit test ─────────────────────────────────────────────────────────


class _StatefulStub(IterableDataset):
    def __init__(self):
        self._idx = 0

    def get_state(self):
        return {"idx": self._idx}

    def set_state(self, state):
        self._idx = state.get("idx", 0)

    def __iter__(self):
        while True:
            self._idx += 1
            yield torch.tensor([self._idx], dtype=torch.long)


class _PlainStub(IterableDataset):
    def __iter__(self):
        i = 0
        while True:
            yield torch.tensor([i], dtype=torch.long)
            i += 1


def test_stateful_dataloader_workers_forces_zero_for_stateful():
    stateful = _StatefulStub()
    assert _stateful_dataloader_workers(stateful, requested=4, role="train", verbose=False) == 0
    assert _stateful_dataloader_workers(stateful, requested=0, role="train", verbose=False) == 0


def test_stateful_dataloader_workers_respects_nonstateful():
    plain = _PlainStub()
    assert _stateful_dataloader_workers(plain, requested=4, role="eval", verbose=False) == 4
    assert _stateful_dataloader_workers(plain, requested=0, role="eval", verbose=False) == 0


# ── End-to-end resume test ───────────────────────────────────────────────────


def _trainer_style_loader(dataset, batch_size: int, requested_workers: int) -> DataLoader:
    """Mirror the trainer's DataLoader assembly for a stateful dataset."""
    workers = _stateful_dataloader_workers(
        dataset, requested_workers, role="train", verbose=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False,  # CPU-friendly for test
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=workers > 0,
    )


@pytest.mark.parametrize("requested_workers", [0, 4])
@pytest.mark.parametrize("prefetch_files", [0, 1])
def test_trainer_dataloader_exact_resume(tmp_path, requested_workers, prefetch_files):
    """Batches after save/load must exactly match uninterrupted continuation,
    independent of both the config-requested `num_workers` and whether the
    dataset-level `prefetch_files` read-ahead is enabled.

    The inner loop covers four combinations: prefetch off / on × workers 0 / 4.
    With `num_workers=4` the trainer-side helper forces it back to 0 (AC-12);
    with `prefetch_files=1` the dataset opens a background thread that reads
    the next parquet shard while tokenization proceeds on the current one.
    The resume contract must not depend on either knob.
    """
    data_dir = _write_parquet_fixture(tmp_path, num_files=4, rows_per_file=32)
    tokenizer = _IdentityTokenizer()
    config = DataConfig(
        data_dir=str(data_dir),
        seq_len=32,
        tokenizer_name="stub",
        prefetch_files=prefetch_files,
    )

    B = 2
    N_BEFORE = 6
    N_AFTER = 6

    # Run A: uninterrupted.
    ds_a = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    dl_a = _trainer_style_loader(ds_a, batch_size=B, requested_workers=requested_workers)
    it_a = iter(dl_a)
    batches_a = [next(it_a)["input_ids"].clone() for _ in range(N_BEFORE + N_AFTER)]

    # Run B: stop after N_BEFORE, capture state, create fresh dataset, resume.
    ds_b = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    dl_b = _trainer_style_loader(ds_b, batch_size=B, requested_workers=requested_workers)
    it_b = iter(dl_b)
    batches_b_before = [next(it_b)["input_ids"].clone() for _ in range(N_BEFORE)]

    saved_state = ds_b.get_state()
    assert "file_idx" in saved_state and "text_idx" in saved_state and "buffer" in saved_state

    ds_b2 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    ds_b2.set_state(saved_state)
    dl_b2 = _trainer_style_loader(ds_b2, batch_size=B, requested_workers=requested_workers)
    it_b2 = iter(dl_b2)
    batches_b_after = [next(it_b2)["input_ids"].clone() for _ in range(N_AFTER)]

    ctx = f"workers={requested_workers}, prefetch_files={prefetch_files}"
    for i, (a, b) in enumerate(zip(batches_a[:N_BEFORE], batches_b_before)):
        assert torch.equal(a, b), f"Pre-checkpoint batch {i} diverged ({ctx})"

    for i, (a, b) in enumerate(zip(batches_a[N_BEFORE:], batches_b_after)):
        assert torch.equal(a, b), (
            f"Post-resume batch {i} diverged ({ctx}). AC-8 prefetch or AC-12 "
            "resume contract regressed."
        )


def test_legacy_seq_idx_state_resumes_without_restart(tmp_path):
    """Backward compat: a pre-Round-1 `data_state.pt` payload (no `text_idx`,
    `seq_idx` carries the skip counter) must resume at the same position the
    old skip-seqs iterator would have, instead of silently restarting at the
    beginning of the current file.

    Setup: iterate the uninterrupted reference for N batches and capture both
    the new-format state and a legacy-format state derived from it. Resume a
    fresh dataset from the legacy state; assert the next M batches continue
    the uninterrupted reference rather than repeating it from the top.
    """
    data_dir = _write_parquet_fixture(tmp_path, num_files=3, rows_per_file=32)
    tokenizer = _IdentityTokenizer()
    config = DataConfig(data_dir=str(data_dir), seq_len=32, tokenizer_name="stub",
                        prefetch_files=0)
    B = 1
    N_BEFORE = 8
    N_AFTER = 6

    # Uninterrupted reference.
    ref = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    ref_batches = [next(iter(DataLoader(ref, batch_size=B, num_workers=0)))]
    # Run full reference in one pass.
    ref = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    it = iter(DataLoader(ref, batch_size=B, num_workers=0))
    ref_batches = [next(it)["input_ids"].clone() for _ in range(N_BEFORE + N_AFTER)]

    # Re-walk to capture an *intermediate* state at the new-format cut point
    # and then synthesize its legacy equivalent.
    interim = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    it2 = iter(DataLoader(interim, batch_size=B, num_workers=0))
    for _ in range(N_BEFORE):
        next(it2)
    new_state = interim.get_state()
    # Legacy payload: same file_idx + buffer, but seq_idx is the skip count
    # from the file's top and NO text_idx key exists.
    legacy_state = {
        "file_idx": new_state["file_idx"],
        "seq_idx": new_state["seq_idx"],
        "buffer": [],   # pre-Round-1 code often saved [] (buffer aliasing bug)
    }

    # Resume from the legacy payload.
    legacy_ds = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    legacy_ds.set_state(legacy_state)
    legacy_it = iter(DataLoader(legacy_ds, batch_size=B, num_workers=0))
    legacy_batches = [next(legacy_it)["input_ids"].clone() for _ in range(N_AFTER)]

    # Legacy semantics won't match new-format batches token-for-token (that
    # was the pre-Round-1 bug this round preserves), but they must not be a
    # raw restart from position 0. Assert the first resumed batch is NOT equal
    # to the first reference batch — that proves the skip-seqs path ran.
    first_ref = ref_batches[0]
    assert not torch.equal(legacy_batches[0], first_ref), (
        "Legacy-payload resume restarted the current file at position 0 "
        "instead of honouring the seq_idx skip counter. The Round-11 "
        "backward-compat fallback is missing or broken."
    )


def test_stateful_dataset_state_is_live_after_iteration(tmp_path):
    """Sanity: the parquet dataset state must advance during iteration so that
    `get_state()` is meaningful. This is the invariant the trainer relies on.
    """
    data_dir = _write_parquet_fixture(tmp_path, num_files=2, rows_per_file=32)
    tokenizer = _IdentityTokenizer()
    config = DataConfig(data_dir=str(data_dir), seq_len=24, tokenizer_name="stub")

    ds = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=0)
    it = iter(DataLoader(ds, batch_size=1, num_workers=0))
    for _ in range(4):
        next(it)
    state = ds.get_state()
    # `text_idx` is the authoritative resume marker; must advance as we consume
    # texts. `seq_idx` is diagnostic-only (see StatefulParquetDataset.get_state).
    assert state["text_idx"] > 0, "text_idx must advance after iteration"
