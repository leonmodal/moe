"""Regression test for the eval-dataloader worker policy.

Round 11 briefly allowed the eval DataLoader to honour `data.num_workers`
directly, but Codex's Round 12 review found that was incorrect for this
codebase: `StatefulParquetDataset` is an `IterableDataset` whose `__iter__`
shards only on `rank`/`world_size` and never consults
`torch.utils.data.get_worker_info()`. With `num_workers > 0` each DataLoader
worker iterates the same file subset, so every validation example is
emitted once per worker → inflated eval step counts and wrong metrics.

Round 12 restored the `_stateful_dataloader_workers` clamp for the eval
DataLoader. This test pins that contract: both train AND eval clamp to 0
for stateful / `IterableDataset` parquet datasets until per-worker
sub-sharding is implemented inside `StatefulParquetDataset.__iter__`.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import run_training


def _trainer_source() -> str:
    return inspect.getsource(run_training)


def test_train_dataloader_clamped_for_stateful_datasets():
    """Train-side clamp must remain (AC-12 resume correctness)."""
    src = _trainer_source()
    assert "_stateful_dataloader_workers" in src, (
        "Trainer no longer calls `_stateful_dataloader_workers`. The train "
        "DataLoader must keep clamping to 0 for stateful datasets so "
        "`dataset.get_state()` stays authoritative at checkpoint time."
    )


def test_eval_dataloader_also_clamped_until_worker_sharding_exists():
    """Eval DataLoader must also clamp to 0 for `StatefulParquetDataset`.

    The dataset's `__iter__` shards only on `rank` / `world_size`; without
    per-worker sub-sharding via `get_worker_info()`, multi-worker eval
    would duplicate every example once per worker.
    """
    src = _trainer_source()
    # Find the eval-side section and assert it routes through the clamp.
    eval_block_start = src.find("eval_dataloader = DataLoader(")
    assert eval_block_start != -1, "eval_dataloader DataLoader(...) not found"
    eval_section_start = src.rfind("if eval_dataset is not None:", 0, eval_block_start)
    eval_section = src[eval_section_start:eval_block_start + 500]
    assert "_stateful_dataloader_workers" in eval_section, (
        "Eval DataLoader must call `_stateful_dataloader_workers` — Round 12 "
        "restored the clamp because `StatefulParquetDataset` is an "
        "IterableDataset without per-worker sharding; multi-worker eval "
        "duplicates validation data."
    )
    # Additionally assert the `role="eval"` label so the banner message and
    # clamp attribution stay distinguishable from the train side.
    assert 'role="eval"' in eval_section, (
        "Eval-side `_stateful_dataloader_workers` call must pass "
        "`role='eval'` for log clarity."
    )


def test_eval_dataloader_banner_notes_iterabledataset_constraint():
    """The inline comment must document *why* eval clamps, not just that it
    does — a future edit rationalising "eval state isn't checkpointed, so
    let it parallelise" would reintroduce the R11 bug. Pin the reason.
    """
    src = _trainer_source()
    # We look for the key phrase flagged by Codex: `IterableDataset` +
    # worker-info sharding story. The exact wording can evolve, but these
    # keywords together prove the comment is about duplication, not resume.
    eval_block_start = src.find("eval_dataloader = DataLoader(")
    eval_section_start = src.rfind("if eval_dataset is not None:", 0, eval_block_start)
    comment_span = src[eval_section_start:eval_block_start + 200]
    assert "IterableDataset" in comment_span or "get_worker_info" in comment_span, (
        "Eval DataLoader clamp must be documented by the IterableDataset / "
        "get_worker_info reason, not by a resume-safety rationale."
    )
