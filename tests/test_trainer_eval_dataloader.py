"""Regression test for the eval-dataloader worker policy.

Round 11 code review found that the trainer was running the
`_stateful_dataloader_workers` clamp on the EVAL DataLoader too, forcing
it to `num_workers=0` whenever the dataset exposed `get_state`/`set_state`.
Eval dataset state is never persisted by `save_checkpoint` / restored by
`load_checkpoint`, so the clamp had no resume-safety benefit and only
regressed throughput on large parquet val sets. This test pins the current
policy: the train DataLoader still clamps to 0 for stateful datasets; the
eval DataLoader honours `data.num_workers` directly.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import run_training


def _trainer_source() -> str:
    return inspect.getsource(run_training)


def test_train_dataloader_still_clamped_for_stateful_datasets():
    """The train-side clamp must remain — AC-12 resume correctness depends on
    the main-process `get_state()` being authoritative."""
    src = _trainer_source()
    # The clamp call must still exist for the train DataLoader.
    assert "_stateful_dataloader_workers" in src, (
        "Trainer no longer calls `_stateful_dataloader_workers`. The train "
        "DataLoader must keep clamping to 0 for stateful datasets so "
        "`dataset.get_state()` stays authoritative at checkpoint time."
    )


def test_eval_dataloader_does_not_clamp_stateful_datasets():
    """Eval state is never persisted; the helper must not clamp eval workers."""
    src = _trainer_source()
    # Locate the eval_dataloader = DataLoader(...) construction and verify it
    # does NOT pass a clamped value. We look for the key signature markers:
    # the explanatory comment that was added in Round 11, and the absence of
    # a second `_stateful_dataloader_workers(eval_dataset, ...)` call.
    eval_block_start = src.find("eval_dataloader = DataLoader(")
    assert eval_block_start != -1, "eval_dataloader DataLoader(...) not found"
    # Back up to find the start of the if eval_dataset is not None: block.
    eval_section_start = src.rfind("if eval_dataset is not None:", 0, eval_block_start)
    eval_section = src[eval_section_start:eval_block_start + 500]
    assert "_stateful_dataloader_workers" not in eval_section, (
        "Eval DataLoader construction still calls `_stateful_dataloader_workers`. "
        "Round 11 reverted that clamp — eval state is never checkpointed."
    )


def test_eval_dataloader_uses_data_num_workers_directly():
    src = _trainer_source()
    # The eval block must compute `eval_workers = max(0, int(data_num_workers))`
    # (or an equivalent pass-through) so config `data.num_workers > 0` takes
    # effect for eval.
    assert "eval_workers = max(0, int(data_num_workers))" in src, (
        "Eval DataLoader must honour `data.num_workers` directly; the "
        "Round 11 fix uses `eval_workers = max(0, int(data_num_workers))`."
    )
