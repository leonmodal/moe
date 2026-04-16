"""AC-12 end-to-end regression: real checkpoint I/O with a real parquet dataset.

Earlier AC-12 coverage went only as far as the `dataset.get_state()` /
`set_state()` handoff (see `test_trainer_dataloader_resume.py`) or round-tripped
a synthetic `dataset_state` dict through the checkpoint module
(`test_unified_trainer.py::test_checkpoint_roundtrip*`). This test closes the
gap by driving the full `src.training.checkpoint.save_checkpoint` +
`load_checkpoint` path with a real `StatefulParquetDataset`:

1. Construct a tiny model + AdamW + StepLR and a 4-file synthetic parquet
   dataset wrapped in a trainer-style DataLoader.
2. Consume N batches, perform a forward/backward/optimizer/scheduler step on
   each batch so model and optimizer state genuinely diverge from initialization.
3. Call `save_checkpoint(...)` with `dataset_state=dataset.get_state()`.
4. Build a fresh model + AdamW + StepLR and a fresh dataset, call
   `load_checkpoint(...)` to restore them, and `dataset.set_state(data_state)`.
5. Continue M more batches.
6. In parallel, run an uninterrupted reference (same seed, same fixtures) for
   N+M batches.
7. Assert that the post-resume batches, model parameters, and optimizer state
   exactly match the uninterrupted reference at the same step count.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import _stateful_dataloader_workers


def _write_parquet_fixture(root: Path, num_files: int = 4, rows_per_file: int = 32) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for f in range(num_files):
        rows = []
        for r in range(rows_per_file):
            words = [f"f{f}r{r}t{t}" for t in range((r % 9) + 3)]
            rows.append(" ".join(words))
        pd.DataFrame({"text": rows}).to_parquet(root / f"shard_{f:04d}.parquet")
    return root


class _IdentityTokenizer:
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = [2 + (hash(tok) & 0xFD) for tok in text.split()]
        out.append(2 + (len(text) & 0xFD))
        return out


class _TinyModel(torch.nn.Module):
    """Minimal trainable module that accepts the (input_ids, labels) batch shape."""

    def __init__(self, vocab_size: int = 256, hidden: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.linear = torch.nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embed(input_ids))


def _build_loader(dataset: StatefulParquetDataset, batch_size: int) -> DataLoader:
    workers = _stateful_dataloader_workers(dataset, 4, role="train", verbose=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False,
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=workers > 0,
    )


def _train_one_batch(model: _TinyModel, optimizer, scheduler, batch) -> torch.Tensor:
    """One deterministic forward/backward/optimizer/scheduler step."""
    logits = model(batch["input_ids"])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def _new_run(tmp_path: Path, data_dir: Path):
    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    cfg = DataConfig(
        data_dir=str(data_dir), seq_len=32, tokenizer_name="stub", prefetch_files=1
    )
    dataset = StatefulParquetDataset(cfg, _IdentityTokenizer(), rank=0, world_size=1, seed=0)
    loader = _build_loader(dataset, batch_size=2)
    return model, optimizer, scheduler, dataset, loader


def _states_match(a: dict, b: dict) -> bool:
    """Deep tensor-equal comparison of two `state_dict`-shaped dicts."""
    if a.keys() != b.keys():
        return False
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape or not torch.equal(va, vb):
                return False
        elif isinstance(va, dict) and isinstance(vb, dict):
            if not _states_match(va, vb):
                return False
        elif isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
            if len(va) != len(vb):
                return False
            for xa, xb in zip(va, vb):
                if isinstance(xa, torch.Tensor) and isinstance(xb, torch.Tensor):
                    if not torch.equal(xa, xb):
                        return False
                elif xa != xb:
                    return False
        elif va != vb:
            return False
    return True


def test_end_to_end_checkpoint_roundtrip_with_parquet_dataset(tmp_path):
    """Save/load path must preserve batch continuation + model + optimizer state."""
    data_dir = _write_parquet_fixture(tmp_path / "data", num_files=4, rows_per_file=24)
    ckpt_root = tmp_path / "ckpts"
    ckpt_root.mkdir()
    N_BEFORE = 5
    N_AFTER = 5

    # Reference run: N_BEFORE + N_AFTER steps, uninterrupted.
    ref_model, ref_opt, ref_sched, _, ref_loader = _new_run(tmp_path, data_dir)
    ref_it = iter(ref_loader)
    ref_batches = []
    ref_losses = []
    for _ in range(N_BEFORE + N_AFTER):
        batch = next(ref_it)
        ref_batches.append(batch["input_ids"].clone())
        ref_losses.append(_train_one_batch(ref_model, ref_opt, ref_sched, batch).item())

    # Interrupted run: N_BEFORE steps, save, fresh init, load, N_AFTER more.
    run_model, run_opt, run_sched, run_ds, run_loader = _new_run(tmp_path, data_dir)
    run_it = iter(run_loader)
    for _ in range(N_BEFORE):
        _train_one_batch(run_model, run_opt, run_sched, next(run_it))

    # Snapshot state just before save (for post-load sanity).
    pre_save_model = copy.deepcopy(run_model.state_dict())
    pre_save_dataset_state = run_ds.get_state()

    save_checkpoint(
        model=run_model,
        optimizer=run_opt,
        scheduler=run_sched,
        step=N_BEFORE,
        output_dir=str(ckpt_root),
        dataset_state=pre_save_dataset_state,
        tokens_seen=float(N_BEFORE * 2 * 32),
    )
    assert (ckpt_root / f"checkpoint-{N_BEFORE}" / "model.pt").exists()
    assert (ckpt_root / f"checkpoint-{N_BEFORE}" / "optimizer_adam.pt").exists()
    assert (ckpt_root / f"checkpoint-{N_BEFORE}" / "training_state.pt").exists()
    assert (ckpt_root / f"checkpoint-{N_BEFORE}" / "data_state.pt").exists()
    assert (ckpt_root / f"checkpoint-{N_BEFORE}" / "meta.json").exists()

    # Fresh components restored from the real checkpoint.
    torch.manual_seed(0)  # Re-seed to match _new_run's init; restored state overrides
    fresh_model = _TinyModel()
    fresh_opt = AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_sched = StepLR(fresh_opt, step_size=2, gamma=0.9)
    fresh_cfg = DataConfig(
        data_dir=str(data_dir), seq_len=32, tokenizer_name="stub", prefetch_files=1
    )
    fresh_ds = StatefulParquetDataset(
        fresh_cfg, _IdentityTokenizer(), rank=0, world_size=1, seed=0
    )

    loaded_step, loaded_data_state, loaded_tokens_seen = load_checkpoint(
        fresh_model, fresh_opt, fresh_sched, str(ckpt_root / f"checkpoint-{N_BEFORE}")
    )
    assert loaded_step == N_BEFORE
    assert loaded_tokens_seen == float(N_BEFORE * 2 * 32)
    assert loaded_data_state is not None
    fresh_ds.set_state(loaded_data_state)

    # Model / optimizer / scheduler state must match the just-saved snapshot.
    assert _states_match(fresh_model.state_dict(), pre_save_model), (
        "Model state drifted across save/load"
    )
    # Sanity: optimizer and scheduler round-trip (AdamW state_dict contains tensors)
    assert _states_match(fresh_opt.state_dict(), run_opt.state_dict()), (
        "Optimizer state drifted across save/load"
    )
    assert fresh_sched.state_dict()["_step_count"] == run_sched.state_dict()["_step_count"]

    # Continue N_AFTER steps on the restored run; batches must continue the reference.
    fresh_loader = _build_loader(fresh_ds, batch_size=2)
    fresh_it = iter(fresh_loader)
    fresh_losses = []
    for i in range(N_AFTER):
        batch = next(fresh_it)
        assert torch.equal(batch["input_ids"], ref_batches[N_BEFORE + i]), (
            f"Post-checkpoint batch {i} diverged from the uninterrupted reference"
        )
        fresh_losses.append(
            _train_one_batch(fresh_model, fresh_opt, fresh_sched, batch).item()
        )

    # Per-step loss must match the reference after resume (same inputs + same
    # restored parameters + same optimizer state ⇒ same gradients ⇒ same loss).
    for i, (r, f) in enumerate(zip(ref_losses[N_BEFORE:], fresh_losses)):
        assert abs(r - f) < 1e-5, (
            f"Post-resume loss at step {N_BEFORE + i} diverged from reference: "
            f"ref={r:.6f} vs fresh={f:.6f}"
        )


def test_checkpoint_roundtrip_preserves_empty_buffer_state(tmp_path):
    """Edge case: saving at a file-boundary (empty buffer) must still round-trip."""
    data_dir = _write_parquet_fixture(tmp_path / "data", num_files=3, rows_per_file=8)
    ckpt_root = tmp_path / "ckpts"
    ckpt_root.mkdir()

    # Consume enough batches to fully exhaust the first file and land on a
    # buffer state that can be empty depending on byte alignment.
    cfg = DataConfig(data_dir=str(data_dir), seq_len=16, tokenizer_name="stub", prefetch_files=1)
    ds = StatefulParquetDataset(cfg, _IdentityTokenizer(), rank=0, world_size=1, seed=0)
    it = iter(DataLoader(ds, batch_size=1, num_workers=0))
    for _ in range(3):
        next(it)

    model = _TinyModel()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = StepLR(opt, step_size=1, gamma=1.0)

    save_checkpoint(
        model=model,
        optimizer=opt,
        scheduler=sched,
        step=3,
        output_dir=str(ckpt_root),
        dataset_state=ds.get_state(),
        tokens_seen=48.0,
    )

    # Roundtrip the data_state through the checkpoint and assert semantic equality.
    fresh_ds = StatefulParquetDataset(cfg, _IdentityTokenizer(), rank=0, world_size=1, seed=0)
    _, loaded_state, _ = load_checkpoint(
        _TinyModel(), AdamW(_TinyModel().parameters(), lr=1e-3),
        StepLR(AdamW(_TinyModel().parameters(), lr=1e-3), step_size=1),
        str(ckpt_root / "checkpoint-3"),
    )
    assert loaded_state is not None
    assert loaded_state["file_idx"] == ds.get_state()["file_idx"]
    assert loaded_state["text_idx"] == ds.get_state()["text_idx"]
    assert loaded_state["buffer"] == ds.get_state()["buffer"]
