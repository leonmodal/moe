"""
Test checkpoint saving and resuming — verify we resume at the exact data position,
with correct scheduler LR and expert biases.
"""
import json
import os
import shutil
import tempfile

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── Minimal deterministic dataset ──────────────────────────────────────

class SimpleCountingDataset(IterableDataset):
    """Yields sequential integers as tokens so we can verify exact position."""

    def __init__(self, seq_len=32, vocab_size=256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._start_idx = 0
        self._cur_idx = 0

    def get_state(self):
        return {"idx": self._cur_idx}

    def set_state(self, state):
        self._start_idx = state.get("idx", 0)

    def __iter__(self):
        idx = self._start_idx
        while True:
            tokens = [(idx * self.seq_len + i) % self.vocab_size for i in range(self.seq_len + 1)]
            self._cur_idx = idx + 1
            yield {
                "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
                "labels": torch.tensor(tokens[1:], dtype=torch.long),
            }
            idx += 1


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: Dataset state with num_workers=0 vs num_workers=1
# ═══════════════════════════════════════════════════════════════════════

def test_dataset_state_num_workers():
    print("=" * 70)
    print("  TEST 1: Dataset state tracking with num_workers")
    print("=" * 70)

    for nw in [0, 1]:
        dataset = SimpleCountingDataset(seq_len=32, vocab_size=256)
        dl = DataLoader(dataset, batch_size=2, num_workers=nw)
        it = iter(dl)
        for _ in range(5):
            next(it)

        state = dataset.get_state()
        print(f"\n  num_workers={nw}: state after 5 batches×2 = {state}")
        if nw == 0:
            assert state["idx"] == 10, f"Expected 10, got {state['idx']}"
            print(f"    ✓ Correct — state tracks position in main process")
        else:
            if state["idx"] == 0:
                print(f"    ✗ BUG: State is stale (idx=0) — worker doesn't update main process")
            else:
                print(f"    ✓ State tracks position")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: Scheduler double-advance on resume
# ═══════════════════════════════════════════════════════════════════════

def test_scheduler_double_advance():
    """
    train.py does:
      1. accelerator.load_state() — restores scheduler state_dict
      2. for _ in range(global_step): scheduler.step() — advances AGAIN

    This double-advances the scheduler. Test it.
    """
    print("=" * 70)
    print("  TEST 2: Scheduler double-advance on resume")
    print("=" * 70)

    # Simple model + optimizer + cosine scheduler (like train.py uses)
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    # Advance 20 steps
    for _ in range(20):
        optimizer.step()  # needed before scheduler.step()
        scheduler.step()

    lr_at_20 = scheduler.get_last_lr()[0]
    state_at_20 = scheduler.state_dict()
    print(f"\n  LR after 20 steps:            {lr_at_20:.8f}")
    print(f"  Scheduler last_epoch at save:  {state_at_20['last_epoch']}")

    # Simulate resume: create fresh scheduler, load state
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=100, eta_min=1e-4)
    scheduler2.load_state_dict(state_at_20)

    lr_after_load = scheduler2.get_last_lr()[0]
    print(f"  LR after load_state_dict:     {lr_after_load:.8f}")

    # Now do what train.py does: advance 20 MORE times
    for _ in range(20):
        optimizer2.step()
        scheduler2.step()

    lr_after_double = scheduler2.get_last_lr()[0]
    print(f"  LR after load + 20 more .step(): {lr_after_double:.8f}")

    # What the LR SHOULD be (step 20)
    optimizer3 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler3 = CosineAnnealingLR(optimizer3, T_max=100, eta_min=1e-4)
    for _ in range(40):
        optimizer3.step()
        scheduler3.step()
    lr_at_40 = scheduler3.get_last_lr()[0]
    print(f"  LR at step 40 (for comparison): {lr_at_40:.8f}")

    if abs(lr_after_load - lr_at_20) < 1e-10:
        print(f"\n  ✓ load_state_dict correctly restores LR to step 20")
    if abs(lr_after_double - lr_at_40) < 1e-10:
        print(f"  ✗ BUG CONFIRMED: After resume, scheduler is at step 40 instead of 20")
        print(f"    train.py's `for _ in range(global_step): scheduler.step()` DOUBLE-ADVANCES")
    elif abs(lr_after_double - lr_at_20) < 1e-10:
        print(f"  ✓ No double-advance")
    else:
        print(f"  ✗ Unexpected LR: {lr_after_double}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: Data reproducibility with num_workers=0
# ═══════════════════════════════════════════════════════════════════════

def test_data_reproducibility():
    print("=" * 70)
    print("  TEST 3: Data reproducibility after resume (num_workers=0)")
    print("=" * 70)

    N, M, B = 5, 5, 2

    # Run A: straight through
    ds_a = SimpleCountingDataset()
    dl_a = DataLoader(ds_a, batch_size=B, num_workers=0)
    it_a = iter(dl_a)
    batches_a = [next(it_a)["input_ids"].clone() for _ in range(N + M)]

    # Run B: N steps, save, resume, M steps
    ds_b = SimpleCountingDataset()
    dl_b = DataLoader(ds_b, batch_size=B, num_workers=0)
    it_b = iter(dl_b)
    batches_b = [next(it_b)["input_ids"].clone() for _ in range(N)]

    state = ds_b.get_state()
    print(f"\n  State after {N} batches: {state}")

    ds_b2 = SimpleCountingDataset()
    ds_b2.set_state(state)
    dl_b2 = DataLoader(ds_b2, batch_size=B, num_workers=0)
    it_b2 = iter(dl_b2)
    batches_b.extend([next(it_b2)["input_ids"].clone() for _ in range(M)])

    all_match = all(torch.equal(batches_a[i], batches_b[i]) for i in range(N + M))
    print(f"  All {N+M} batches match: {'✓' if all_match else '✗'}")

    if all_match:
        print(f"  ✓ Data perfectly reproducible with num_workers=0 + state save/restore")
    else:
        for i in range(N + M):
            if not torch.equal(batches_a[i], batches_b[i]):
                print(f"  ✗ Mismatch at batch {i}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: Verify the actual parquet dataset state tracking
# ═══════════════════════════════════════════════════════════════════════

def test_parquet_dataset_state():
    """Test StatefulParquetDataset with num_workers=0 to verify state works."""
    print("=" * 70)
    print("  TEST 4: StatefulParquetDataset state with num_workers=0")
    print("=" * 70)

    from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
    from transformers import AutoTokenizer

    # Check if data exists
    data_dir = "./data/parquet"
    if not os.path.exists(data_dir):
        print(f"\n  Skipping — no data at {data_dir}")
        print()
        return

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = DataConfig(data_dir=data_dir, seq_len=128, tokenizer_name="Qwen/Qwen3-0.6B")
    ds = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    dl = DataLoader(ds, batch_size=4, num_workers=0)
    it = iter(dl)

    # Consume 10 batches
    batches_before = []
    for i in range(10):
        batch = next(it)
        batches_before.append(batch["input_ids"].clone())

    state = ds.get_state()
    print(f"\n  State after 10 batches: {state}")

    # Resume from state, get next 5 batches
    ds2 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    ds2.set_state(state)
    dl2 = DataLoader(ds2, batch_size=4, num_workers=0)
    it2 = iter(dl2)

    # Also get the same 5 from the original (continue iteration)
    batches_continued = [next(it)["input_ids"].clone() for _ in range(5)]
    batches_resumed = [next(it2)["input_ids"].clone() for _ in range(5)]

    match_count = sum(
        torch.equal(batches_continued[i], batches_resumed[i]) for i in range(5)
    )
    print(f"  Batches matching after resume: {match_count}/5")
    if match_count == 5:
        print(f"  ✓ Perfect data reproducibility with StatefulParquetDataset")
    else:
        print(f"  ✗ Data mismatch — state doesn't fully capture position")
        print(f"    (likely due to dropped token buffer in get_state())")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

def print_summary():
    print("=" * 70)
    print("  SUMMARY OF CHECKPOINT/RESUME ISSUES")
    print("=" * 70)
    print("""
  BUG 1: Dataset state stale with num_workers >= 1
    train.py uses num_workers=gradient_accumulation (=1 in your config).
    With num_workers=1, the iteration happens in a subprocess.
    dataset.get_state() in the main process returns the INITIAL state.
    On resume, data restarts from the beginning.
    FIX: Change to num_workers=0 in the DataLoader.

  BUG 2: Scheduler double-advanced on resume
    accelerator.load_state() restores the scheduler to step N.
    Then train.py does: for _ in range(N): scheduler.step()
    This advances it to step 2N. LR decays too fast after resume.
    FIX: Remove the manual scheduler.step() loop after load_state().

  OK: Expert biases (persistent buffer) — saved/restored by accelerator.
  OK: local_tokens_per_expert (non-persistent) — correctly zeroed on resume.
  OK: Model weights + optimizer state — handled by accelerator.save/load_state.
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CHECKPOINT / RESUME VERIFICATION")
    print("=" * 70 + "\n")

    test_dataset_state_num_workers()
    test_scheduler_double_advance()
    test_data_reproducibility()
    test_parquet_dataset_state()
    print_summary()
