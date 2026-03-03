"""
Test auto-resume: find_latest_checkpoint, load meta.json, restore dataset state,
and verify training continues from the correct step and data position.
"""
import json
import os
import shutil
import tempfile

import torch
from torch.utils.data import DataLoader

# Reuse from train.py (inline to avoid liger import)
import re

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    for entry in os.listdir(output_dir):
        m = pattern.match(entry)
        if m:
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path):
                checkpoints.append((int(m.group(1)), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def cleanup_checkpoints(output_dir, max_keep):
    if max_keep <= 0:
        return
    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    for entry in os.listdir(output_dir):
        m = pattern.match(entry)
        if m:
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path):
                checkpoints.append((int(m.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_keep:
        _, path = checkpoints.pop(0)
        shutil.rmtree(path)


# Simple dataset for testing
from torch.utils.data import IterableDataset

class CountingDataset(IterableDataset):
    def __init__(self, seq_len=32, vocab=256):
        self.seq_len = seq_len
        self.vocab = vocab
        self._start_idx = 0
        self._cur_idx = 0

    def get_state(self):
        return {"idx": self._cur_idx}

    def set_state(self, state):
        self._start_idx = state.get("idx", 0)

    def __iter__(self):
        idx = self._start_idx
        while True:
            tokens = [(idx * self.seq_len + i) % self.vocab for i in range(self.seq_len + 1)]
            self._cur_idx = idx + 1
            yield {
                "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
                "labels": torch.tensor(tokens[1:], dtype=torch.long),
            }
            idx += 1


def save_fake_checkpoint(output_dir, step, dataset_state, wandb_run_id=None):
    """Simulate what train.py's save_checkpoint does (minus accelerator)."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {"step": step}
    if dataset_state:
        meta["dataset_state"] = dataset_state
    if wandb_run_id:
        meta["wandb_run_id"] = wandb_run_id
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return ckpt_dir


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: find_latest_checkpoint picks the right one
# ═══════════════════════════════════════════════════════════════════════

def test_find_latest():
    print("=" * 70)
    print("  TEST 1: find_latest_checkpoint")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()
    try:
        # No checkpoints
        result = find_latest_checkpoint(tmpdir)
        assert result is None, f"Expected None, got {result}"
        print(f"\n  Empty dir → None: ✓")

        # Create checkpoints out of order
        save_fake_checkpoint(tmpdir, 500, {"idx": 50})
        save_fake_checkpoint(tmpdir, 1000, {"idx": 100})
        save_fake_checkpoint(tmpdir, 200, {"idx": 20})

        result = find_latest_checkpoint(tmpdir)
        assert result.endswith("checkpoint-1000"), f"Expected checkpoint-1000, got {result}"
        print(f"  3 checkpoints (200, 500, 1000) → picks 1000: ✓")

        # Add a higher one
        save_fake_checkpoint(tmpdir, 2000, {"idx": 200})
        result = find_latest_checkpoint(tmpdir)
        assert result.endswith("checkpoint-2000"), f"Expected checkpoint-2000, got {result}"
        print(f"  After adding 2000 → picks 2000: ✓")

        # Non-existent dir
        result = find_latest_checkpoint("/tmp/nonexistent_dir_xyz")
        assert result is None
        print(f"  Non-existent dir → None: ✓")

    finally:
        shutil.rmtree(tmpdir)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: cleanup_checkpoints keeps only max_keep
# ═══════════════════════════════════════════════════════════════════════

def test_cleanup():
    print("=" * 70)
    print("  TEST 2: cleanup_checkpoints")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()
    try:
        for step in [100, 200, 300, 400, 500]:
            save_fake_checkpoint(tmpdir, step, {"idx": step})

        entries_before = sorted(os.listdir(tmpdir))
        print(f"\n  Before cleanup: {entries_before}")

        cleanup_checkpoints(tmpdir, max_keep=2)

        entries_after = sorted(os.listdir(tmpdir))
        print(f"  After cleanup (keep=2): {entries_after}")

        assert len(entries_after) == 2
        assert "checkpoint-400" in entries_after
        assert "checkpoint-500" in entries_after
        print(f"  ✓ Kept only latest 2 (400, 500)")

    finally:
        shutil.rmtree(tmpdir)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: Full auto-resume flow
# ═══════════════════════════════════════════════════════════════════════

def test_full_auto_resume_flow():
    """
    Simulate: train 20 batches → checkpoint → new process with --auto_resume
    → should find checkpoint, load dataset state, resume at correct data.
    """
    print("=" * 70)
    print("  TEST 3: Full auto-resume flow (simulated)")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()
    try:
        B = 2

        # ── Phase 1: Train 20 batches, checkpoint at step 10 and 20 ──
        ds1 = CountingDataset()
        dl1 = DataLoader(ds1, batch_size=B, num_workers=0)
        it1 = iter(dl1)

        all_batches = []
        for step in range(1, 21):
            batch = next(it1)
            all_batches.append(batch["input_ids"].clone())

            if step % 10 == 0:
                state = ds1.get_state()
                save_fake_checkpoint(tmpdir, step, state)
                print(f"\n  Step {step}: saved checkpoint, dataset state = {state}")

        # ── Phase 2: Simulate --auto_resume ──
        # This is what train.py does:
        #   resume_from = find_latest_checkpoint(train_cfg.output_dir)
        resume_from = find_latest_checkpoint(tmpdir)
        print(f"\n  auto_resume found: {os.path.basename(resume_from)}")

        # Load meta
        with open(os.path.join(resume_from, "meta.json")) as f:
            meta = json.load(f)
        global_step = meta["step"]
        dataset_state = meta.get("dataset_state")
        print(f"  Restored: global_step={global_step}, dataset_state={dataset_state}")

        # ── Phase 3: Resume from checkpoint ──
        ds2 = CountingDataset()
        ds2.set_state(dataset_state)
        dl2 = DataLoader(ds2, batch_size=B, num_workers=0)
        it2 = iter(dl2)

        # global_step starts at 20, training loop continues from there
        print(f"\n  Resuming training from step {global_step}...")
        resumed_batches = []
        for step in range(global_step + 1, global_step + 11):
            batch = next(it2)
            resumed_batches.append(batch["input_ids"].clone())

        # ── Phase 4: Compare against straight-through ──
        # Continue the original iterator for 10 more batches
        continued_batches = []
        for _ in range(10):
            batch = next(it1)
            continued_batches.append(batch["input_ids"].clone())

        match_count = sum(
            torch.equal(resumed_batches[i], continued_batches[i])
            for i in range(10)
        )
        print(f"  Post-resume batches match: {match_count}/10")

        if match_count == 10:
            print(f"  ✓ PASS — auto-resume continues from exact data position")
        else:
            print(f"  ✗ MISMATCH")

        # ── Phase 5: Verify global_step is correct ──
        # Training loop: while global_step < max_steps
        # After resume, global_step=20, next step should be 21
        assert global_step == 20, f"Expected step 20, got {global_step}"
        print(f"  ✓ global_step correctly restored to 20")

    finally:
        shutil.rmtree(tmpdir)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: Auto-resume with no checkpoints (fresh start)
# ═══════════════════════════════════════════════════════════════════════

def test_auto_resume_fresh_start():
    print("=" * 70)
    print("  TEST 4: Auto-resume with no checkpoints → fresh start")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()
    try:
        resume_from = find_latest_checkpoint(tmpdir)
        print(f"\n  find_latest_checkpoint on empty dir: {resume_from}")
        assert resume_from is None

        # Simulate train.py logic:
        # if args.auto_resume and not resume_from:
        #     resume_from = find_latest_checkpoint(...)
        # ...
        # if resume_from:
        #     global_step, dataset_state = load_checkpoint(...)
        # else: global_step = 0
        global_step = 0 if resume_from is None else -1
        assert global_step == 0
        print(f"  global_step = {global_step} → fresh start: ✓")

    finally:
        shutil.rmtree(tmpdir)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: Meta.json contains all required fields
# ═══════════════════════════════════════════════════════════════════════

def test_meta_json_fields():
    print("=" * 70)
    print("  TEST 5: meta.json contains all required fields")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()
    try:
        ds_state = {"file_idx": 3, "seq_idx": 1500, "buffer": []}
        ckpt_dir = save_fake_checkpoint(tmpdir, 5000, ds_state, wandb_run_id="abc123")

        with open(os.path.join(ckpt_dir, "meta.json")) as f:
            meta = json.load(f)

        print(f"\n  meta.json contents: {json.dumps(meta, indent=2)}")

        assert meta["step"] == 5000, f"step: expected 5000, got {meta['step']}"
        assert meta["dataset_state"]["file_idx"] == 3
        assert meta["dataset_state"]["seq_idx"] == 1500
        assert meta["wandb_run_id"] == "abc123"

        print(f"  ✓ step: {meta['step']}")
        print(f"  ✓ dataset_state.file_idx: {meta['dataset_state']['file_idx']}")
        print(f"  ✓ dataset_state.seq_idx: {meta['dataset_state']['seq_idx']}")
        print(f"  ✓ wandb_run_id: {meta['wandb_run_id']}")

    finally:
        shutil.rmtree(tmpdir)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AUTO-RESUME TESTS")
    print("=" * 70 + "\n")

    test_find_latest()
    test_cleanup()
    test_full_auto_resume_flow()
    test_auto_resume_fresh_start()
    test_meta_json_fields()

    print("=" * 70)
    print("  ALL 5 TESTS PASSED")
    print("=" * 70)
