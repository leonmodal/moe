"""
Test edge cases in StatefulParquetDataset:
  1. Buffer loss at file boundaries — does dropped buffer cause data mismatch?
  2. Seed determinism — same file order across fresh instantiations?
  3. Multi-rank sharding — each rank sees disjoint files?
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset


DATA_DIR = "./data/parquet"


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: Buffer loss at file boundary
# ═══════════════════════════════════════════════════════════════════════

def test_buffer_at_file_boundary():
    """
    get_state() saves buffer: []. If we checkpoint right after crossing
    a file boundary, leftover tokens from the previous file are lost.
    Test: does this cause data mismatch on resume?
    """
    print("=" * 70)
    print("  TEST 1: Buffer loss at file boundary")
    print("=" * 70)

    if not os.path.exists(DATA_DIR):
        print("  No data — skipping\n")
        return

    tokenizer = get_tokenizer()
    config = DataConfig(data_dir=DATA_DIR, seq_len=128)
    B = 4

    # Run straight through and record when file_idx changes
    ds = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    dl = DataLoader(ds, batch_size=B, num_workers=0)
    it = iter(dl)

    # Find the file boundary — iterate until file_idx changes
    prev_file_idx = 0
    boundary_batch = None
    all_batches = []

    for i in range(5000):
        batch = next(it)
        all_batches.append(batch["input_ids"].clone())
        state = ds.get_state()
        if state["file_idx"] != prev_file_idx:
            boundary_batch = i
            print(f"\n  File boundary at batch {i}: file_idx {prev_file_idx} → {state['file_idx']}")
            print(f"  State at boundary: {state}")
            break
        prev_file_idx = state["file_idx"]

    if boundary_batch is None:
        print("  Could not find file boundary in 5000 batches — single large file")
        print("  Buffer issue doesn't apply (only matters across files)\n")
        return

    # Now test: checkpoint 5 batches BEFORE the boundary, resume, cross boundary
    # This should work fine because we're still in the same file
    ckpt_batch = boundary_batch - 5
    if ckpt_batch < 0:
        ckpt_batch = 0

    # Get state at ckpt_batch
    ds2 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    dl2 = DataLoader(ds2, batch_size=B, num_workers=0)
    it2 = iter(dl2)
    for i in range(ckpt_batch):
        next(it2)
    state_before = ds2.get_state()

    print(f"\n  Checkpoint at batch {ckpt_batch}: state = {state_before}")
    print(f"  Buffer saved as: {state_before['buffer']}  (always empty!)")

    # Resume from that state
    ds3 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    ds3.set_state(state_before)
    dl3 = DataLoader(ds3, batch_size=B, num_workers=0)
    it3 = iter(dl3)

    # Compare batches from ckpt_batch to boundary+10
    n_compare = boundary_batch - ckpt_batch + 10
    match_count = 0
    first_mismatch = None
    for i in range(n_compare):
        resumed_batch = next(it3)["input_ids"]
        original_batch = all_batches[ckpt_batch + i]
        if torch.equal(resumed_batch, original_batch):
            match_count += 1
        elif first_mismatch is None:
            first_mismatch = ckpt_batch + i

    print(f"\n  Comparing batches {ckpt_batch} to {ckpt_batch + n_compare - 1}:")
    print(f"  Matched: {match_count}/{n_compare}")

    if match_count == n_compare:
        print(f"  ✓ PASS — data matches perfectly across file boundary")
        print(f"    Buffer is reconstructed by the skip logic (re-tokenizes file from scratch)")
    else:
        print(f"  ✗ MISMATCH at batch {first_mismatch}")

    # Now test: checkpoint AT the file boundary (in the new file)
    # This is the problematic case — leftover tokens from prev file are lost
    print(f"\n  --- Checkpoint AT file boundary (batch {boundary_batch}) ---")
    state_at_boundary = ds.get_state()  # ds was iterated to boundary_batch
    print(f"  State: {state_at_boundary}")

    ds4 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    ds4.set_state(state_at_boundary)
    dl4 = DataLoader(ds4, batch_size=B, num_workers=0)
    it4 = iter(dl4)

    # Get next 10 batches from resumed and original
    n_after = 10
    match_after = 0
    first_mismatch_after = None
    for i in range(n_after):
        try:
            resumed = next(it4)["input_ids"]
        except StopIteration:
            print(f"  Resumed dataset exhausted at batch {i}")
            break
        original = all_batches[boundary_batch + 1 + i]  # +1 because boundary batch was already consumed
        if torch.equal(resumed, original):
            match_after += 1
        elif first_mismatch_after is None:
            first_mismatch_after = i
            n_diff = (resumed != original).sum().item()
            print(f"  Batch {i}: {n_diff}/{original.numel()} tokens differ")

    print(f"  After-boundary matches: {match_after}/{n_after}")
    if match_after == n_after:
        print(f"  ✓ PASS — even AT file boundary, resume is exact")
    else:
        print(f"  ✗ MISMATCH — buffer from previous file is lost")
        print(f"    First mismatch at batch {first_mismatch_after} after boundary")
        print(f"    This means the first few sequences after resuming at a file boundary")
        print(f"    will differ because leftover tokens from the previous file are gone")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: Seed determinism
# ═══════════════════════════════════════════════════════════════════════

def test_seed_determinism():
    """Two fresh dataset instances with the same seed should get the same file order."""
    print("=" * 70)
    print("  TEST 2: Seed determinism — same file order across instantiations")
    print("=" * 70)

    if not os.path.exists(DATA_DIR):
        print("  No data — skipping\n")
        return

    tokenizer = get_tokenizer()
    config = DataConfig(data_dir=DATA_DIR, seq_len=128)

    # Create two datasets with default seed
    ds1 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=42)
    ds2 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=42)

    files_match = ds1.files == ds2.files
    print(f"\n  Dataset 1: {len(ds1.files)} files, first 3: {[os.path.basename(f) for f in ds1.files[:3]]}")
    print(f"  Dataset 2: {len(ds2.files)} files, first 3: {[os.path.basename(f) for f in ds2.files[:3]]}")
    print(f"  File order matches: {'✓' if files_match else '✗'}")

    # Different seed should give different order
    ds3 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=123)
    diff_seed = ds1.files != ds3.files
    print(f"  Different seed gives different order: {'✓' if diff_seed else '✗'}")

    # Also verify the actual data matches
    dl1 = DataLoader(ds1, batch_size=4, num_workers=0)
    dl2 = DataLoader(ds2, batch_size=4, num_workers=0)
    it1, it2 = iter(dl1), iter(dl2)
    data_match = all(
        torch.equal(next(it1)["input_ids"], next(it2)["input_ids"])
        for _ in range(20)
    )
    print(f"  First 20 batches identical: {'✓' if data_match else '✗'}")

    assert files_match and diff_seed and data_match
    print(f"  ✓ PASS — deterministic file ordering, reproducible data\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: Multi-rank sharding
# ═══════════════════════════════════════════════════════════════════════

def test_multi_rank_sharding():
    """Each rank gets disjoint files, same order within each shard."""
    print("=" * 70)
    print("  TEST 3: Multi-rank sharding — disjoint file sets")
    print("=" * 70)

    if not os.path.exists(DATA_DIR):
        print("  No data — skipping\n")
        return

    tokenizer = get_tokenizer()
    config = DataConfig(data_dir=DATA_DIR, seq_len=128)
    world_size = 8

    rank_files = []
    for rank in range(world_size):
        ds = StatefulParquetDataset(config, tokenizer, rank=rank, world_size=world_size, seed=42)
        rank_files.append(set(ds.files))
        print(f"  Rank {rank}: {len(ds.files)} files")

    # Check disjoint
    all_disjoint = True
    for i in range(world_size):
        for j in range(i + 1, world_size):
            overlap = rank_files[i] & rank_files[j]
            if overlap:
                print(f"  ✗ Rank {i} and {j} share {len(overlap)} files!")
                all_disjoint = False

    # Check coverage (all files assigned to some rank)
    all_assigned = set()
    for rf in rank_files:
        all_assigned |= rf

    ds_all = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1, seed=42)
    total_files = len(ds_all.files)
    coverage = len(all_assigned) == total_files

    print(f"  Disjoint: {'✓' if all_disjoint else '✗'}")
    print(f"  Total files: {total_files}, assigned: {len(all_assigned)}, coverage: {'✓' if coverage else '✗'}")

    # Check determinism: second instantiation with same params gives same sharding
    ds_check = StatefulParquetDataset(config, tokenizer, rank=3, world_size=world_size, seed=42)
    ds_check2 = StatefulParquetDataset(config, tokenizer, rank=3, world_size=world_size, seed=42)
    shard_deterministic = ds_check.files == ds_check2.files
    print(f"  Rank 3 shard deterministic across instances: {'✓' if shard_deterministic else '✗'}")

    assert all_disjoint and coverage and shard_deterministic
    print(f"  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: train.py doesn't pass seed to dataset — verify default works
# ═══════════════════════════════════════════════════════════════════════

def test_train_seed_path():
    """
    train.py does:
        set_seed(args.seed + accelerator.process_index)
        ...
        dataset = StatefulParquetDataset(config, tokenizer, rank=..., world_size=...)

    It does NOT pass seed= to the dataset. The dataset uses seed=42 by default.
    This is fine because the dataset's shuffle uses its own random.Random(seed=42),
    isolated from torch/numpy/python global RNGs.
    """
    print("=" * 70)
    print("  TEST 4: Dataset seed isolation from global RNG")
    print("=" * 70)

    if not os.path.exists(DATA_DIR):
        print("  No data — skipping\n")
        return

    import random
    import numpy as np

    tokenizer = get_tokenizer()
    config = DataConfig(data_dir=DATA_DIR, seq_len=128)

    # Set different global seeds
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    ds1 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)

    torch.manual_seed(9999)
    random.seed(9999)
    np.random.seed(9999)
    ds2 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)

    files_match = ds1.files == ds2.files
    print(f"\n  Different global RNG seeds, same dataset seed (42)")
    print(f"  File order matches: {'✓' if files_match else '✗'}")

    assert files_match, "Dataset file order should be independent of global RNG!"
    print(f"  ✓ PASS — dataset uses its own isolated random.Random(seed=42)\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PARQUET DATASET EDGE CASE TESTS")
    print("=" * 70 + "\n")

    test_buffer_at_file_boundary()
    test_seed_determinism()
    test_multi_rank_sharding()
    test_train_seed_path()

    print("=" * 70)
    print("  DONE")
    print("=" * 70)
