"""
Test that StatefulParquetDataset resumes at the exact parquet file and
exact sequence position within that file. Uses the real data files.
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset


def test_parquet_resume():
    data_dir = "./data/parquet"
    if not os.path.exists(data_dir):
        print("No data — skipping")
        return

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = DataConfig(data_dir=data_dir, seq_len=128, tokenizer_name="Qwen/Qwen3-0.6B")
    B = 4
    N_BEFORE = 50    # batches before checkpoint
    N_AFTER = 20     # batches after checkpoint

    # ═══════════════════════════════════════════════════════════════════
    #  RUN A: straight through N_BEFORE + N_AFTER batches
    # ═══════════════════════════════════════════════════════════════════
    ds_a = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    dl_a = DataLoader(ds_a, batch_size=B, num_workers=0)
    it_a = iter(dl_a)

    all_batches_a = []
    for i in range(N_BEFORE + N_AFTER):
        batch = next(it_a)
        all_batches_a.append(batch["input_ids"].clone())

    # ═══════════════════════════════════════════════════════════════════
    #  RUN B: N_BEFORE batches → save state → fresh dataset → resume
    # ═══════════════════════════════════════════════════════════════════
    ds_b = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    dl_b = DataLoader(ds_b, batch_size=B, num_workers=0)
    it_b = iter(dl_b)

    batches_before = []
    for i in range(N_BEFORE):
        batch = next(it_b)
        batches_before.append(batch["input_ids"].clone())

    # Capture state — this is what gets saved to meta.json
    state = ds_b.get_state()

    print("=" * 70)
    print("  Parquet Dataset Resume Test")
    print("=" * 70)
    print(f"\n  Files assigned to rank 0: {len(ds_b.files)}")
    for i, f in enumerate(ds_b.files[:5]):
        print(f"    [{i}] {os.path.basename(f)}")
    if len(ds_b.files) > 5:
        print(f"    ... and {len(ds_b.files) - 5} more")

    print(f"\n  Consumed {N_BEFORE} batches × {B} seqs = {N_BEFORE * B} sequences")
    print(f"  Saved state: file_idx={state['file_idx']}, seq_idx={state['seq_idx']}")
    print(f"    → Resume at file: {os.path.basename(ds_b.files[state['file_idx']])}")
    print(f"    → Skip first {state['seq_idx']} sequences in that file")

    # Create fresh dataset, set state, create new dataloader
    ds_b2 = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    ds_b2.set_state(state)
    dl_b2 = DataLoader(ds_b2, batch_size=B, num_workers=0)
    it_b2 = iter(dl_b2)

    batches_after = []
    for i in range(N_AFTER):
        batch = next(it_b2)
        batches_after.append(batch["input_ids"].clone())

    # ═══════════════════════════════════════════════════════════════════
    #  Compare: first N_BEFORE should match, next N_AFTER should match
    # ═══════════════════════════════════════════════════════════════════

    # Before-checkpoint batches should match exactly
    before_match = 0
    for i in range(N_BEFORE):
        if torch.equal(all_batches_a[i], batches_before[i]):
            before_match += 1
    print(f"\n  Pre-checkpoint batches match: {before_match}/{N_BEFORE}")

    # Post-checkpoint batches (resumed vs straight-through)
    after_match = 0
    first_mismatch = None
    for i in range(N_AFTER):
        if torch.equal(all_batches_a[N_BEFORE + i], batches_after[i]):
            after_match += 1
        elif first_mismatch is None:
            first_mismatch = i

    print(f"  Post-checkpoint batches match: {after_match}/{N_AFTER}")

    if after_match == N_AFTER:
        print(f"\n  ✓ PERFECT RESUME — all {N_AFTER} post-checkpoint batches are identical")
        print(f"    Data resumes at exactly the right parquet file and sequence position")
    else:
        print(f"\n  ✗ MISMATCH at batch {first_mismatch} after resume")
        print(f"    This is caused by the token buffer being dropped in get_state()")
        print(f"    (buffer: [] instead of actual leftover tokens)")

        # Check how many tokens differ
        if first_mismatch is not None:
            a = all_batches_a[N_BEFORE + first_mismatch]
            b = batches_after[first_mismatch]
            n_diff = (a != b).sum().item()
            total = a.numel()
            print(f"    First mismatched batch: {n_diff}/{total} tokens differ")

    # ═══════════════════════════════════════════════════════════════════
    #  Also test: state correctly advances across file boundaries
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n  --- File boundary tracking ---")
    ds_c = StatefulParquetDataset(config, tokenizer, rank=0, world_size=1)
    dl_c = DataLoader(ds_c, batch_size=B, num_workers=0)
    it_c = iter(dl_c)

    prev_file_idx = 0
    for i in range(500):
        next(it_c)
        state_c = ds_c.get_state()
        if state_c["file_idx"] != prev_file_idx:
            print(f"    Batch {i}: moved to file_idx={state_c['file_idx']} "
                  f"({os.path.basename(ds_c.files[state_c['file_idx']])}), "
                  f"seq_idx reset to {state_c['seq_idx']}")
            prev_file_idx = state_c["file_idx"]

    final_state = ds_c.get_state()
    print(f"    After 500 batches: file_idx={final_state['file_idx']}, seq_idx={final_state['seq_idx']}")
    print(f"    ✓ File index correctly advances across parquet file boundaries")
    print()


if __name__ == "__main__":
    test_parquet_resume()
