# Data Loading

## Data Format

The sole supported data format is **sharded parquet**. The token-bin format has been removed; see `legacy/speedrun/` for the archived `src/data/token_bin_dataset.py` implementation.

## StatefulParquetDataset

`src/data/parquet_dataset.py` provides `StatefulParquetDataset`:

- Loads parquet shards from a directory
- Tokenizes text on-the-fly using a configurable HuggingFace tokenizer
- Deterministically shards files across distributed ranks
- Stateful: `get_state()` and `set_state()` support exact-resume checkpointing

### Configuration

```yaml
data:
  data_dir: ./data/parquet
  text_column: text
  seq_len: 1024
  tokenizer_name: Qwen/Qwen3-0.6B
  num_workers: 0        # see "DataLoader workers" below
  prefetch_files: 1     # dataset-level file read-ahead; see "Throughput" below
```

### Sharding

Files are sorted deterministically, then shuffled with a fixed seed. Each DDP rank receives a non-overlapping subset of files based on `rank` and `world_size`.

### Resume State (exact)

`get_state()` returns:

| Field | Meaning |
|-------|---------|
| `file_idx` | index of the file currently being consumed |
| `text_idx` | index of the **next** text row to tokenize in that file |
| `buffer` | the live token buffer (tokens accumulated from already-consumed texts but not yet emitted as full sequences) |
| `seq_idx` | diagnostic counter: sequences yielded from the current file (not used by `set_state`) |

`set_state()` restores `(file_idx, text_idx, buffer)`. `seq_idx` is intentionally ignored by `set_state()` — it is diagnostic only.

With this contract, resumed batches match the uninterrupted continuation **tensor-for-tensor**. The regression test is `tests/test_trainer_dataloader_resume.py::test_trainer_dataloader_exact_resume`, parametrized over the requested `num_workers` values `{0, 4}`.

### DataLoader workers — deterministic-resume policy

`StatefulParquetDataset` is an `IterableDataset` whose live state attributes (`_cur_file_idx`, `_cur_text_idx`, `_live_buffer`) are mutated inside `__iter__`. When PyTorch spawns `num_workers > 0` worker subprocesses, those attributes are mutated on the worker's dataset copy, not on the main-process object. A checkpoint save reads state from the main-process object and would get stale data.

To keep `get_state()` authoritative, `src/training/trainer.py` always constructs the training DataLoader with `num_workers=0` when the dataset exposes `get_state`/`set_state`, regardless of the `data.num_workers` config value. The trainer logs a one-line notice on rank 0 if the config requested a nonzero value. See `_stateful_dataloader_workers` in `src/training/trainer.py`.

### Throughput — dataset-level file read-ahead (`prefetch_files`)

Because the DataLoader runs with `num_workers=0` for correctness, throughput-oriented parallelism lives **inside** the dataset: `StatefulParquetDataset.__iter__` uses a dedicated single-worker `ThreadPoolExecutor` to read the next parquet shard in the background while the main process tokenizes the current one. The background thread is keyed strictly by file order, so prefetch never skips or duplicates content and is indistinguishable from synchronous loading from a resume standpoint.

The policy is controlled by `DataConfig.prefetch_files`:

| Value | Behavior |
|-------|----------|
| `0` | Disabled — each shard is loaded synchronously on the critical path. |
| `1` *(default)* | Read the next shard in a background thread while the current one is being tokenized. |

Resume-safety: `get_state()` / `set_state()` capture only `(file_idx, text_idx, buffer)`. A prefetched-but-not-yet-consumed shard is simply re-prefetched when iteration resumes — there is no additional state to persist. This is verified by `tests/test_trainer_dataloader_resume.py::test_trainer_dataloader_exact_resume`, which parametrizes over `prefetch_files ∈ {0, 1}` and `requested_workers ∈ {0, 4}` (four combinations, all passing).

End-to-end correctness through the real checkpoint I/O path is covered by `tests/test_checkpoint_e2e_parquet.py::test_end_to_end_checkpoint_roundtrip_with_parquet_dataset` (drives `save_checkpoint()` + `load_checkpoint()` with a real `StatefulParquetDataset` and asserts tensor-equal batch continuation and loss match after resume).

#### Throughput benchmark

Run `uv run python scripts/benchmark_parquet_prefetch.py` to measure prefetch on/off walltime on a synthetic fixture. The benchmark simulates real file-I/O latency via `time.sleep` inside `_load_file` so that file-read vs tokenization overlap is visible on a small machine.

Representative result on this workstation (8 shards × 64 rows, 60 batches, 50ms simulated I/O, 30µs/token):

```
prefetch_files=0 → 0.380s
prefetch_files=1 → 0.279s
speedup: 1.36x
```

Heavier fixture (12 shards × 96 rows, 120 batches, 100ms simulated I/O, 40µs/token):

```
prefetch_files=0 → 1.027s
prefetch_files=1 → 0.730s
speedup: 1.41x
```

On real parquet shards (hundreds of megabytes, multi-second reads) the absolute numbers will differ but the sign of the delta — prefetch-on faster than prefetch-off — holds as long as file I/O is non-negligible relative to tokenization cost. The in-flight background thread is keyed strictly by file order, so the speedup comes from overlap, not from parallel tokenization.

### Train/Val Split

Training data uses `split="all"` with `holdout_fraction=0.0` (all data for training). Eval data uses `split="val"` with configurable `holdout_fraction` (default 0.05). The val split is deterministic given `seed` and disjoint from the train split — see `tests/test_parquet_edge_cases.py::test_holdout_split_is_deterministic_and_disjoint`.

## Download

```bash
uv run python scripts/download_data.py --max_shards 64
```

For Modal, use `modal run modal_train.py::download_data --max-shards 64`.
