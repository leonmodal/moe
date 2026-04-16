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

Throughput-oriented parallel loading belongs at the dataset level (file read-ahead inside `StatefulParquetDataset` itself), not at the DataLoader worker level; that path is tracked under AC-8.

### Train/Val Split

Training data uses `split="all"` with `holdout_fraction=0.0` (all data for training). Eval data uses `split="val"` with configurable `holdout_fraction` (default 0.05). The val split is deterministic given `seed` and disjoint from the train split — see `tests/test_parquet_edge_cases.py::test_holdout_split_is_deterministic_and_disjoint`.

## Download

```bash
uv run python scripts/download_data.py --max_shards 64
```

For Modal, use `modal run modal_train.py::download_data --max-shards 64`.
