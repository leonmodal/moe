# Data Loading

## Data Format

The sole supported data format is **sharded parquet**. Token-bin format has been removed.

## StatefulParquetDataset

`src/data/parquet_dataset.py` provides `StatefulParquetDataset`:

- Loads parquet shards from a directory
- Tokenizes text on-the-fly using a configurable tokenizer
- Supports deterministic sharding across distributed workers
- Stateful: can save/restore position for training resume

### Configuration

```yaml
data:
  data_dir: ./data/parquet
  text_column: text
  seq_len: 1024
  tokenizer_name: Qwen/Qwen3-0.6B
  num_workers: 4
```

### Sharding

Files are sorted deterministically, then shuffled with a fixed seed. Each worker gets a non-overlapping subset of files based on `rank` and `world_size`.

### Resume State

`get_state()` returns `file_idx` and `seq_idx` for the current position. `set_state()` restores from this. Note: the internal token buffer is not saved — exact token-level reproducibility is not guaranteed across resume, but file-level ordering is preserved.

### Train/Val Split

Training data uses `split="all"` with `holdout_fraction=0.0` (all data for training). Eval data uses `split="val"` with configurable `holdout_fraction` (default 0.05).

## Download

```bash
python scripts/download_data.py --max_shards 64
```

For Modal, use `modal run modal_train.py::download_data --max-shards 64`.
