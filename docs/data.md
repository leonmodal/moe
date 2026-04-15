# Data Loading

This document covers dataset formats, data loading, tokenization, and stateful resumption.

## Table of Contents

- [1. Stateful Parquet Dataset (Primary)](#1-stateful-parquet-dataset-primary)
- [2. Token Bin Dataset (Legacy/Speedrun)](#2-token-bin-dataset-legacyspeedrun)
- [3. Configuration](#3-configuration)

---

## 1. Stateful Parquet Dataset (Primary)

**File**: `src/data/parquet_dataset.py`
**Class**: `StatefulParquetDataset`

The primary data loading backend. Streams from a directory of Parquet files containing text data.

### How it works

1. **File discovery**: Scans `data_dir` for `.parquet` files, sorts deterministically
2. **DDP sharding**: Files are assigned to ranks round-robin (rank 0 gets files 0, world_size, 2*world_size, ...)
3. **Tokenization**: Each parquet row's text column is tokenized on-the-fly using `AutoTokenizer`
4. **Sequence packing**: Tokens are concatenated into a buffer, then chunked into `seq_len`-length sequences
5. **Train/val split**: File-level holdout -- bottom `holdout_fraction` of files are held out for validation

### Stateful resumption

The dataset tracks its exact position for checkpoint resumption:

```python
state = {
    "current_file_idx": 3,           # which file we're reading
    "sequences_yielded": 1247,       # sequences produced so far
    "leftover_tokens": [12, 45, ...]  # partial sequence buffer
}
```

On resume, the dataset:
1. Seeks to `current_file_idx`
2. Skips `sequences_yielded` sequences
3. Restores the leftover token buffer

This ensures **exact reproducibility** -- resuming from a checkpoint produces the same data sequence as uninterrupted training.

### Configuration

```yaml
data:
  format: parquet
  data_dir: ./data/fineweb         # directory of .parquet files
  text_column: text                 # column name in parquet
  seq_len: 2048                     # sequence length
  tokenizer_name: Qwen/Qwen3-0.6B  # HuggingFace tokenizer
  num_workers: 4                    # dataloader workers
  split: train                      # train | val | all
  holdout_fraction: 0.05            # fraction of files for validation
```

---

## 2. Token Bin Dataset (Legacy/Speedrun)

**File**: `src/data/token_bin_dataset.py`
**Class**: `StatefulTokenBinDataset`

Pre-tokenized binary format used by the modded-nanogpt speedrun. Tokens are stored as uint16/uint32 in `.bin` files with a 1024-byte header.

### Binary format

```
[Header: 256 int32 values = 1024 bytes]
  header[0] = 20240520  (magic number)
  header[1] = 1         (version)
  header[2] = token_count
[Tokens: token_count * dtype_size bytes]
  uint16 or uint32 token IDs
```

### DDP sharding

Rank-strided: rank `i` starts at token `i * seq_len` and advances by `world_size * seq_len` tokens. This is simpler than file-level sharding but requires all ranks to see all files.

### BOS alignment

Optional: when `align_to_bos=True`, each sequence is aligned to a document boundary (BOS token, default 50256). This consumes up to 2x tokens to find boundaries but ensures sequences start at document starts.

### Configuration

```yaml
data:
  format: token_bin
  files_glob: data/fineweb10B/*.bin
  seq_len: 2048
  header_bytes: 1024
  token_dtype: uint16
  max_tokens: null          # optional cap
  shuffle_files: false
  repeat: true              # cycle through files
  align_to_bos: false
  bos_token_id: 50256
```

---

## 3. Configuration

### Data download scripts

| Script | Description |
|--------|-------------|
| `scripts/download_data.py` | Download FineWeb-Edu parquet data |
| `scripts/download_fineweb10b_gpt2_bins.py` | Download pre-tokenized GPT-2 .bin files |
| `scripts/create_official_finewebedu_sample_parquet.py` | Create parquet sample from FineWeb-Edu |
