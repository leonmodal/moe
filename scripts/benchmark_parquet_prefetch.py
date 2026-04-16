#!/usr/bin/env python3
"""Benchmark the StatefulParquetDataset file-read-ahead.

The benchmark uses small synthetic parquet fixtures and artificially slows each
file read (via a subclass that inserts a sleep before `pd.read_parquet`) so the
overlap between file I/O and tokenization is visible on a small machine without
needing a real multi-GB dataset. On real parquet shards the absolute numbers
will differ, but the sign of the delta (prefetch-on faster than prefetch-off)
should match as long as file read time is non-negligible compared to
tokenization time.

Usage:
    uv run python scripts/benchmark_parquet_prefetch.py
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset


class _SlowTokenizer:
    """Deterministic hash tokenizer with an adjustable per-token cost."""

    eos_token_id = 1

    def __init__(self, per_token_us: float = 0.0):
        self.per_token_us = per_token_us

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.per_token_us > 0.0:
            time.sleep(self.per_token_us * 1e-6 * len(text.split()))
        out = [2 + (hash(tok) & 0xFD) for tok in text.split()]
        out.append(2 + (len(text) & 0xFD))
        return out


def _fixture(root: Path, num_files: int, rows_per_file: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for f in range(num_files):
        rows = [" ".join([f"f{f}r{r}t{t}" for t in range((r % 13) + 5)]) for r in range(rows_per_file)]
        pd.DataFrame({"text": rows}).to_parquet(root / f"shard_{f:04d}.parquet")
    return root


class _SlowLoadDataset(StatefulParquetDataset):
    """Subclass that artificially slows each parquet file read for benchmarking."""

    def __init__(self, *args, io_latency_s: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._io_latency_s = io_latency_s

    def _load_file(self, path: str) -> pd.DataFrame:
        if self._io_latency_s > 0:
            time.sleep(self._io_latency_s)
        return super()._load_file(path)


def _walltime(prefetch_files: int, num_files: int, rows_per_file: int,
              io_latency_s: float, per_token_us: float, data_dir: Path,
              batches: int) -> float:
    cfg = DataConfig(
        data_dir=str(data_dir),
        seq_len=32,
        tokenizer_name="stub",
        prefetch_files=prefetch_files,
    )
    ds = _SlowLoadDataset(
        cfg, _SlowTokenizer(per_token_us=per_token_us),
        rank=0, world_size=1, seed=0,
        io_latency_s=io_latency_s,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    it = iter(loader)
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-files", type=int, default=6)
    parser.add_argument("--rows-per-file", type=int, default=128)
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--io-latency-s", type=float, default=0.05,
                        help="Simulated per-file read latency (seconds)")
    parser.add_argument("--per-token-us", type=float, default=20.0,
                        help="Simulated per-token tokenization latency (microseconds)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        data_dir = _fixture(Path(td) / "data", args.num_files, args.rows_per_file)

        # Warm-up both configurations to amortize first-call overhead.
        _walltime(0, args.num_files, args.rows_per_file, args.io_latency_s,
                  args.per_token_us, data_dir, batches=4)
        _walltime(1, args.num_files, args.rows_per_file, args.io_latency_s,
                  args.per_token_us, data_dir, batches=4)

        off = _walltime(0, args.num_files, args.rows_per_file, args.io_latency_s,
                        args.per_token_us, data_dir, batches=args.batches)
        on = _walltime(1, args.num_files, args.rows_per_file, args.io_latency_s,
                       args.per_token_us, data_dir, batches=args.batches)

    speedup = off / on if on > 0 else float("inf")
    print("# StatefulParquetDataset prefetch benchmark")
    print(f"  num_files={args.num_files}, rows_per_file={args.rows_per_file}, batches={args.batches}")
    print(f"  simulated io_latency_s={args.io_latency_s}, per_token_us={args.per_token_us}")
    print()
    print(f"  prefetch_files=0 → {off:.3f}s")
    print(f"  prefetch_files=1 → {on:.3f}s")
    print(f"  speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
