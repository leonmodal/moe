"""
Materialize a small local parquet corpus from the official FineWebEdu dataset.

Example:
  uv run python scripts/create_official_finewebedu_sample_parquet.py \
    --out-dir data/parquet_finewebedu_official_sample_8x10k \
    --config sample-10BT \
    --num-shards 8 \
    --rows-per-shard 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="Hugging Face dataset id to stream from.",
    )
    parser.add_argument(
        "--config",
        default="sample-10BT",
        help="Dataset config name to stream from.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream from.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for parquet shards.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of parquet shards to write.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=10000,
        help="Rows per parquet shard.",
    )
    return parser.parse_args()


def _flush_rows(out_dir: Path, shard_idx: int, rows: list[dict[str, object]]) -> None:
    out_path = out_dir / f"shard_{shard_idx:05d}.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} rows={len(df)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
    )

    rows: list[dict[str, object]] = []
    written = 0
    shard_idx = 0

    for example in stream:
        text = example.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        rows.append({"text": text})
        if len(rows) >= args.rows_per_shard:
            _flush_rows(out_dir, shard_idx, rows)
            shard_idx += 1
            written += len(rows)
            rows = []
            if shard_idx >= args.num_shards:
                break

    if rows and shard_idx < args.num_shards:
        _flush_rows(out_dir, shard_idx, rows)
        written += len(rows)
        shard_idx += 1

    print(f"done: shards={shard_idx} rows={written} out_dir={out_dir}")


if __name__ == "__main__":
    main()
