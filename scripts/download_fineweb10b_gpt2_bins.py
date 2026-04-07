#!/usr/bin/env python3
"""Download cached FineWeb GPT-2 token bins used by modded-nanogpt."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="data/fineweb10B_gpt2",
        help="Directory to store downloaded .bin shards",
    )
    parser.add_argument(
        "--num-train-chunks",
        type=int,
        default=5,
        help="Number of train shards to download (train shards are 1-indexed)",
    )
    parser.add_argument(
        "--repo-id",
        default="kjj0/fineweb10B-gpt2",
        help="HF dataset repo containing the cached GPT-2 bins",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = ["fineweb_val_000000.bin"]
    files.extend(f"fineweb_train_{idx:06d}.bin" for idx in range(1, args.num_train_chunks + 1))

    for name in files:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=name,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()
