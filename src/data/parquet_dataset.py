"""
Stateful Parquet dataset for language model pretraining.

State tracked:
  - current file index
  - sequences already yielded from that file (for skip-on-resume)
  - leftover token buffer (serialized as a list)

State is saved as JSON alongside model checkpoints and restored
at the start of a resumed training run.
"""
import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd
import torch
from torch.utils.data import IterableDataset


@dataclass
class DataConfig:
    data_dir: str = "./data/parquet"
    text_column: str = "text"
    seq_len: int = 2048
    tokenizer_name: str = "gpt2"
    num_workers: int = 4
    split: str = "all"             # all | train | val
    holdout_fraction: float = 0.0  # file-level holdout used when split != all


class StatefulParquetDataset(IterableDataset):
    """
    Streams (input_ids, labels) pairs from a directory of Parquet files.

    Files are sorted lexicographically and distributed across DDP ranks
    so each rank sees a disjoint shard.  Within a rank the dataset is
    fully stateful: call `get_state()` to capture position and
    `set_state(state)` to restore it on resume.
    """

    def __init__(
        self,
        config: DataConfig,
        tokenizer,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size

        all_files = sorted(set(
            glob.glob(os.path.join(config.data_dir, "**/*.parquet"), recursive=True)
            + glob.glob(os.path.join(config.data_dir, "*.parquet"))
        ))
        if not all_files:
            raise FileNotFoundError(f"No parquet files found in {config.data_dir}")

        # Shuffle with fixed seed — deterministic across runs for resumability
        rng = random.Random(seed)
        rng.shuffle(all_files)

        split = getattr(config, "split", "all")
        holdout_fraction = float(getattr(config, "holdout_fraction", 0.0) or 0.0)
        if split not in {"all", "train", "val"}:
            raise ValueError(f"Unknown dataset split: {split}")
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")
        if split != "all" and holdout_fraction > 0.0:
            min_val_files = min(world_size, max(1, len(all_files) - 1))
            val_count = max(1, int(round(len(all_files) * holdout_fraction)))
            val_count = max(val_count, min_val_files)
            if val_count >= len(all_files):
                raise ValueError(
                    f"holdout_fraction={holdout_fraction} leaves no training files in {config.data_dir}"
                )
            if split == "val":
                all_files = all_files[:val_count]
            else:
                all_files = all_files[val_count:]
        elif split == "val":
            raise ValueError("split='val' requires holdout_fraction > 0 or a dedicated eval data_dir")

        # Shard files across ranks deterministically
        self.files = [f for i, f in enumerate(all_files) if i % world_size == rank]
        if not self.files:
            raise RuntimeError(
                f"Rank {rank} received 0 parquet files from {len(all_files)} total files "
                f"with world_size={world_size}. Need at least {world_size} parquet files "
                "or different data sharding."
            )

        # Resumption state
        self._start_file_idx: int = 0
        self._start_seq_skip: int = 0  # how many sequences to skip in the start file
        self._start_buffer: list[int] = []

        # Live tracking (updated during __iter__)
        self._cur_file_idx: int = 0
        self._cur_seq_idx: int = 0

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        return {
            "file_idx": self._cur_file_idx,
            "seq_idx": self._cur_seq_idx,
            "buffer": [],  # buffer is small; drop it for simplicity
        }

    def set_state(self, state: dict) -> None:
        self._start_file_idx = state.get("file_idx", 0)
        self._start_seq_skip = state.get("seq_idx", 0)
        self._start_buffer = state.get("buffer", [])

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_state(), f)

    def load_state(self, path: str) -> None:
        if os.path.exists(path):
            with open(path) as f:
                self.set_state(json.load(f))

    # ------------------------------------------------------------------ #
    #  Iteration                                                           #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        seq_len = self.config.seq_len
        eos = self.tokenizer.eos_token_id or 0

        token_buffer: list[int] = list(self._start_buffer)
        skip_seqs = self._start_seq_skip

        for file_idx, file_path in enumerate(self.files):
            if file_idx < self._start_file_idx:
                continue

            self._cur_file_idx = file_idx
            df = self._load_file(file_path)
            file_seq_count = 0

            for text in df[self.config.text_column]:
                if not isinstance(text, str) or not text.strip():
                    continue

                ids = self.tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(ids)
                token_buffer.append(eos)

                while len(token_buffer) >= seq_len + 1:
                    # Skip sequences when resuming within a file
                    if file_idx == self._start_file_idx and skip_seqs > 0:
                        skip_seqs -= 1
                        token_buffer = token_buffer[seq_len:]
                        continue

                    chunk = token_buffer[: seq_len + 1]
                    token_buffer = token_buffer[seq_len:]  # slide by seq_len (1-token overlap)
                    file_seq_count += 1
                    self._cur_seq_idx = file_seq_count

                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }

    def _load_file(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path, columns=[self.config.text_column])
