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

        # Resumption state (authoritative position marker is (file_idx, text_idx);
        # buffer holds the in-flight leftover tokens from texts already consumed
        # into the buffer but not yet yielded as full sequences).
        self._start_file_idx: int = 0
        self._start_text_idx: int = 0
        self._start_buffer: list[int] = []

        # Live tracking (updated during __iter__)
        self._cur_file_idx: int = 0
        self._cur_text_idx: int = 0
        self._cur_seq_idx: int = 0  # diagnostic: seqs yielded from current file
        self._live_buffer: list[int] = []

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        """Capture the current iteration position for deterministic resume.

        Authoritative markers for set_state(): `file_idx`, `text_idx`, `buffer`.
        `seq_idx` is a diagnostic counter (sequences yielded from the current
        file since iteration started) and is intentionally NOT restored by
        set_state() — do not rely on it for resume.
        """
        return {
            "file_idx": self._cur_file_idx,
            "text_idx": self._cur_text_idx,
            "seq_idx": self._cur_seq_idx,
            "buffer": list(self._live_buffer),
        }

    def set_state(self, state: dict) -> None:
        """Restore iteration position. Reads `file_idx`, `text_idx`, `buffer`.

        `seq_idx` is ignored by design (see get_state docstring).
        """
        self._start_file_idx = int(state.get("file_idx", 0))
        self._start_text_idx = int(state.get("text_idx", 0))
        self._start_buffer = list(state.get("buffer", []))

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

        # The token buffer is mutated in place for the lifetime of iteration
        # so that `self._live_buffer` remains a live view used by get_state().
        token_buffer: list[int] = list(self._start_buffer)
        self._live_buffer = token_buffer

        start_text_idx = self._start_text_idx
        for file_idx, file_path in enumerate(self.files):
            if file_idx < self._start_file_idx:
                continue

            self._cur_file_idx = file_idx
            df = self._load_file(file_path)
            self._cur_seq_idx = 0

            # Only the start file honours the resumption text offset; subsequent
            # files always begin at text 0.
            first_text = start_text_idx if file_idx == self._start_file_idx else 0
            self._cur_text_idx = first_text

            texts = df[self.config.text_column]
            for text_idx in range(first_text, len(texts)):
                text = texts.iat[text_idx]
                if isinstance(text, str) and text.strip():
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(ids)
                    token_buffer.append(eos)

                # Text `text_idx` is now fully absorbed into the buffer (or
                # skipped as empty). Advance the "next text to consume" marker
                # BEFORE yielding so a save between yields still points at the
                # correct next text and doesn't re-tokenize text `text_idx`.
                self._cur_text_idx = text_idx + 1

                while len(token_buffer) >= seq_len + 1:
                    chunk = token_buffer[: seq_len + 1]
                    # Advance in place to preserve `_live_buffer` aliasing
                    # and keep get_state() authoritative during iteration.
                    del token_buffer[:seq_len]
                    self._cur_seq_idx += 1

                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }

            # Finishing a file resets the per-file start offset.
            start_text_idx = 0

    def _load_file(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path, columns=[self.config.text_column])
