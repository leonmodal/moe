"""
Stateful Parquet dataset for language model pretraining.

Resume contract (authoritative markers used by set_state):
  - file_idx: index of the file currently being consumed
  - text_idx: index of the next text row to tokenize in that file
  - buffer:   the live token buffer (tokens from already-consumed texts
              that have not yet been emitted as a full seq_len+1 chunk)

Also captured for diagnostics but NOT used on resume:
  - seq_idx:  number of sequences yielded from the current file since
              iteration started

State is saved as JSON alongside model checkpoints and restored at the
start of a resumed run. With num_workers=0 (forced by the trainer for
stateful datasets) the live attributes on the main-process dataset object
are the authoritative source for get_state(), and resumed batches match
the uninterrupted continuation tensor-for-tensor.

Throughput optimization: `DataConfig.prefetch_files` enables a single-file
read-ahead inside the dataset itself (a dedicated background thread reads
the next parquet shard while the main process tokenizes the current one).
Prefetch is resume-safe: the persisted state is unchanged, and on resume
we simply re-schedule the same file order from `_start_file_idx`.
"""
import concurrent.futures
import glob
import json
import os
import random
from dataclasses import dataclass
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
    # Dataset-level read-ahead of parquet files. 1 = read one file ahead in a
    # background thread while the main process tokenizes; 0 disables prefetch.
    # Prefetch is keyed strictly by file order so resume stays exact. The trainer
    # forces DataLoader num_workers=0 for deterministic resume correctness, so this is the
    # correct layer to add parallelism for throughput.
    prefetch_files: int = 1


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

        # Shard files across ranks deterministically. Truncate the shared
        # pool to `(N // world_size) * world_size` files so every rank sees
        # exactly the same number — otherwise ranks with fewer files finish
        # the epoch earlier and the collective reductions on remaining ranks
        # stall on `all_reduce`. With large datasets the dropped files are
        # negligible; with a small `val` split we pad up to world_size instead.
        usable = (len(all_files) // world_size) * world_size
        if usable == 0:
            # Fewer files than ranks — repeat files round-robin so every rank
            # still sees at least one file. Eval telemetry is mean-of-means
            # across ranks, so the repeated shards introduce at most a small
            # duplication bias. Train splits should never hit this path in
            # practice; config validation should flag < world_size shards.
            if split == "val" and len(all_files) > 0:
                reps = (world_size + len(all_files) - 1) // len(all_files)
                padded = (all_files * reps)[:world_size]
                self.files = [padded[rank]]
            else:
                raise RuntimeError(
                    f"Need at least {world_size} parquet files in "
                    f"{config.data_dir} for world_size={world_size}; found "
                    f"only {len(all_files)}."
                )
        else:
            pool = all_files[:usable]
            self.files = [f for i, f in enumerate(pool) if i % world_size == rank]

        # Resumption state (authoritative position marker is (file_idx, text_idx);
        # buffer holds the in-flight leftover tokens from texts already consumed
        # into the buffer but not yet yielded as full sequences).
        self._start_file_idx: int = 0
        self._start_text_idx: int = 0
        self._start_buffer: list[int] = []
        # Backward-compat: if a pre-Round-1 `data_state.pt` is restored
        # (only `file_idx` / `seq_idx` / `buffer`, no `text_idx`), we fall back
        # to the old skip-seqs resume semantics by tokenising the start file
        # from the top and dropping the first `_start_seq_skip` seq_len chunks
        # before yielding. Always 0 for post-Round-1 state dicts.
        self._start_seq_skip: int = 0

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
        """Restore iteration position.

        Post-Round-1 payloads carry `text_idx` and that marker is authoritative.
        If the payload predates that change (a legacy `data_state.pt` produced
        by the skip-seqs resume code, with `seq_idx` but no `text_idx`) this
        falls back to the old semantics: start the current file from text 0,
        prepend the saved `buffer`, and drop the first `seq_idx` full seq_len
        chunks during iteration. That reproduces the pre-Round-1 resume path
        so auto-resume across the Round-1 upgrade does not silently reset to
        the start of the current file.
        """
        self._start_file_idx = int(state.get("file_idx", 0))
        if "text_idx" in state:
            # Post-Round-1 payload: buffer is authoritative (the new iter
            # keeps `_live_buffer` aliased with the live `token_buffer`).
            self._start_text_idx = int(state["text_idx"])
            self._start_buffer = list(state.get("buffer", []))
            self._start_seq_skip = 0
        else:
            # Legacy payload — no `text_idx` was ever written AND the saved
            # `buffer` is unreliable: pre-Round-1 `__iter__` rebound
            # `token_buffer` on every yielded chunk without updating the
            # `_live_buffer` alias, so `get_state()` serialised a stale
            # (often frozen-at-first-text or empty) buffer. Reproduce the
            # old skip-seqs semantics from a clean slate instead: re-walk
            # the file from text 0, drop the first `seq_idx` full chunks,
            # then yield normally.
            self._start_text_idx = 0
            self._start_buffer = []
            self._start_seq_skip = int(state.get("seq_idx", 0))

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

        files = self.files
        n_files = len(files)
        start = self._start_file_idx
        if start >= n_files:
            return

        executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="parquet-prefetch"
            )
            if self.config.prefetch_files > 0 else None
        )

        try:
            df = self._load_file(files[start])
            next_future = (
                executor.submit(self._load_file, files[start + 1])
                if executor is not None and start + 1 < n_files else None
            )

            start_text_idx = self._start_text_idx
            # Only applies to a legacy-format restore; consumed and cleared
            # while walking the start file.
            skip_seqs = self._start_seq_skip
            cur_file_idx = start

            def _drain_buffered_chunks():
                """Yield every complete seq_len+1 chunk already in the buffer.

                Called at two points: (1) once at the very start of the
                per-file loop so any chunks remaining in `_start_buffer` (a
                mid-drain save point) are emitted *before* the next text is
                tokenised — this is what preserves tensor-for-tensor resume
                when a text produced more chunks than fit between saves;
                (2) immediately after each text is absorbed, to drain new
                chunks that text contributed. Both call sites share the
                legacy skip-seqs semantics so a pre-Round-1 resume lands at
                the same position the old code did.
                """
                nonlocal skip_seqs
                while len(token_buffer) >= seq_len + 1:
                    if cur_file_idx == start and skip_seqs > 0:
                        del token_buffer[:seq_len]
                        skip_seqs -= 1
                        continue
                    chunk = token_buffer[: seq_len + 1]
                    # Advance in place to preserve `_live_buffer` aliasing
                    # and keep get_state() authoritative during iteration.
                    del token_buffer[:seq_len]
                    self._cur_seq_idx += 1
                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }

            while True:
                self._cur_file_idx = cur_file_idx
                self._cur_seq_idx = 0

                # Only the start file honours the resumption text offset;
                # subsequent files always begin at text 0.
                first_text = start_text_idx if cur_file_idx == start else 0
                self._cur_text_idx = first_text

                # Pre-loop drain: emit any complete chunks already in
                # `_start_buffer` *before* the first text of this file is
                # tokenised. This fixes the tensor-for-tensor resume
                # guarantee for mid-drain save points — without this, new
                # text tokens get appended ahead of buffered continuation
                # chunks (or dropped if the last text ran into its own
                # inner drain and no more texts remain).
                yield from _drain_buffered_chunks()

                texts = df[self.config.text_column]
                for text_idx in range(first_text, len(texts)):
                    text = texts.iat[text_idx]
                    if isinstance(text, str) and text.strip():
                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        token_buffer.extend(ids)
                        token_buffer.append(eos)

                    # Text `text_idx` is now fully absorbed into the buffer
                    # (or skipped as empty). Advance the "next text to
                    # consume" marker BEFORE yielding so a save between
                    # yields still points at the next un-consumed text;
                    # the pre-loop drain above handles the case where that
                    # save leaves complete chunks behind.
                    self._cur_text_idx = text_idx + 1

                    yield from _drain_buffered_chunks()

                # Finishing a file resets the per-file start offset.
                start_text_idx = 0
                cur_file_idx += 1
                if cur_file_idx >= n_files:
                    break

                # Block on the pre-submitted next-file future (or load sync
                # if prefetch is disabled), then kick off the file after that.
                if next_future is not None:
                    df = next_future.result()
                    next_future = None
                else:
                    df = self._load_file(files[cur_file_idx])
                if executor is not None and cur_file_idx + 1 < n_files:
                    next_future = executor.submit(
                        self._load_file, files[cur_file_idx + 1]
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def _load_file(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path, columns=[self.config.text_column])
