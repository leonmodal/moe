"""
Stateful dataset for GPT-2 token `.bin` shards.

The format matches the cached FineWeb bins used by modded-nanogpt/llm.c:

- 256 little-endian int32 header values (`1024` bytes total)
- header[0] = magic (`20240520`)
- header[1] = version (`1`)
- header[2] = token count
- payload = uint16 token ids

This dataset yields fixed-length `(input_ids, labels)` examples and keeps
enough state to resume deterministically from checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
import random
import struct
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
EXPECTED_MAGIC = 20240520
EXPECTED_VERSION = 1


@dataclass
class TokenBinConfig:
    files_glob: str
    seq_len: int = 2048
    header_bytes: int = HEADER_BYTES
    token_dtype: str = "uint16"
    max_tokens: int | None = None
    shuffle_files: bool = False
    repeat: bool = False


def _dtype_from_name(name: str) -> np.dtype:
    try:
        return np.dtype(name)
    except TypeError as exc:
        raise ValueError(f"Unsupported token dtype: {name}") from exc


def _read_bin_header(path: str, header_bytes: int) -> tuple[int, int, int]:
    if header_bytes == 0:
        return 0, 0, os.path.getsize(path) // np.dtype(np.uint16).itemsize
    if header_bytes < 12:
        raise ValueError(f"header_bytes must be at least 12, got {header_bytes}")
    with open(path, "rb") as f:
        header = f.read(header_bytes)
    if len(header) < 12:
        raise ValueError(f"{path} is too small to contain a valid bin header")
    magic, version, token_count = struct.unpack_from("<III", header, 0)
    return magic, version, token_count


class StatefulTokenBinDataset(IterableDataset):
    """
    Streams fixed-length sequences from cached GPT-2 token bins.

    Files are processed in a deterministic order, and each DDP rank walks a
    disjoint strided set of sequence start positions within every file:

    - rank `0` starts at token `0`
    - rank `1` starts at token `seq_len`
    - ...
    - all ranks advance by `world_size * seq_len`
    """

    def __init__(
        self,
        config: TokenBinConfig,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self._token_dtype = _dtype_from_name(config.token_dtype)

        files = sorted(glob.glob(config.files_glob))
        if not files:
            raise FileNotFoundError(f"No token bin files found for glob: {config.files_glob}")

        if config.shuffle_files:
            rng = random.Random(seed)
            rng.shuffle(files)

        raw_counts: list[int] = []
        for path in files:
            magic, version, token_count = _read_bin_header(path, config.header_bytes)
            if config.header_bytes:
                if magic != EXPECTED_MAGIC:
                    raise ValueError(f"{path} has unexpected magic {magic}, expected {EXPECTED_MAGIC}")
                if version != EXPECTED_VERSION:
                    raise ValueError(f"{path} has unexpected version {version}, expected {EXPECTED_VERSION}")
            raw_counts.append(int(token_count))

        token_limits: list[int] = []
        remaining = config.max_tokens
        for count in raw_counts:
            if remaining is None:
                token_limits.append(count)
                continue
            if remaining <= 0:
                break
            allowed = min(count, int(remaining))
            token_limits.append(allowed)
            remaining -= allowed

        if not token_limits:
            raise ValueError(f"max_tokens={config.max_tokens} leaves no readable tokens for {config.files_glob}")

        self.files = files[: len(token_limits)]
        self.file_token_counts = token_limits
        self.total_tokens = int(sum(token_limits))
        self.total_sequences = int(sum(self._num_sequences_in_file(n_tokens) for n_tokens in token_limits))

        self._default_token_pos = self.rank * self.config.seq_len
        self._start_file_idx = 0
        self._start_token_pos = self._default_token_pos
        self._cur_file_idx = 0
        self._cur_token_pos = self._default_token_pos

    def _num_sequences_in_file(self, token_count: int) -> int:
        seq_len = self.config.seq_len
        start = self.rank * seq_len
        max_start = token_count - (seq_len + 1)
        stride = self.world_size * seq_len
        if max_start < start:
            return 0
        return 1 + (max_start - start) // stride

    def get_state(self) -> dict:
        return {
            "file_idx": self._cur_file_idx,
            "token_pos": self._cur_token_pos,
        }

    def set_state(self, state: dict) -> None:
        self._start_file_idx = int(state.get("file_idx", 0))
        self._start_token_pos = int(state.get("token_pos", self._default_token_pos))

    def _next_state(self, file_idx: int, next_pos: int, max_start: int) -> tuple[int, int]:
        if next_pos <= max_start:
            return file_idx, next_pos
        return file_idx + 1, self._default_token_pos

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        seq_len = self.config.seq_len
        stride = self.world_size * seq_len

        while True:
            file_idx = self._start_file_idx
            while file_idx < len(self.files):
                path = self.files[file_idx]
                token_count = self.file_token_counts[file_idx]
                if token_count < seq_len + 1:
                    file_idx += 1
                    continue

                token_pos = self._start_token_pos if file_idx == self._start_file_idx else self._default_token_pos
                max_start = token_count - (seq_len + 1)
                if token_pos > max_start:
                    file_idx += 1
                    continue

                tokens = np.memmap(
                    path,
                    mode="r",
                    dtype=self._token_dtype,
                    offset=self.config.header_bytes,
                    shape=(token_count,),
                )

                while token_pos <= max_start:
                    next_file_idx, next_token_pos = self._next_state(file_idx, token_pos + stride, max_start)
                    self._cur_file_idx = next_file_idx
                    self._cur_token_pos = next_token_pos

                    chunk = np.asarray(tokens[token_pos : token_pos + seq_len + 1], dtype=np.int64)
                    yield {
                        "input_ids": torch.from_numpy(chunk[:-1].copy()).to(torch.long),
                        "labels": torch.from_numpy(chunk[1:].copy()).to(torch.long),
                    }
                    token_pos += stride

                file_idx += 1
                self._start_token_pos = self._default_token_pos

            if not self.config.repeat:
                self._cur_file_idx = len(self.files)
                self._cur_token_pos = self._default_token_pos
                return

            self._start_file_idx = 0
            self._start_token_pos = self._default_token_pos
