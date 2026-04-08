import struct

import numpy as np

from src.data.token_bin_dataset import (
    EXPECTED_MAGIC,
    EXPECTED_VERSION,
    HEADER_INTS,
    StatefulTokenBinDataset,
    TokenBinConfig,
)


def _write_token_bin(path, tokens) -> None:
    header = [0] * HEADER_INTS
    header[0] = EXPECTED_MAGIC
    header[1] = EXPECTED_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{HEADER_INTS}I", *header))
        np.asarray(tokens, dtype=np.uint16).tofile(f)


def test_token_bin_dataset_rank_sharding_and_header(tmp_path):
    path = tmp_path / "sample.bin"
    _write_token_bin(path, list(range(20)))

    cfg = TokenBinConfig(files_glob=str(path), seq_len=4)
    rank0 = StatefulTokenBinDataset(cfg, rank=0, world_size=2)
    rank1 = StatefulTokenBinDataset(cfg, rank=1, world_size=2)

    rank0_items = list(rank0)
    rank1_items = list(rank1)

    assert [item["input_ids"].tolist() for item in rank0_items] == [
        [0, 1, 2, 3],
        [8, 9, 10, 11],
    ]
    assert [item["labels"].tolist() for item in rank0_items] == [
        [1, 2, 3, 4],
        [9, 10, 11, 12],
    ]
    assert [item["input_ids"].tolist() for item in rank1_items] == [
        [4, 5, 6, 7],
        [12, 13, 14, 15],
    ]


def test_token_bin_dataset_resume_state(tmp_path):
    path = tmp_path / "sample.bin"
    _write_token_bin(path, list(range(20)))

    cfg = TokenBinConfig(files_glob=str(path), seq_len=4)
    ds = StatefulTokenBinDataset(cfg, rank=0, world_size=1)
    it = iter(ds)
    first = next(it)
    state = ds.get_state()
    second = next(it)

    resumed = StatefulTokenBinDataset(cfg, rank=0, world_size=1)
    resumed.set_state(state)
    resumed_second = next(iter(resumed))

    assert first["input_ids"].tolist() == [0, 1, 2, 3]
    assert state == {"file_idx": 0, "token_pos": 4}
    assert second["input_ids"].tolist() == [4, 5, 6, 7]
    assert resumed_second["input_ids"].tolist() == [4, 5, 6, 7]
    assert resumed_second["labels"].tolist() == [5, 6, 7, 8]


def test_token_bin_dataset_max_tokens_cap(tmp_path):
    path = tmp_path / "sample.bin"
    _write_token_bin(path, list(range(25)))

    cfg = TokenBinConfig(files_glob=str(path), seq_len=4, max_tokens=10)
    ds = StatefulTokenBinDataset(cfg, rank=0, world_size=1)
    items = list(ds)

    assert len(items) == 2
    assert [item["input_ids"].tolist() for item in items] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert [item["labels"].tolist() for item in items] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]


def test_token_bin_dataset_align_to_bos_shards_and_resumes(tmp_path):
    path = tmp_path / "aligned.bin"
    bos = 50256
    tokens = [
        bos, 10, 11, 12,
        bos, 20, 21, 22,
        bos, 30, 31, 32,
        bos, 40, 41, 42,
        bos, 50, 51, 52,
    ]
    _write_token_bin(path, tokens)

    cfg = TokenBinConfig(files_glob=str(path), seq_len=3, align_to_bos=True, bos_token_id=bos)
    rank0 = StatefulTokenBinDataset(cfg, rank=0, world_size=2)
    rank1 = StatefulTokenBinDataset(cfg, rank=1, world_size=2)

    rank0_items = list(rank0)
    rank1_items = list(rank1)

    assert [item["input_ids"].tolist() for item in rank0_items] == [
        [bos, 10, 11],
        [bos, 30, 31],
    ]
    assert [item["input_ids"].tolist() for item in rank1_items] == [
        [bos, 20, 21],
        [bos, 40, 41],
    ]

    ds = StatefulTokenBinDataset(cfg, rank=0, world_size=2)
    it = iter(ds)
    first = next(it)
    state = ds.get_state()
    resumed = StatefulTokenBinDataset(cfg, rank=0, world_size=2)
    resumed.set_state(state)
    resumed_second = next(iter(resumed))

    assert first["input_ids"].tolist() == [bos, 10, 11]
    assert state == {"file_idx": 0, "token_pos": 8}
    assert resumed_second["input_ids"].tolist() == [bos, 30, 31]
