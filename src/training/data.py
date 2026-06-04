"""Dataset construction for training: sharded parquet or synthetic streams."""

from __future__ import annotations

from src.data import (
    DataConfig as ParquetDataConfig,
    StatefulParquetDataset,
    build_synthetic_dataset,
)

_SYNTHETIC_FORMATS = {"synthetic_linear_map", "synthetic_cellular_automata"}


def get_data_format(cfg_dict: dict) -> str:
    """Determine the data format from config.

    Supported: 'parquet', 'synthetic_linear_map', 'synthetic_cellular_automata'.
    """
    fmt = cfg_dict.get("format", "parquet")
    if fmt == "parquet" or fmt in _SYNTHETIC_FORMATS:
        return fmt
    raise ValueError(
        f"Unsupported data format: '{fmt}'. Supported: 'parquet', "
        f"{sorted(_SYNTHETIC_FORMATS)}. Token-bin format has been removed."
    )


def is_synthetic_format(cfg_dict: dict) -> bool:
    return cfg_dict.get("format", "parquet") in _SYNTHETIC_FORMATS


def build_dataset_from_config(
    cfg_dict: dict,
    *,
    rank: int,
    world_size: int,
    seed: int,
    tokenizer=None,
):
    """Build a dataset from a `data:` config dict.

    Returns either a `StatefulParquetDataset` or a synthetic dataset
    (`LinearMapDataset` / `CellularAutomataDataset`). Both expose the
    `get_state` / `set_state` resume interface the trainer needs.
    """
    fmt = get_data_format(cfg_dict)
    if fmt in _SYNTHETIC_FORMATS:
        # Synthetic datasets ignore the tokenizer. The `task_seed` controls
        # ground-truth A / rules and MUST be the same across train and eval;
        # it is read from `data.seed` (or `data.task_seed` if set explicitly).
        # The `stream_seed` controls per-example sampling and is set from the
        # builder's `seed` argument so train and eval get distinct streams
        # while sharing the same ground-truth task.
        synth_cfg = dict(cfg_dict)
        synth_cfg.setdefault("task_seed", cfg_dict.get("seed", 0))
        synth_cfg["stream_seed"] = seed
        return build_synthetic_dataset(synth_cfg, rank=rank, world_size=world_size)

    data_cfg = ParquetDataConfig(
        data_dir=cfg_dict["data_dir"],
        text_column=cfg_dict.get("text_column", "text"),
        seq_len=cfg_dict.get("seq_len", 2048),
        tokenizer_name=cfg_dict.get("tokenizer_name", "gpt2"),
        num_workers=cfg_dict.get("num_workers", 4),
        split=cfg_dict.get("split", "all"),
        holdout_fraction=cfg_dict.get("holdout_fraction", 0.0),
        # Copy prefetch_files through so `data.prefetch_files: 0` in
        # YAML actually disables the dataset-level read-ahead.
        # Default matches `DataConfig` (one-file read-ahead).
        prefetch_files=cfg_dict.get("prefetch_files", 1),
    )
    return StatefulParquetDataset(
        config=data_cfg,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )


def _synthetic_task_seed(cfg: dict) -> int:
    """Read the shared synthetic-task seed from the original `data:` block.

    `data.task_seed` controls the ground-truth A / rules; it MUST be the same
    for train and eval so they evaluate the same task. Falls back to
    `data.seed` for legacy configs; defaults to 0.
    """
    data_cfg = cfg.get("data", {})
    return int(data_cfg.get("task_seed", data_cfg.get("seed", 0)))


def build_train_dataset(
    cfg: dict,
    *,
    tokenizer,
    rank: int,
    world_size: int,
):
    """Build the training dataset from full config."""
    dcfg = dict(cfg.get("data", {}))
    dcfg["split"] = "all"
    dcfg["holdout_fraction"] = 0.0
    if is_synthetic_format(dcfg):
        # Pin the synthetic task_seed to its shared source before the merged
        # dcfg can drift it (eval merges in `eval.seed` which would otherwise
        # land here on resume / branching).
        dcfg["task_seed"] = _synthetic_task_seed(cfg)
    return build_dataset_from_config(
        dcfg,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=dcfg.get("seed", 42),
    )


def build_eval_dataset(
    cfg: dict,
    *,
    tokenizer,
    rank: int,
    world_size: int,
):
    """Build the eval dataset from full config, if eval is enabled."""
    eval_cfg = cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return None

    dcfg = dict(cfg.get("data", {}))
    dcfg.update(eval_cfg)

    if is_synthetic_format(dcfg):
        # Eval shares the task_seed with train so the same ground-truth A /
        # rules govern both phases. Only the stream_seed differs, which the
        # builder pulls from the `seed=` argument below.
        dcfg["task_seed"] = _synthetic_task_seed(cfg)
        return build_dataset_from_config(
            dcfg,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            seed=eval_cfg.get("seed", 1234),
        )

    eval_data_dir = eval_cfg.get("data_dir", dcfg.get("data_dir"))
    eval_split = eval_cfg.get("split")
    if eval_split is None:
        eval_split = "all" if "data_dir" in eval_cfg else "val"
    dcfg["data_dir"] = eval_data_dir
    dcfg["split"] = eval_split
    dcfg["holdout_fraction"] = eval_cfg.get(
        "holdout_fraction",
        0.05 if eval_split == "val" else 0.0,
    )

    return build_dataset_from_config(
        dcfg,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=eval_cfg.get("seed", 1234),
    )
