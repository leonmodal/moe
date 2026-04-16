"""Dataset construction for training: sharded parquet only."""

from __future__ import annotations

from src.data import DataConfig as ParquetDataConfig, StatefulParquetDataset


def get_data_format(cfg_dict: dict) -> str:
    """Determine the data format from config. Only 'parquet' is supported."""
    fmt = cfg_dict.get("format", "parquet")
    if fmt != "parquet":
        raise ValueError(
            f"Unsupported data format: '{fmt}'. Only 'parquet' is supported. "
            f"Token-bin format has been removed."
        )
    return "parquet"


def build_dataset_from_config(
    cfg_dict: dict,
    *,
    rank: int,
    world_size: int,
    seed: int,
    tokenizer=None,
) -> StatefulParquetDataset:
    """Build a StatefulParquetDataset from config dict."""
    data_cfg = ParquetDataConfig(
        data_dir=cfg_dict["data_dir"],
        text_column=cfg_dict.get("text_column", "text"),
        seq_len=cfg_dict.get("seq_len", 2048),
        tokenizer_name=cfg_dict.get("tokenizer_name", "gpt2"),
        num_workers=cfg_dict.get("num_workers", 4),
        split=cfg_dict.get("split", "all"),
        holdout_fraction=cfg_dict.get("holdout_fraction", 0.0),
        # Copy prefetch_files through so `data.prefetch_files: 0` in YAML
        # actually disables the dataset-level read-ahead added in Round 3.
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


def build_train_dataset(
    cfg: dict,
    *,
    tokenizer,
    rank: int,
    world_size: int,
) -> StatefulParquetDataset:
    """Build the training dataset from full config."""
    dcfg = dict(cfg.get("data", {}))
    dcfg["split"] = "all"
    dcfg["holdout_fraction"] = 0.0
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
) -> StatefulParquetDataset | None:
    """Build the eval dataset from full config, if eval is enabled."""
    eval_cfg = cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return None

    dcfg = dict(cfg.get("data", {}))
    dcfg.update(eval_cfg)

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
