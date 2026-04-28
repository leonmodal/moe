#!/usr/bin/env python3
"""Unified training CLI entrypoint.

Usage:
    # Single GPU
    python scripts/train.py --config configs/standard_moe.yaml

    # DDP (multi-GPU)
    torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml

    # FSDP
    torchrun --nproc_per_node=8 scripts/train.py --config configs/standard_moe.yaml --dist-strategy fsdp

Supported model types: dense, standard_moe, global_moe, moe_everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

torch.backends.cuda.preferred_blas_library("cublaslt")

from src.training.config import load_config, build_training_config
from src.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified MoE training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--max_checkpoints", type=int, default=0, help="Max checkpoints to keep (0=unlimited)")
    parser.add_argument("--auto_resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument(
        "--dist-strategy",
        choices=("none", "ddp", "fsdp"),
        default="ddp",
        help="Distributed training strategy",
    )
    parser.add_argument("--init-from-config", default=None, help="Source config for weight initialization")
    parser.add_argument(
        "--init-strategy",
        choices=("global_to_alternating_sanity",),
        default=None,
        help="Weight initialization strategy",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Canonical-block resolution: resolve `load_balancing_method` once and
    # AUTO-ZERO any legacy coefficients that conflict with it BEFORE either
    # build runs.
    # This way both `build_training_config` and `build_model` see a
    # method-consistent view of the coefficients (e.g. with method=aux_loss,
    # `bias_update_rate` and `seq_aux_loss_coef` resolve to 0 even if the
    # yaml carried legacy non-zero values, with a one-time deprecation
    # warning describing the auto-zero).
    from src.training.balancing_fields import normalize_balancing_config
    normalize_balancing_config(cfg)
    train_cfg = build_training_config(cfg)

    # Apply CLI overrides to train_cfg
    if args.max_checkpoints:
        from dataclasses import replace
        train_cfg = replace(train_cfg, max_checkpoints=args.max_checkpoints)

    run_training(cfg, train_cfg, args)


if __name__ == "__main__":
    main()
