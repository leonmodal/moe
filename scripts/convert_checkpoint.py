#!/usr/bin/env python3
"""Convert model checkpoints to safetensors format.

Usage:
    python scripts/convert_checkpoint.py path/to/checkpoint-1000
    python scripts/convert_checkpoint.py path/to/checkpoint-1000 --output model.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import convert_to_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert checkpoint to safetensors")
    parser.add_argument("checkpoint_dir", help="Directory containing the checkpoint")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: model.safetensors in checkpoint dir)")
    args = parser.parse_args()

    output_path = convert_to_safetensors(args.checkpoint_dir, args.output)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
