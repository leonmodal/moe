"""AC-9 subprocess smoke: the unified trainer actually runs under DDP and FSDP.

The in-process smoke matrix in `test_unified_trainer.py` builds models directly
and bypasses `scripts/train.py`, the distributed wrappers, and the real
DataLoader / checkpoint pipeline. This test launches the actual training
entrypoint via `torchrun --standalone --nproc_per_node=2` against a synthetic
parquet fixture, completes at least one optimizer step, and asserts that the
resulting checkpoint directory contains the separate state files required by
AC-12 (`model.pt`, `optimizer_adam.pt`, `training_state.pt`, `data_state.pt`,
`meta.json`).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"


def _write_parquet_shards(data_dir: Path, num_files: int = 2, rows_per_file: int = 64) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in range(num_files):
        rows = [
            " ".join([f"file{f} row{r} word{w}" for w in range((r % 9) + 5)])
            for r in range(rows_per_file)
        ]
        pd.DataFrame({"text": rows}).to_parquet(data_dir / f"shard_{f:04d}.parquet")


def _write_tiny_dense_config(path: Path, *, output_dir: Path, data_dir: Path) -> None:
    path.write_text(textwrap.dedent(f"""\
        experiment_name: dist_smoke
        model:
          type: dense
          vocab_size: 50304
          hidden_size: 64
          num_hidden_layers: 2
          head_dim: 16
          num_attention_heads: 4
          num_key_value_heads: 2
          intermediate_size: 128
          max_position_embeddings: 128
          rms_norm_eps: 1.0e-6
          rope_theta: 1000000.0
          tie_word_embeddings: true
          attention_bias: false
          attention_dropout: 0.0
        training:
          learning_rate: 1.0e-3
          weight_decay: 0.0
          max_grad_norm: 1.0
          lr_scheduler: constant
          warmup_steps: 0
          max_steps: 2
          min_lr_ratio: 1.0
          batch_size: 1
          gradient_accumulation: 1
          mixed_precision: bf16
          log_every: 1
          save_every: 1
          output_dir: {output_dir}
          wandb_project: null
          optimizer: adamw
          max_checkpoints: 0
          disable_liger: true
        data:
          data_dir: {data_dir}
          text_column: text
          seq_len: 32
          tokenizer_name: gpt2
          num_workers: 0
          prefetch_files: 0
        eval:
          enabled: false
        checkpoint:
          resume_from: null
    """))


def _required_checkpoint_files(ckpt_dir: Path) -> list[Path]:
    return [
        ckpt_dir / "model.pt",
        ckpt_dir / "optimizer_adam.pt",
        ckpt_dir / "training_state.pt",
        ckpt_dir / "data_state.pt",
        ckpt_dir / "meta.json",
    ]


def _find_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return candidates[-1] if candidates else None


def _run_trainer_subprocess(tmp_path: Path, dist_strategy: str, timeout: int = 600) -> tuple[subprocess.CompletedProcess, Path]:
    data_dir = tmp_path / "data"
    _write_parquet_shards(data_dir, num_files=2, rows_per_file=64)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_tiny_dense_config(config_path, output_dir=output_dir, data_dir=data_dir)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{_REPO_ROOT}" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["WANDB_DISABLED"] = "true"
    # Quiet NCCL chatter so failures are visible.
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(_TRAIN_SCRIPT),
        "--config", str(config_path),
        "--dist-strategy", dist_strategy,
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(_REPO_ROOT))
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        pytest.fail(
            f"torchrun {dist_strategy} exited with code {result.returncode} after {elapsed:.1f}s\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result, output_dir


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
def test_unified_trainer_ddp_subprocess_smoke(tmp_path):
    """`torchrun --nproc_per_node=2 scripts/train.py --dist-strategy ddp` runs to
    completion, executes ≥1 optimizer step, and writes a checkpoint directory
    with the full set of separate state files.
    """
    result, output_dir = _run_trainer_subprocess(tmp_path, dist_strategy="ddp")
    assert "step=" in result.stdout or "step=" in result.stderr, (
        f"Expected at least one logged optimizer step in stdout/stderr; got:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    ckpt = _find_checkpoint(output_dir)
    assert ckpt is not None, f"No checkpoint-* directory under {output_dir}"
    for req in _required_checkpoint_files(ckpt):
        assert req.exists(), f"Expected {req} after DDP smoke; not found"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
def test_unified_trainer_fsdp_subprocess_smoke(tmp_path):
    """Same shape as the DDP smoke but with `--dist-strategy fsdp`."""
    result, output_dir = _run_trainer_subprocess(tmp_path, dist_strategy="fsdp")
    assert "step=" in result.stdout or "step=" in result.stderr, (
        f"Expected at least one logged optimizer step in stdout/stderr; got:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    ckpt = _find_checkpoint(output_dir)
    assert ckpt is not None, f"No checkpoint-* directory under {output_dir}"
    for req in _required_checkpoint_files(ckpt):
        assert req.exists(), f"Expected {req} after FSDP smoke; not found"
