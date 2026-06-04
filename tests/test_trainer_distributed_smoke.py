"""AC-9 subprocess smoke: the unified trainer actually runs under DDP and FSDP.

The in-process smoke matrix in `test_unified_trainer.py` builds models directly
and bypasses `scripts/train.py`, the distributed wrappers, and the real
DataLoader / checkpoint pipeline. This test launches the actual training
entrypoint via `torchrun --standalone --nproc_per_node=2` against a synthetic
parquet fixture, completes at least one optimizer step, and asserts that the
resulting checkpoint directory contains the separate state files required by
AC-12 (`model.pt`, `optimizer_adam.pt`, `training_state.pt`, `data_state.pt`,
`meta.json`).

Parametrized across model families so AC-9's "all supported models" wording is
exercised end-to-end, not just inferred from an in-process smoke.
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


_COMMON_TRAINING_YAML = textwrap.dedent("""\
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
      gradient_checkpointing: {gradient_checkpointing}
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
""")


# Shared model-section fragments assembled into the per-variant YAMLs below.
_BASE_MODEL_FIELDS = """  vocab_size: 50304
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
  attention_dropout: 0.0"""

_MOE_COMMON = """  num_experts: 4
  num_experts_per_tok: 2
  moe_intermediate_size: 32
  norm_topk_prob: true
  router_aux_loss_coef: 0.001
  output_router_logits: true"""

_DEEPSEEK_EXTRA = """  topk_scaling_factor: 2.5
  num_groups: 2
  group_topk: 1"""


_MODEL_YAMLS: dict[str, str] = {
    # Dense baseline — no routing; proves the trainer path itself.
    "dense": (
        "experiment_name: dist_smoke_dense\n"
        "model:\n"
        "  type: dense\n"
    ) + _BASE_MODEL_FIELDS + "\n",
    # standard_moe: per-layer MoE under both routing modes.
    "standard_moe_softmax": (
        "experiment_name: dist_smoke_standard_moe_softmax\n"
        "model:\n"
        "  type: standard_moe\n"
        "  router_type: softmax\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n",
    "standard_moe_deepseek": (
        "experiment_name: dist_smoke_standard_moe_deepseek\n"
        "model:\n"
        "  type: standard_moe\n"
        "  router_type: deepseek\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n" + _DEEPSEEK_EXTRA + "\n",
    # global_moe: shared expert pool across all layers.
    "global_moe_softmax": (
        "experiment_name: dist_smoke_global_moe_softmax\n"
        "model:\n"
        "  type: global_moe\n"
        "  router_type: softmax\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n",
    "global_moe_deepseek": (
        "experiment_name: dist_smoke_global_moe_deepseek\n"
        "model:\n"
        "  type: global_moe\n"
        "  router_type: deepseek\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n" + _DEEPSEEK_EXTRA + "\n",
    # moe_everything: branch routing + per-head attention expert banks.
    "moe_everything_no_recompute": (
        "experiment_name: dist_smoke_moe_everything_no_recompute\n"
        "model:\n"
        "  type: moe_everything\n"
        "  router_type: softmax\n"
        "  num_attn_experts: 4\n"
        "  num_attn_experts_per_tok: 1\n"
        "  attn_expert_mode: per_head_no_recompute\n"
        "  attn_routing_bundle: q_k_v_o\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n",
    "moe_everything_recompute_k": (
        "experiment_name: dist_smoke_moe_everything_recompute_k\n"
        "model:\n"
        "  type: moe_everything\n"
        "  router_type: softmax\n"
        "  num_attn_experts: 4\n"
        "  num_attn_experts_per_tok: 1\n"
        "  attn_expert_mode: per_head_recompute_k\n"
        "  attn_routing_bundle: qkvo\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n",
    "moe_everything_recompute_kv": (
        "experiment_name: dist_smoke_moe_everything_recompute_kv\n"
        "model:\n"
        "  type: moe_everything\n"
        "  router_type: softmax\n"
        "  num_attn_experts: 4\n"
        "  num_attn_experts_per_tok: 1\n"
        "  attn_expert_mode: per_head_recompute_kv\n"
        "  attn_routing_bundle: qkvo\n"
    ) + _BASE_MODEL_FIELDS + "\n" + _MOE_COMMON + "\n",
}


def _write_config(path: Path, *, model_variant: str, output_dir: Path, data_dir: Path) -> None:
    model_block = _MODEL_YAMLS[model_variant]
    training_block = _COMMON_TRAINING_YAML.format(
        output_dir=output_dir,
        data_dir=data_dir,
        gradient_checkpointing="true" if model_variant.startswith("moe_everything") else "false",
    )
    path.write_text(model_block + training_block)


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


def _run_trainer_subprocess(
    tmp_path: Path,
    *,
    model_variant: str,
    dist_strategy: str,
    fsdp_sharding_strategy: str | None = None,
    timeout: int = 600,
) -> tuple[subprocess.CompletedProcess, Path]:
    data_dir = tmp_path / "data"
    _write_parquet_shards(data_dir, num_files=2, rows_per_file=64)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, model_variant=model_variant,
                  output_dir=output_dir, data_dir=data_dir)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{_REPO_ROOT}" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["WANDB_DISABLED"] = "true"
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
    if fsdp_sharding_strategy is not None:
        cmd += ["--fsdp-sharding-strategy", fsdp_sharding_strategy]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(_REPO_ROOT))
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        pytest.fail(
            f"torchrun {dist_strategy} ({model_variant}) exited with code "
            f"{result.returncode} after {elapsed:.1f}s\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result, output_dir


_SUPPORTED_VARIANTS = [
    "dense",
    "standard_moe_softmax",
    "standard_moe_deepseek",
    "global_moe_softmax",
    "global_moe_deepseek",
    "moe_everything_no_recompute",
    "moe_everything_recompute_k",
    "moe_everything_recompute_kv",
]


_SMOKE_MATRIX = [
    (variant, strategy)
    for variant in _SUPPORTED_VARIANTS
    for strategy in ("ddp", "fsdp")
]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
@pytest.mark.parametrize("model_variant,dist_strategy", _SMOKE_MATRIX)
def test_unified_trainer_subprocess_smoke(tmp_path, model_variant, dist_strategy):
    """Launch `scripts/train.py` under `torchrun --nproc_per_node=2` for the
    parametrized (model, strategy) pair. Each combination executes at least
    one optimizer step, writes a full AC-12 checkpoint directory, and
    reports the effective wrapper class in the banner so the assertions
    below can pin the real distributed runtime (no silent substitutions).
    """
    result, output_dir = _run_trainer_subprocess(
        tmp_path, model_variant=model_variant, dist_strategy=dist_strategy,
    )
    combined = result.stdout + result.stderr
    assert "step=" in combined, (
        f"Expected at least one logged optimizer step for {model_variant} "
        f"under {dist_strategy}; got:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    # The trainer banner must report the actual wrapper (FSDP or DDP), so a
    # future regression that silently substitutes one for the other surfaces
    # here. moe_everything must use real FSDP under --dist-strategy fsdp;
    # every other (variant, fsdp) pair must use FSDP too; every ddp pair
    # must use DDP.
    if dist_strategy == "fsdp":
        assert "Wrapper   : FSDP(" in combined, (
            f"{model_variant}/fsdp must use a real FSDP wrapper — the banner "
            f"should read 'Wrapper   : FSDP(...)'. Got:\n{combined}"
        )
        if model_variant.startswith("moe_everything"):
            assert "NO_SHARD" in combined and "auto_wrap=yes" in combined, (
                "moe_everything × fsdp must use NO_SHARD + an auto_wrap_policy "
                "so sparse-gradient sub-modules stay in their own FSDP units. "
                f"Got banner line from:\n{combined}"
            )
            assert "Gradient checkpointing enabled." in combined
        else:
            assert "FULL_SHARD" in combined, (
                f"{model_variant}/fsdp must use FULL_SHARD sharding. Got:\n{combined}"
            )
    else:  # ddp
        assert "Wrapper   : DDP" in combined, (
            f"{model_variant}/ddp must use a DDP wrapper. Got:\n{combined}"
        )
        if model_variant.startswith("moe_everything"):
            assert "Gradient checkpointing enabled." in combined

    ckpt = _find_checkpoint(output_dir)
    assert ckpt is not None, (
        f"No checkpoint-* directory under {output_dir} for "
        f"{model_variant}/{dist_strategy}"
    )
    for req in _required_checkpoint_files(ckpt):
        assert req.exists(), (
            f"Expected {req} after {model_variant}/{dist_strategy} smoke; not found"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
def test_fsdp_no_shard_override_smoke(tmp_path):
    """Pin the FSDP NO_SHARD override used by the Modal launcher."""
    result, _output_dir = _run_trainer_subprocess(
        tmp_path,
        model_variant="standard_moe_softmax",
        dist_strategy="fsdp",
        fsdp_sharding_strategy="no_shard",
    )
    combined = result.stdout + result.stderr
    assert "Wrapper   : FSDP(NO_SHARD" in combined
