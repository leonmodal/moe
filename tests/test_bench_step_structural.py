"""Structural test for `scripts/bench_step.py`. The bench runs end-
to-end through the actual `load_config -> build_model -> warmup ->
measured iters` sequence on a tiny CPU-friendly config and emits a
schema-conforming JSON record. The per-config tokens/sec figure is
not asserted (CPU is too noisy) — but the schema and the script's
end-to-end invocation contract are.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def tiny_config_yaml(tmp_path):
    """A minimal moe_everything yaml the bench can build + run on
    CPU in a few seconds."""
    yaml_path = tmp_path / "tiny_bench.yaml"
    yaml_path.write_text(
        """experiment_name: bench_structural
model:
  type: moe_everything
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 1
  head_dim: 8
  num_attention_heads: 2
  num_key_value_heads: 2
  num_experts: 2
  num_experts_per_tok: 1
  moe_intermediate_size: 16
  intermediate_size: 16
  norm_topk_prob: true
  branch_router_aux_loss_coef: 0.0
  router_exploration_rate: 0.0
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_fully_independent
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
  per_head_compute_mode: dense
  use_deepseek_routing: true
  branch_deepseek: true
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 32
  tie_word_embeddings: true
  output_router_logits: false
  attn_implementation: eager
training:
  learning_rate: 1.0e-3
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 0
  max_steps: 100
  min_lr_ratio: 0.1
  batch_size: 1
  gradient_accumulation: 1
  mixed_precision: ""
  log_every: 1
  save_every: 100000
  output_dir: /tmp/bench_structural_out
  router_aux_loss_coef: 0.0
  seq_aux_loss_coef: 0.0
  bias_update_rate: 0.0
"""
    )
    return yaml_path


def _run_bench(args, cwd):
    cmd = [sys.executable, "scripts/bench_step.py"] + args
    env = {"PYTHONPATH": str(REPO)}
    import os
    env_full = {**os.environ, **env}
    result = subprocess.run(
        cmd, cwd=str(cwd), env=env_full, capture_output=True, text=True,
    )
    return result


def test_bench_step_emits_schema_conforming_record(tiny_config_yaml, tmp_path):
    """End-to-end smoke test: the bench script should run on the
    tiny CPU config, print a JSON record to stdout, and append the
    same record to the output JSON file."""
    output_path = tmp_path / "results.json"
    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "1",
            "--measure", "2",
            "--device", "cpu",
            "--output", str(output_path),
        ],
        cwd=REPO,
    )
    assert result.returncode == 0, (
        f"bench_step.py failed:\nSTDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
    record = json.loads(result.stdout)
    expected_keys = {
        "config", "model_type", "device", "warmup_iters", "measure_iters",
        "batch_size", "seq_len", "tokens_per_step", "wall_seconds_mean",
        "wall_seconds_std", "tokens_per_second", "peak_gpu_memory_bytes",
        "loss_mean", "loss_std", "search_mode", "search_max_batch",
    }
    assert set(record.keys()) == expected_keys, (
        f"output schema drift: expected {expected_keys}, got {set(record.keys())}"
    )
    assert record["device"] == "cpu"
    assert record["warmup_iters"] == 1
    assert record["measure_iters"] == 2
    assert record["model_type"] == "moe_everything"
    assert record["search_mode"] is False
    assert record["wall_seconds_mean"] > 0
    assert record["tokens_per_second"] > 0

    on_disk = json.loads(output_path.read_text())
    assert isinstance(on_disk, list) and len(on_disk) == 1
    assert on_disk[0] == record


def test_bench_step_search_mode_smoke(tiny_config_yaml, tmp_path):
    """`--search` should find a max stable batch size in the given
    range and report `search_mode=True` plus a non-null
    `search_max_batch`."""
    output_path = tmp_path / "results.json"
    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "1",
            "--measure", "2",
            "--device", "cpu",
            "--search",
            "--batch-min", "1",
            "--batch-max", "2",
            "--output", str(output_path),
        ],
        cwd=REPO,
    )
    assert result.returncode == 0, (
        f"bench_step.py --search failed:\nSTDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
    record = json.loads(result.stdout)
    assert record["search_mode"] is True
    assert record["search_max_batch"] in (1, 2)


def test_bench_step_appends_to_existing_results(tiny_config_yaml, tmp_path):
    """Two consecutive bench invocations against the same output
    path append to the JSON list rather than overwriting it. The
    AC-22 sweep depends on this so the 39 config records co-exist
    in `bench/results.json`."""
    output_path = tmp_path / "results.json"
    output_path.write_text(json.dumps([{"prior": True}]))

    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "1",
            "--measure", "2",
            "--device", "cpu",
            "--output", str(output_path),
        ],
        cwd=REPO,
    )
    assert result.returncode == 0
    on_disk = json.loads(output_path.read_text())
    assert len(on_disk) == 2
    assert on_disk[0] == {"prior": True}
    assert on_disk[1]["model_type"] == "moe_everything"


if __name__ == "__main__":
    print(
        "Run via pytest:\n"
        "  PYTHONPATH=. python -m pytest tests/test_bench_step_structural.py"
    )
