"""Structural test for `scripts/bench_step.py`. The bench runs end-
to-end through the actual `load_config -> build_model -> warmup ->
measured iters` sequence on a tiny CPU-friendly config and emits a
schema-conforming JSON record. The per-config tokens/sec figure is
not asserted (CPU is too noisy) — but the schema and the script's
end-to-end invocation contract are.
"""
from __future__ import annotations

import json
import os
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


_EXPECTED_KEYS = {
    "config", "model_type", "device", "warmup_iters", "measure_iters",
    "batch_size", "gradient_accumulation", "gradient_checkpointing",
    "chunked_ce", "seq_len", "tokens_per_step",
    "wall_seconds_median", "wall_seconds_p5", "wall_seconds_p95",
    "tokens_per_second_median", "peak_gpu_memory_bytes",
    "headroom_bytes", "loss_first", "loss_last", "search_mode", "oom",
}


def test_bench_step_emits_ac19_compliant_median_schema(tiny_config_yaml, tmp_path):
    """End-to-end smoke test: the bench should run the production
    optimizer-step + post-step bias-update path on the tiny CPU
    config and emit the AC-19 / AC-24 median-schema record (median,
    p5, p95 — NOT mean / std).
    """
    output_path = tmp_path / "results.json"
    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "2",
            "--measure", "3",
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
    assert set(record.keys()) == _EXPECTED_KEYS, (
        f"output schema drift: expected {_EXPECTED_KEYS}, got {set(record.keys())}"
    )
    assert record["device"] == "cpu"
    assert record["warmup_iters"] == 2
    assert record["measure_iters"] == 3
    assert record["model_type"] == "moe_everything"
    assert record["search_mode"] is False
    assert record["oom"] is False
    assert record["wall_seconds_median"] > 0
    assert record["tokens_per_second_median"] > 0
    # p5 <= median <= p95 (ordering invariant on the timing distribution).
    assert record["wall_seconds_p5"] <= record["wall_seconds_median"]
    assert record["wall_seconds_median"] <= record["wall_seconds_p95"]

    on_disk = json.loads(output_path.read_text())
    assert isinstance(on_disk, list) and len(on_disk) == 1
    assert on_disk[0] == record


def test_bench_step_search_mode_emits_multi_dim_records(tiny_config_yaml, tmp_path):
    """`--search` sweeps the AC-24 axes (batch / grad_accum /
    grad_ckpt / chunked_ce) and emits one record per grid point.
    Records carry `search_mode=true`. OOM rows would have `oom=true`
    but the tiny CPU config does not OOM, so all records here have
    `oom=false`.
    """
    output_path = tmp_path / "results.json"
    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "1",
            "--measure", "2",
            "--device", "cpu",
            "--search",
            "--search-batch-min", "1",
            "--search-batch-max", "2",
            "--search-grad-accum", "1,2",
            "--search-grad-ckpt", "false",
            "--search-chunked-ce", "false",
            "--output", str(output_path),
        ],
        cwd=REPO,
    )
    assert result.returncode == 0, (
        f"bench_step.py --search failed:\nSTDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
    records = json.loads(result.stdout)
    assert isinstance(records, list)
    # 2 batches x 2 grad_accum x 1 grad_ckpt x 1 chunked_ce = 4 rows.
    assert len(records) == 4
    distinct_points = {
        (r["batch_size"], r["gradient_accumulation"],
         r["gradient_checkpointing"], r["chunked_ce"])
        for r in records
    }
    assert distinct_points == {
        (1, 1, False, False), (1, 2, False, False),
        (2, 1, False, False), (2, 2, False, False),
    }
    for r in records:
        assert r["search_mode"] is True
        assert r["oom"] is False


def test_bench_step_search_records_oom_without_aborting(tiny_config_yaml, tmp_path):
    """When no GPU is available, the search loop simply runs every
    grid point on CPU; the AC-24 OOM-row contract still requires
    that a real OOM in the matrix is recorded with `oom=true`
    rather than aborting. We can verify the contract with a
    bench-internal probe on the OOM-record helper without needing
    actual CUDA OOMs.
    """
    repo = REPO
    cmd = [
        sys.executable, "-c",
        """
import json
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import importlib.util
spec = importlib.util.spec_from_file_location("bench_step", "scripts/bench_step.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Argparse stub.
import argparse
args = argparse.Namespace(
    config="configs/8_layers/standard_moe.yaml", warmup=10, measure=90,
)
record = mod._oom_record(
    {
        "args": args, "cfg_template": {"model": {"type": "moe_everything"}},
        "device": "cuda:0",
        "batch_size": 64, "gradient_accumulation": 4,
        "gradient_checkpointing": True, "chunked_ce": False,
        "seq_len": 2048,
    },
    "CUDA out of memory: tried to allocate ...",
)
print(json.dumps(record))
"""
    ]
    env = {"PYTHONPATH": str(repo), **os.environ}
    result = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"oom-record probe failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    record = json.loads(result.stdout)
    assert record["oom"] is True
    assert record["wall_seconds_median"] is None
    assert record["batch_size"] == 64
    assert record["gradient_accumulation"] == 4


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


def test_bench_step_single_config_records_oom_without_aborting():
    """Single-config mode used to call `_bench_one_config` directly,
    so an OOM in the warmup window aborted the run. The new path
    routes single-config through `_safe_bench` so OOMs become
    structured records — the same contract the search matrix uses.
    Verify by patching `_bench_one_config` to raise an OOM-signaling
    RuntimeError.
    """
    repo = REPO
    cmd = [
        sys.executable, "-c",
        """
import importlib.util, json, sys, types
from pathlib import Path
spec = importlib.util.spec_from_file_location("bench_step", "scripts/bench_step.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
def _raise_oom(*a, **kw):
    raise RuntimeError("CUDA out of memory: probe")
mod._bench_one_config = _raise_oom
import argparse
args = argparse.Namespace(
    config='probe.yaml', warmup=10, measure=90,
)
record = mod._safe_bench(
    {"model": {"type": "moe_everything"}}, args=args,
    batch_size=8, gradient_accumulation=1,
    gradient_checkpointing=False, chunked_ce=False,
    seq_len=2048, device="cuda:0",
)
print(json.dumps(record))
"""
    ]
    env = {"PYTHONPATH": str(repo), **os.environ}
    result = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"single-config OOM path probe failed:\n"
        f"STDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    record = json.loads(result.stdout)
    assert record["oom"] is True
    assert record["batch_size"] == 8


def test_bench_step_search_emit_max_row_schema(tiny_config_yaml, tmp_path):
    """`--search --emit-max-row` collapses the grid into a single
    per-config max-row summary with the AC-24 schema."""
    output_path = tmp_path / "results.json"
    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "1",
            "--measure", "2",
            "--device", "cpu",
            "--search",
            "--search-batch-min", "1",
            "--search-batch-max", "2",
            "--search-grad-accum", "1,2",
            "--search-grad-ckpt", "false",
            "--search-chunked-ce", "false",
            "--emit-max-row",
            "--output", str(output_path),
        ],
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    if isinstance(payload, list):
        # When `--emit-grid` is also passed, the payload contains both;
        # without it, the payload is a list with the max-row only.
        max_row = payload[-1]
    else:
        max_row = payload
    expected_keys = {
        "config", "device", "seq_len", "summary",
        "max_per_rank_batch", "max_global_batch",
        "grad_accum_at_max", "grad_ckpt_at_max", "chunked_ce_at_max",
        "median_step_s_at_max", "tokens_per_sec_at_max",
        "peak_mem_gb_at_max", "n_grid_points", "n_eligible",
    }
    assert set(max_row.keys()) == expected_keys, (
        f"max-row schema drift: expected {expected_keys}, got {set(max_row.keys())}"
    )
    assert max_row["summary"] == "max_row"
    assert max_row["n_grid_points"] == 4  # 2 batch x 2 grad_accum
    assert max_row["max_per_rank_batch"] in (1, 2)
    assert max_row["grad_accum_at_max"] in (1, 2)
    assert max_row["max_global_batch"] == max_row["max_per_rank_batch"] * max_row["grad_accum_at_max"]


def test_bench_step_rejects_chunked_ce_until_runtime_implemented(tiny_config_yaml, tmp_path):
    """`--search-chunked-ce=true` must raise `NotImplementedError`
    rather than silently writing metadata-only rows. The
    chunked-CE / fused-linear-CE runtime path is not yet
    implemented in the model code; bench should surface that gap
    instead of pretending the lever is exercised.
    """
    output_path = tmp_path / "results.json"
    result = _run_bench(
        [
            "--config", str(tiny_config_yaml),
            "--warmup", "1",
            "--measure", "2",
            "--device", "cpu",
            "--search",
            "--search-batch-min", "1",
            "--search-batch-max", "1",
            "--search-grad-accum", "1",
            "--search-grad-ckpt", "false",
            "--search-chunked-ce", "true",
            "--output", str(output_path),
        ],
        cwd=REPO,
    )
    assert result.returncode != 0, (
        f"bench should have rejected chunked_ce=true; got return 0\n"
        f"STDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    assert "chunked_ce" in result.stderr or "chunked_ce" in result.stdout, (
        f"expected error to mention chunked_ce; got\n{result.stderr}"
    )


def test_bench_step_disables_wandb_eval_heatmap_paths():
    """The bench is required by AC-19/DEC-8 to disable W&B,
    checkpoint, eval, and heatmap paths. The
    `_disable_w_and_b_eval_heatmaps` helper is the production
    enforcement point — verify it sets every gate.
    """
    repo = REPO
    cmd = [
        sys.executable, "-c",
        """
import importlib.util, json
spec = importlib.util.spec_from_file_location("bench_step", "scripts/bench_step.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg = {"training": {
    "wandb_project": "real-project",
    "save_every": 100, "log_every": 10,
    "routing_log_every": 50, "heatmap_every": 100,
}, "eval": {"every": 100}}
out = mod._disable_w_and_b_eval_heatmaps(cfg)
print(json.dumps(out))
"""
    ]
    env = {"PYTHONPATH": str(repo), **os.environ}
    result = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["training"]["wandb_project"] is None
    assert out["training"]["save_every"] >= 10**8
    assert out["training"]["log_every"] >= 10**8
    assert out["training"]["heatmap_every"] == 0
    assert out["eval"]["every"] == 0


if __name__ == "__main__":
    print(
        "Run via pytest:\n"
        "  PYTHONPATH=. python -m pytest tests/test_bench_step_structural.py"
    )
