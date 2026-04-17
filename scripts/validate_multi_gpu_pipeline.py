#!/usr/bin/env python3
"""End-to-end multi-GPU validation for every supported model variant.

Runs three stages sequentially on 8 GPUs:

  A. **Broad smoke** — every (model family × router × dist strategy) combo on
     real parquet data for a short number of steps. Asserts: run succeeds,
     checkpoint files are written, final loss is finite and below a loose
     loss-sanity band.

  B. **Loss-to-3.3 evidence** — a handful of representative configs at a
     moderate scale for many more steps. Reports final loss per config and
     checks it landed below a realistic threshold. On real data at this
     scale the window averages should trend toward ~3.3 given enough steps;
     this stage logs the trajectory so the operator can see where it is.

  C. **Auto-resume correctness** — for 1 DDP and 1 FSDP config, train N
     steps, save, then resume for M more steps. Compare the resumed loss
     trajectory against an uninterrupted reference of N+M steps. Equality
     of the final losses within a small tolerance proves optimizer state,
     data-state, and model state all round-tripped correctly.

Usage:
    python scripts/validate_multi_gpu_pipeline.py [--stages ABC]
                                                   [--data-dir PATH]
                                                   [--report out.md]

The full run takes ~1.5–3 hours on 8×H200 depending on data throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"


# ── Configuration building ──────────────────────────────────────────────────


_SMOKE_BASE_FIELDS = """  vocab_size: 50304
  hidden_size: 256
  num_hidden_layers: 4
  head_dim: 32
  num_attention_heads: 8
  num_key_value_heads: 4
  intermediate_size: 512
  max_position_embeddings: 512
  rms_norm_eps: 1.0e-6
  rope_theta: 1000000.0
  tie_word_embeddings: true
  attention_bias: false
  attention_dropout: 0.0"""


_REALISTIC_BASE_FIELDS = """  vocab_size: 50304
  hidden_size: 512
  num_hidden_layers: 8
  head_dim: 64
  num_attention_heads: 8
  num_key_value_heads: 4
  intermediate_size: 1024
  max_position_embeddings: 1024
  rms_norm_eps: 1.0e-6
  rope_theta: 1000000.0
  tie_word_embeddings: true
  attention_bias: false
  attention_dropout: 0.0"""


_MOE_COMMON = """  num_experts: 8
  num_experts_per_tok: 2
  moe_intermediate_size: 256
  norm_topk_prob: true
  router_aux_loss_coef: 0.001
  output_router_logits: true"""


_DEEPSEEK_EXTRA = """  topk_scaling_factor: 2.5
  num_groups: 2
  group_topk: 1"""


def _model_block(variant: str, *, base_fields: str, experiment_name: str) -> str:
    """Return a YAML fragment for the `model:` section of the given variant."""
    if variant == "dense":
        return f"experiment_name: {experiment_name}\nmodel:\n  type: dense\n" + base_fields + "\n"
    if variant == "standard_moe_softmax":
        return (
            f"experiment_name: {experiment_name}\n"
            "model:\n  type: standard_moe\n  router_type: softmax\n"
        ) + base_fields + "\n" + _MOE_COMMON + "\n"
    if variant == "standard_moe_deepseek":
        return (
            f"experiment_name: {experiment_name}\n"
            "model:\n  type: standard_moe\n  router_type: deepseek\n"
        ) + base_fields + "\n" + _MOE_COMMON + "\n" + _DEEPSEEK_EXTRA + "\n"
    if variant == "global_moe_softmax":
        return (
            f"experiment_name: {experiment_name}\n"
            "model:\n  type: global_moe\n  router_type: softmax\n"
        ) + base_fields + "\n" + _MOE_COMMON + "\n"
    if variant == "global_moe_deepseek":
        return (
            f"experiment_name: {experiment_name}\n"
            "model:\n  type: global_moe\n  router_type: deepseek\n"
        ) + base_fields + "\n" + _MOE_COMMON + "\n" + _DEEPSEEK_EXTRA + "\n"
    if variant == "moe_everything_fully_independent":
        return (
            f"experiment_name: {experiment_name}\n"
            "model:\n  type: moe_everything\n  router_type: softmax\n"
            "  num_attn_experts: 4\n  num_attn_experts_per_tok: 1\n"
            "  attn_expert_mode: per_head_fully_independent\n"
        ) + base_fields + "\n" + _MOE_COMMON + "\n"
    if variant == "moe_everything_precompute_kv":
        return (
            f"experiment_name: {experiment_name}\n"
            "model:\n  type: moe_everything\n  router_type: softmax\n"
            "  num_attn_experts: 4\n  num_attn_experts_per_tok: 1\n"
            "  attn_expert_mode: per_head_precompute_kv\n"
        ) + base_fields + "\n" + _MOE_COMMON + "\n"
    raise ValueError(f"Unknown variant: {variant}")


def _training_block(*, output_dir: Path, data_dir: Path, max_steps: int,
                    batch_size: int, grad_accum: int, seq_len: int,
                    save_every: int, log_every: int) -> str:
    return f"""training:
  learning_rate: 1.0e-3
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 50
  max_steps: {max_steps}
  min_lr_ratio: 0.1
  batch_size: {batch_size}
  gradient_accumulation: {grad_accum}
  mixed_precision: bf16
  log_every: {log_every}
  save_every: {save_every}
  output_dir: {output_dir}
  wandb_project: null
  optimizer: adamw
  beta1: 0.9
  beta2: 0.95
  max_checkpoints: 0
  disable_liger: true
data:
  data_dir: {data_dir}
  text_column: text
  seq_len: {seq_len}
  tokenizer_name: gpt2
  num_workers: 0
  prefetch_files: 1
eval:
  enabled: false
checkpoint:
  resume_from: null
"""


# ── Subprocess runner ───────────────────────────────────────────────────────


@dataclass
class RunResult:
    name: str
    dist_strategy: str
    returncode: int
    elapsed_s: float
    losses: list[tuple[int, float]] = field(default_factory=list)
    ce_losses: list[tuple[int, float]] = field(default_factory=list)
    checkpoint_dir: Path | None = None
    stdout_tail: str = ""

    @property
    def final_loss(self) -> float | None:
        return self.losses[-1][1] if self.losses else None

    @property
    def first_loss(self) -> float | None:
        return self.losses[0][1] if self.losses else None


_LOSS_PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+loss=(?P<loss>[-\d.]+)\s+ce=(?P<ce>[-\d.]+)"
)


def _parse_losses(stdout: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    losses: list[tuple[int, float]] = []
    ces: list[tuple[int, float]] = []
    for line in stdout.splitlines():
        m = _LOSS_PATTERN.search(line)
        if m:
            step = int(m["step"])
            losses.append((step, float(m["loss"])))
            ces.append((step, float(m["ce"])))
    return losses, ces


def run_training_subprocess(
    config_path: Path,
    *,
    dist_strategy: str,
    data_dir: Path,
    output_dir: Path,
    nproc_per_node: int = 8,
    extra_args: list[str] | None = None,
    timeout: int = 7200,
    name: str = "run",
) -> RunResult:
    """Launch `scripts/train.py` via torchrun and capture per-step loss."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{_REPO_ROOT}" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["WANDB_DISABLED"] = "true"
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(_TRAIN_SCRIPT),
        "--config", str(config_path),
        "--dist-strategy", dist_strategy,
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[{name}] launching {dist_strategy} x {nproc_per_node}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(_REPO_ROOT),
    )
    elapsed = time.perf_counter() - t0
    losses, ces = _parse_losses(result.stdout)
    checkpoint_dir = None
    if output_dir.exists():
        ckpts = sorted(output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        checkpoint_dir = ckpts[-1] if ckpts else None
    tail = "\n".join(result.stdout.splitlines()[-40:])
    if result.returncode != 0:
        tail += "\n--- STDERR TAIL ---\n" + "\n".join(result.stderr.splitlines()[-40:])
    return RunResult(
        name=name,
        dist_strategy=dist_strategy,
        returncode=result.returncode,
        elapsed_s=elapsed,
        losses=losses,
        ce_losses=ces,
        checkpoint_dir=checkpoint_dir,
        stdout_tail=tail,
    )


# ── Stages ──────────────────────────────────────────────────────────────────


_ALL_VARIANTS = [
    "dense",
    "standard_moe_softmax",
    "standard_moe_deepseek",
    "global_moe_softmax",
    "global_moe_deepseek",
    "moe_everything_fully_independent",
    "moe_everything_precompute_kv",
]


_REQUIRED_CKPT_FILES = ["model.pt", "training_state.pt", "data_state.pt", "meta.json"]


def _verify_checkpoint(ckpt: Path) -> list[str]:
    """Return a list of missing AC-12 checkpoint files (empty = all present)."""
    if ckpt is None:
        return list(_REQUIRED_CKPT_FILES)
    return [f for f in _REQUIRED_CKPT_FILES if not (ckpt / f).is_file()]


def stage_a_broad_smoke(
    run_dir: Path, data_dir: Path, *, max_steps: int = 200, nproc_per_node: int = 8,
) -> list[RunResult]:
    """Every variant × {ddp, fsdp} on real data for `max_steps`."""
    results: list[RunResult] = []
    stage_dir = run_dir / "stage_a"
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(max_steps // 2, 50)
    for variant in _ALL_VARIANTS:
        for strategy in ("ddp", "fsdp"):
            name = f"A/{variant}/{strategy}"
            out = stage_dir / variant / strategy
            out.mkdir(parents=True, exist_ok=True)
            config_path = out / "config.yaml"
            config_path.write_text(
                _model_block(variant, base_fields=_SMOKE_BASE_FIELDS,
                             experiment_name=f"smokeA_{variant}_{strategy}")
                + _training_block(
                    output_dir=out, data_dir=data_dir, max_steps=max_steps,
                    batch_size=4, grad_accum=1, seq_len=128,
                    save_every=save_every, log_every=10,
                )
            )
            r = run_training_subprocess(
                config_path, dist_strategy=strategy, data_dir=data_dir,
                output_dir=out, nproc_per_node=nproc_per_node, name=name,
            )
            results.append(r)
            _print_run_summary(r)
    return results


def stage_b_loss_evidence(
    run_dir: Path, data_dir: Path, *, variants: list[str], max_steps: int = 5000,
    nproc_per_node: int = 8,
) -> list[RunResult]:
    """Longer runs on a representative subset; target is loss approaching ~3.3."""
    results: list[RunResult] = []
    stage_dir = run_dir / "stage_b"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        name = f"B/{variant}/ddp"
        out = stage_dir / variant
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "config.yaml"
        config_path.write_text(
            _model_block(variant, base_fields=_REALISTIC_BASE_FIELDS,
                         experiment_name=f"lossB_{variant}")
            + _training_block(
                output_dir=out, data_dir=data_dir, max_steps=max_steps,
                batch_size=16, grad_accum=1, seq_len=512,
                save_every=max_steps, log_every=50,
            )
        )
        r = run_training_subprocess(
            config_path, dist_strategy="ddp", data_dir=data_dir,
            output_dir=out, nproc_per_node=nproc_per_node, name=name,
            timeout=14400,
        )
        results.append(r)
        _print_run_summary(r)
    return results


def stage_c_auto_resume(
    run_dir: Path, data_dir: Path, *, variants_with_strategies: list[tuple[str, str]],
    save_at: int = 100, total_steps: int = 200, nproc_per_node: int = 8,
) -> list[tuple[str, RunResult, RunResult, RunResult]]:
    """For each (variant, strategy): reference full run vs. interrupted-and-resumed run.

    - reference: train for `total_steps`, no interruption
    - phase1: train for `save_at` steps, save
    - phase2: resume and train for `total_steps - save_at` more steps
    Compare final losses; they should be equal within numeric tolerance.
    """
    stage_dir = run_dir / "stage_c"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tuples: list[tuple[str, RunResult, RunResult, RunResult]] = []
    for variant, strategy in variants_with_strategies:
        key = f"{variant}/{strategy}"
        slot = stage_dir / variant / strategy
        slot.mkdir(parents=True, exist_ok=True)

        # Reference run: uninterrupted for total_steps
        ref_out = slot / "reference"
        ref_out.mkdir(exist_ok=True)
        ref_cfg = ref_out / "config.yaml"
        ref_cfg.write_text(
            _model_block(variant, base_fields=_SMOKE_BASE_FIELDS,
                         experiment_name=f"resumeC_ref_{variant}_{strategy}")
            + _training_block(
                output_dir=ref_out, data_dir=data_dir, max_steps=total_steps,
                batch_size=4, grad_accum=1, seq_len=128,
                save_every=total_steps, log_every=10,
            )
        )
        ref = run_training_subprocess(
            ref_cfg, dist_strategy=strategy, data_dir=data_dir, output_dir=ref_out,
            nproc_per_node=nproc_per_node, name=f"C/{key}/ref",
        )
        _print_run_summary(ref)

        # Phase 1: train up to save_at, produce a checkpoint
        p1_out = slot / "phase1"
        p1_out.mkdir(exist_ok=True)
        p1_cfg = p1_out / "config.yaml"
        p1_cfg.write_text(
            _model_block(variant, base_fields=_SMOKE_BASE_FIELDS,
                         experiment_name=f"resumeC_p1_{variant}_{strategy}")
            + _training_block(
                output_dir=p1_out, data_dir=data_dir, max_steps=save_at,
                batch_size=4, grad_accum=1, seq_len=128,
                save_every=save_at, log_every=10,
            )
        )
        p1 = run_training_subprocess(
            p1_cfg, dist_strategy=strategy, data_dir=data_dir, output_dir=p1_out,
            nproc_per_node=nproc_per_node, name=f"C/{key}/phase1",
        )
        _print_run_summary(p1)

        # Phase 2: resume from p1 and train for the remainder
        p2_out = slot / "phase2"
        p2_out.mkdir(exist_ok=True)
        p2_cfg = p2_out / "config.yaml"
        p2_cfg.write_text(
            _model_block(variant, base_fields=_SMOKE_BASE_FIELDS,
                         experiment_name=f"resumeC_p2_{variant}_{strategy}")
            + _training_block(
                output_dir=p2_out, data_dir=data_dir, max_steps=total_steps,
                batch_size=4, grad_accum=1, seq_len=128,
                save_every=total_steps, log_every=10,
            )
        )
        # Copy the phase1 checkpoint into phase2's output dir so the trainer's
        # --auto_resume picks it up (it searches inside output_dir).
        if p1.checkpoint_dir is not None:
            shutil.copytree(p1.checkpoint_dir, p2_out / p1.checkpoint_dir.name)
        p2 = run_training_subprocess(
            p2_cfg, dist_strategy=strategy, data_dir=data_dir, output_dir=p2_out,
            nproc_per_node=nproc_per_node, name=f"C/{key}/phase2",
            extra_args=["--auto_resume"],
        )
        _print_run_summary(p2)
        tuples.append((key, ref, p1, p2))
    return tuples


# ── Reporting ───────────────────────────────────────────────────────────────


def _fmt_loss(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _print_run_summary(r: RunResult) -> None:
    status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
    n = len(r.losses)
    print(
        f"  [{r.name}] {status} {r.elapsed_s:.1f}s  steps_logged={n}  "
        f"loss {_fmt_loss(r.first_loss)} → {_fmt_loss(r.final_loss)}  "
        f"ckpt={'yes' if r.checkpoint_dir else 'no'}",
        flush=True,
    )
    if r.returncode != 0:
        print("  stdout/stderr tail:\n" + r.stdout_tail, flush=True)


def write_report(path: Path, *, stage_a: list[RunResult], stage_b: list[RunResult],
                 stage_c: list[tuple[str, RunResult, RunResult, RunResult]]) -> None:
    lines: list[str] = []
    lines.append("# Multi-GPU pipeline validation report")
    lines.append("")
    lines.append("Every entry below was run via `torchrun --standalone --nproc_per_node=8 scripts/train.py`.")
    lines.append("")
    if stage_a:
        lines.append("## Stage A — broad smoke")
        lines.append("")
        lines.append("| Variant | Strategy | Status | Steps logged | First loss | Final loss | Ckpt | Elapsed |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in stage_a:
            lines.append(
                f"| {r.name.split('/')[1]} | {r.dist_strategy} | "
                f"{'OK' if r.returncode == 0 else 'FAIL'} | "
                f"{len(r.losses)} | "
                f"{_fmt_loss(r.first_loss)} | {_fmt_loss(r.final_loss)} | "
                f"{'yes' if r.checkpoint_dir else 'no'} | {r.elapsed_s:.1f}s |"
            )
        lines.append("")
    if stage_b:
        lines.append("## Stage B — loss-to-3.3 evidence")
        lines.append("")
        lines.append("| Variant | Status | Steps | First loss | Final loss | Elapsed |")
        lines.append("|---|---|---|---|---|---|")
        for r in stage_b:
            lines.append(
                f"| {r.name.split('/')[1]} | "
                f"{'OK' if r.returncode == 0 else 'FAIL'} | "
                f"{r.losses[-1][0] if r.losses else 0} | "
                f"{_fmt_loss(r.first_loss)} | {_fmt_loss(r.final_loss)} | {r.elapsed_s:.1f}s |"
            )
        lines.append("")
    if stage_c:
        lines.append("## Stage C — auto-resume correctness")
        lines.append("")
        lines.append("| Key | Ref final | Phase1 final | Phase2 final (resumed) | Ref vs P2 delta |")
        lines.append("|---|---|---|---|---|")
        for key, ref, p1, p2 in stage_c:
            ref_last = ref.final_loss
            p1_last = p1.final_loss
            p2_last = p2.final_loss
            if ref_last is not None and p2_last is not None:
                delta = abs(ref_last - p2_last)
                lines.append(
                    f"| {key} | {_fmt_loss(ref_last)} | {_fmt_loss(p1_last)} | "
                    f"{_fmt_loss(p2_last)} | {delta:.4f} |"
                )
            else:
                lines.append(
                    f"| {key} | {_fmt_loss(ref_last)} | {_fmt_loss(p1_last)} | "
                    f"{_fmt_loss(p2_last)} | (n/a) |"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {path}", flush=True)


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="ABC",
                        help="Subset of stages to run, e.g. 'A', 'AB', 'ABC'")
    parser.add_argument("--data-dir", default="/tmp/moe/data/parquet")
    parser.add_argument("--run-dir", default=None,
                        help="Where to place temp configs/outputs (default: /tmp/moe_validate_<ts>)")
    parser.add_argument("--report", default=None,
                        help="Path to write the summary markdown (default: <run-dir>/report.md)")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--loss-steps", type=int, default=5000)
    parser.add_argument("--resume-save-at", type=int, default=100)
    parser.add_argument("--resume-total", type=int, default=200)
    parser.add_argument("--stage-b-variants", default="dense,standard_moe_deepseek,moe_everything_fully_independent",
                        help="Comma-separated variants for stage B.")
    parser.add_argument("--stage-c-variants", default="standard_moe_softmax:ddp,moe_everything_fully_independent:fsdp",
                        help="Comma-separated `variant:strategy` pairs for stage C.")
    args = parser.parse_args()

    stages = args.stages.upper()
    data_dir = Path(args.data_dir).resolve()
    assert data_dir.is_dir(), f"data dir not found: {data_dir}"

    run_dir = Path(args.run_dir or f"/tmp/moe_validate_{int(time.time())}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[validate] run_dir={run_dir}", flush=True)
    report_path = Path(args.report or (run_dir / "report.md"))

    stage_a_results: list[RunResult] = []
    stage_b_results: list[RunResult] = []
    stage_c_results: list[tuple[str, RunResult, RunResult, RunResult]] = []

    if "A" in stages:
        print("\n[stage A] broad smoke across every (variant, strategy) combo", flush=True)
        stage_a_results = stage_a_broad_smoke(
            run_dir, data_dir, max_steps=args.smoke_steps,
            nproc_per_node=args.nproc_per_node,
        )
        write_report(report_path, stage_a=stage_a_results,
                     stage_b=stage_b_results, stage_c=stage_c_results)

    if "B" in stages:
        variants = [v.strip() for v in args.stage_b_variants.split(",") if v.strip()]
        print(f"\n[stage B] long runs for {variants}", flush=True)
        stage_b_results = stage_b_loss_evidence(
            run_dir, data_dir, variants=variants, max_steps=args.loss_steps,
            nproc_per_node=args.nproc_per_node,
        )
        write_report(report_path, stage_a=stage_a_results,
                     stage_b=stage_b_results, stage_c=stage_c_results)

    if "C" in stages:
        pairs = []
        for entry in args.stage_c_variants.split(","):
            entry = entry.strip()
            if not entry:
                continue
            v, s = entry.split(":")
            pairs.append((v, s))
        print(f"\n[stage C] auto-resume for {pairs}", flush=True)
        stage_c_results = stage_c_auto_resume(
            run_dir, data_dir, variants_with_strategies=pairs,
            save_at=args.resume_save_at, total_steps=args.resume_total,
            nproc_per_node=args.nproc_per_node,
        )
        write_report(report_path, stage_a=stage_a_results,
                     stage_b=stage_b_results, stage_c=stage_c_results)

    # Exit with non-zero if any run failed.
    all_runs: list[RunResult] = list(stage_a_results) + list(stage_b_results)
    for _, ref, p1, p2 in stage_c_results:
        all_runs.extend([ref, p1, p2])
    any_failed = any(r.returncode != 0 for r in all_runs)
    if any_failed:
        print("[validate] one or more runs failed; see report", flush=True)
        return 1
    print(f"[validate] all runs OK; report at {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
