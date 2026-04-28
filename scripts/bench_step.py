"""Per-step throughput / memory benchmark.

Loads a yaml config, builds the model + optimizer + scheduler the way
`scripts/train.py` does, and runs a small fixed-step benchmark loop:

  1. Warmup iterations (default 5) — never measured; flush the dataset
     prefetcher, autocast caches, JIT compilation, and any allocator
     warm-up work so the measurement window does not include first-call
     overhead.
  2. Measured iterations (default 20) — full forward + backward +
     optimizer.step on synthetic random batches. Per-iter wall time,
     tokens/sec, peak GPU memory, and average loss are recorded; the
     summary is printed and (optionally) appended to bench/results.json.

The synthetic-batch path (`--synthetic`) is the default so the bench
does not depend on the data pipeline; the dataset path
(`--dataset-config`) is reserved for the AC-22 8xH200 sweep where the
end-to-end Modal training data flow matters.

The `--search` mode wraps the bench in a binary search over per-rank
batch size to find the largest stable batch (no OOM) that still passes
a forward + backward + step. The output records the discovered max
batch and the corresponding tokens/sec.

Usage:

    python scripts/bench_step.py \\
        --config configs/8_layers/moe_everything_per_head_fully_independent.yaml \\
        --warmup 5 --measure 20 --output bench/results.json

    python scripts/bench_step.py \\
        --config configs/8_layers/moe_everything_per_head_fully_independent.yaml \\
        --search --batch-min 1 --batch-max 64

Output schema (one record per run, appended to results.json as a JSON
list):

    {
      "config": "<yaml path>",
      "model_type": "moe_everything",
      "device": "cuda" | "cpu",
      "warmup_iters": 5,
      "measure_iters": 20,
      "batch_size": 4,
      "seq_len": 2048,
      "tokens_per_step": <int>,
      "wall_seconds_mean": <float>,
      "wall_seconds_std": <float>,
      "tokens_per_second": <float>,
      "peak_gpu_memory_bytes": <int | null>,
      "loss_mean": <float>,
      "loss_std": <float>,
      "search_mode": false,
      "search_max_batch": <int | null>
    }
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Make `src.*` imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _bypass_training_pkg_init_then_import():
    """`src.training.__init__` re-exports `src.training.data`, which
    pulls in pandas via `src.data.parquet_dataset`. Bench runs in
    environments (CI / pre-Modal sandboxes) that do not have pandas
    installed, so we import the two specific submodules we need with
    direct file-loading instead of going through the package init.
    """
    import importlib.util
    import sys as _sys
    import types as _types

    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in _sys.modules:
        pkg = _types.ModuleType("src.training")
        pkg.__path__ = [str(repo / "src" / "training")]
        _sys.modules["src.training"] = pkg

    def _load(modname, relpath):
        spec = importlib.util.spec_from_file_location(modname, str(repo / relpath))
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("src.training.config", "src/training/config.py")
    factory_mod = _load("src.training.model_factory", "src/training/model_factory.py")
    return cfg_mod.load_config, cfg_mod.build_training_config, factory_mod.build_model


load_config, build_training_config, build_model = _bypass_training_pkg_init_then_import()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iters")
    parser.add_argument("--measure", type=int, default=20, help="Measured iters")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override training.batch_size for the bench",
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Override sequence length for the bench (default reads "
             "max_position_embeddings or falls back to 2048)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device (cuda/cpu); default chooses cuda if available",
    )
    parser.add_argument(
        "--output", default=None,
        help="Append result record to this JSON file (creates if missing). "
             "Use bench/results.json for the AC-20 results landing zone.",
    )
    parser.add_argument(
        "--search", action="store_true",
        help="Binary-search over per-rank batch size to find the largest "
             "stable batch (no OOM, completes one full step). The reported "
             "tokens_per_second is for that max batch.",
    )
    parser.add_argument("--batch-min", type=int, default=1)
    parser.add_argument("--batch-max", type=int, default=64)
    return parser.parse_args()


def _make_synthetic_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    """Per-iter random-token batch matching the training data shape."""
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long,
    )
    return {"input_ids": input_ids}


def _run_one_step(
    model, optimizer, scheduler, batch: dict[str, torch.Tensor],
    *, gradient_accumulation: int = 1,
) -> tuple[float, float]:
    """Run one full optimizer step. Returns (wall_seconds, loss_value)."""
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
    loss = output.loss / gradient_accumulation
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    return wall, float(output.loss.detach().item())


def _build_optimizer(model, train_cfg) -> tuple[Any, Any]:
    """Minimal optimizer + scheduler suitable for the bench. Real
    training uses a larger surface area (Muon parameter groups, LR
    schedulers, weight decay grouping); the bench just needs something
    that exercises a forward + backward + step end-to-end."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(train_cfg, "learning_rate", 3e-4),
        weight_decay=getattr(train_cfg, "weight_decay", 0.0),
    )
    return optimizer, None


def _resolve_seq_len(cfg: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(cfg.get("model", {}).get("max_position_embeddings", 2048))


def _resolve_batch_size(cfg: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(cfg.get("training", {}).get("batch_size", 4))


def _resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bench_at_batch(
    args: argparse.Namespace, cfg: dict, batch_size: int, seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the warmup + measure loop for a given batch size. Returns
    a result dict with timing + memory stats, or raises (so a search
    sweep can catch OOM at higher batch sizes)."""
    model, _model_cfg = build_model(cfg)
    model = model.to(device)
    train_cfg = build_training_config(cfg)
    optimizer, scheduler = _build_optimizer(model, train_cfg)
    model.train()

    vocab_size = int(cfg["model"]["vocab_size"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.warmup):
        batch = _make_synthetic_batch(batch_size, seq_len, vocab_size, device)
        _run_one_step(model, optimizer, scheduler, batch)

    walls: list[float] = []
    losses: list[float] = []
    for _ in range(args.measure):
        batch = _make_synthetic_batch(batch_size, seq_len, vocab_size, device)
        wall, loss = _run_one_step(model, optimizer, scheduler, batch)
        walls.append(wall)
        losses.append(loss)

    tokens_per_step = batch_size * seq_len
    wall_mean = statistics.mean(walls)
    wall_std = statistics.stdev(walls) if len(walls) > 1 else 0.0
    return {
        "config": args.config,
        "model_type": cfg.get("model", {}).get("type"),
        "device": str(device),
        "warmup_iters": args.warmup,
        "measure_iters": args.measure,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "wall_seconds_mean": wall_mean,
        "wall_seconds_std": wall_std,
        "tokens_per_second": tokens_per_step / wall_mean if wall_mean > 0 else 0.0,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "loss_mean": statistics.mean(losses),
        "loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
        "search_mode": False,
        "search_max_batch": None,
    }


def _bench_search_max_batch(
    args: argparse.Namespace, cfg: dict, seq_len: int, device: torch.device,
) -> dict[str, Any]:
    """Binary-search the largest stable batch size in [batch-min,
    batch-max]. Reports the bench result at the discovered max."""
    lo, hi = args.batch_min, args.batch_max
    best_batch = None
    best_record: dict[str, Any] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            record = _bench_at_batch(args, cfg, mid, seq_len, device)
            best_batch = mid
            best_record = record
            lo = mid + 1
        except (RuntimeError, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hi = mid - 1
    if best_record is None:
        raise RuntimeError(
            f"No batch size in [{args.batch_min}, {args.batch_max}] "
            f"completed a step. Lower --batch-min or fix the model "
            f"build under this config."
        )
    best_record["search_mode"] = True
    best_record["search_max_batch"] = best_batch
    return best_record


def _append_result(output_path: str, record: dict[str, Any]) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = json.loads(p.read_text())
        if not isinstance(existing, list):
            raise ValueError(
                f"{output_path} exists and is not a JSON list; refusing to append."
            )
    else:
        existing = []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seq_len = _resolve_seq_len(cfg, args.seq_len)
    batch_size = _resolve_batch_size(cfg, args.batch_size)
    device = _resolve_device(args.device)

    if args.search:
        record = _bench_search_max_batch(args, cfg, seq_len, device)
    else:
        record = _bench_at_batch(args, cfg, batch_size, seq_len, device)

    print(json.dumps(record, indent=2))
    if args.output:
        _append_result(args.output, record)


if __name__ == "__main__":
    main()
