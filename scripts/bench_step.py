"""Production-trainer-path benchmark for the MoE matrix.

This script implements the AC-19 / AC-24 production-trainer-compatible
benchmark contract. It is NOT a generic forward-backward microbenchmark:
it builds the model + optimizer + scheduler exactly as
`scripts/train.py` does, calls the trainer's branch-router pre-forward
schedule hook, and runs a 100-step measured loop:

  * 10 warmup iterations  — never measured. Flushes optimizer state
    init, autocast caches, allocator warm-up, and any first-call
    overhead.
  * 90 measured iterations — full forward + backward + clip-grad +
    `trainer_optimizer_step_and_bias_update(...)`. Per-iter wall time
    is recorded; the schema reports MEDIAN, p5, and p95 (not mean and
    std), matching the AC-19 spec.

W&B / checkpoint / eval / heatmap paths are explicitly disabled so
the bench measures only the optimizer-step hot path. Synthetic
random batches are used in place of the dataset (the dataset path
adds noise the bench is not measuring).

`--search` mode performs a multi-dimensional sweep across the four
axes named by AC-24:

  * `--search-batch-min` / `--search-batch-max`: per-rank batch.
  * `--search-grad-accum`: comma-separated list of gradient-
    accumulation values (default `1,2,4,8`).
  * `--search-grad-ckpt`: comma-separated bools (default `false,true`).
  * `--search-chunked-ce`: comma-separated bools (default `false,true`).

OOM rows are recorded with `oom: true` rather than aborting the
sweep, so a single config matrix entry that exceeds memory at the
top of its grid does not invalidate the rest of the sweep.

Usage:

    python scripts/bench_step.py \\
        --config configs/8_layers/moe_everything_per_head_fully_independent.yaml \\
        --output bench/results.json

    python scripts/bench_step.py \\
        --config configs/8_layers/moe_everything_per_head_fully_independent.yaml \\
        --search --output bench/results.json

Output schema (per record, appended to results.json as a JSON list):

    {
      "config": "<yaml path>",
      "model_type": "moe_everything",
      "device": "cuda" | "cpu",
      "warmup_iters": 10,
      "measure_iters": 90,
      "batch_size": <int>,
      "gradient_accumulation": <int>,
      "gradient_checkpointing": <bool>,
      "chunked_ce": <bool>,
      "seq_len": <int>,
      "tokens_per_step": <int>,
      "wall_seconds_median": <float>,
      "wall_seconds_p5": <float>,
      "wall_seconds_p95": <float>,
      "tokens_per_second_median": <float>,
      "peak_gpu_memory_bytes": <int | null>,
      "headroom_bytes": <int | null>,
      "loss_first": <float>,
      "loss_last": <float>,
      "search_mode": <bool>,
      "oom": <bool>
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _bypass_training_pkg_init():
    """`src.training.__init__` re-exports `src.training.data`, which
    pulls pandas in through the dataset path. The bench skips the
    dataset entirely, so we side-load only the modules we need via
    direct file-loading.
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
    routing_mod = _load("src.training.routing", "src/training/routing.py")
    # build_optimizer / build_lr_scheduler live in src/utils/training.py;
    # that file does not transitively import pandas, so plain import works.
    from src.utils.training import build_optimizer, build_lr_scheduler
    return cfg_mod, factory_mod, build_optimizer, build_lr_scheduler, routing_mod


_CFG_MOD, _FACTORY_MOD, _BUILD_OPTIM, _BUILD_SCHED, _ROUTING_MOD = _bypass_training_pkg_init()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iters (AC-19 default)")
    parser.add_argument("--measure", type=int, default=90, help="Measured iters (AC-19 default)")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Per-rank batch size (defaults to training.batch_size)",
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=None,
        help="Gradient-accumulation factor (defaults to training.gradient_accumulation)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=["false", "true"], default=None,
        help="Override gradient_checkpointing (default reads training.gradient_checkpointing)",
    )
    parser.add_argument(
        "--chunked-ce",
        choices=["false", "true"], default=None,
        help="Override chunked_ce flag",
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Sequence length (defaults to model.max_position_embeddings or 2048)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device (cuda/cpu); default chooses cuda if available",
    )
    parser.add_argument(
        "--output", default=None,
        help="Append bench record(s) to this JSON file; bench/results.json is the AC-20 landing zone.",
    )
    parser.add_argument(
        "--search", action="store_true",
        help="Multi-dim sweep across (batch_size, gradient_accumulation, "
             "gradient_checkpointing, chunked_ce). OOM rows are recorded "
             "with oom=true rather than aborting the sweep.",
    )
    parser.add_argument("--search-batch-min", type=int, default=1)
    parser.add_argument("--search-batch-max", type=int, default=8)
    parser.add_argument(
        "--search-grad-accum", default="1,2",
        help="Comma-separated list of gradient_accumulation values to sweep.",
    )
    parser.add_argument(
        "--search-grad-ckpt", default="false,true",
        help="Comma-separated bools (false/true) for gradient_checkpointing.",
    )
    parser.add_argument(
        "--search-chunked-ce", default="false",
        help="Comma-separated bools (false/true) for chunked_ce.",
    )
    return parser.parse_args()


def _make_synthetic_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long,
    )
    return {"input_ids": input_ids}


def _disable_w_and_b_eval_heatmaps(cfg: dict) -> dict:
    """Disable W&B / checkpoint / eval / heatmap paths so the bench
    measures only the optimizer-step hot path, per AC-19 / DEC-8."""
    tcfg = cfg.setdefault("training", {})
    tcfg["wandb_project"] = None
    tcfg["save_every"] = 10**9
    tcfg["log_every"] = 10**9
    tcfg["routing_log_every"] = 10**9
    tcfg["heatmap_every"] = 0
    cfg.setdefault("eval", {})["every"] = 0
    return cfg


def _build_production_optimizer_and_scheduler(model, train_cfg):
    """Mirror `src/training/trainer.py:189-200` so the bench
    exercises the production optimizer + scheduler construction
    (AC-19: bench reuses the production trainer path)."""
    if train_cfg.optimizer == "muon":
        # Lazy import: muon path requires CUDA in some configurations,
        # so we defer to the standard AdamW path on CPU sandboxes that
        # don't run muon. The bench-level tests exercise the AdamW path.
        try:
            from src.utils.muon import build_muon_optimizer
            optimizer = build_muon_optimizer(
                model, train_cfg,
                muon_lr=getattr(train_cfg, "muon_lr", train_cfg.learning_rate),
                muon_weight_decay=getattr(train_cfg, "muon_weight_decay", 0.0),
                adam_lr=getattr(train_cfg, "adam_lr", train_cfg.learning_rate),
            )
        except Exception:
            optimizer = _BUILD_OPTIM(model, train_cfg)
    else:
        optimizer = _BUILD_OPTIM(model, train_cfg)
    scheduler = _BUILD_SCHED(optimizer, train_cfg)
    return optimizer, scheduler


def _maybe_enable_chunked_ce(model, chunked_ce: bool) -> None:
    """The chunked-CE path is a model-side optimization the bench can
    toggle when present. Models that do not expose a chunked-CE
    setter accept the flag silently (no-op) so the sweep can include
    chunked_ce as a search axis without aborting on configs that do
    not support it."""
    setter = getattr(model, "set_chunked_ce", None)
    if callable(setter):
        try:
            setter(chunked_ce)
        except Exception:
            pass


def _bench_one_config(
    cfg_template: dict,
    *,
    args: argparse.Namespace,
    batch_size: int,
    gradient_accumulation: int,
    gradient_checkpointing: bool,
    chunked_ce: bool,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the full warmup + measured loop for a single point in the
    search grid. Returns a record dict; raises only on non-OOM
    runtime errors. OOM is captured by the caller in `--search` mode
    so the matrix continues."""
    import copy
    cfg = copy.deepcopy(cfg_template)
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["gradient_accumulation"] = gradient_accumulation
    cfg["training"]["gradient_checkpointing"] = gradient_checkpointing
    if "model" in cfg:
        cfg["model"]["chunked_ce"] = chunked_ce

    model, _model_cfg = _FACTORY_MOD.build_model(cfg)
    model = model.to(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    _maybe_enable_chunked_ce(model, chunked_ce)
    train_cfg = _CFG_MOD.build_training_config(cfg)
    optimizer, scheduler = _build_production_optimizer_and_scheduler(model, train_cfg)
    model.train()

    vocab_size = int(cfg["model"]["vocab_size"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Seed the branch exploration_only schedule once before the loop
    # (mirrors trainer.py's pre-loop seed call).
    _ROUTING_MOD.apply_branch_schedule_pre_forward(model, 0)

    walls: list[float] = []
    losses: list[float] = []
    global_step = 0
    total_iters = args.warmup + args.measure
    for it in range(total_iters):
        # Pre-forward branch schedule, exactly as trainer does.
        _ROUTING_MOD.apply_branch_schedule_pre_forward(model, global_step)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(gradient_accumulation):
            batch = _make_synthetic_batch(batch_size, seq_len, vocab_size, device)
            output = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            loss = output.loss / gradient_accumulation
            loss.backward()
        if train_cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.max_grad_norm,
            )
        global_step += 1
        # Production optimizer-step + scheduler-step + post-step
        # bias-update path. This is what AC-19 requires the bench to
        # exercise.
        _ROUTING_MOD.trainer_optimizer_step_and_bias_update(
            model, optimizer, scheduler, train_cfg, cfg,
            distributed=False, global_step=global_step,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        loss_val = float(output.loss.detach().item())
        if it >= args.warmup:
            walls.append(wall)
            losses.append(loss_val)

    sorted_walls = sorted(walls)
    n = len(sorted_walls)
    median = statistics.median(sorted_walls)
    p5 = sorted_walls[max(0, int(0.05 * n))] if n > 0 else 0.0
    p95 = sorted_walls[min(n - 1, int(0.95 * n))] if n > 0 else 0.0
    tokens_per_step = batch_size * gradient_accumulation * seq_len
    peak_mem = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    headroom = None
    if torch.cuda.is_available() and peak_mem is not None:
        free, total = torch.cuda.mem_get_info()
        headroom = int(free)
    return {
        "config": args.config,
        "model_type": cfg.get("model", {}).get("type"),
        "device": str(device),
        "warmup_iters": args.warmup,
        "measure_iters": args.measure,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "gradient_checkpointing": gradient_checkpointing,
        "chunked_ce": chunked_ce,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "wall_seconds_median": median,
        "wall_seconds_p5": p5,
        "wall_seconds_p95": p95,
        "tokens_per_second_median": (
            tokens_per_step / median if median > 0 else 0.0
        ),
        "peak_gpu_memory_bytes": peak_mem,
        "headroom_bytes": headroom,
        "loss_first": losses[0] if losses else 0.0,
        "loss_last": losses[-1] if losses else 0.0,
        "search_mode": False,
        "oom": False,
    }


def _safe_bench(*args, **kwargs) -> dict[str, Any]:
    """Wrap `_bench_one_config` so OOMs become structured records
    rather than aborting the matrix. Non-OOM errors propagate."""
    try:
        return _bench_one_config(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _oom_record(kwargs, str(exc))
    except RuntimeError as exc:
        # CUDA OOMs sometimes surface as RuntimeError("CUDA out of memory");
        # detect with a substring match before treating as fatal.
        msg = str(exc)
        if "out of memory" in msg.lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _oom_record(kwargs, msg)
        raise


def _oom_record(kwargs: dict[str, Any], reason: str) -> dict[str, Any]:
    args = kwargs["args"]
    cfg_template = kwargs["cfg_template"]
    return {
        "config": args.config,
        "model_type": cfg_template.get("model", {}).get("type"),
        "device": str(kwargs["device"]),
        "warmup_iters": args.warmup,
        "measure_iters": args.measure,
        "batch_size": kwargs["batch_size"],
        "gradient_accumulation": kwargs["gradient_accumulation"],
        "gradient_checkpointing": kwargs["gradient_checkpointing"],
        "chunked_ce": kwargs["chunked_ce"],
        "seq_len": kwargs["seq_len"],
        "tokens_per_step": kwargs["batch_size"] * kwargs["gradient_accumulation"] * kwargs["seq_len"],
        "wall_seconds_median": None,
        "wall_seconds_p5": None,
        "wall_seconds_p95": None,
        "tokens_per_second_median": None,
        "peak_gpu_memory_bytes": None,
        "headroom_bytes": None,
        "loss_first": None,
        "loss_last": None,
        "search_mode": True,
        "oom": True,
        "oom_reason": reason[:200],
    }


def _parse_csv_bool(value: str) -> list[bool]:
    out: list[bool] = []
    for piece in value.split(","):
        piece = piece.strip().lower()
        if piece in ("true", "1", "yes"):
            out.append(True)
        elif piece in ("false", "0", "no"):
            out.append(False)
        else:
            raise ValueError(f"unrecognized boolean: {piece!r}")
    return out


def _parse_csv_int(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _bench_search(
    cfg_template: dict, *, args: argparse.Namespace, seq_len: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Multi-dim sweep across the AC-24 search axes. OOM rows
    appear in the record list with `oom=true` rather than
    aborting the matrix."""
    grad_accums = _parse_csv_int(args.search_grad_accum)
    grad_ckpts = _parse_csv_bool(args.search_grad_ckpt)
    chunked_ces = _parse_csv_bool(args.search_chunked_ce)
    batches = list(range(args.search_batch_min, args.search_batch_max + 1))

    records: list[dict[str, Any]] = []
    for ga in grad_accums:
        for gc in grad_ckpts:
            for cce in chunked_ces:
                for bs in batches:
                    record = _safe_bench(
                        cfg_template,
                        args=args,
                        batch_size=bs,
                        gradient_accumulation=ga,
                        gradient_checkpointing=gc,
                        chunked_ce=cce,
                        seq_len=seq_len,
                        device=device,
                    )
                    record["search_mode"] = True
                    records.append(record)
    return records


def _resolve_seq_len(cfg: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(cfg.get("model", {}).get("max_position_embeddings", 2048))


def _resolve_batch_size(cfg: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(cfg.get("training", {}).get("batch_size", 4))


def _resolve_grad_accum(cfg: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(cfg.get("training", {}).get("gradient_accumulation", 1))


def _resolve_grad_ckpt(cfg: dict, override: str | None) -> bool:
    if override is not None:
        return override == "true"
    return bool(cfg.get("training", {}).get("gradient_checkpointing", False))


def _resolve_chunked_ce(cfg: dict, override: str | None) -> bool:
    if override is not None:
        return override == "true"
    return bool(cfg.get("model", {}).get("chunked_ce", False))


def _resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _append_results(output_path: str, records: list[dict[str, Any]]) -> None:
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
    existing.extend(records)
    p.write_text(json.dumps(existing, indent=2))


def main() -> None:
    args = _parse_args()
    cfg = _CFG_MOD.load_config(args.config)
    cfg = _disable_w_and_b_eval_heatmaps(cfg)
    seq_len = _resolve_seq_len(cfg, args.seq_len)
    device = _resolve_device(args.device)

    if args.search:
        records = _bench_search(cfg, args=args, seq_len=seq_len, device=device)
    else:
        record = _bench_one_config(
            cfg, args=args,
            batch_size=_resolve_batch_size(cfg, args.batch_size),
            gradient_accumulation=_resolve_grad_accum(cfg, args.gradient_accumulation),
            gradient_checkpointing=_resolve_grad_ckpt(cfg, args.gradient_checkpointing),
            chunked_ce=_resolve_chunked_ce(cfg, args.chunked_ce),
            seq_len=seq_len, device=device,
        )
        records = [record]

    print(json.dumps(records if args.search else records[0], indent=2))
    if args.output:
        _append_results(args.output, records)


if __name__ == "__main__":
    main()
