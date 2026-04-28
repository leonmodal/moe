#!/usr/bin/env python3
"""Benchmark router-exploration warmup on tiny debug configs.

Runs one `standard_moe` and one `moe_everything` debug model for a fixed
number of optimizer steps with (a) warmup disabled and (b) warmup ramping
from a chosen start value to the configured target rate over a portion of
the run. Records walltime and final loss for each condition. Used to pin the
"Early-step capacity warmup" technique against its baseline before
recommending it as an integrated default.

Usage:
    uv run python scripts/benchmark_router_exploration_warmup.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.model_factory import build_model
from src.training.routing import apply_router_exploration_rate, exploration_rate_schedule


def _standard_moe_cfg(target_rate: float) -> dict:
    return {
        "model": {
            "type": "standard_moe",
            "router_type": "deepseek",
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "head_dim": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "intermediate_size": 128,
            "max_position_embeddings": 128,
            "router_aux_loss_coef": 0.001,
            "norm_topk_prob": True,
            "router_exploration_rate": target_rate,
            "topk_scaling_factor": 2.5,
            "num_groups": 2,
            "group_topk": 1,
        },
        "training": {},
    }


def _moe_everything_cfg(target_rate: float) -> dict:
    return {
        "model": {
            "type": "moe_everything",
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "head_dim": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_attn_experts": 4,
            "num_attn_experts_per_tok": 1,
            "attn_expert_mode": "per_head_fully_independent",
            "moe_intermediate_size": 32,
            "intermediate_size": 128,
            "max_position_embeddings": 128,
            "router_exploration_rate": target_rate,
        },
        "training": {},
    }


def _run(cfg_fn, *, steps: int, warmup_start: float, warmup_steps: int,
         batch_size: int, seq_len: int, lr: float, seed: int,
         target_rate: float) -> tuple[float, float]:
    """Return (final_loss, walltime_s) for one configuration."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_fn(target_rate)
    model, _ = build_model(cfg)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, cfg["model"]["vocab_size"], (batch_size, seq_len), device=device)
    labels = input_ids

    is_dense = cfg["model"]["type"] == "dense"
    t0 = time.perf_counter()
    losses = []
    for step in range(steps):
        rate = exploration_rate_schedule(step, target_rate, warmup_start, warmup_steps)
        apply_router_exploration_rate(model, rate)
        out = model(input_ids=input_ids, labels=labels,
                    **({} if is_dense else {"output_router_logits": True}))
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.item()))
    walltime = time.perf_counter() - t0
    return losses[-1], walltime


def _fmt_row(label: str, final_loss: float, walltime: float) -> str:
    return f"  {label:<36}final_loss={final_loss:.4f}  walltime={walltime:.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-start", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--target-rate", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; running on CPU (results are indicative only)")

    print("# router-exploration warmup benchmark")
    print(f"  steps={args.steps}, batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"lr={args.lr}, target_rate={args.target_rate}")
    print()

    for name, cfg_fn in [("standard_moe", _standard_moe_cfg),
                         ("moe_everything", _moe_everything_cfg)]:
        # Baseline: no warmup — rate = target_rate for every step.
        loss_no_warmup, t_no_warmup = _run(
            cfg_fn, steps=args.steps,
            warmup_start=args.target_rate,  # degenerate: start=target means constant
            warmup_steps=0,
            batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr, seed=args.seed,
            target_rate=args.target_rate,
        )
        # Warmup: rate ramps from warmup_start to target_rate over warmup_steps.
        loss_warmup, t_warmup = _run(
            cfg_fn, steps=args.steps,
            warmup_start=args.warmup_start,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr, seed=args.seed,
            target_rate=args.target_rate,
        )
        print(f"{name}")
        print(_fmt_row("no_warmup (rate==target)", loss_no_warmup, t_no_warmup))
        print(_fmt_row(f"warmup (start={args.warmup_start}, steps={args.warmup_steps})",
                       loss_warmup, t_warmup))
        rel = 100.0 * (loss_warmup - loss_no_warmup) / loss_no_warmup
        print(f"  delta: {loss_warmup - loss_no_warmup:+.4f} ({rel:+.2f}% vs baseline)")
        print()


if __name__ == "__main__":
    main()
