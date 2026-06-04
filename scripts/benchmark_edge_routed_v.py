"""Microbenchmark row-routed V vs edge-routed V attention.

This is a focused proxy for the "router(concat(q_t, k_i)) chooses V"
idea. It does not implement a new model path. It measures the expensive
part that differs from current recompute-kv attention:

* row-routed V: one V expert per query token/head group, so each query
  row reads a whole V expert table.
* edge-routed V: one V expert per (query token, key token) edge, so each
  attention cell can gather from a different expert V table.

The edge router is linear on concat(q, k), implemented as
q @ Wq + k @ Wk to avoid materializing concat(q, k), but it still has
pairwise logits with shape [B, H, T, T, E].
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    experts: int


def _parse_shape(spec: str) -> Shape:
    parts = dict(item.split("=") for item in spec.split(","))
    return Shape(
        batch=int(parts["B"]),
        heads=int(parts["H"]),
        seq_len=int(parts["T"]),
        head_dim=int(parts["D"]),
        experts=int(parts["E"]),
    )


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def _time_ms(fn, *, warmup: int, measure: int) -> tuple[float, float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
        # Keep the tensor live until synchronization has completed.
        if out.numel() == 0:
            raise RuntimeError("empty output")

    peak = torch.cuda.max_memory_allocated()
    return statistics.median(samples), max(samples), peak


def _make_inputs(shape: Shape, dtype: torch.dtype, device: torch.device):
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    q = torch.randn(
        shape.batch,
        shape.heads,
        shape.seq_len,
        shape.head_dim,
        device=device,
        dtype=dtype,
        generator=g,
    )
    k = torch.randn_like(q)
    v_bank = torch.randn(
        shape.experts,
        shape.batch,
        shape.heads,
        shape.seq_len,
        shape.head_dim,
        device=device,
        dtype=dtype,
        generator=g,
    )
    wq = torch.randn(shape.head_dim, shape.experts, device=device, dtype=dtype, generator=g)
    wk = torch.randn(shape.head_dim, shape.experts, device=device, dtype=dtype, generator=g)
    row_route = torch.randint(
        0,
        shape.experts,
        (shape.batch, shape.heads, shape.seq_len),
        device=device,
        generator=g,
    )
    edge_route = torch.randint(
        0,
        shape.experts,
        (shape.batch, shape.heads, shape.seq_len, shape.seq_len),
        device=device,
        generator=g,
    )
    causal = torch.ones(shape.seq_len, shape.seq_len, device=device, dtype=torch.bool).tril()
    return q, k, v_bank, wq, wk, row_route, edge_route, causal


def _attention_probs(q: torch.Tensor, k: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    return torch.softmax(scores.float(), dim=-1).to(q.dtype)


def _edge_route(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor) -> torch.Tensor:
    q_logits = torch.matmul(q, wq)
    k_logits = torch.matmul(k, wk)
    logits = q_logits.unsqueeze(3) + k_logits.unsqueeze(2)
    return logits.argmax(dim=-1)


def _edge_gather_mix(
    probs: torch.Tensor,
    v_bank: torch.Tensor,
    edge_route: torch.Tensor,
) -> torch.Tensor:
    # v_by_key: [B, H, key_T, E, D]
    v_by_key = v_bank.permute(1, 2, 3, 0, 4)
    B, H, T, E, D = v_by_key.shape
    v_expanded = v_by_key.unsqueeze(2).expand(B, H, T, T, E, D)
    gather_idx = edge_route.unsqueeze(-1).unsqueeze(-1).expand(B, H, T, T, 1, D)
    v_edge = torch.gather(v_expanded, dim=4, index=gather_idx).squeeze(4)
    return (probs.unsqueeze(-1) * v_edge).sum(dim=3)


def _normal_sdpa(q: torch.Tensor, k: torch.Tensor, v_bank: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v_bank[0], is_causal=True)


def _row_routed_dense_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v_bank: torch.Tensor,
    row_route: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(q)
    active = row_route.unique().tolist()
    for expert in active:
        attn = F.scaled_dot_product_attention(q, k, v_bank[expert], is_causal=True)
        out = out + attn * (row_route == expert).unsqueeze(-1).to(attn.dtype)
    return out


def _edge_routed_precomputed(
    q: torch.Tensor,
    k: torch.Tensor,
    v_bank: torch.Tensor,
    edge_route: torch.Tensor,
    causal: torch.Tensor,
) -> torch.Tensor:
    probs = _attention_probs(q, k, causal)
    return _edge_gather_mix(probs, v_bank, edge_route)


def _edge_routed_with_router(
    q: torch.Tensor,
    k: torch.Tensor,
    v_bank: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    causal: torch.Tensor,
) -> torch.Tensor:
    probs = _attention_probs(q, k, causal)
    edge_route = _edge_route(q, k, wq, wk)
    return _edge_gather_mix(probs, v_bank, edge_route)


def _route_only(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor) -> torch.Tensor:
    return _edge_route(q, k, wq, wk)


def _format_shape(shape: Shape) -> str:
    return (
        f"B={shape.batch},H={shape.heads},T={shape.seq_len},"
        f"D={shape.head_dim},E={shape.experts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="Shape as B=1,H=8,T=1024,D=128,E=16. May be passed multiple times.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    shapes = [
        _parse_shape(spec)
        for spec in (
            args.shape
            or [
                "B=1,H=8,T=512,D=128,E=16",
                "B=1,H=8,T=1024,D=128,E=16",
                "B=4,H=8,T=512,D=128,E=16",
                "B=1,H=8,T=1024,D=128,E=64",
            ]
        )
    ]

    dtype = _dtype(args.dtype)
    device = torch.device("cuda")
    records = []

    for shape in shapes:
        q, k, v_bank, wq, wk, row_route, edge_route, causal = _make_inputs(shape, dtype, device)
        benchmarks = {
            "normal_sdpa_one_v_table": lambda: _normal_sdpa(q, k, v_bank),
            "row_routed_v_dense_sdpa": lambda: _row_routed_dense_sdpa(q, k, v_bank, row_route),
            "edge_route_only_qk_router": lambda: _route_only(q, k, wq, wk),
            "edge_routed_v_precomputed_routes": lambda: _edge_routed_precomputed(
                q, k, v_bank, edge_route, causal
            ),
            "edge_routed_v_with_qk_router": lambda: _edge_routed_with_router(
                q, k, v_bank, wq, wk, causal
            ),
        }
        print(f"\nshape: {_format_shape(shape)} dtype={args.dtype}", flush=True)
        shape_records = []
        for name, fn in benchmarks.items():
            try:
                median_ms, max_ms, peak_bytes = _time_ms(fn, warmup=args.warmup, measure=args.measure)
                record = {
                    "shape": _format_shape(shape),
                    "benchmark": name,
                    "median_ms": median_ms,
                    "max_ms": max_ms,
                    "peak_gb": peak_bytes / (1024**3),
                }
                shape_records.append(record)
                records.append(record)
                print(
                    f"  {name:34s} {median_ms:9.3f} ms"
                    f"   peak={record['peak_gb']:7.2f} GiB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record = {
                    "shape": _format_shape(shape),
                    "benchmark": name,
                    "oom": True,
                }
                shape_records.append(record)
                records.append(record)
                print(f"  {name:34s} OOM", flush=True)

        base = next((r for r in shape_records if r["benchmark"] == "row_routed_v_dense_sdpa" and not r.get("oom")), None)
        edge = next((r for r in shape_records if r["benchmark"] == "edge_routed_v_with_qk_router" and not r.get("oom")), None)
        if base and edge:
            print(
                f"  edge_with_router / row_dense_sdpa: "
                f"{edge['median_ms'] / base['median_ms']:.2f}x slower",
                flush=True,
            )
        normal = next((r for r in shape_records if r["benchmark"] == "normal_sdpa_one_v_table" and not r.get("oom")), None)
        if normal and edge:
            print(
                f"  edge_with_router / normal_sdpa:    "
                f"{edge['median_ms'] / normal['median_ms']:.2f}x slower",
                flush=True,
            )

    if args.json:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
