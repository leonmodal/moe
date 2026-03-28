"""
Benchmark sparse vs dense per-head attention dispatch as attention-token fraction changes.

Example:
  uv run python scripts/benchmark_per_head_attention_dispatch.py --mode per_head_precompute_kv
"""
from __future__ import annotations

import argparse
import contextlib
import time

import torch

from src.models.mixture_of_everything import AttentionExpertBank, MoEverythingConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding


def build_config(args, mode: str) -> MoEverythingConfig:
    return MoEverythingConfig(
        vocab_size=256,
        hidden_size=args.hidden_size,
        num_hidden_layers=1,
        head_dim=args.head_dim,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=max(2048, args.seq_len),
        num_attn_experts=args.num_attn_experts,
        num_attn_experts_per_tok=1,
        attn_expert_mode=mode,
        per_head_compute_mode="sparse",
    )


def build_inputs(args, config, device, dtype, attn_fraction):
    B, T = args.batch_size, args.seq_len
    base_dtype = torch.float32
    hidden_states = torch.randn(B, T, config.hidden_size, device=device, dtype=base_dtype)
    rotary = Qwen3MoeRotaryEmbedding(config=config).to(device)
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary(hidden_states, position_ids=position_ids)

    K_old = torch.randn(B, config.num_key_value_heads, T, config.head_dim, device=device, dtype=base_dtype)
    V_old = torch.randn(B, config.num_key_value_heads, T, config.head_dim, device=device, dtype=base_dtype)
    causal_mask = torch.triu(
        torch.full((T, T), float("-inf"), device=device, dtype=base_dtype),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    token_mask = torch.zeros(B, T, 1, device=device, dtype=torch.bool)
    attn_tokens = max(1, min(T, int(round(T * attn_fraction))))
    token_mask[:, :attn_tokens, 0] = True
    return hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask


def run_case(bank, mode: str, hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask):
    if mode == "sparse":
        if bank.mode == "per_head_precompute_kv":
            return bank.project_and_attend_per_head_precompute_kv_sparse(
                hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask, depth_idx=0
            )[0]
        return bank.project_and_attend_per_head_fully_independent_sparse(
            hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask, depth_idx=0
        )[0]

    if bank.mode == "per_head_precompute_kv":
        return bank.project_and_attend_per_head_precompute_kv_dense_mixed(
            hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask, depth_idx=0
        )[0]

    Q, K_fresh, V_fresh = bank.project(hidden_states, position_embeddings, depth_idx=0)
    attn_mask_kv = token_mask.unsqueeze(1)
    K_blend = torch.where(attn_mask_kv, K_fresh, K_old)
    V_blend = torch.where(attn_mask_kv, V_fresh, V_old)
    return bank.attend(Q, K_blend, V_blend, causal_mask, depth_idx=0)


def benchmark_one(bank, mode: str, args, hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask):
    autocast_enabled = hidden_states.is_cuda and args.dtype != "fp32"

    for _ in range(args.warmup):
        bank.zero_grad(set_to_none=True)
        with (
            torch.autocast(
                device_type=hidden_states.device.type,
                dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype],
            )
            if autocast_enabled else contextlib.nullcontext()
        ):
            out = run_case(bank, mode, hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask)
            loss = out.float().square().mean()
        loss.backward()

    if hidden_states.is_cuda:
        torch.cuda.synchronize(hidden_states.device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        bank.zero_grad(set_to_none=True)
        with (
            torch.autocast(
                device_type=hidden_states.device.type,
                dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype],
            )
            if autocast_enabled else contextlib.nullcontext()
        ):
            out = run_case(bank, mode, hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask)
            loss = out.float().square().mean()
        loss.backward()
    if hidden_states.is_cuda:
        torch.cuda.synchronize(hidden_states.device)
    elapsed = time.perf_counter() - t0
    return elapsed / args.iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["per_head_fully_independent", "per_head_precompute_kv"], default="per_head_precompute_kv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-attn-experts", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    config = build_config(args, args.mode)
    bank = AttentionExpertBank(config).to(device=device).train()

    print(
        f"mode={args.mode} batch={args.batch_size} seq={args.seq_len} "
        f"hidden={args.hidden_size} dtype={args.dtype} device={device}"
    )
    print("attn_frac  sparse_sec  dense_sec  faster")
    for fraction in args.fractions:
        hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask = build_inputs(
            args, config, device, dtype, fraction
        )
        sparse_sec = benchmark_one(
            bank, "sparse", args, hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask
        )
        dense_sec = benchmark_one(
            bank, "dense", args, hidden_states, position_embeddings, K_old, V_old, token_mask, causal_mask
        )
        faster = "sparse" if sparse_sec < dense_sec else "dense"
        print(f"{fraction:8.3f}  {sparse_sec:10.4f}  {dense_sec:9.4f}  {faster}")


if __name__ == "__main__":
    main()
