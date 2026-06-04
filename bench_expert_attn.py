"""Benchmark dense vs sparse expert-table attention for per_head_recompute_kv."""
import torch
import torch.nn.functional as F
import time



def bench_dense(Q, flat, k_proj, v_proj, k_norm_weight, idx, position_embeddings,
                attention_mask, num_kv_heads, num_kv_groups, head_dim, eps, scaling,
                query_group_mask, slot_experts, active_experts):
    """Current approach: full attention per expert, mask afterward."""
    B, num_heads, T, _ = Q.shape
    cos, sin = position_embeddings
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)

    attn_output = Q.new_zeros(B, num_heads, T, head_dim)
    for expert in active_experts:
        K_e = flat @ k_proj[expert]
        V_e = flat @ v_proj[expert]
        # norm
        K_f = K_e.float()
        var = K_f.pow(2).mean(-1, keepdim=True)
        K_e = (k_norm_weight[expert] * (K_f * torch.rsqrt(var + eps)).to(K_e.dtype))

        K_e = K_e.view(B, T, 1, head_dim).transpose(1, 2)
        V_e = V_e.view(B, T, 1, head_dim).transpose(1, 2)
        K_e = (K_e * cos_u) + (torch.cat((-K_e[..., head_dim//2:], K_e[..., :head_dim//2]), -1) * sin_u)

        attn_e = F.scaled_dot_product_attention(
            Q,
            K_e.expand(-1, num_kv_heads, -1, -1).repeat_interleave(num_kv_groups, dim=1),
            V_e.expand(-1, num_kv_heads, -1, -1).repeat_interleave(num_kv_groups, dim=1),
            attn_mask=attention_mask,
            scale=scaling,
        )
        group_mask = (slot_experts == expert) & query_group_mask
        head_mask = group_mask.unsqueeze(-1).repeat_interleave(num_kv_groups, dim=1)
        attn_output = attn_output + attn_e * head_mask.to(attn_e.dtype)
    return attn_output


def bench_sparse(Q, flat, k_proj, v_proj, k_norm_weight, idx, position_embeddings,
                 attention_mask, num_kv_heads, num_kv_groups, head_dim, eps, scaling,
                 query_group_mask, slot_experts, active_experts):
    """Efficient: only run attention for Q heads routed to each expert."""
    B, num_heads, T, _ = Q.shape
    cos, sin = position_embeddings
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
    q_per_kv = num_kv_groups  # query heads per kv group

    attn_output = Q.new_zeros(B, num_heads, T, head_dim)
    for expert in active_experts:
        K_e = flat @ k_proj[expert]
        V_e = flat @ v_proj[expert]
        K_f = K_e.float()
        var = K_f.pow(2).mean(-1, keepdim=True)
        K_e = (k_norm_weight[expert] * (K_f * torch.rsqrt(var + eps)).to(K_e.dtype))

        K_e = K_e.view(B, T, 1, head_dim).transpose(1, 2)
        V_e = V_e.view(B, T, 1, head_dim).transpose(1, 2)
        K_e = (K_e * cos_u) + (torch.cat((-K_e[..., head_dim//2:], K_e[..., :head_dim//2]), -1) * sin_u)

        # Find which (batch, kv_group) pairs use this expert at ANY token position
        # group_uses_expert: (B, num_kv_heads) — True if any token in this batch uses expert for this group
        group_uses_expert = (slot_experts == expert) & query_group_mask  # (B, num_kv_heads, T)
        any_uses = group_uses_expert.any(dim=2)  # (B, num_kv_heads)

        if not any_uses.any():
            continue

        # Gather relevant Q heads: expand group mask to query heads
        # head_uses: (B, num_heads)
        head_uses = any_uses.repeat_interleave(q_per_kv, dim=1)  # (B, num_heads)

        # For simplicity with batched SDPA, gather per-batch
        for b in range(B):
            active_heads = head_uses[b].nonzero(as_tuple=True)[0]
            if active_heads.numel() == 0:
                continue
            Q_sub = Q[b:b+1, active_heads]  # (1, n_active, T, head_dim)
            K_sub = K_e[b:b+1].expand(-1, active_heads.shape[0], -1, -1)
            V_sub = V_e[b:b+1].expand(-1, active_heads.shape[0], -1, -1)

            attn_sub = F.scaled_dot_product_attention(
                Q_sub, K_sub, V_sub,
                attn_mask=attention_mask[:, :1] if attention_mask is not None else None,
                scale=scaling,
            )

            # Mask by token: only keep results where this group actually selected this expert
            kv_groups_for_heads = active_heads // q_per_kv
            token_mask = group_uses_expert[b, kv_groups_for_heads]  # (n_active, T)
            attn_sub = attn_sub.squeeze(0) * token_mask.unsqueeze(-1).to(attn_sub.dtype)
            attn_output[b, active_heads] = attn_output[b, active_heads] + attn_sub

    return attn_output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for dtype_name, dtype, use_autocast in [("fp32", torch.float32, False), ("bf16", torch.bfloat16, True)]:
        for B, T, num_heads, num_kv_heads, E, head_dim in [
            (32, 256, 16, 8, 32, 128),
            (32, 512, 16, 8, 64, 128),
            (32, 1024, 16, 8, 64, 128),
        ]:
            num_kv_groups = num_heads // num_kv_heads
            hidden = 1024
            eps = 1e-6
            scaling = head_dim ** -0.5

            print(f"\n--- {dtype_name} | B={B} T={T} heads={num_heads}/{num_kv_heads} E={E} ---")

            torch.manual_seed(42)
            flat = torch.randn(B * T, hidden, device=device, dtype=torch.float32)
            k_proj = torch.randn(E, hidden, head_dim, device=device, dtype=torch.float32) * 0.02
            v_proj = torch.randn(E, hidden, head_dim, device=device, dtype=torch.float32) * 0.02
            k_norm_weight = torch.ones(E, head_dim, device=device, dtype=torch.float32)
            Q = torch.randn(B, num_heads, T, head_dim, device=device, dtype=torch.float32)

            # Random routing: each token picks one expert per KV group
            idx = torch.randint(0, E, (B * T, num_kv_heads), device=device)
            slot_experts = idx.view(B, T, num_kv_heads).permute(0, 2, 1)  # (B, num_kv_heads, T)
            active_experts = idx.unique().tolist()
            query_group_mask = torch.ones(B, num_kv_heads, T, device=device, dtype=torch.bool)

            # Position embeddings
            pos = torch.arange(T, device=device)
            theta = 1000000.0
            freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
            angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
            cos = torch.cos(angles).repeat(1, 2).unsqueeze(0).expand(B, -1, -1)
            sin = torch.sin(angles).repeat(1, 2).unsqueeze(0).expand(B, -1, -1)
            position_embeddings = (cos, sin)

            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1).unsqueeze(0).unsqueeze(0)

            args = (Q, flat, k_proj, v_proj, k_norm_weight, idx, position_embeddings,
                    causal_mask, num_kv_heads, num_kv_groups, head_dim, eps, scaling,
                    query_group_mask, slot_experts, active_experts)

            # Warmup
            for fn in [bench_dense, bench_sparse]:
                for _ in range(3):
                    if use_autocast and device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _ = fn(*args)
                    else:
                        _ = fn(*args)
                if device == "cuda":
                    torch.cuda.synchronize()

            # Benchmark
            n_iters = 20
            for name, fn in [("dense", bench_dense), ("sparse", bench_sparse)]:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    if use_autocast and device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = fn(*args)
                    else:
                        out = fn(*args)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) / n_iters * 1000
                print(f"  {name:8s}: {elapsed:.3f} ms  (active experts: {len(active_experts)})")

            # Verify correctness
            if use_autocast and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_dense = bench_dense(*args)
                    out_sparse = bench_sparse(*args)
            else:
                out_dense = bench_dense(*args)
                out_sparse = bench_sparse(*args)
            diff = (out_dense.float() - out_sparse.float()).abs().max().item()
            print(f"  max diff: {diff:.6e}")


if __name__ == "__main__":
    main()
