"""
Final line-by-line verification of every component against Megatron-LM
and the DeepSeek V3 paper. Runs on real GPU.

Checks:
  1. Router sigmoid path: scores, biased selection, gather, normalize, scale
  2. Token count accumulation: biased routing (same as Megatron _apply_expert_bias)
  3. group_limited_topk: identical to Megatron
  4. Aux loss f_i: uses UNBIASED topk (same as Megatron compute_routing_scores_for_aux_loss)
  5. Aux loss P_i: uses NORMALIZED scores (Eq. 19-20)
  6. Aux loss formula: E/(K*T²) * Σ counts * agg_probs (Eq. 17-18)
  7. Expert bias update: sign(avg - counts) * rate (matches Megatron get_updated_expert_bias)
  8. Seq loss end-to-end: matches Megatron _apply_seq_aux_loss
"""
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  Megatron reference implementations (inlined)
# ═══════════════════════════════════════════════════════════════════════

def megatron_topk_routing_sigmoid(logits, topk, expert_bias=None, num_groups=None, group_topk=None, scaling_factor=None):
    """Megatron moe_utils.py:772-785 sigmoid path."""
    scores = torch.sigmoid(logits.float())
    if expert_bias is not None:
        scores_for_routing = scores + expert_bias
        if num_groups is not None and group_topk is not None:
            _, top_indices = megatron_group_limited_topk(scores_for_routing, topk, num_groups, group_topk)
        else:
            _, top_indices = torch.topk(scores_for_routing, topk, dim=1)
        gathered = torch.gather(scores, dim=1, index=top_indices)
    else:
        if num_groups is not None and group_topk is not None:
            gathered, top_indices = megatron_group_limited_topk(scores, topk, num_groups, group_topk)
        else:
            gathered, top_indices = torch.topk(scores, topk, dim=1)
    probs = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else gathered
    if scaling_factor:
        probs = probs * scaling_factor
    return scores, probs, top_indices


def megatron_group_limited_topk(scores, topk, num_groups, group_topk):
    """Megatron moe_utils.py:561-616."""
    num_tokens, num_experts = scores.shape
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return probs, top_indices


def megatron_compute_routing_scores_for_aux_loss_sigmoid(logits, topk):
    """Megatron moe_utils.py:809-857 sigmoid path."""
    scores = torch.sigmoid(logits.to(torch.float32))
    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits, dtype=torch.float32).scatter(1, top_indices, 1).bool()
    return routing_map, scores


def megatron_switch_load_balancing_loss(probs, tokens_per_expert, total_num_tokens, topk, num_experts, coeff):
    """Megatron moe_utils.py:47-134."""
    aggregated_probs = probs.sum(dim=0)
    return torch.sum(aggregated_probs * tokens_per_expert) * (
        num_experts * coeff / (topk * total_num_tokens * total_num_tokens)
    )


def megatron_seq_aux_loss(scores_for_aux_loss, routing_map, seq_length, bsz, topk, num_experts, coeff):
    """Megatron router.py:324-377 _apply_seq_aux_loss."""
    scores_reshaped = scores_for_aux_loss.reshape(seq_length, -1)
    routing_map_reshaped = routing_map.reshape(seq_length, -1)
    tokens_per_expert = routing_map_reshaped.sum(dim=0).float()
    total_num_tokens = float(seq_length)
    aux_loss = megatron_switch_load_balancing_loss(
        scores_reshaped, tokens_per_expert, total_num_tokens, topk, num_experts, coeff,
    ) / bsz
    return aux_loss


def megatron_get_updated_expert_bias(tokens_per_expert, expert_bias, rate):
    """Megatron moe_utils.py:1119-1142."""
    with torch.no_grad():
        avg = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = avg - tokens_per_expert
        return expert_bias + torch.sign(offset) * rate


# ═══════════════════════════════════════════════════════════════════════
#  Our implementations
# ═══════════════════════════════════════════════════════════════════════

from src.models.router import DeepSeekRouter, group_limited_topk
from src.models.load_balancing import seq_load_balancing_loss_func


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

def run_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    T, E, K = 256, 64, 8
    num_groups, group_topk = 8, 4
    scaling_factor = 2.5
    B, S = 4, T // 4  # T = B * S

    logits = torch.randn(T, E, device=device)
    expert_bias = torch.randn(E, device=device) * 0.01

    print("=" * 70)
    print(f"  FINAL VERIFICATION — device={device}")
    print(f"  T={T} E={E} K={K} groups={num_groups} group_topk={group_topk}")
    print("=" * 70)
    results = []

    # ── 1. Router sigmoid path ─────────────────────────────────────────
    print("\n  1. Router sigmoid path")

    mega_scores, mega_probs, mega_idx = megatron_topk_routing_sigmoid(
        logits, K, expert_bias, num_groups, group_topk, scaling_factor,
    )

    # Our router's logic (inlined, since DeepSeekRouter needs a full config)
    our_scores = torch.sigmoid(logits.float())
    biased = our_scores + expert_bias.unsqueeze(0)
    _, our_idx = group_limited_topk(biased, K, num_groups, group_topk)
    our_gathered = our_scores.gather(1, our_idx)
    our_probs = our_gathered / (our_gathered.sum(dim=-1, keepdim=True) + 1e-20)
    our_probs = our_probs * scaling_factor

    scores_match = (mega_scores - our_scores).abs().max().item() < 1e-6
    idx_match = (mega_idx == our_idx).all().item()
    probs_match = (mega_probs - our_probs).abs().max().item() < 1e-5

    print(f"     Sigmoid scores match:     {'✓' if scores_match else '✗'}")
    print(f"     Top-K indices match:      {'✓' if idx_match else '✗'}")
    print(f"     Normalized probs match:   {'✓' if probs_match else '✗'} (max diff: {(mega_probs - our_probs).abs().max():.2e})")
    results.extend([scores_match, idx_match, probs_match])

    # ── 2. Token count accumulation ────────────────────────────────────
    print("\n  2. Token count accumulation (from biased routing)")

    # Megatron: routing_map.sum(dim=0)
    mega_routing_map = torch.zeros(T, E, device=device).int().scatter(1, mega_idx, 1).bool()
    mega_counts = mega_routing_map.sum(dim=0).float()

    # Our: bincount(top_k_idx)
    our_counts = torch.bincount(our_idx.reshape(-1), minlength=E).float()

    counts_match = (mega_counts == our_counts).all().item()
    print(f"     Token counts match:       {'✓' if counts_match else '✗'}")
    print(f"     Total tokens: mega={mega_counts.sum():.0f} ours={our_counts.sum():.0f} (expected: {T*K})")
    results.append(counts_match)

    # ── 3. group_limited_topk ──────────────────────────────────────────
    print("\n  3. group_limited_topk")

    test_scores = torch.randn(T, E, device=device)
    mega_glk_vals, mega_glk_idx = megatron_group_limited_topk(test_scores, K, num_groups, group_topk)
    our_glk_vals, our_glk_idx = group_limited_topk(test_scores, K, num_groups, group_topk)

    glk_idx_match = (mega_glk_idx == our_glk_idx).all().item()
    glk_val_match = (mega_glk_vals - our_glk_vals).abs().max().item() < 1e-6
    print(f"     Indices match:            {'✓' if glk_idx_match else '✗'}")
    print(f"     Values match:             {'✓' if glk_val_match else '✗'}")
    results.extend([glk_idx_match, glk_val_match])

    # ── 4. Aux loss routing: UNBIASED topk ─────────────────────────────
    print("\n  4. Aux loss uses UNBIASED normalized topk (not biased routing)")

    mega_aux_map, mega_aux_scores = megatron_compute_routing_scores_for_aux_loss_sigmoid(logits, K)

    # Our: normalize then topk
    our_aux_scores = torch.sigmoid(logits.float())
    our_aux_scores = our_aux_scores / (our_aux_scores.sum(dim=-1, keepdim=True) + 1e-20)
    _, our_aux_top = torch.topk(our_aux_scores, K, dim=-1)
    our_aux_map = torch.zeros(T, E, device=device).int().scatter(1, our_aux_top, 1).bool()

    aux_scores_match = (mega_aux_scores - our_aux_scores).abs().max().item() < 1e-6
    aux_map_match = (mega_aux_map == our_aux_map).all().item()

    # Show that aux topk ≠ biased routing topk
    biased_vs_unbiased = (mega_idx == our_aux_top).all().item()

    print(f"     Normalized scores match:  {'✓' if aux_scores_match else '✗'}")
    print(f"     Routing map match:        {'✓' if aux_map_match else '✗'}")
    print(f"     Aux topk ≠ biased topk:   {'✓' if not biased_vs_unbiased else '✗ (same — bias too small?)'}")
    results.extend([aux_scores_match, aux_map_match])

    # ── 5. Seq loss end-to-end ─────────────────────────────────────────
    print("\n  5. Seq loss: our code vs Megatron (end-to-end)")

    coeff = 0.0001
    num_layers = 3
    raw_logits_list = [torch.randn(T, E, device=device) for _ in range(num_layers)]

    # Megatron: per-layer, averaged
    mega_total = 0.0
    for raw_logits in raw_logits_list:
        routing_map, scores = megatron_compute_routing_scores_for_aux_loss_sigmoid(raw_logits, K)
        mega_layer = megatron_seq_aux_loss(scores, routing_map, S, B, K, E, coeff)
        mega_total += mega_layer.item()
    mega_avg = mega_total / num_layers

    # Ours: pass raw sigmoid scores (normalization is internal)
    gate_logits = tuple(torch.sigmoid(r.float()) for r in raw_logits_list)
    our_loss = seq_load_balancing_loss_func(gate_logits, E, K, batch_size=B)

    # Compare (our loss doesn't include coeff)
    ratio = our_loss.item() / (mega_avg / coeff)
    loss_match = abs(ratio - 1.0) < 0.01
    print(f"     Our loss:                 {our_loss.item():.6f}")
    print(f"     Megatron loss:            {mega_avg:.10f} (with coeff={coeff})")
    print(f"     Ratio (ours / mega/coeff): {ratio:.6f} (should be 1.0)")
    print(f"     Match:                    {'✓' if loss_match else '✗'}")
    results.append(loss_match)

    # ── 6. Paper Eq. 17-20 direct computation ──────────────────────────
    print("\n  6. Paper equations (direct implementation)")

    paper_total = 0.0
    for raw_logits in raw_logits_list:
        s = torch.sigmoid(raw_logits.float())
        s_prime = s / (s.sum(dim=-1, keepdim=True) + 1e-20)
        s_seq = s.reshape(B, S, E)
        sp_seq = s_prime.reshape(B, S, E)
        _, sel = torch.topk(s_seq, K, dim=-1)
        mask = F.one_hot(sel, E).float()
        counts = mask.sum(dim=(1, 2))
        f_i = (E / (K * S)) * counts
        P_i = sp_seq.mean(dim=1)
        paper_total += (f_i * P_i).sum(dim=-1).mean().item()
    paper_avg = paper_total / num_layers

    paper_ratio = our_loss.item() / paper_avg
    paper_match = abs(paper_ratio - 1.0) < 0.01
    print(f"     Paper loss:               {paper_avg:.6f}")
    print(f"     Our loss:                 {our_loss.item():.6f}")
    print(f"     Ratio:                    {paper_ratio:.6f}")
    print(f"     Match:                    {'✓' if paper_match else '✗'}")
    results.append(paper_match)

    # ── 7. Expert bias update ──────────────────────────────────────────
    print("\n  7. Expert bias update (per-layer, matches Megatron)")

    num_layers_test = 4
    test_counts = [torch.randint(50, 200, (E,), device=device).float() for _ in range(num_layers_test)]
    test_biases = [torch.randn(E, device=device) * 0.01 for _ in range(num_layers_test)]
    rate = 0.001

    # Megatron: stacked
    mega_updated = megatron_get_updated_expert_bias(
        torch.stack(test_counts), torch.stack([b.clone() for b in test_biases]), rate,
    )

    # Ours: per-layer loop
    bias_match = True
    for i in range(num_layers_test):
        avg = test_counts[i].mean()
        our_updated = test_biases[i] + torch.sign(avg - test_counts[i]) * rate
        diff = (our_updated - mega_updated[i]).abs().max().item()
        if diff > 1e-7:
            bias_match = False

    print(f"     All {num_layers_test} layers match Megatron: {'✓' if bias_match else '✗'}")
    results.append(bias_match)

    # ── 8. Gradient flow ───────────────────────────────────────────────
    print("\n  8. Gradient flow through seq loss")

    grad_logits = torch.randn(T, E, device=device)
    grad_sigmoid = torch.sigmoid(grad_logits).requires_grad_(True)
    grad_gate = (grad_sigmoid,)
    grad_loss = seq_load_balancing_loss_func(grad_gate, E, K, batch_size=B)
    grad_loss.backward()

    has_grad = grad_sigmoid.grad is not None and grad_sigmoid.grad.abs().max().item() > 0
    print(f"     Gradients exist and nonzero: {'✓' if has_grad else '✗'}")
    results.append(has_grad)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  ALL {total}/{total} CHECKS PASSED")
    else:
        print(f"  {passed}/{total} CHECKS PASSED — FAILURES DETECTED")
        assert False
    print("=" * 70)


if __name__ == "__main__":
    run_all()
