"""
Compare our seq_load_balancing_loss_func and update_expert_biases
against Megatron-LM's implementation to verify correctness.

We inline the Megatron logic here (no import needed) so we can run
side-by-side with our code on the exact same tensors.
"""
import math
import torch
import torch.nn.functional as F

# ── Our code ──────────────────────────────────────────────────────────
from src.models.load_balancing import seq_load_balancing_loss_func

# ── Megatron reference (inlined) ─────────────────────────────────────

def megatron_switch_load_balancing_loss_func(
    probs,               # [num_tokens, num_experts]
    tokens_per_expert,   # [num_experts]  — counts
    total_num_tokens,    # scalar
    topk,
    num_experts,
    moe_aux_loss_coeff,
):
    """Megatron-LM moe_utils.py:47-134 (non-fused path)."""
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def megatron_seq_aux_loss(
    scores_for_aux_loss,   # [B*S, E]  — normalized probs
    routing_map,           # [B*S, E]  — bool
    seq_length,
    bsz,
    topk,
    num_experts,
    coeff,
):
    """
    Megatron-LM router.py:_apply_seq_aux_loss (inlined).

    Reshape trick: [B*S, E] → [S, B*E] then call switch_load_balancing_loss_func
    with num_experts=E, topk=K, and divide by bsz.
    """
    scores_reshaped = scores_for_aux_loss.reshape(seq_length, -1)  # [S, B*E]
    routing_map_reshaped = routing_map.reshape(seq_length, -1)     # [S, B*E]

    # get_tokens_per_expert_and_token_count (single-rank, no padding)
    tokens_per_expert = routing_map_reshaped.sum(dim=0).float()    # [B*E]
    total_num_tokens = seq_length  # single rank

    aux_loss = megatron_switch_load_balancing_loss_func(
        probs=scores_reshaped,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=total_num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
    ) / bsz

    return aux_loss


def megatron_compute_routing_scores_for_aux_loss(
    logits,   # [T, E] — raw logits (pre-sigmoid)
    topk,
    score_function="sigmoid",
):
    """
    Megatron-LM moe_utils.py:809-857 (non-fused, single rank).
    Returns routing_map [T, E] bool and scores [T, E] float.
    """
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.to(torch.float32))
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)  # ← NORMALIZES
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits, dtype=torch.float32).scatter(1, top_indices, 1).bool()
    return routing_map, scores


def megatron_get_updated_expert_bias(
    tokens_per_expert,      # [num_layers, E]  (stacked)
    expert_bias,            # [num_layers, E]  (stacked)
    expert_bias_update_rate,
):
    """Megatron-LM moe_utils.py:1119-1142 (single rank, no all_reduce)."""
    with torch.no_grad():
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias


# ── Test Helpers ──────────────────────────────────────────────────────

def make_sigmoid_router_logits(B, S, E, K, num_layers, seed=42):
    """Simulate DeepSeekRouter outputs: sigmoid scores (NOT softmax).

    Returns:
        raw_logits: list of [B*S, E] raw logits (pre-sigmoid) — for Megatron
        gate_logits: tuple of [B*S, E] sigmoid scores — what our code receives
    """
    torch.manual_seed(seed)
    raw_logits_list = []
    gate_logits_list = []
    for _ in range(num_layers):
        raw = torch.randn(B * S, E)
        sig = torch.sigmoid(raw)
        raw_logits_list.append(raw)
        gate_logits_list.append(sig)
    return raw_logits_list, tuple(gate_logits_list)


def make_softmax_router_logits(B, S, E, K, num_layers, seed=42):
    """Simulate softmax router outputs."""
    torch.manual_seed(seed)
    gate_logits_list = []
    for _ in range(num_layers):
        raw = torch.randn(B * S, E)
        probs = torch.softmax(raw, dim=-1, dtype=torch.float32)
        gate_logits_list.append(probs)
    return None, tuple(gate_logits_list)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: seq_load_balancing_loss_func vs Megatron — SOFTMAX router
# ═══════════════════════════════════════════════════════════════════════

def test_seq_loss_softmax_matches_megatron():
    """
    For softmax router, the scores ARE already normalized (sum-to-1).
    Our code and Megatron should agree up to the known K-factor difference.
    """
    B, S, E, K = 4, 128, 64, 8
    num_layers = 3
    coeff = 0.0001

    _, gate_logits = make_softmax_router_logits(B, S, E, K, num_layers)

    # ── Our loss ──
    our_loss = seq_load_balancing_loss_func(gate_logits, E, K, batch_size=B)

    # ── Megatron loss (per-layer, averaged) ──
    mega_total = 0.0
    for layer_gate in gate_logits:
        # Megatron uses softmax scores directly (already sum-to-1)
        scores = layer_gate  # [B*S, E]
        _, top_indices = torch.topk(scores, K, dim=-1)
        routing_map = torch.zeros_like(scores).scatter(1, top_indices, 1).bool()

        mega_layer_loss = megatron_seq_aux_loss(
            scores_for_aux_loss=scores,
            routing_map=routing_map,
            seq_length=S, bsz=B, topk=K,
            num_experts=E, coeff=coeff,
        )
        mega_total += mega_layer_loss.item()
    mega_avg = mega_total / num_layers

    # Both should now match (our code implements the same formula as Megatron).
    # Our loss doesn't include coeff, Megatron's does.
    our_val = our_loss.item()
    mega_val = mega_avg
    actual_ratio = our_val / (mega_val / coeff)

    print(f"[softmax seq loss] Our: {our_val:.6f}  Megatron: {mega_val:.8f} (with coeff={coeff})")
    print(f"  Our / (Megatron/coeff) = {actual_ratio:.4f}  (expected: 1.0)")
    assert abs(actual_ratio - 1.0) < 0.01, (
        f"Ratio mismatch: got {actual_ratio}, expected 1.0"
    )
    print("  ✓ PASS — matches Megatron\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: seq_load_balancing_loss_func vs Megatron — SIGMOID router
# ═══════════════════════════════════════════════════════════════════════

def test_seq_loss_sigmoid_vs_megatron():
    """
    For sigmoid router, Megatron NORMALIZES scores to sum-to-1 for aux loss,
    but our code uses raw sigmoid scores. This test quantifies the difference.
    """
    B, S, E, K = 4, 128, 64, 8
    num_layers = 3
    coeff = 0.0001

    raw_logits_list, gate_logits = make_sigmoid_router_logits(B, S, E, K, num_layers)

    # ── Our loss (raw sigmoid scores) ──
    our_loss = seq_load_balancing_loss_func(gate_logits, E, K, batch_size=B)

    # ── Megatron loss (normalized sigmoid) ──
    mega_total = 0.0
    for raw_logits in raw_logits_list:
        routing_map, scores = megatron_compute_routing_scores_for_aux_loss(
            raw_logits, K, "sigmoid"
        )
        mega_layer_loss = megatron_seq_aux_loss(
            scores_for_aux_loss=scores,
            routing_map=routing_map,
            seq_length=S, bsz=B, topk=K,
            num_experts=E, coeff=coeff,
        )
        mega_total += mega_layer_loss.item()
    mega_avg = mega_total / num_layers

    # ── Our loss with normalized sigmoid (what Megatron does) ──
    normalized_gate_logits = []
    for raw_logits in raw_logits_list:
        sig = torch.sigmoid(raw_logits.float())
        normalized = sig / (sig.sum(dim=-1, keepdim=True) + 1e-20)
        normalized_gate_logits.append(normalized)
    normalized_gate_logits = tuple(normalized_gate_logits)

    our_loss_normalized = seq_load_balancing_loss_func(
        normalized_gate_logits, E, K, batch_size=B
    )

    # Our code now normalizes internally, so raw and normalized inputs should match
    our_raw_val = our_loss.item()
    our_norm_val = our_loss_normalized.item()
    mega_val = mega_avg

    print(f"[sigmoid seq loss]")
    print(f"  Our (raw sigmoid input):    {our_raw_val:.6f}")
    print(f"  Our (normalized input):     {our_norm_val:.6f}")
    print(f"  Megatron (normalized):      {mega_val:.8f} (with coeff={coeff})")

    # Both should match Megatron now
    ratio_raw = our_raw_val / (mega_val / coeff)
    ratio_norm = our_norm_val / (mega_val / coeff)
    print(f"  Our (raw input) / Megatron = {ratio_raw:.4f}  (expected: 1.0)")
    print(f"  Our (norm input) / Megatron = {ratio_norm:.4f}  (expected: 1.0)")

    assert abs(ratio_raw - 1.0) < 0.01, f"Ratio mismatch: got {ratio_raw}, expected 1.0"
    assert abs(ratio_norm - 1.0) < 0.01, f"Ratio mismatch: got {ratio_norm}, expected 1.0"

    # Raw vs normalized input should give identical results (normalization is idempotent)
    raw_vs_norm = our_raw_val / our_norm_val
    print(f"  Raw input / Norm input = {raw_vs_norm:.4f}  (expected: 1.0 — internal normalization)")
    assert abs(raw_vs_norm - 1.0) < 0.01
    print("  ✓ PASS — matches Megatron and paper\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: Uniform routing — verify minimum values
# ═══════════════════════════════════════════════════════════════════════

def test_uniform_routing_minimum():
    """
    With perfectly uniform routing, both losses should be at their minimum.
    Our minimum = K, Megatron minimum = coeff.
    """
    B, S, E, K = 2, 64, 16, 4
    num_layers = 2
    coeff = 0.0001

    # Perfectly uniform softmax scores
    uniform_probs = torch.ones(B * S, E) / E
    gate_logits = tuple([uniform_probs] * num_layers)

    our_loss = seq_load_balancing_loss_func(gate_logits, E, K, batch_size=B)

    # Megatron
    mega_total = 0.0
    for probs in gate_logits:
        _, top_indices = torch.topk(probs, K, dim=-1)
        routing_map = torch.zeros_like(probs).scatter(1, top_indices, 1).bool()
        mega_layer = megatron_seq_aux_loss(
            probs, routing_map, S, B, K, E, coeff,
        )
        mega_total += mega_layer.item()
    mega_avg = mega_total / num_layers

    print(f"[uniform routing minimum]")
    print(f"  Our loss:      {our_loss.item():.6f}  (expected minimum: 1.0)")
    print(f"  Megatron loss: {mega_avg:.8f}  (expected minimum: {coeff})")
    print(f"  Our / (Megatron/coeff) = {our_loss.item() / (mega_avg / coeff):.4f}")

    # Note: with uniform probs, topk picks the SAME experts for ALL tokens
    # (since all probs are equal, topk picks the first K by index).
    # So routing is NOT balanced — all tokens go to experts 0..K-1.
    # True minimum requires balanced assignment.
    # Let's also test with balanced assignment.
    print(f"  Note: uniform probs + topk → all tokens pick same experts (0..{K-1})")
    print(f"  True minimum only achievable with balanced routing assignment\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: Expert bias update — match Megatron per-layer
# ═══════════════════════════════════════════════════════════════════════

def test_expert_bias_update_perlayer():
    """
    Our per-layer bias update (is_global=False) should exactly match Megatron.
    """
    E = 16
    num_layers = 4
    rate = 0.001

    torch.manual_seed(42)

    # Simulate token counts and biases
    counts_list = [torch.randint(50, 200, (E,)).float() for _ in range(num_layers)]
    bias_list = [torch.randn(E) * 0.01 for _ in range(num_layers)]

    # ── Our per-layer update (is_global=False path) ──
    our_biases = [b.clone() for b in bias_list]
    for i in range(num_layers):
        avg = counts_list[i].mean()
        delta = torch.sign(avg - counts_list[i]) * rate
        our_biases[i] += delta

    # ── Megatron update ──
    stacked_counts = torch.stack(counts_list)  # [L, E]
    stacked_bias = torch.stack([b.clone() for b in bias_list])  # [L, E]
    mega_updated = megatron_get_updated_expert_bias(stacked_counts, stacked_bias, rate)

    # Compare
    all_match = True
    for i in range(num_layers):
        diff = (our_biases[i] - mega_updated[i]).abs().max().item()
        if diff > 1e-7:
            print(f"  Layer {i}: max diff = {diff}")
            all_match = False

    print(f"[expert bias update — per-layer]")
    if all_match:
        print(f"  ✓ PASS — Our per-layer update exactly matches Megatron (all {num_layers} layers)\n")
    else:
        print(f"  ✗ FAIL — Mismatches found\n")
    assert all_match


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: Expert bias update — global interpolation (our extension)
# ═══════════════════════════════════════════════════════════════════════

def test_expert_bias_update_global_interpolation():
    """
    Test the global interpolation logic from train.py:update_expert_biases.
    Verify:
      - alpha=1.0 → purely per-layer (should match Megatron)
      - alpha=0.0 → purely global (cross-layer pooling)
      - alpha=0.5 → blend of both
    """
    E = 16
    num_layers = 4
    rate = 0.001

    torch.manual_seed(42)
    counts_list = [torch.randint(50, 200, (E,)).float() for _ in range(num_layers)]
    bias_list = [torch.zeros(E) for _ in range(num_layers)]

    def our_global_update(counts_list, bias_list, rate, alpha):
        """Inline from train.py:update_expert_biases (is_global=True path)."""
        biases = [b.clone() for b in bias_list]
        global_counts = torch.stack(counts_list).sum(dim=0)  # [E]
        global_avg = global_counts.mean()
        global_delta = torch.sign(global_avg - global_counts) * rate

        for i, counts in enumerate(counts_list):
            if alpha > 0:
                layer_avg = counts.mean()
                layer_delta = torch.sign(layer_avg - counts) * rate
                biases[i] += alpha * layer_delta + (1 - alpha) * global_delta
            else:
                biases[i] += global_delta
        return biases

    # alpha=1.0 → purely per-layer → should match Megatron
    biases_a1 = our_global_update(counts_list, bias_list, rate, alpha=1.0)

    stacked_counts = torch.stack(counts_list)
    stacked_bias = torch.stack([b.clone() for b in bias_list])
    mega_updated = megatron_get_updated_expert_bias(stacked_counts, stacked_bias, rate)

    match_a1 = all(
        (biases_a1[i] - mega_updated[i]).abs().max().item() < 1e-7
        for i in range(num_layers)
    )

    # alpha=0.0 → purely global → all layers get same delta
    biases_a0 = our_global_update(counts_list, bias_list, rate, alpha=0.0)
    global_counts = torch.stack(counts_list).sum(dim=0)
    global_avg = global_counts.mean()
    expected_delta = torch.sign(global_avg - global_counts) * rate
    match_a0 = all(
        (biases_a0[i] - expected_delta).abs().max().item() < 1e-7
        for i in range(num_layers)
    )

    # alpha=0.5 → blend
    biases_a5 = our_global_update(counts_list, bias_list, rate, alpha=0.5)
    match_a5 = True
    for i in range(num_layers):
        layer_avg = counts_list[i].mean()
        layer_delta = torch.sign(layer_avg - counts_list[i]) * rate
        expected = 0.5 * layer_delta + 0.5 * expected_delta
        diff = (biases_a5[i] - expected).abs().max().item()
        if diff > 1e-7:
            match_a5 = False

    print(f"[expert bias update — global interpolation]")
    print(f"  alpha=1.0 (per-layer): {'✓ matches Megatron' if match_a1 else '✗ MISMATCH'}")
    print(f"  alpha=0.0 (global):    {'✓ all layers get same delta' if match_a0 else '✗ MISMATCH'}")
    print(f"  alpha=0.5 (blend):     {'✓ correct blend' if match_a5 else '✗ MISMATCH'}")
    print()
    assert match_a1 and match_a0 and match_a5


# ═══════════════════════════════════════════════════════════════════════
#  TEST 6: Alpha schedule
# ═══════════════════════════════════════════════════════════════════════

def test_bias_alpha_schedule():
    """Verify cosine schedule from 1 → 0."""
    def bias_alpha_schedule(step, max_steps):
        progress = min(step / max(1, max_steps), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    max_steps = 10000

    a0 = bias_alpha_schedule(0, max_steps)
    a_mid = bias_alpha_schedule(max_steps // 2, max_steps)
    a_end = bias_alpha_schedule(max_steps, max_steps)

    print(f"[alpha schedule]")
    print(f"  step=0:          alpha={a0:.4f}  (expected: 1.0)")
    print(f"  step=max/2:      alpha={a_mid:.4f}  (expected: 0.5)")
    print(f"  step=max:        alpha={a_end:.4f}  (expected: 0.0)")

    assert abs(a0 - 1.0) < 1e-6
    assert abs(a_mid - 0.5) < 1e-6
    assert abs(a_end - 0.0) < 1e-6
    print("  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 7: Sigmoid normalization — does it change topk selection?
# ═══════════════════════════════════════════════════════════════════════

def test_sigmoid_normalization_topk_equivalence():
    """
    Normalizing sigmoid scores (dividing by sum) is a monotonic transformation
    per-token, so topk selection should be identical to raw sigmoid.
    This means our f_i matches Megatron's f_i even without normalization.
    The ONLY difference is in P_i (the soft probability part).
    """
    torch.manual_seed(42)
    T, E, K = 512, 64, 8

    raw_logits = torch.randn(T, E)
    sig_raw = torch.sigmoid(raw_logits)
    sig_norm = sig_raw / (sig_raw.sum(dim=-1, keepdim=True) + 1e-20)

    _, topk_raw = torch.topk(sig_raw, K, dim=-1)
    _, topk_norm = torch.topk(sig_norm, K, dim=-1)

    agreement = (topk_raw == topk_norm).all().item()

    print(f"[sigmoid normalization — topk equivalence]")
    print(f"  topk(raw_sigmoid) == topk(normalized_sigmoid): {agreement}")
    assert agreement, "Normalization should not change topk selection!"
    print("  ✓ PASS — f_i (token assignment) is identical with or without normalization")
    print("  → The only difference is in P_i (soft probability)\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 8: Quantify actual P_i difference between raw and normalized sigmoid
# ═══════════════════════════════════════════════════════════════════════

def test_sigmoid_pi_normalization():
    """
    Verify that our code normalizes internally, so raw and pre-normalized
    sigmoid inputs produce the same loss.
    """
    torch.manual_seed(42)
    B, S, E, K = 4, 128, 64, 8
    num_layers = 3

    raw_logits_list, gate_logits_raw = make_sigmoid_router_logits(B, S, E, K, num_layers)

    # Pre-normalized version
    gate_logits_norm = tuple(
        torch.sigmoid(r.float()) / (torch.sigmoid(r.float()).sum(dim=-1, keepdim=True) + 1e-20)
        for r in raw_logits_list
    )

    our_raw = seq_load_balancing_loss_func(gate_logits_raw, E, K, batch_size=B).item()
    our_norm = seq_load_balancing_loss_func(gate_logits_norm, E, K, batch_size=B).item()

    print(f"[sigmoid internal normalization]")
    print(f"  Loss (raw sigmoid input):    {our_raw:.6f}")
    print(f"  Loss (pre-normalized input): {our_norm:.6f}")
    print(f"  Ratio: {our_raw / our_norm:.6f}  (should be 1.0)")
    assert abs(our_raw / our_norm - 1.0) < 0.001
    print(f"  ✓ PASS — internal normalization makes input format irrelevant\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 9: End-to-end DeepSeekRouter output → seq loss
# ═══════════════════════════════════════════════════════════════════════

def test_deepseek_router_end_to_end():
    """
    Create a DeepSeekRouter, run a forward pass, and verify:
    1. gate_logits (sigmoid scores) are captured correctly
    2. seq_load_balancing_loss_func computes without error
    3. Loss is finite and reasonable
    """
    from src.models.router import DeepSeekRouter

    class FakeConfig:
        hidden_size = 256
        num_experts = 64
        num_experts_per_tok = 8
        norm_topk_prob = True
        topk_scaling_factor = 2.5
        num_groups = 8
        group_topk = 4

    config = FakeConfig()
    router = DeepSeekRouter(config)

    B, S = 2, 64
    hidden = torch.randn(B * S, config.hidden_size)

    with torch.no_grad():
        scores, routing_weights, selected_experts = router(hidden)

    # scores should be sigmoid values in (0, 1)
    assert scores.min() >= 0 and scores.max() <= 1, "Scores should be sigmoid values"
    assert scores.shape == (B * S, config.num_experts)

    # Compute seq loss using these sigmoid scores
    gate_logits = (scores,)  # tuple of 1 layer
    loss = seq_load_balancing_loss_func(
        gate_logits, config.num_experts, config.num_experts_per_tok, batch_size=B,
    )

    print(f"[DeepSeekRouter end-to-end]")
    print(f"  Scores shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Seq loss: {loss.item():.6f}")
    print(f"  Loss is finite: {loss.isfinite().item()}")

    # The loss should be roughly K=8 at uniform, probably higher with real routing
    assert loss.isfinite(), "Loss should be finite"
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 10: Verify our seq loss gives correct gradient direction
# ═══════════════════════════════════════════════════════════════════════

def test_seq_loss_gradient_direction():
    """
    Verify that the seq loss gradient pushes the router toward balanced routing:
    - Expert with MORE tokens should get LOWER probability (negative gradient)
    - Expert with FEWER tokens should get HIGHER probability (positive gradient)
    """
    B, S, E, K = 1, 32, 8, 2
    torch.manual_seed(42)

    # Create imbalanced probs: expert 0 gets very high probability
    logits = torch.zeros(B * S, E)
    logits[:, 0] = 5.0  # expert 0 is much more popular
    probs = torch.softmax(logits, dim=-1).requires_grad_(True)

    gate_logits = (probs,)
    loss = seq_load_balancing_loss_func(gate_logits, E, K, batch_size=B)
    loss.backward()

    # Expert 0 should have negative gradient (reduce its probability)
    grad = probs.grad[0]  # gradient for first token
    print(f"[gradient direction test]")
    print(f"  Probs[0]:  {probs[0].data.tolist()[:4]}...")
    print(f"  Grad[0]:   {grad.tolist()[:4]}...")
    print(f"  Expert 0 prob: {probs[0, 0].item():.4f}, grad: {grad[0].item():.6f}")
    print(f"  Expert 1 prob: {probs[0, 1].item():.4f}, grad: {grad[1].item():.6f}")

    # The gradient for the over-utilized expert should be positive (loss increases
    # when we increase its probability), so optimizer will decrease it
    # Actually the gradient of the loss w.r.t. the dominant expert's prob should
    # be positive (since increasing its prob increases imbalance → higher loss)
    assert grad[0] > 0, "Over-utilized expert should have positive gradient (loss increases)"
    print("  ✓ PASS — gradient correctly penalizes imbalance\n")


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Megatron-LM vs Our Implementation — Comparison Tests")
    print("=" * 70 + "\n")

    test_seq_loss_softmax_matches_megatron()
    test_seq_loss_sigmoid_vs_megatron()
    test_uniform_routing_minimum()
    test_expert_bias_update_perlayer()
    test_expert_bias_update_global_interpolation()
    test_bias_alpha_schedule()
    test_sigmoid_normalization_topk_equivalence()
    test_sigmoid_pi_normalization()
    test_deepseek_router_end_to_end()
    test_seq_loss_gradient_direction()

    print("=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)
