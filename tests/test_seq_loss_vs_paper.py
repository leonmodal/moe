"""
Definitive test: compare our seq_load_balancing_loss_func against
the EXACT DeepSeek V3 paper formulation (Eqs. 17-20) and Megatron-LM.

DeepSeek V3 paper (arxiv 2412.19437) defines:

  Eq. 15:  s_{i,t}  = sigmoid(u_t^T e_i)           — raw sigmoid affinity
  Eq. 19:  s'_{i,t} = s_{i,t} / Σ_j s_{j,t}        — NORMALIZED to sum-to-1
  Eq. 18:  f_i = (N_r / (K_r * T)) * Σ_t 1[expert i in topK]   — note: /K in denom
  Eq. 20:  P_i = (1/T) * Σ_t s'_{i,t}              — uses NORMALIZED scores
  Eq. 17:  L_Bal = α * Σ_i f_i * P_i

Two differences vs our code:
  1. P_i uses NORMALIZED sigmoid (s'_{i,t}), we use raw sigmoid s_{i,t}
  2. f_i divides by K, we don't (our one_hot + mean over tokens gives count/T, not count/(T*K))
"""
import torch
import torch.nn.functional as F

from src.models.load_balancing import seq_load_balancing_loss_func


def deepseek_v3_paper_seq_loss(
    raw_logits_per_layer: list[torch.Tensor],  # [B*S, E] raw logits (pre-sigmoid)
    num_experts: int,
    top_k: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Exact DeepSeek V3 paper formulation (Eqs. 17-20), per-sequence, averaged.
    """
    total_loss = torch.tensor(0.0)

    for raw_logits in raw_logits_per_layer:
        T, E = raw_logits.shape
        seq_len = T // batch_size

        # Eq. 15: raw sigmoid affinities
        s = torch.sigmoid(raw_logits.float())  # [B*S, E]

        # Eq. 19: normalized affinities (sum-to-1 per token)
        s_prime = s / (s.sum(dim=-1, keepdim=True) + 1e-20)  # [B*S, E]

        # Reshape per-sequence
        s_seq = s.reshape(batch_size, seq_len, E)           # (B, S, E)
        s_prime_seq = s_prime.reshape(batch_size, seq_len, E)  # (B, S, E)

        # Eq. 18: f_i = (N_r / (K * T)) * Σ_t 1[expert i in topK(s)]
        # Note: topK is on raw sigmoid (unbiased)
        _, selected = torch.topk(s_seq, top_k, dim=-1)      # (B, S, K)
        expert_mask = F.one_hot(selected, E).float()          # (B, S, K, E)
        # Sum indicator over tokens: Σ_t 1[...] for each expert
        expert_counts = expert_mask.sum(dim=(1, 2))           # (B, E)
        # f_i = (N_r / (K * S)) * counts
        f_i = (E / (top_k * seq_len)) * expert_counts        # (B, E)

        # Eq. 20: P_i = (1/T) * Σ_t s'_{i,t}
        P_i = s_prime_seq.mean(dim=1)                         # (B, E)

        # Eq. 17: L_Bal = Σ_i f_i * P_i  (per sequence, no alpha — we add it outside)
        per_seq_loss = (f_i * P_i).sum(dim=-1)                # (B,)
        total_loss = total_loss + per_seq_loss.mean()

    return total_loss / len(raw_logits_per_layer)


def megatron_seq_loss(
    raw_logits_per_layer: list[torch.Tensor],
    num_experts: int,
    top_k: int,
    batch_size: int,
    coeff: float,
) -> torch.Tensor:
    """Megatron-LM seq_aux_loss implementation (inlined from router.py)."""
    total_loss = torch.tensor(0.0)

    for raw_logits in raw_logits_per_layer:
        T, E = raw_logits.shape
        seq_len = T // batch_size

        # Megatron: compute_routing_scores_for_aux_loss (sigmoid path)
        scores = torch.sigmoid(raw_logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)  # NORMALIZE
        _, top_indices = torch.topk(scores, top_k, dim=-1)
        routing_map = torch.zeros_like(raw_logits, dtype=torch.float32).scatter(
            1, top_indices, 1
        ).bool()

        # Megatron: _apply_seq_aux_loss reshape trick
        scores_reshaped = scores.reshape(seq_len, -1)      # [S, B*E]
        routing_map_reshaped = routing_map.reshape(seq_len, -1)  # [S, B*E]

        # get_tokens_per_expert_and_token_count (single rank)
        tokens_per_expert = routing_map_reshaped.sum(dim=0).float()  # [B*E]
        total_num_tokens = float(seq_len)

        # switch_load_balancing_loss_func
        aggregated_probs = scores_reshaped.sum(dim=0)  # [B*E]
        layer_loss = torch.sum(aggregated_probs * tokens_per_expert) * (
            E * coeff / (top_k * total_num_tokens * total_num_tokens)
        )
        layer_loss = layer_loss / batch_size

        total_loss = total_loss + layer_loss

    return total_loss / len(raw_logits_per_layer)


def test_all_three():
    """Compare paper, Megatron, and our code on identical inputs."""
    B, S, E, K = 4, 128, 64, 8
    num_layers = 3
    alpha = 0.0001  # DeepSeek V3 coefficient

    torch.manual_seed(42)
    raw_logits_list = [torch.randn(B * S, E) for _ in range(num_layers)]

    # ── 1. DeepSeek V3 paper (ground truth) ──
    paper_loss = deepseek_v3_paper_seq_loss(raw_logits_list, E, K, B)

    # ── 2. Megatron-LM ──
    mega_loss = megatron_seq_loss(raw_logits_list, E, K, B, coeff=alpha)

    # ── 3. Our code (raw sigmoid, no /K) ──
    gate_logits_raw = tuple(torch.sigmoid(r.float()) for r in raw_logits_list)
    our_loss_raw = seq_load_balancing_loss_func(gate_logits_raw, E, K, batch_size=B)

    # ── 4. Our code with normalized sigmoid (fixing P_i only) ──
    gate_logits_norm = tuple(
        torch.sigmoid(r.float()) / (torch.sigmoid(r.float()).sum(dim=-1, keepdim=True) + 1e-20)
        for r in raw_logits_list
    )
    our_loss_norm = seq_load_balancing_loss_func(gate_logits_norm, E, K, batch_size=B)

    print("=" * 70)
    print("  DeepSeek V3 Paper vs Megatron vs Our Code")
    print("=" * 70)
    print()
    print(f"Config: B={B}, S={S}, E={E}, K={K}, layers={num_layers}, alpha={alpha}")
    print()

    # Paper loss doesn't include alpha
    paper_val = paper_loss.item()
    mega_val = mega_loss.item()
    our_raw_val = our_loss_raw.item()
    our_norm_val = our_loss_norm.item()

    print(f"{'Implementation':<35} {'Loss (no coeff)':<20} {'Loss (×alpha)':<20}")
    print(f"{'-'*75}")
    print(f"{'DeepSeek V3 paper (Eqs 17-20)':<35} {paper_val:<20.6f} {paper_val * alpha:<20.8f}")
    print(f"{'Megatron-LM':<35} {mega_val / alpha:<20.6f} {mega_val:<20.8f}")
    print(f"{'Our code (raw sigmoid)':<35} {our_raw_val:<20.6f} {our_raw_val * alpha:<20.8f}")
    print(f"{'Our code (normalized sigmoid)':<35} {our_norm_val:<20.6f} {our_norm_val * alpha:<20.8f}")
    print()

    # ── Check: Paper ≈ Megatron? ──
    paper_vs_mega = paper_val / (mega_val / alpha)
    print(f"Paper / Megatron = {paper_vs_mega:.6f}  (should be ~1.0)")

    # ── Check: Our (raw sigmoid input) matches paper ──
    our_raw_vs_paper = our_raw_val / paper_val
    print(f"Our (raw sigmoid input) / Paper = {our_raw_vs_paper:.6f}  (should be ~1.0)")

    # ── Check: Our (normalized sigmoid input) matches paper ──
    our_norm_vs_paper = our_norm_val / paper_val
    print(f"Our (normalized sigmoid input) / Paper = {our_norm_vs_paper:.6f}  (should be ~1.0)")
    print()

    # ── Verify Megatron matches paper ──
    assert abs(paper_vs_mega - 1.0) < 0.01, f"Paper vs Megatron: expected ~1.0, got {paper_vs_mega}"
    print("✓ Megatron matches DeepSeek V3 paper formulation")

    # ── Verify our code matches paper (regardless of input normalization) ──
    assert abs(our_raw_vs_paper - 1.0) < 0.01, f"Our (raw) vs paper: expected ~1.0, got {our_raw_vs_paper}"
    print("✓ Our code (raw sigmoid input) matches paper — normalization applied internally")

    assert abs(our_norm_vs_paper - 1.0) < 0.01, f"Our (norm) vs paper: expected ~1.0, got {our_norm_vs_paper}"
    print("✓ Our code (pre-normalized input) matches paper — double normalization is no-op")

    print()


def test_with_actual_config():
    """Test with your actual config values: E=256, K=4."""
    B, S, E, K = 4, 128, 256, 4
    num_layers = 2
    alpha = 0.0001

    torch.manual_seed(42)
    raw_logits_list = [torch.randn(B * S, E) for _ in range(num_layers)]

    # Paper
    paper_loss = deepseek_v3_paper_seq_loss(raw_logits_list, E, K, B)

    # Our code (raw sigmoid)
    gate_logits_raw = tuple(torch.sigmoid(r.float()) for r in raw_logits_list)
    our_loss_raw = seq_load_balancing_loss_func(gate_logits_raw, E, K, batch_size=B)

    paper_val = paper_loss.item()
    our_raw_val = our_loss_raw.item()
    ratio = our_raw_val / paper_val

    print("=" * 70)
    print("  With Your Actual Config (E=256, K=4)")
    print("=" * 70)
    print()
    print(f"  Paper loss (no coeff):    {paper_val:.4f}")
    print(f"  Our loss (no coeff):      {our_raw_val:.4f}")
    print(f"  Ratio (our / paper):      {ratio:.1f}×")
    print()
    print(f"  Your coeff:               {alpha}")
    print(f"  Effective paper-equivalent: {alpha * ratio:.6f}")
    print(f"  To match paper α={alpha}:  set coeff = {alpha / ratio:.8f}")
    print()


if __name__ == "__main__":
    test_all_three()
    test_with_actual_config()


def test_seq_loss_token_masks_match_active_token_subset():
    gate_logits = torch.tensor([
        [0.90, 0.10],
        [0.80, 0.20],
        [0.10, 0.90],
        [0.20, 0.80],
    ])
    selected = torch.tensor([[0], [0], [1], [1]])
    token_mask = torch.tensor([True, True, False, False])

    masked_loss = seq_load_balancing_loss_func(
        (gate_logits,),
        num_experts=2,
        top_k=1,
        batch_size=1,
        selected_experts=(selected,),
        token_masks=(token_mask,),
    )
    active_only_loss = seq_load_balancing_loss_func(
        (gate_logits[token_mask],),
        num_experts=2,
        top_k=1,
        batch_size=1,
        selected_experts=(selected[token_mask],),
    )

    torch.testing.assert_close(masked_loss, active_only_loss)


# AC-5 / task10: sequence-aux paper-formula extensions.
#
# Per `docs/plan.md` AC-5 contract: `seq_load_balancing_loss_func` must
# (1) normalize sigmoid scores to sum-to-1 (Eq. 19) so uniform routing
#     yields exactly `1.0`,
# (2) honor a `token_masks` parameter for the per-rank skipped-token case,
# (3) accept caller-provided `selected_experts` (when provided, f_i is
#     computed from those indices rather than re-deriving via top-k of
#     scores). Reaching parity between the explicit and the implicit path
#     proves the loss can route through whatever the model actually picked
#     (e.g. with exploration / `branch_router` overrides).

def test_seq_loss_uniform_routing_yields_one():
    """AC-5: with perfect uniform routing the loss must equal 1.0.

    The DeepSeek-V3 paper (Eqs. 17-20) calibrates the seq aux loss so that
    the perfect-uniform baseline is `1.0` — any deviation indicates load
    imbalance. This test sets every token's pre-normalize sigmoid score to
    a constant, which after Eq. 19 normalization is 1/E. With explicit
    selected_experts that perfectly uniform-tile the experts (each expert
    selected exactly `K/E` of the time per token, summed to `T*K/E` total
    selections), `f_i = 1.0` and `P_i = 1/E`, so `L_Bal = E * (1/E) = 1.0`.
    """
    B, S, E, K = 1, 64, 4, 2
    T = B * S
    # Constant scores: after Eq. 19 normalization → uniform 1/E.
    gate_logits = torch.full((T, E), 0.25)

    # Build a selected_experts that uniformly tiles every (token, expert) cell:
    # rotate the topK selection through the experts so each expert appears
    # exactly K*T/E times across all tokens.
    base_selection = torch.arange(K, dtype=torch.long)        # (K,) e.g. [0, 1]
    selections = []
    for t in range(T):
        rotated = (base_selection + t) % E                    # rotate per token
        selections.append(rotated)
    selected = torch.stack(selections, dim=0)                  # (T, K)

    loss = seq_load_balancing_loss_func(
        (gate_logits,),
        num_experts=E,
        top_k=K,
        batch_size=B,
        selected_experts=(selected,),
    )
    torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_seq_loss_selected_experts_parity_with_default_topk():
    """AC-5: passing `selected_experts` derived from `topk(scores)` must
    produce the SAME loss as the implicit top-k fallback.

    This locks the contract that `selected_experts` is a faithful pass-
    through path (no off-by-one re-derivation, no unintended re-sorting).
    """
    B, S, E, K = 2, 16, 6, 2
    torch.manual_seed(20260428)
    T = B * S
    raw = torch.randn(T, E)
    gate_logits = torch.sigmoid(raw)

    # Default path: function derives selection internally via topk(scores).
    loss_default = seq_load_balancing_loss_func(
        (gate_logits,),
        num_experts=E,
        top_k=K,
        batch_size=B,
        selected_experts=None,
    )

    # Explicit path: caller hands the SAME topk indices in.
    scores_normalized = gate_logits / (gate_logits.sum(dim=-1, keepdim=True) + 1e-20)
    _, idx = torch.topk(scores_normalized.reshape(B, S, E), K, dim=-1)
    explicit = idx.reshape(T, K)
    loss_explicit = seq_load_balancing_loss_func(
        (gate_logits,),
        num_experts=E,
        top_k=K,
        batch_size=B,
        selected_experts=(explicit,),
    )

    torch.testing.assert_close(loss_default, loss_explicit)


def test_seq_loss_zero_active_layers_returns_zero():
    """AC-5: when every token is masked out (e.g. branch routing sent all
    tokens through the OTHER branch), the function must return a finite
    zero rather than NaN/Inf — no live aux contribution, no divide-by-zero.
    """
    B, S, E, K = 1, 4, 2, 1
    T = B * S
    gate_logits = torch.tensor([[0.9, 0.1]] * T)
    selected = torch.tensor([[0]] * T)
    all_masked = torch.zeros(T, dtype=torch.bool)  # all inactive

    loss = seq_load_balancing_loss_func(
        (gate_logits,),
        num_experts=E,
        top_k=K,
        batch_size=B,
        selected_experts=(selected,),
        token_masks=(all_masked,),
    )
    assert torch.isfinite(loss).all(), f"empty-active-token path returned non-finite: {loss}"
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_seq_loss_skipped_layer_excluded_from_average():
    """AC-5: when one layer has zero active tokens (token_mask all False)
    and another has active tokens, the per-layer average must divide by
    the number of LIVE layers, not by the total layer count.
    """
    B, S, E, K = 1, 4, 2, 1
    T = B * S
    gate_logits_active = torch.tensor([[0.9, 0.1]] * T)
    gate_logits_dead = torch.tensor([[0.5, 0.5]] * T)
    selected_active = torch.tensor([[0]] * T)
    selected_dead = torch.tensor([[0]] * T)
    mask_all_active = torch.ones(T, dtype=torch.bool)
    mask_all_inactive = torch.zeros(T, dtype=torch.bool)

    loss_two_layers = seq_load_balancing_loss_func(
        (gate_logits_active, gate_logits_dead),
        num_experts=E,
        top_k=K,
        batch_size=B,
        selected_experts=(selected_active, selected_dead),
        token_masks=(mask_all_active, mask_all_inactive),
    )

    # Reference: just the active layer alone.
    loss_one_layer = seq_load_balancing_loss_func(
        (gate_logits_active,),
        num_experts=E,
        top_k=K,
        batch_size=B,
        selected_experts=(selected_active,),
        token_masks=(mask_all_active,),
    )

    torch.testing.assert_close(loss_two_layers, loss_one_layer)
