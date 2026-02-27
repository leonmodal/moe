"""
Diagnostic test: Why do Global MoE and Standard MoE have the exact same CE loss?

We test several hypotheses:
  H1. Global experts aren't actually in the computation graph (no grads)
  H2. Global experts aren't actually changing the output (dead code path)
  H3. The loss subtraction/addition in GlobalMoEForCausalLM.forward() is wrong
  H4. Both models happen to produce similar CE at init (not a bug)
  H5. The router in global MoE isn't actually routing to global experts
"""
import torch
import torch.nn as nn
from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM
from src.models.global_moe import GlobalMoEModel, GlobalSparseMoeBlock


def tiny_standard_config():
    return Qwen3MoeConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=2,
        head_dim=32, num_attention_heads=2, num_key_value_heads=1,
        num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=32, intermediate_size=128,
        max_position_embeddings=128, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
    )


def tiny_global_config():
    return GlobalMoEConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=2,
        head_dim=32, num_attention_heads=2, num_key_value_heads=1,
        num_experts=8, num_experts_per_tok=2,  # global pool = 2 layers × 4
        moe_intermediate_size=32, intermediate_size=128,
        max_position_embeddings=128, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
    )


def _batch(vocab=256, B=2, T=16):
    torch.manual_seed(42)
    ids = torch.randint(0, vocab, (B, T))
    return ids, ids


# ── H1: Do global experts get gradients? ──────────────────────────────────────

def test_global_experts_get_gradients():
    """If global experts are in the computation graph, they must get gradients."""
    model = GlobalMoEForCausalLM(tiny_global_config()).train()
    ids, labels = _batch()
    out = model(input_ids=ids, labels=labels, output_router_logits=True)
    out.loss.backward()

    ge = model.model.global_experts
    for name, p in ge.named_parameters():
        assert p.grad is not None, f"global_experts.{name} has NO gradient!"
        assert p.grad.abs().sum() > 0, f"global_experts.{name} has zero gradient!"
    print("PASS: Global experts get non-zero gradients")


# ── H2: Does perturbing global expert weights change the output? ──────────────

def test_perturbing_global_experts_changes_loss():
    """If we perturb global expert weights, the CE loss must change."""
    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg).eval()
    ids, labels = _batch()

    with torch.no_grad():
        out1 = model(input_ids=ids, labels=labels, output_router_logits=True)
        loss1 = out1.loss.item()

        # Perturb the expert weights significantly
        for p in model.model.global_experts.parameters():
            p.add_(torch.randn_like(p) * 10.0)

        out2 = model(input_ids=ids, labels=labels, output_router_logits=True)
        loss2 = out2.loss.item()

    print(f"  Loss before perturbation: {loss1:.6f}")
    print(f"  Loss after perturbation:  {loss2:.6f}")
    assert abs(loss1 - loss2) > 0.01, (
        f"Perturbing global experts by 10x std didn't change the loss! "
        f"Before={loss1:.6f}, After={loss2:.6f}"
    )
    print("PASS: Perturbing global experts changes the loss")


# ── H3: Is the aux loss subtraction/addition correct? ─────────────────────────

def test_aux_loss_substitution():
    """Check that GlobalMoEForCausalLM correctly substitutes aux loss.

    The super().forward() adds HF aux loss to the total.
    Then we subtract old and add new. If old == new, the CE is unaffected.
    """
    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg).eval()
    ids, labels = _batch()

    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, output_router_logits=True)

    total = out.loss.item()
    aux = out.aux_loss.item()
    coef = model.router_aux_loss_coef
    ce = total - coef * aux

    print(f"  Total loss: {total:.6f}")
    print(f"  Aux loss:   {aux:.6f}")
    print(f"  Coef:       {coef}")
    print(f"  Derived CE: {ce:.6f}")

    # CE must be positive and near ln(vocab_size) at init
    import math
    expected_init = math.log(256)
    print(f"  Expected init CE (ln(256)): {expected_init:.4f}")
    assert ce > 0, f"CE loss is non-positive: {ce}"
    assert abs(ce - expected_init) < 2.0, f"CE loss {ce} is far from expected init {expected_init}"
    print("PASS: Aux loss substitution looks correct")


# ── H4: Compare CE losses directly ───────────────────────────────────────────

def test_compare_ce_losses_at_init():
    """Both models should have SIMILAR but NOT IDENTICAL CE at init."""
    std_model = StandardMoEModel(tiny_standard_config()).eval()
    glb_model = GlobalMoEForCausalLM(tiny_global_config()).eval()

    ids, labels = _batch()

    with torch.no_grad():
        std_out = std_model(input_ids=ids, labels=labels, output_router_logits=True)
        glb_out = glb_model(input_ids=ids, labels=labels, output_router_logits=True)

    std_ce = std_out.loss.item() - std_model.router_aux_loss_coef * std_out.aux_loss.item()
    glb_ce = glb_out.loss.item() - glb_model.router_aux_loss_coef * glb_out.aux_loss.item()

    print(f"  Standard CE: {std_ce:.6f}")
    print(f"  Global CE:   {glb_ce:.6f}")
    print(f"  Difference:  {abs(std_ce - glb_ce):.6f}")

    # They should be different (different random weights, different routing)
    # If they're EXACTLY the same, something is very wrong
    # But note: at init both are near ln(256) so they could be close
    print(f"  (Both near ln(256) = {5.545:.3f} at init — closeness is expected)")


# ── H5: Verify different routing per layer into global pool ───────────────────

def test_different_layers_route_differently():
    """Each layer has its own router. Verify they produce different routing patterns."""
    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg).eval()
    ids, _ = _batch()

    with torch.no_grad():
        out = model(input_ids=ids, output_router_logits=True)

    logits = out.router_logits
    assert len(logits) == 2, f"Expected 2 layers of router logits, got {len(logits)}"

    # Each should be [B*T, num_experts_global=8]
    for i, rl in enumerate(logits):
        print(f"  Layer {i} router logits shape: {rl.shape}")
        assert rl.shape[-1] == cfg.num_experts, (
            f"Layer {i} router logits have {rl.shape[-1]} experts, expected {cfg.num_experts}"
        )

    # Different layers should produce different routing decisions
    _, sel0 = logits[0].topk(cfg.num_experts_per_tok, dim=-1)
    _, sel1 = logits[1].topk(cfg.num_experts_per_tok, dim=-1)

    match_rate = (sel0 == sel1).float().mean().item()
    print(f"  Expert selection match rate between layers: {match_rate:.3f}")
    print("PASS: Router logits have correct global pool size")


# ── H6: Training step comparison ─────────────────────────────────────────────

def test_training_diverges():
    """After several gradient steps on the same data, losses should diverge."""
    torch.manual_seed(99)
    ids, labels = _batch()

    # Standard
    torch.manual_seed(0)
    std = StandardMoEModel(tiny_standard_config()).train()
    std_opt = torch.optim.AdamW(std.parameters(), lr=1e-3)

    # Global
    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(tiny_global_config()).train()
    glb_opt = torch.optim.AdamW(glb.parameters(), lr=1e-3)

    std_ces = []
    glb_ces = []

    for step in range(20):
        std_out = std(input_ids=ids, labels=labels, output_router_logits=True)
        std_ce = std_out.loss.item() - std.router_aux_loss_coef * std_out.aux_loss.item()
        std_out.loss.backward()
        std_opt.step()
        std_opt.zero_grad()
        std_ces.append(std_ce)

        glb_out = glb(input_ids=ids, labels=labels, output_router_logits=True)
        glb_ce = glb_out.loss.item() - glb.router_aux_loss_coef * glb_out.aux_loss.item()
        glb_out.loss.backward()
        glb_opt.step()
        glb_opt.zero_grad()
        glb_ces.append(glb_ce)

    print(f"\n  {'Step':>4}  {'Std CE':>10}  {'Glb CE':>10}  {'Diff':>10}")
    for i in [0, 4, 9, 14, 19]:
        print(f"  {i:4d}  {std_ces[i]:10.6f}  {glb_ces[i]:10.6f}  {abs(std_ces[i]-glb_ces[i]):10.6f}")

    # Check: are the final losses EXACTLY the same?
    if abs(std_ces[-1] - glb_ces[-1]) < 1e-6:
        print("\n  WARNING: Losses are still identical after 20 steps!")
    else:
        print(f"\n  Losses diverged by {abs(std_ces[-1]-glb_ces[-1]):.6f} after 20 steps")


# ── H7: Check if standard MoE's loss function is actually being swapped ──────

def test_hf_loss_vs_custom_loss_equivalence():
    """
    Check if the HF load_balancing_loss_func (in modeling_qwen3_moe.py) produces
    the same result as our custom one (in load_balancing.py).
    If they're identical, the subtract-and-add in forward() is a no-op.
    """
    from src.models.load_balancing import load_balancing_loss_func as custom_loss
    from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func as hf_loss

    torch.manual_seed(42)
    # Simulate router logits (softmax probs) for 2 layers, 8 experts
    logits = (
        torch.softmax(torch.randn(32, 8), dim=-1),
        torch.softmax(torch.randn(32, 8), dim=-1),
    )

    custom = custom_loss(logits, num_experts=8, top_k=2)
    hf = hf_loss(logits, num_experts=8, top_k=2)

    print(f"  Custom loss: {custom.item():.6f}")
    print(f"  HF loss:     {hf.item():.6f}")
    print(f"  Difference:  {abs(custom.item() - hf.item()):.10f}")

    if abs(custom.item() - hf.item()) < 1e-6:
        print("  >> CONFIRMED: HF loss == custom loss. The subtract-and-add is a NO-OP!")
        print("  >> This means both models use the same loss function from modeling_qwen3_moe.py")
    else:
        print("  >> Losses differ — the substitution has an effect")


# ── H8: Check GlobalMoEForCausalLM super().forward() calls ──────────────────

def test_global_model_actually_uses_global_experts_module():
    """Verify that GlobalMoEForCausalLM.model is a GlobalMoEModel, not Qwen3MoeModel."""
    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg)

    assert isinstance(model.model, GlobalMoEModel), (
        f"model.model is {type(model.model).__name__}, expected GlobalMoEModel"
    )

    # Check layers are GlobalMoEDecoderLayer
    from src.models.global_moe import GlobalMoEDecoderLayer
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer, GlobalMoEDecoderLayer), (
            f"Layer {i} is {type(layer).__name__}, expected GlobalMoEDecoderLayer"
        )

    # Check each layer's mlp is GlobalSparseMoeBlock
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp, GlobalSparseMoeBlock), (
            f"Layer {i} mlp is {type(layer.mlp).__name__}, expected GlobalSparseMoeBlock"
        )

    print("PASS: Model architecture is correct (GlobalMoEModel + GlobalMoEDecoderLayer)")


# ── H9: Check that the HF Qwen3MoeForCausalLM.forward() actually uses
#         our GlobalMoEModel, not a secretly-created Qwen3MoeModel ─────────────

def test_no_hidden_standard_moe_layers():
    """Make sure no standard Qwen3MoeSparseMoeBlock exists in the global model."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg)

    for name, mod in model.named_modules():
        assert not isinstance(mod, Qwen3MoeSparseMoeBlock), (
            f"Found Qwen3MoeSparseMoeBlock at {name}! "
            f"Global MoE should only have GlobalSparseMoeBlock."
        )
    print("PASS: No hidden standard MoE blocks found")


# ── H10: Check if the problem is that expert weights are the same ─────────────

def test_expert_weights_differ_from_standard():
    """
    Check that the global expert pool is a single Qwen3MoeExperts with num_experts=8,
    not multiple small pools.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    cfg = tiny_global_config()
    model = GlobalMoEForCausalLM(cfg)

    ge = model.model.global_experts
    assert isinstance(ge, Qwen3MoeExperts), f"global_experts is {type(ge)}, expected Qwen3MoeExperts"

    # Check dimensions: gate_up_proj should be [num_experts, hidden_size, 2*moe_intermediate_size]
    gup = ge.gate_up_proj
    print(f"  gate_up_proj shape: {gup.shape}")
    print(f"  Expected: [{cfg.num_experts}, {cfg.hidden_size}, {2 * cfg.moe_intermediate_size}]")
    assert gup.shape[0] == cfg.num_experts, (
        f"gate_up_proj dim 0 is {gup.shape[0]}, expected {cfg.num_experts}"
    )
    print("PASS: Global expert pool has correct dimensions")


if __name__ == "__main__":
    print("=" * 70)
    print("CE LOSS DIAGNOSIS")
    print("=" * 70)

    tests = [
        ("H1: Global experts get gradients", test_global_experts_get_gradients),
        ("H2: Perturbing experts changes loss", test_perturbing_global_experts_changes_loss),
        ("H3: Aux loss substitution", test_aux_loss_substitution),
        ("H4: Compare CE at init", test_compare_ce_losses_at_init),
        ("H5: Different routing per layer", test_different_layers_route_differently),
        ("H6: Training divergence", test_training_diverges),
        ("H7: HF vs custom loss equivalence", test_hf_loss_vs_custom_loss_equivalence),
        ("H8: Model architecture check", test_global_model_actually_uses_global_experts_module),
        ("H9: No hidden standard MoE blocks", test_no_hidden_standard_moe_layers),
        ("H10: Expert weight dimensions", test_expert_weights_differ_from_standard),
    ]

    for name, fn in tests:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}")
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
