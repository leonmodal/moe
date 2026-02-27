"""
Definitive test: Are global MoE and standard MoE CE losses truly identical?

Uses actual xs config dimensions on GPU.
Tests both same-seed init and training divergence.
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM


def xs_standard_config():
    """Matches xs_standard.yaml exactly."""
    return Qwen3MoeConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
    )


def xs_global_config():
    """Matches xs_global.yaml (updated: 16 layers, tied embeddings)."""
    return GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=16,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=True,
    )


def test_ce_at_init():
    """Compare CE at init with actual config dimensions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (2, 128), device=device)

    # Standard model
    torch.manual_seed(0)
    std = StandardMoEModel(xs_standard_config()).to(device=device, dtype=dtype).eval()

    # Global model
    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(xs_global_config()).to(device=device, dtype=dtype).eval()

    with torch.no_grad():
        std_out = std(input_ids=ids, labels=ids, output_router_logits=True)
        glb_out = glb(input_ids=ids, labels=ids, output_router_logits=True)

    std_ce = std_out.loss.item() - std.router_aux_loss_coef * std_out.aux_loss.item()
    glb_ce = glb_out.loss.item() - glb.router_aux_loss_coef * glb_out.aux_loss.item()

    print(f"Standard CE: {std_ce:.8f}")
    print(f"Global CE:   {glb_ce:.8f}")
    print(f"Difference:  {abs(std_ce - glb_ce):.8f}")
    print(f"Relative:    {abs(std_ce - glb_ce) / std_ce * 100:.4f}%")

    print(f"\nStandard total loss: {std_out.loss.item():.8f}")
    print(f"Standard aux loss:   {std_out.aux_loss.item():.8f}")
    print(f"Global total loss:   {glb_out.loss.item():.8f}")
    print(f"Global aux loss:     {glb_out.aux_loss.item():.8f}")

    if abs(std_ce - glb_ce) < 1e-4:
        print("\n*** WARNING: CE losses are suspiciously close! ***")
    else:
        print(f"\nCE losses differ by {abs(std_ce - glb_ce):.6f} — this is expected.")

    del std, glb
    torch.cuda.empty_cache() if device == "cuda" else None


def test_training_divergence_gpu():
    """Train both models for a few steps on GPU and check CE divergence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (2, 128), device=device)

    # Standard
    torch.manual_seed(0)
    std = StandardMoEModel(xs_standard_config()).to(device=device, dtype=dtype).train()
    std_opt = torch.optim.AdamW(std.parameters(), lr=1e-3)

    # Global
    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(xs_global_config()).to(device=device, dtype=dtype).train()
    glb_opt = torch.optim.AdamW(glb.parameters(), lr=1e-3)

    print(f"\n{'Step':>4}  {'Std CE':>12}  {'Glb CE':>12}  {'Diff':>12}  {'Std Aux':>10}  {'Glb Aux':>10}")
    print("-" * 75)

    for step in range(10):
        # Standard step
        std_out = std(input_ids=ids, labels=ids, output_router_logits=True)
        std_ce = std_out.loss.item() - std.router_aux_loss_coef * std_out.aux_loss.item()
        std_aux = std_out.aux_loss.item()
        std_out.loss.backward()
        std_opt.step()
        std_opt.zero_grad()

        # Global step
        glb_out = glb(input_ids=ids, labels=ids, output_router_logits=True)
        glb_ce = glb_out.loss.item() - glb.router_aux_loss_coef * glb_out.aux_loss.item()
        glb_aux = glb_out.aux_loss.item()
        glb_out.loss.backward()
        glb_opt.step()
        glb_opt.zero_grad()

        diff = abs(std_ce - glb_ce)
        flag = " <<<" if diff < 1e-4 else ""
        print(f"{step:4d}  {std_ce:12.6f}  {glb_ce:12.6f}  {diff:12.6f}  {std_aux:10.4f}  {glb_aux:10.4f}{flag}")

    del std, glb
    torch.cuda.empty_cache() if device == "cuda" else None


def test_check_loss_composition():
    """
    Verify that CE = total - coef * aux is correct by also computing CE directly.
    This ensures the aux loss substitution isn't corrupting the CE calculation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(xs_global_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (2, 128), device=device)

    with torch.no_grad():
        # Full forward with aux loss
        out_full = glb(input_ids=ids, labels=ids, output_router_logits=True)
        total = out_full.loss.item()
        aux = out_full.aux_loss.item()
        derived_ce = total - glb.router_aux_loss_coef * aux

        # Forward WITHOUT router logits — should give pure CE
        out_no_router = glb(input_ids=ids, labels=ids, output_router_logits=False)
        pure_ce = out_no_router.loss.item()

    print(f"Total loss (with aux):     {total:.8f}")
    print(f"Aux loss:                  {aux:.8f}")
    print(f"Derived CE (total-c*aux):  {derived_ce:.8f}")
    print(f"Pure CE (no router logits):{pure_ce:.8f}")
    print(f"Difference:                {abs(derived_ce - pure_ce):.10f}")

    if abs(derived_ce - pure_ce) < 1e-3:
        print("PASS: Derived CE matches pure CE")
    else:
        print(f"*** FAIL: Derived CE != Pure CE by {abs(derived_ce - pure_ce):.6f} ***")
        print("This means the aux loss substitution is corrupting the CE calculation!")

    del glb
    torch.cuda.empty_cache() if device == "cuda" else None


if __name__ == "__main__":
    print("=" * 75)
    print("CE IDENTITY TEST — Actual XS Config Dimensions on GPU")
    print("=" * 75)

    print("\n--- Test 1: CE at init ---")
    test_ce_at_init()

    print("\n--- Test 2: Training divergence ---")
    test_training_divergence_gpu()

    print("\n--- Test 3: Loss composition check ---")
    test_check_loss_composition()

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)
