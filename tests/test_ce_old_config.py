"""
Test with the OLD config: both 8 layers, to check if CE was identical there.
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM


def old_standard_config():
    """xs_standard.yaml — 8 layers, 16 experts/layer."""
    return Qwen3MoeConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
    )


def old_global_config():
    """OLD xs_global.yaml — 8 layers (SAME as standard), 128 global experts."""
    return GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
    )


def test_old_config_ce():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (4, 256), device=device)  # bigger batch for more stable stats

    # Standard
    torch.manual_seed(0)
    std = StandardMoEModel(old_standard_config()).to(device=device, dtype=dtype).train()
    std_opt = torch.optim.AdamW(std.parameters(), lr=1e-3)

    # Global
    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config()).to(device=device, dtype=dtype).train()
    glb_opt = torch.optim.AdamW(glb.parameters(), lr=1e-3)

    # Count params
    std_total = sum(p.numel() for p in std.parameters())
    glb_total = sum(p.numel() for p in glb.parameters())
    std_expert = sum(p.numel() for n, p in std.named_parameters()
                     if "gate_up_proj" in n or ("down_proj" in n and "mlp" in n))
    glb_expert = sum(p.numel() for n, p in glb.named_parameters()
                     if "gate_up_proj" in n or ("down_proj" in n and "global_experts" in n))

    print(f"Standard: {std_total/1e6:.1f}M total, {std_expert/1e6:.1f}M expert params")
    print(f"Global:   {glb_total/1e6:.1f}M total, {glb_expert/1e6:.1f}M expert params")
    print(f"Param diff: {(glb_total - std_total)/1e6:.1f}M")

    print(f"\n{'Step':>4}  {'Std CE':>12}  {'Glb CE':>12}  {'AbsDiff':>12}  {'RelDiff%':>10}")
    print("-" * 65)

    for step in range(15):
        std_out = std(input_ids=ids, labels=ids, output_router_logits=True)
        std_ce = std_out.loss.item() - std.router_aux_loss_coef * std_out.aux_loss.item()
        std_out.loss.backward()
        std_opt.step(); std_opt.zero_grad()

        glb_out = glb(input_ids=ids, labels=ids, output_router_logits=True)
        glb_ce = glb_out.loss.item() - glb.router_aux_loss_coef * glb_out.aux_loss.item()
        glb_out.loss.backward()
        glb_opt.step(); glb_opt.zero_grad()

        diff = abs(std_ce - glb_ce)
        rel = diff / std_ce * 100
        print(f"{step:4d}  {std_ce:12.6f}  {glb_ce:12.6f}  {diff:12.6f}  {rel:10.4f}")

    del std, glb
    torch.cuda.empty_cache() if device == "cuda" else None


if __name__ == "__main__":
    print("=" * 65)
    print("OLD CONFIG TEST: Both 8 layers")
    print("=" * 65)
    test_old_config_ce()
    print("=" * 65)
