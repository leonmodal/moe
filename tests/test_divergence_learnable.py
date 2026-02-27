"""
Divergence test with LEARNABLE data.

Uses a small fixed dataset of repeated batches so both models can actually
learn (overfit) and reveal whether they follow different loss trajectories.
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM


def old_standard_config():
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
    return GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
    )

def new_global_config():
    return GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=16,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=True,
    )


def make_learnable_data(num_batches=8, batch_size=4, seq_len=128, device="cuda"):
    """Create structured data with patterns the model can learn."""
    torch.manual_seed(123)
    batches = []
    for _ in range(num_batches):
        # Use a small vocab subset so patterns are learnable
        ids = torch.randint(0, 5000, (batch_size, seq_len), device=device)
        batches.append(ids)
    return batches


def run_comparison(std_cfg, glb_cfg, label, num_steps=500):
    device = "cuda"
    dtype = torch.bfloat16

    batches = make_learnable_data(device=device)

    torch.manual_seed(0)
    std = StandardMoEModel(std_cfg).to(device=device, dtype=dtype).train()
    std_opt = torch.optim.AdamW(std.parameters(), lr=1e-4, weight_decay=0.01,
                                 betas=(0.9, 0.95))

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(glb_cfg).to(device=device, dtype=dtype).train()
    glb_opt = torch.optim.AdamW(glb.parameters(), lr=1e-4, weight_decay=0.01,
                                 betas=(0.9, 0.95))

    std_total = sum(p.numel() for p in std.parameters())
    glb_total = sum(p.numel() for p in glb.parameters())
    print(f"  Standard: {std_total/1e6:.1f}M params")
    print(f"  Global:   {glb_total/1e6:.1f}M params")

    std_ces = []
    glb_ces = []

    for step in range(num_steps):
        ids = batches[step % len(batches)]

        std_out = std(input_ids=ids, labels=ids, output_router_logits=True)
        std_ce = std_out.loss.item() - std.router_aux_loss_coef * std_out.aux_loss.item()
        std_out.loss.backward()
        torch.nn.utils.clip_grad_norm_(std.parameters(), 1.0)
        std_opt.step(); std_opt.zero_grad()

        glb_out = glb(input_ids=ids, labels=ids, output_router_logits=True)
        glb_ce = glb_out.loss.item() - glb.router_aux_loss_coef * glb_out.aux_loss.item()
        glb_out.loss.backward()
        torch.nn.utils.clip_grad_norm_(glb.parameters(), 1.0)
        glb_opt.step(); glb_opt.zero_grad()

        std_ces.append(std_ce)
        glb_ces.append(glb_ce)

    print(f"\n  {'Step':>5}  {'Std CE':>10}  {'Glb CE':>10}  {'Diff':>10}  {'Rel%':>8}")
    print(f"  {'-'*50}")
    checkpoints = [0, 9, 24, 49, 99, 149, 199, 299, 399, 499]
    for i in [c for c in checkpoints if c < num_steps]:
        diff = glb_ces[i] - std_ces[i]
        rel = diff / std_ces[i] * 100 if std_ces[i] != 0 else 0
        print(f"  {i:5d}  {std_ces[i]:10.4f}  {glb_ces[i]:10.4f}  {diff:+10.4f}  {rel:+8.3f}%")

    last_n = min(50, num_steps)
    avg_std = sum(std_ces[-last_n:]) / last_n
    avg_glb = sum(glb_ces[-last_n:]) / last_n
    avg_diff = avg_glb - avg_std
    avg_rel = avg_diff / avg_std * 100

    print(f"\n  Last {last_n} steps average:")
    print(f"    Std CE:  {avg_std:.4f}")
    print(f"    Glb CE:  {avg_glb:.4f}")
    print(f"    Diff:    {avg_diff:+.4f} ({avg_rel:+.3f}%)")
    print(f"    CE has dropped from ~{std_ces[0]:.1f} to ~{avg_std:.1f} "
          f"({(1 - avg_std/std_ces[0])*100:.1f}% reduction)")

    del std, glb
    torch.cuda.empty_cache()
    return std_ces, glb_ces


if __name__ == "__main__":
    print("=" * 60)
    print("DIVERGENCE TEST WITH LEARNABLE DATA")
    print("=" * 60)

    print("\n--- OLD CONFIG: Both 8 layers (same capacity) ---")
    old_s, old_g = run_comparison(
        old_standard_config(), old_global_config(),
        "old", num_steps=500
    )

    print("\n\n--- NEW CONFIG: 16-layer global vs 8-layer standard ---")
    new_s, new_g = run_comparison(
        old_standard_config(), new_global_config(),
        "new", num_steps=500
    )

    print("\n" + "=" * 60)
