"""
Divergence test using the ACTUAL training data pipeline.
Runs both old config (both 8 layers) and new config (16 vs 8) side-by-side.
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
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


def get_dataloader(batch_size=4, seq_len=1024):
    data_cfg = DataConfig(
        data_dir="./data/parquet",
        text_column="text",
        seq_len=seq_len,
        tokenizer_name="Qwen/Qwen3-0.6B",
    )
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = StatefulParquetDataset(config=data_cfg, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)


def run_comparison(std_cfg, glb_cfg, label, num_steps=500, batch_size=4, seq_len=1024):
    device = "cuda"
    dtype = torch.bfloat16

    dataloader = get_dataloader(batch_size=batch_size, seq_len=seq_len)
    data_iter = iter(dataloader)

    torch.manual_seed(0)
    std = StandardMoEModel(std_cfg).to(device=device, dtype=dtype).train()
    std_opt = torch.optim.AdamW(std.parameters(), lr=3e-4, weight_decay=0.01,
                                 betas=(0.9, 0.95))

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(glb_cfg).to(device=device, dtype=dtype).train()
    glb_opt = torch.optim.AdamW(glb.parameters(), lr=3e-4, weight_decay=0.01,
                                 betas=(0.9, 0.95))

    std_total = sum(p.numel() for p in std.parameters())
    glb_total = sum(p.numel() for p in glb.parameters())
    print(f"  Standard: {std_total/1e6:.1f}M params")
    print(f"  Global:   {glb_total/1e6:.1f}M params")

    std_ces = []
    glb_ces = []

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Standard
        std_out = std(input_ids=input_ids, labels=labels, output_router_logits=True)
        std_ce = std_out.loss.item() - std.router_aux_loss_coef * std_out.aux_loss.item()
        std_out.loss.backward()
        torch.nn.utils.clip_grad_norm_(std.parameters(), 1.0)
        std_opt.step(); std_opt.zero_grad()

        # Global
        glb_out = glb(input_ids=input_ids, labels=labels, output_router_logits=True)
        glb_ce = glb_out.loss.item() - glb.router_aux_loss_coef * glb_out.aux_loss.item()
        glb_out.loss.backward()
        torch.nn.utils.clip_grad_norm_(glb.parameters(), 1.0)
        glb_opt.step(); glb_opt.zero_grad()

        std_ces.append(std_ce)
        glb_ces.append(glb_ce)

        if step % 50 == 0:
            diff = glb_ce - std_ce
            print(f"  step {step:4d}  std={std_ce:.4f}  glb={glb_ce:.4f}  diff={diff:+.4f}")

    print(f"\n  {'Step':>5}  {'Std CE':>10}  {'Glb CE':>10}  {'Diff':>10}  {'Rel%':>8}")
    print(f"  {'-'*50}")
    checkpoints = [0, 24, 49, 99, 199, 299, 399, 499]
    for i in [c for c in checkpoints if c < num_steps]:
        diff = glb_ces[i] - std_ces[i]
        rel = diff / std_ces[i] * 100 if std_ces[i] != 0 else 0
        print(f"  {i:5d}  {std_ces[i]:10.4f}  {glb_ces[i]:10.4f}  {diff:+10.4f}  {rel:+8.3f}%")

    # Smoothed averages (window=20) at end
    w = 20
    for label_pt, idx in [("step 100", 100), ("step 200", 200), ("step 300", 300),
                            ("step 400", 400), ("step 500", 500)]:
        if idx <= num_steps:
            s = max(0, idx - w)
            e = min(num_steps, idx)
            avg_s = sum(std_ces[s:e]) / (e - s)
            avg_g = sum(glb_ces[s:e]) / (e - s)
            d = avg_g - avg_s
            print(f"  Smoothed @ {label_pt}: std={avg_s:.4f}  glb={avg_g:.4f}  diff={d:+.4f} ({d/avg_s*100:+.3f}%)")

    del std, glb
    torch.cuda.empty_cache()
    return std_ces, glb_ces


if __name__ == "__main__":
    print("=" * 65)
    print("DIVERGENCE TEST WITH REAL TRAINING DATA")
    print("=" * 65)

    print("\n--- OLD CONFIG: Both 8 layers (same capacity) ---")
    run_comparison(old_standard_config(), old_global_config(), "old", num_steps=500)

    print("\n\n--- NEW CONFIG: 16-layer global vs 8-layer standard ---")
    run_comparison(old_standard_config(), new_global_config(), "new", num_steps=500)

    print("\n" + "=" * 65)
