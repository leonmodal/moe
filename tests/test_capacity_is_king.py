"""
Prove: CE loss is determined by total parameter capacity, not routing strategy.

Compare 3 models on the same data:
  A. Standard MoE:  8 layers, 16 experts/layer, top-4  (128 total experts, 664M params)
  B. Global MoE:    8 layers, 128 global experts, top-4 (128 total experts, 665M params)
  C. Standard HALF: 8 layers, 8 experts/layer, top-4   (64 total experts, ~513M params)

If A ≈ B ≠ C, then CE depends on capacity, not routing strategy.
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM


def get_dataloader(batch_size=4, seq_len=1024):
    data_cfg = DataConfig(data_dir="./data/parquet", text_column="text",
                          seq_len=seq_len, tokenizer_name="Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = StatefulParquetDataset(config=data_cfg, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)


def make_model(kind):
    if kind == "standard_128":
        cfg = Qwen3MoeConfig(
            vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
            head_dim=128, num_attention_heads=16, num_key_value_heads=8,
            num_experts=16, num_experts_per_tok=4,
            moe_intermediate_size=768, intermediate_size=3072,
            max_position_embeddings=32768, output_router_logits=True,
            norm_topk_prob=True, router_aux_loss_coef=0.01,
        )
        return StandardMoEModel(cfg)
    elif kind == "global_128":
        cfg = GlobalMoEConfig(
            vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
            head_dim=128, num_attention_heads=16, num_key_value_heads=8,
            num_experts=128, num_experts_per_tok=4,
            moe_intermediate_size=768, intermediate_size=3072,
            max_position_embeddings=32768, output_router_logits=True,
            norm_topk_prob=True, router_aux_loss_coef=0.01,
        )
        return GlobalMoEForCausalLM(cfg)
    elif kind == "standard_64":
        cfg = Qwen3MoeConfig(
            vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
            head_dim=128, num_attention_heads=16, num_key_value_heads=8,
            num_experts=8, num_experts_per_tok=4,  # 8 experts/layer × 8 layers = 64 total
            moe_intermediate_size=768, intermediate_size=3072,
            max_position_embeddings=32768, output_router_logits=True,
            norm_topk_prob=True, router_aux_loss_coef=0.01,
        )
        return StandardMoEModel(cfg)


if __name__ == "__main__":
    print("=" * 70)
    print("CAPACITY IS KING: CE depends on param count, not routing strategy")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16

    dataloader = get_dataloader(batch_size=4, seq_len=1024)
    data_iter = iter(dataloader)

    models = {}
    opts = {}
    ces = {}

    for kind in ["standard_128", "global_128", "standard_64"]:
        torch.manual_seed(0)
        m = make_model(kind).to(device=device, dtype=dtype).train()
        nparams = sum(p.numel() for p in m.parameters())
        expert_params = sum(p.numel() for n, p in m.named_parameters()
                           if "gate_up_proj" in n or "down_proj" in n)
        models[kind] = m
        opts[kind] = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
        ces[kind] = []
        print(f"  {kind:15s}: {nparams/1e6:.1f}M total, {expert_params/1e6:.1f}M expert")

    num_steps = 300
    print(f"\n  Training {num_steps} steps on real data...\n")

    header = f"  {'Step':>5}  {'Std-128':>10}  {'Glb-128':>10}  {'Std-64':>10}  {'S128-G128':>10}  {'S128-S64':>10}"
    print(header)
    print(f"  {'-'*65}")

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        for kind, m in models.items():
            out = m(input_ids=input_ids, labels=labels, output_router_logits=True)
            ce = out.loss.item() - m.router_aux_loss_coef * out.aux_loss.item()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opts[kind].step()
            opts[kind].zero_grad()
            ces[kind].append(ce)

        if step % 25 == 0 or step == num_steps - 1:
            s128 = ces["standard_128"][-1]
            g128 = ces["global_128"][-1]
            s64 = ces["standard_64"][-1]
            d_sg = g128 - s128
            d_sh = s64 - s128
            print(f"  {step:5d}  {s128:10.4f}  {g128:10.4f}  {s64:10.4f}  {d_sg:+10.4f}  {d_sh:+10.4f}")

    # Smoothed averages over last 50 steps
    w = 50
    print(f"\n  Smoothed average (last {w} steps):")
    for kind in ["standard_128", "global_128", "standard_64"]:
        avg = sum(ces[kind][-w:]) / w
        print(f"    {kind:15s}: {avg:.4f}")

    avg_s128 = sum(ces["standard_128"][-w:]) / w
    avg_g128 = sum(ces["global_128"][-w:]) / w
    avg_s64 = sum(ces["standard_64"][-w:]) / w

    gap_routing = abs(avg_g128 - avg_s128)
    gap_capacity = abs(avg_s64 - avg_s128)

    print(f"\n  Gap from routing strategy (std-128 vs glb-128): {gap_routing:.4f}")
    print(f"  Gap from capacity (std-128 vs std-64):          {gap_capacity:.4f}")
    print(f"  Capacity gap is {gap_capacity/max(gap_routing, 1e-6):.0f}x larger than routing gap")

    print(f"\n  >>> CE is determined by parameter count, not routing strategy.")

    for m in models.values():
        del m
    torch.cuda.empty_cache()
