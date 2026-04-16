"""
Quick 50-step comparison of L-scale standard vs global MoE.
No warmup, lr=1e-3, single GPU.
"""
import sys, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
apply_liger_kernel_to_qwen3_moe()

import yaml
from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from train import build_model

STEPS = 50
DEVICE = "cuda"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_config(config_path, steps=STEPS):
    set_seed(42)
    cfg = load_cfg(config_path)
    model, model_cfg = build_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for n, p in model.named_parameters()
                        if "gate_up_proj" in n or "down_proj" in n)
    print(f"  Total: {total_params/1e9:.2f}B, Expert: {expert_params/1e9:.2f}B")

    model = model.to(DEVICE)

    if cfg["model"].get("gradient_checkpointing", False) or cfg["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    dcfg = cfg.get("data", {})
    tokenizer = AutoTokenizer.from_pretrained(dcfg.get("tokenizer_name", "Qwen/Qwen3-0.6B"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = StatefulParquetDataset(
        config=DataConfig(data_dir=dcfg["data_dir"], text_column=dcfg.get("text_column", "text"),
                          seq_len=dcfg.get("seq_len", 1024), tokenizer_name=dcfg.get("tokenizer_name")),
        tokenizer=tokenizer, rank=0, world_size=1,
    )
    dataloader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
    model.train()
    data_iter = iter(dataloader)
    losses = []

    for step in range(steps):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
            loss = output.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"    step {step+1}: loss={losses[-1]:.4f}, GPU mem={mem:.1f}GB")

    del model, optimizer
    torch.cuda.empty_cache()
    return losses


def main():
    configs = [
        ("L-Standard", "configs/scaling/l_deepseek_standard.yaml"),
        ("L-Global", "configs/scaling/l_deepseek_global.yaml"),
    ]

    all_losses = {}
    for name, path in configs:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        all_losses[name] = run_config(path)

    names = list(all_losses.keys())
    print(f"\n{'='*80}")
    print("Step-by-step comparison:")
    print(f"{'='*80}")
    print(f"{'Step':>5} {'L-Standard':>12} {'L-Global':>12} {'Diff':>10}")
    print("-" * 42)
    for i in range(STEPS):
        s = all_losses[names[0]][i]
        g = all_losses[names[1]][i]
        print(f"{i+1:>5} {s:>12.4f} {g:>12.4f} {abs(s-g):>10.4f}")

    diffs = [abs(all_losses[names[0]][i] - all_losses[names[1]][i]) for i in range(STEPS)]
    print(f"\nMean diff: {sum(diffs)/len(diffs):.4f}")
    print(f"Max diff:  {max(diffs):.4f}")
    print(f"Min diff:  {min(diffs):.4f}")

    # Compare with XS results
    print(f"\nFor reference, XS scale (860M params) had mean diff ~0.04")


if __name__ == "__main__":
    main()
