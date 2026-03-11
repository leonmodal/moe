"""
Key test: does changing the seed (same arch) produce MORE divergence
than changing the architecture (standard vs global)?

If seed diff > arch diff, something is suppressing the architectural difference.
"""
import sys, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def run(config_path, seed, steps=STEPS):
    set_seed(seed)
    cfg = load_cfg(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE)

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
        tokenizer=tokenizer, rank=0, world_size=1, seed=seed,
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

    del model, optimizer
    torch.cuda.empty_cache()
    return losses


def mean_abs_diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def main():
    std_cfg = "configs/scaling/xs_deepseek_standard.yaml"
    glb_cfg = "configs/scaling/xs_deepseek_global.yaml"

    print("Running 4 experiments:")
    print("  A: Standard seed=42")
    print("  B: Standard seed=123")
    print("  C: Global seed=42")
    print("  D: Global seed=123")
    print()

    a = run(std_cfg, seed=42)
    print(f"  A done: {a[0]:.4f} -> {a[-1]:.4f}")
    b = run(std_cfg, seed=123)
    print(f"  B done: {b[0]:.4f} -> {b[-1]:.4f}")
    c = run(glb_cfg, seed=42)
    print(f"  C done: {c[0]:.4f} -> {c[-1]:.4f}")
    d = run(glb_cfg, seed=123)
    print(f"  D done: {d[0]:.4f} -> {d[-1]:.4f}")

    print("\n" + "=" * 90)
    print("Step-by-step:")
    print(f"{'Step':>5} {'Std s42':>10} {'Std s123':>10} {'Glb s42':>10} {'Glb s123':>10} | {'Std-Std':>8} {'Std-Glb':>8} {'Glb-Glb':>8}")
    print("-" * 90)
    for i in range(STEPS):
        if i < 10 or i % 5 == 4:
            print(f"{i+1:>5} {a[i]:>10.4f} {b[i]:>10.4f} {c[i]:>10.4f} {d[i]:>10.4f} | "
                  f"{abs(a[i]-b[i]):>8.4f} {abs(a[i]-c[i]):>8.4f} {abs(c[i]-d[i]):>8.4f}")

    same_arch_diff_seed = mean_abs_diff(a, b)
    diff_arch_same_seed = mean_abs_diff(a, c)
    diff_arch_diff_seed = mean_abs_diff(a, d)
    same_arch_glb = mean_abs_diff(c, d)

    print(f"\n{'='*60}")
    print(f"SUMMARY (mean absolute diff over {STEPS} steps):")
    print(f"{'='*60}")
    print(f"  Standard seed42 vs Standard seed123:  {same_arch_diff_seed:.4f}  (same arch, diff seed)")
    print(f"  Standard seed42 vs Global seed42:     {diff_arch_same_seed:.4f}  (diff arch, same seed)")
    print(f"  Standard seed42 vs Global seed123:    {diff_arch_diff_seed:.4f}  (diff arch, diff seed)")
    print(f"  Global seed42 vs Global seed123:      {same_arch_glb:.4f}  (same arch, diff seed)")
    print()

    if same_arch_diff_seed > diff_arch_same_seed:
        print("  >>> SAME arch with different seeds diverges MORE than different archs!")
        print("  >>> This means the architectural difference is SMALLER than random init variance.")
        print("  >>> The models are effectively equivalent at this scale.")
    else:
        print("  >>> Different archs diverge MORE than same arch with different seeds.")
        print("  >>> The architectural difference IS meaningful.")


if __name__ == "__main__":
    main()
