"""
1000-step comparison: Dense vs Standard MoE vs Global MoE.
batch_size=32, lr=1e-3, no warmup, real data.
Prints every 50 steps + saves full results to CSV.
"""
import sys, csv, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
apply_liger_kernel_to_qwen3_moe()

from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models.load_balancing import seq_load_balancing_loss_func
from train import build_model, load_config

STEPS = 1000
DEVICE = "cuda"
BATCH_SIZE = 32
LR = 1e-3


def run(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE).train()
    is_dense = cfg["model"]["type"] == "dense"
    if is_dense:
        model.gradient_checkpointing_enable()

    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params/1e6:.0f}M")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = StatefulParquetDataset(
        config=DataConfig(data_dir="./data/parquet", text_column="text",
                          seq_len=1024, tokenizer_name="Qwen/Qwen3-0.6B"),
        tokenizer=tokenizer, rank=0, world_size=1, seed=42)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    data_iter = iter(dataloader)
    losses = []

    for step in range(STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()

        kwargs = {"input_ids": input_ids, "labels": labels}
        if not is_dense:
            kwargs["output_router_logits"] = True

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(**kwargs)
            loss = output.loss
            if not is_dense and seq_aux_loss_coef > 0 and getattr(output, "router_logits", None) is not None:
                seq_aux = seq_load_balancing_loss_func(
                    output.router_logits, model_cfg.num_experts,
                    model_cfg.num_experts_per_tok, batch_size=input_ids.shape[0])
                loss = loss + seq_aux_loss_coef * seq_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"    step {step+1:5d}: loss={losses[-1]:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()
    return losses


def main():
    configs = [
        ("dense", "configs/scaling/xs_dense_baseline.yaml"),
        ("standard", "configs/scaling/xs_deepseek_standard.yaml"),
        ("global", "configs/scaling/xs_deepseek_global.yaml"),
    ]

    all_losses = {}
    for name, path in configs:
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"{'='*40}")
        all_losses[name] = run(path)

    # Save to CSV
    csv_path = "outputs/compare_1000_steps.csv"
    Path("outputs").mkdir(exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "dense", "standard", "global"])
        for i in range(STEPS):
            w.writerow([i + 1, all_losses["dense"][i], all_losses["standard"][i], all_losses["global"][i]])
    print(f"\nSaved to {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Step':>6} {'Dense':>10} {'Standard':>10} {'Global':>10} | {'D-Std':>8} {'D-Glb':>8} {'Std-Glb':>8}")
    print("-" * 80)
    for i in range(STEPS):
        if (i + 1) % 50 == 0 or i < 10:
            d, s, g = all_losses["dense"][i], all_losses["standard"][i], all_losses["global"][i]
            print(f"{i+1:>6} {d:>10.4f} {s:>10.4f} {g:>10.4f} | {abs(d-s):>8.4f} {abs(d-g):>8.4f} {abs(s-g):>8.4f}")

    # Summary
    ds = [abs(all_losses["dense"][i] - all_losses["standard"][i]) for i in range(STEPS)]
    dg = [abs(all_losses["dense"][i] - all_losses["global"][i]) for i in range(STEPS)]
    sg = [abs(all_losses["standard"][i] - all_losses["global"][i]) for i in range(STEPS)]

    for window_start, window_end, label in [(0, 100, "first 100"), (400, 600, "mid 400-600"), (900, 1000, "last 100")]:
        ds_w = ds[window_start:window_end]
        dg_w = dg[window_start:window_end]
        sg_w = sg[window_start:window_end]
        print(f"\n  {label} steps:")
        print(f"    Dense vs Standard: {sum(ds_w)/len(ds_w):.4f}")
        print(f"    Dense vs Global:   {sum(dg_w)/len(dg_w):.4f}")
        print(f"    Standard vs Global:{sum(sg_w)/len(sg_w):.4f}")


if __name__ == "__main__":
    main()
