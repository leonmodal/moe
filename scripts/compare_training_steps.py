"""
Run 50 real training steps with each config and compare step-by-step losses.
Uses the actual training pipeline (real data, real dataloader).
"""
import sys
import os
import copy
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models.load_balancing import seq_load_balancing_loss_func

# Import the same build_model from train.py
from train import build_model, load_config

CONFIGS = [
    "configs/scaling/xs_deepseek_standard.yaml",
    "configs/scaling/xs_deepseek_global.yaml",
    "configs/scaling/xs_deepseek_global_nointerp.yaml",
]

STEPS = 50
DEVICE = "cuda"


def run_config(config_path, steps=STEPS):
    """Run N training steps and return per-step losses."""
    set_seed(42)

    cfg = load_config(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE)

    # Attach seq_aux_loss_coef
    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    # Data
    dcfg = cfg.get("data", {})
    data_cfg = DataConfig(
        data_dir=dcfg["data_dir"],
        text_column=dcfg.get("text_column", "text"),
        seq_len=dcfg.get("seq_len", 1024),
        tokenizer_name=dcfg.get("tokenizer_name", "Qwen/Qwen3-0.6B"),
    )
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StatefulParquetDataset(
        config=data_cfg, tokenizer=tokenizer, rank=0, world_size=1,
    )
    dataloader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=0, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(cfg["training"].get("beta1", 0.9), cfg["training"].get("beta2", 0.95)),
    )

    model.train()
    scaler = torch.amp.GradScaler("cuda")
    data_iter = iter(dataloader)

    losses = []
    ce_losses = []

    for step in range(steps):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
            loss = output.loss

            # Add seq aux loss (same as train.py)
            if seq_aux_loss_coef > 0 and output.router_logits is not None:
                seq_aux = seq_load_balancing_loss_func(
                    output.router_logits,
                    model_cfg.num_experts,
                    model_cfg.num_experts_per_tok,
                    batch_size=input_ids.shape[0],
                )
                loss = loss + seq_aux_loss_coef * seq_aux

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss = loss.item()
        losses.append(total_loss)

        # Compute CE only (subtract aux components)
        with torch.no_grad():
            aux = getattr(output, "aux_loss", None)
            aux_val = aux.item() if aux is not None else 0.0
            aux_coef = getattr(model, "router_aux_loss_coef", 0.0)
            ce = total_loss - aux_coef * aux_val
            if seq_aux_loss_coef > 0 and output.router_logits is not None:
                seq_aux_val = seq_load_balancing_loss_func(
                    output.router_logits, model_cfg.num_experts,
                    model_cfg.num_experts_per_tok, batch_size=input_ids.shape[0],
                )
                sav = seq_aux_val.item() if isinstance(seq_aux_val, torch.Tensor) else seq_aux_val
                ce -= seq_aux_loss_coef * sav
            ce_losses.append(ce)

    return losses, ce_losses


def main():
    print("Running 50 real training steps for each config...")
    print("=" * 90)

    all_losses = {}
    all_ce = {}

    for cfg_path in CONFIGS:
        name = Path(cfg_path).stem
        print(f"\nTraining: {name}")
        losses, ce = run_config(cfg_path)
        all_losses[name] = losses
        all_ce[name] = ce
        print(f"  Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")

    names = list(all_losses.keys())

    # Print step-by-step comparison
    print("\n" + "=" * 90)
    print(f"{'Step':>5s}", end="")
    for n in names:
        short = n.replace("xs_deepseek_", "")[:12]
        print(f"  {short:>12s}", end="")
    # Print pairwise diffs
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            si = names[i].replace("xs_deepseek_", "")[:6]
            sj = names[j].replace("xs_deepseek_", "")[:6]
            print(f"  {si}-{sj:>6s}", end="")
    print()
    print("-" * 90)

    for step in range(STEPS):
        print(f"{step:>5d}", end="")
        for n in names:
            print(f"  {all_losses[n][step]:>12.6f}", end="")
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                diff = abs(all_losses[names[i]][step] - all_losses[names[j]][step])
                print(f"  {diff:>12.6f}", end="")
        print()

    # Summary stats
    print("\n" + "=" * 90)
    print("SUMMARY: Pairwise loss differences over 50 steps")
    print("=" * 90)
    import numpy as np
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            diffs = [abs(all_losses[names[i]][s] - all_losses[names[j]][s]) for s in range(STEPS)]
            corr = np.corrcoef(all_losses[names[i]], all_losses[names[j]])[0, 1]
            print(f"  {names[i]} vs {names[j]}:")
            print(f"    Mean diff: {np.mean(diffs):.6f}")
            print(f"    Max diff:  {np.max(diffs):.6f}")
            print(f"    Correlation: {corr:.8f}")

    # Also print CE losses
    print("\n" + "=" * 90)
    print("CE LOSS (excluding aux) comparison")
    print("=" * 90)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            diffs = [abs(all_ce[names[i]][s] - all_ce[names[j]][s]) for s in range(STEPS)]
            corr = np.corrcoef(all_ce[names[i]], all_ce[names[j]])[0, 1]
            print(f"  {names[i]} vs {names[j]}:")
            print(f"    Mean CE diff: {np.mean(diffs):.6f}")
            print(f"    Max CE diff:  {np.max(diffs):.6f}")
            print(f"    CE Correlation: {corr:.8f}")


if __name__ == "__main__":
    main()
