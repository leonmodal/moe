"""
Crank up the LR to amplify architectural differences.
Test lr=1e-2, 1e-1, and even 1.0 to see if models diverge.
"""
import sys, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from train import build_model, load_config

STEPS = 50
DEVICE = "cuda"


def run(config_path, lr, seed=42):
    set_seed(seed)
    cfg = load_config(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE).train()
    model._seq_aux_loss_coef = 0.0001

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = StatefulParquetDataset(
        config=DataConfig(data_dir="./data/parquet", text_column="text",
                          seq_len=1024, tokenizer_name="Qwen/Qwen3-0.6B"),
        tokenizer=tokenizer, rank=0, world_size=1, seed=42)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    data_iter = iter(dataloader)
    losses = []

    for step in range(STEPS):
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


def main():
    std_cfg = "configs/scaling/xs_deepseek_standard.yaml"
    glb_cfg = "configs/scaling/xs_deepseek_global.yaml"

    for lr in [1e-3, 1e-2, 1e-1]:
        print(f"\n{'='*70}")
        print(f"LR = {lr}")
        print(f"{'='*70}")

        std = run(std_cfg, lr=lr)
        glb = run(glb_cfg, lr=lr)

        diffs = [abs(std[i] - glb[i]) for i in range(STEPS)]

        print(f"  Standard: {std[0]:.4f} -> {std[-1]:.4f}")
        print(f"  Global:   {glb[0]:.4f} -> {glb[-1]:.4f}")
        print(f"  Mean diff: {sum(diffs)/len(diffs):.4f}")
        print(f"  Max diff:  {max(diffs):.4f}")
        print()
        print(f"  {'Step':>5} {'Standard':>10} {'Global':>10} {'Diff':>10}")
        print(f"  {'-'*38}")
        for i in range(STEPS):
            if i < 10 or i % 5 == 4:
                print(f"  {i+1:>5} {std[i]:>10.4f} {glb[i]:>10.4f} {diffs[i]:>10.4f}")


if __name__ == "__main__":
    main()
