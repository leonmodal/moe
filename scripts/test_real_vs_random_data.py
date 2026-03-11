"""
Does real language data produce less divergence than random tokens?
Same models, same lr, same steps — only difference is the data.
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
BATCH_SIZE = 64
SEQ_LEN = 1024


def run(config_path, use_real_data, seed=42):
    set_seed(seed)
    cfg = load_config(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE).train()
    model._seq_aux_loss_coef = 0.0001

    if use_real_data:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        tokenizer.pad_token = tokenizer.eos_token
        dataset = StatefulParquetDataset(
            config=DataConfig(data_dir="./data/parquet", text_column="text",
                              seq_len=SEQ_LEN, tokenizer_name="Qwen/Qwen3-0.6B"),
            tokenizer=tokenizer, rank=0, world_size=1, seed=seed)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        data_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
    losses = []

    # Pre-generate random batches so both models see the SAME random data
    if not use_real_data:
        torch.manual_seed(999)  # fixed seed for random data, separate from model seed
        random_batches = []
        for _ in range(STEPS):
            ids = torch.randint(0, 151936, (BATCH_SIZE, SEQ_LEN + 1), device=DEVICE)
            random_batches.append({
                "input_ids": ids[:, :-1],
                "labels": ids[:, 1:],
            })

    for step in range(STEPS):
        if use_real_data:
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
        else:
            input_ids = random_batches[step]["input_ids"]
            labels = random_batches[step]["labels"]

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

    print("Running 4 experiments (50 steps each, lr=1e-3, no warmup)...")
    print()

    std_real = run(std_cfg, use_real_data=True)
    print(f"  Standard + real data:   {std_real[0]:.4f} -> {std_real[-1]:.4f}")
    glb_real = run(glb_cfg, use_real_data=True)
    print(f"  Global + real data:     {glb_real[0]:.4f} -> {glb_real[-1]:.4f}")
    std_rand = run(std_cfg, use_real_data=False)
    print(f"  Standard + random data: {std_rand[0]:.4f} -> {std_rand[-1]:.4f}")
    glb_rand = run(glb_cfg, use_real_data=False)
    print(f"  Global + random data:   {glb_rand[0]:.4f} -> {glb_rand[-1]:.4f}")

    # Compare
    print("\n" + "=" * 90)
    print(f"{'Step':>5} {'Real: Std':>10} {'Real: Glb':>10} {'Real diff':>10} | "
          f"{'Rand: Std':>10} {'Rand: Glb':>10} {'Rand diff':>10}")
    print("-" * 90)
    for i in range(STEPS):
        rd = abs(std_real[i] - glb_real[i])
        ad = abs(std_rand[i] - glb_rand[i])
        if i < 15 or i % 5 == 4:
            print(f"{i+1:>5} {std_real[i]:>10.4f} {glb_real[i]:>10.4f} {rd:>10.4f} | "
                  f"{std_rand[i]:>10.4f} {glb_rand[i]:>10.4f} {ad:>10.4f}")

    real_diffs = [abs(std_real[i] - glb_real[i]) for i in range(STEPS)]
    rand_diffs = [abs(std_rand[i] - glb_rand[i]) for i in range(STEPS)]

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"  Real data:   mean_diff={sum(real_diffs)/len(real_diffs):.4f}, max={max(real_diffs):.4f}")
    print(f"  Random data: mean_diff={sum(rand_diffs)/len(rand_diffs):.4f}, max={max(rand_diffs):.4f}")
    print()
    if sum(rand_diffs) > sum(real_diffs) * 1.5:
        print("  >>> Random data produces MORE divergence than real data!")
        print("  >>> Real language data has structure that both models exploit similarly.")
    elif sum(real_diffs) > sum(rand_diffs) * 1.5:
        print("  >>> Real data produces MORE divergence than random data!")
    else:
        print("  >>> Similar divergence on both real and random data.")
        print("  >>> Data is NOT the issue.")


if __name__ == "__main__":
    main()
