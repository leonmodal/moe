"""
XS model, 100 steps, NO liger kernel.
Correct weight comparison.
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

STEPS = 100
DEVICE = "cuda"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_config(config_path, steps=STEPS):
    set_seed(42)
    cfg = load_cfg(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE)

    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    # Snapshot ALL params with their norms
    init_weights = {}
    for n, p in model.named_parameters():
        init_weights[n] = p.data.clone().cpu()

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

    final_weights = {}
    for n, p in model.named_parameters():
        final_weights[n] = p.data.clone().cpu()

    del model, optimizer
    torch.cuda.empty_cache()
    return losses, init_weights, final_weights


def classify_param(name):
    if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attention"
    if "gate_up_proj" in name or ("down_proj" in name and "expert" in name):
        return "expert"
    if "down_proj" in name and "expert" not in name:
        # This could be a dense MLP down_proj — check if it's in an expert module
        return "expert"  # all down_proj in this codebase are expert
    if "gate.weight" in name:
        return "router"
    if "embed" in name or "lm_head" in name:
        return "embedding"
    if "norm" in name or "layernorm" in name:
        return "norm"
    return "other"


def main():
    print("=" * 80)
    print("XS model, 100 steps, NO liger kernel, lr=1e-3, no warmup")
    print("=" * 80)

    configs = [
        ("Standard", "configs/scaling/xs_deepseek_standard.yaml"),
        ("Global", "configs/scaling/xs_deepseek_global.yaml"),
    ]

    all_losses = {}
    all_init = {}
    all_final = {}

    for name, path in configs:
        print(f"\nRunning: {name}")
        losses, init_w, final_w = run_config(path)
        all_losses[name] = losses
        all_init[name] = init_w
        all_final[name] = final_w
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # === 1. Initial weights ===
    print("\n" + "=" * 80)
    print("1. INITIAL WEIGHTS — are they the same?")
    print("=" * 80)
    std_init = all_init["Standard"]
    glb_init = all_init["Global"]

    print(f"\n  Standard params: {len(std_init)}")
    print(f"  Global params:   {len(glb_init)}")

    # Check shared param names
    common = set(std_init.keys()) & set(glb_init.keys())
    std_only = set(std_init.keys()) - set(glb_init.keys())
    glb_only = set(glb_init.keys()) - set(std_init.keys())

    print(f"\n  Common param names: {len(common)}")
    print(f"  Standard-only params: {len(std_only)}")
    for k in sorted(std_only)[:5]:
        print(f"    {k}: {std_init[k].shape}")
    print(f"  Global-only params: {len(glb_only)}")
    for k in sorted(glb_only)[:5]:
        print(f"    {k}: {glb_init[k].shape}")

    # Check if common params are equal
    print(f"\n  Are common params initialized identically?")
    n_equal = 0
    n_diff = 0
    for k in sorted(common):
        if std_init[k].shape == glb_init[k].shape:
            if torch.equal(std_init[k], glb_init[k]):
                n_equal += 1
            else:
                n_diff += 1
                if n_diff <= 3:
                    d = (std_init[k].float() - glb_init[k].float()).abs().max().item()
                    print(f"    DIFFERENT: {k} max_diff={d:.8f}")
    print(f"    Equal: {n_equal}, Different: {n_diff}")

    # === 2. Losses ===
    print("\n" + "=" * 80)
    print("2. PER-STEP LOSSES (100 steps)")
    print("=" * 80)

    print(f"\n{'Step':>5} {'Standard':>12} {'Global':>12} {'Diff':>10}")
    print("-" * 42)
    for i in range(STEPS):
        s = all_losses["Standard"][i]
        g = all_losses["Global"][i]
        d = abs(s - g)
        if i < 10 or i % 10 == 9:
            print(f"{i+1:>5} {s:>12.4f} {g:>12.4f} {d:>10.4f}")

    diffs = [abs(all_losses["Standard"][i] - all_losses["Global"][i]) for i in range(STEPS)]
    print(f"\n  Mean diff (all):      {sum(diffs)/len(diffs):.4f}")
    print(f"  Mean diff (first 10): {sum(diffs[:10])/10:.4f}")
    print(f"  Mean diff (last 10):  {sum(diffs[-10:])/10:.4f}")
    print(f"  Max diff:             {max(diffs):.4f}")

    # === 3. Weight deltas ===
    print("\n" + "=" * 80)
    print("3. WEIGHT CHANGES after 100 steps")
    print("=" * 80)

    for model_name in ["Standard", "Global"]:
        init = all_init[model_name]
        final = all_final[model_name]

        deltas = {}
        for k in init:
            d = (final[k].float() - init[k].float()).norm().item()
            cat = classify_param(k)
            deltas.setdefault(cat, []).append((k, d))

        print(f"\n  {model_name}:")
        for cat in ["attention", "expert", "router", "embedding", "norm", "other"]:
            if cat in deltas:
                total_delta = sum(d for _, d in deltas[cat])
                count = len(deltas[cat])
                print(f"    {cat:>12}: {count:>3} params, total_delta={total_delta:>10.2f}")
                # Show individual expert params
                if cat == "expert":
                    for k, d in deltas[cat]:
                        print(f"      {k}: delta={d:.4f}")

    # === 4. Are final weights same? ===
    print("\n" + "=" * 80)
    print("4. Are FINAL weights same between Standard and Global?")
    print("=" * 80)

    std_final = all_final["Standard"]
    glb_final = all_final["Global"]

    for k in sorted(common):
        if std_final[k].shape == glb_final[k].shape:
            d = (std_final[k].float() - glb_final[k].float()).norm().item()
            if "layers.0" in k and ("q_proj" in k or "embed" in k or "norm" in k):
                print(f"  {k}: norm_diff={d:.4f}")

    # Embedding comparison
    for k in sorted(common):
        if "embed_tokens" in k:
            d = (std_final[k].float() - glb_final[k].float()).norm().item()
            print(f"  {k}: norm_diff={d:.4f}")


if __name__ == "__main__":
    main()
