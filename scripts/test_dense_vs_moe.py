"""
The real test: does MoE actually do anything different from dense?
If dense has the same loss curve as both MoE variants, the experts are useless.
If dense differs but both MoEs match, the MoE architecture is working but
standard vs global happens to converge the same way.

Also test: what does the loss look like with NO expert layer (zero out MoE output)?
"""
import sys, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models.load_balancing import seq_load_balancing_loss_func
from train import build_model, load_config

STEPS = 50
DEVICE = "cuda"


def run(config_path, seed=42, batch_size=64):
    set_seed(seed)
    cfg = load_config(config_path)
    # Force same batch size for fair comparison
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["gradient_accumulation"] = 1
    model, model_cfg = build_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    active_params = total_params  # for dense
    if hasattr(model_cfg, 'num_experts') and model_cfg.num_experts > 0:
        expert_total = sum(p.numel() for n, p in model.named_parameters()
                          if "gate_up_proj" in n or "down_proj" in n)
        # Active params = total - expert_total + (expert_total / num_experts * num_experts_per_tok)
        per_expert = expert_total / model_cfg.num_experts if model_cfg.num_experts > 0 else 0
        active_expert = per_expert * model_cfg.num_experts_per_tok
        active_params = total_params - expert_total + active_expert

    print(f"  Total: {total_params/1e6:.0f}M, Active: {active_params/1e6:.0f}M")

    model = model.to(DEVICE).train()
    is_dense = cfg["model"]["type"] == "dense"
    if is_dense:
        model.gradient_checkpointing_enable()

    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = StatefulParquetDataset(
        config=DataConfig(data_dir="./data/parquet", text_column="text",
                          seq_len=1024, tokenizer_name="Qwen/Qwen3-0.6B"),
        tokenizer=tokenizer, rank=0, world_size=1, seed=42)  # same data seed always
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
    data_iter = iter(dataloader)
    losses = []

    for step in range(STEPS):
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

    del model, optimizer
    torch.cuda.empty_cache()
    return losses


def main():
    configs = [
        ("Dense", "configs/scaling/xs_dense_baseline.yaml"),
        ("Standard MoE", "configs/scaling/xs_deepseek_standard.yaml"),
        ("Global MoE", "configs/scaling/xs_deepseek_global.yaml"),
    ]

    all_losses = {}
    for name, path in configs:
        print(f"\n{name}:")
        all_losses[name] = run(path, batch_size=32)
        print(f"  Loss: {all_losses[name][0]:.4f} -> {all_losses[name][-1]:.4f}")

    names = list(all_losses.keys())
    print(f"\n{'='*80}")
    print(f"{'Step':>5}", end="")
    for n in names:
        print(f" {n:>14}", end="")
    print(f" {'D-Std':>8} {'D-Glb':>8} {'Std-Glb':>8}")
    print("-" * 80)

    for i in range(STEPS):
        if i < 15 or i % 5 == 4:
            d, s, g = all_losses["Dense"][i], all_losses["Standard MoE"][i], all_losses["Global MoE"][i]
            print(f"{i+1:>5} {d:>14.4f} {s:>14.4f} {g:>14.4f} {abs(d-s):>8.4f} {abs(d-g):>8.4f} {abs(s-g):>8.4f}")

    # Summary
    ds = [abs(all_losses["Dense"][i] - all_losses["Standard MoE"][i]) for i in range(STEPS)]
    dg = [abs(all_losses["Dense"][i] - all_losses["Global MoE"][i]) for i in range(STEPS)]
    sg = [abs(all_losses["Standard MoE"][i] - all_losses["Global MoE"][i]) for i in range(STEPS)]

    print(f"\n{'='*60}")
    print(f"Mean absolute diffs over {STEPS} steps:")
    print(f"  Dense vs Standard MoE:    {sum(ds)/len(ds):.4f}")
    print(f"  Dense vs Global MoE:      {sum(dg)/len(dg):.4f}")
    print(f"  Standard vs Global MoE:   {sum(sg)/len(sg):.4f}")
    print()

    if sum(ds)/len(ds) > sum(sg)/len(sg) * 3:
        print("  >>> Dense is clearly different from MoE — experts ARE doing something.")
        print("  >>> But Standard and Global MoE converge similarly.")
    elif sum(ds)/len(ds) < sum(sg)/len(sg) * 1.5:
        print("  >>> Dense loss is similar to MoE — experts may not be contributing!")


if __name__ == "__main__":
    main()
