"""
Test if liger kernel is making the models produce more similar outputs.
Run the same 30 steps with and without liger, compare the divergence.
"""
import sys
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models.load_balancing import seq_load_balancing_loss_func


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_pair(use_liger: bool, steps=30):
    """Run standard + global for N steps, return per-step CE losses."""
    if use_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
        apply_liger_kernel_to_qwen3_moe()

    # Must import AFTER liger patching
    from train import build_model

    results = {}
    for config_path in [
        "configs/scaling/xs_deepseek_standard.yaml",
        "configs/scaling/xs_deepseek_global.yaml",
    ]:
        set_seed(42)
        name = Path(config_path).stem.replace("xs_deepseek_", "")

        cfg = load_cfg(config_path)
        model, model_cfg = build_model(cfg)
        model = model.cuda()

        seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
        if seq_aux_loss_coef > 0:
            model._seq_aux_loss_coef = seq_aux_loss_coef

        # Same data pipeline as train.py
        dcfg = cfg.get("data", {})
        tokenizer = AutoTokenizer.from_pretrained(dcfg.get("tokenizer_name", "Qwen/Qwen3-0.6B"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = StatefulParquetDataset(
            config=DataConfig(data_dir=dcfg["data_dir"], text_column=dcfg.get("text_column", "text"),
                              seq_len=dcfg.get("seq_len", 1024), tokenizer_name=dcfg.get("tokenizer_name", "Qwen/Qwen3-0.6B")),
            tokenizer=tokenizer, rank=0, world_size=1,
        )
        dataloader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=0, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01,
                                       betas=(0.9, 0.95))

        model.train()
        data_iter = iter(dataloader)
        losses = []

        for step in range(steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, labels=labels, output_router_logits=True)
                loss = output.loss
                if seq_aux_loss_coef > 0 and output.router_logits is not None:
                    seq_aux = seq_load_balancing_loss_func(
                        output.router_logits, model_cfg.num_experts,
                        model_cfg.num_experts_per_tok, batch_size=input_ids.shape[0])
                    loss = loss + seq_aux_loss_coef * seq_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        results[name] = losses
        del model, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    # Run WITHOUT liger first (in a subprocess to avoid contamination)
    # Since liger monkey-patches globally, we test liger=True in this process
    # and compare with values from a no-liger subprocess
    import subprocess, json

    # No-liger subprocess
    print("Running WITHOUT liger kernel...")
    result = subprocess.run(
        [sys.executable, "-c", """
import sys, json, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
sys.path.insert(0, ".")
from scripts.test_liger_effect import run_pair
results = run_pair(use_liger=False, steps=30)
# Convert to JSON-serializable
print(json.dumps({k: v for k, v in results.items()}))
"""],
        capture_output=True, text=True, cwd="/tmp/moe"
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        return
    no_liger = json.loads(result.stdout.strip().split('\n')[-1])

    print("Running WITH liger kernel...")
    result2 = subprocess.run(
        [sys.executable, "-c", """
import sys, json, torch
torch.backends.cuda.preferred_blas_library("cublaslt")
sys.path.insert(0, ".")
from scripts.test_liger_effect import run_pair
results = run_pair(use_liger=True, steps=30)
print(json.dumps({k: v for k, v in results.items()}))
"""],
        capture_output=True, text=True, cwd="/tmp/moe"
    )
    if result2.returncode != 0:
        print("STDERR:", result2.stderr[-2000:])
        return
    with_liger = json.loads(result2.stdout.strip().split('\n')[-1])

    # Compare
    print("\n" + "=" * 80)
    print("WITHOUT liger (lr=1e-3, no warmup):")
    print("=" * 80)
    print("{:>5} {:>12} {:>12} {:>10}".format("Step", "Standard", "Global", "Diff"))
    for i in range(30):
        s, g = no_liger["standard"][i], no_liger["global"][i]
        print("{:>5} {:>12.4f} {:>12.4f} {:>10.4f}".format(i, s, g, abs(s-g)))

    no_liger_diffs = [abs(no_liger["standard"][i] - no_liger["global"][i]) for i in range(30)]

    print("\n" + "=" * 80)
    print("WITH liger (lr=1e-3, no warmup):")
    print("=" * 80)
    print("{:>5} {:>12} {:>12} {:>10}".format("Step", "Standard", "Global", "Diff"))
    for i in range(30):
        s, g = with_liger["standard"][i], with_liger["global"][i]
        print("{:>5} {:>12.4f} {:>12.4f} {:>10.4f}".format(i, s, g, abs(s-g)))

    liger_diffs = [abs(with_liger["standard"][i] - with_liger["global"][i]) for i in range(30)]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("No liger:   mean_diff={:.4f}, max_diff={:.4f}".format(
        sum(no_liger_diffs)/len(no_liger_diffs), max(no_liger_diffs)))
    print("With liger: mean_diff={:.4f}, max_diff={:.4f}".format(
        sum(liger_diffs)/len(liger_diffs), max(liger_diffs)))


if __name__ == "__main__":
    main()
