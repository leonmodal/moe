"""
XS model, 100 steps, NO liger kernel.
Compare standard vs global:
  1. Are initial weights the same?
  2. Per-step loss divergence
  3. Are weights the same after training?
"""
import sys, torch, copy
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NO liger kernel import

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


def snapshot_weights(model):
    """Return a dict of param_name -> cloned tensor."""
    return {n: p.data.clone().cpu() for n, p in model.named_parameters()}


def compare_weights(w1, w2, label1, label2):
    """Compare two weight dicts, find shared param names and check equality."""
    # Find params with matching shapes (ignore name differences)
    # Group by role: attention, embedding, expert, router
    roles = {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "norm": ["layernorm", "norm"],
        "embedding": ["embed_tokens", "lm_head"],
        "expert": ["gate_up_proj", "down_proj"],
        "router": ["gate.weight"],
    }

    print(f"\n  Weight comparison: {label1} vs {label2}")
    print(f"  {'Role':<15} {'Param':<50} {'Shape':>20} {'Equal?':>8} {'MaxDiff':>12}")
    print("  " + "-" * 110)

    for role, keywords in roles.items():
        for n1, p1 in w1.items():
            for kw in keywords:
                if kw in n1:
                    # Find matching param in w2
                    # For standard vs global, names differ but structure is similar
                    # Match by: same layer index + same role
                    matched = False
                    for n2, p2 in w2.items():
                        if kw in n2 and p1.shape == p2.shape:
                            # Try to match layer index
                            l1 = [s for s in n1.split(".") if s.isdigit()]
                            l2 = [s for s in n2.split(".") if s.isdigit()]
                            if l1 == l2 or (not l1 and not l2):
                                is_equal = torch.equal(p1, p2)
                                max_diff = (p1 - p2).abs().max().item()
                                print(f"  {role:<15} {n1[:50]:<50} {str(list(p1.shape)):>20} {'YES' if is_equal else 'NO':>8} {max_diff:>12.8f}")
                                matched = True
                                break
                    if matched:
                        break  # only show one param per role per layer
            else:
                continue
            break  # only show first match per role


def run_config(config_path, steps=STEPS):
    set_seed(42)
    cfg = load_cfg(config_path)
    model, model_cfg = build_model(cfg)
    model = model.to(DEVICE)

    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    # Snapshot initial weights
    init_weights = snapshot_weights(model)

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

    final_weights = snapshot_weights(model)

    del model, optimizer
    torch.cuda.empty_cache()
    return losses, init_weights, final_weights


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

    # === 1. Compare initial weights ===
    print("\n" + "=" * 80)
    print("1. INITIAL WEIGHT COMPARISON")
    print("=" * 80)

    # Check attention weights (should be DIFFERENT - different seed consumption order)
    std_init = all_init["Standard"]
    glb_init = all_init["Global"]

    print("\n  Checking if specific layers share initial weights...")
    # Attention layer 0
    for key_part in ["self_attn.q_proj.weight", "self_attn.k_proj.weight"]:
        std_key = [k for k in std_init if key_part in k and ".0." in k]
        glb_key = [k for k in glb_init if key_part in k and ".0." in k]
        if std_key and glb_key:
            s, g = std_init[std_key[0]], glb_init[glb_key[0]]
            eq = torch.equal(s, g)
            md = (s - g).abs().max().item()
            print(f"  Layer 0 {key_part}: equal={eq}, max_diff={md:.8f}")

    # Embedding
    for key_part in ["embed_tokens.weight"]:
        std_key = [k for k in std_init if key_part in k]
        glb_key = [k for k in glb_init if key_part in k]
        if std_key and glb_key:
            s, g = std_init[std_key[0]], glb_init[glb_key[0]]
            eq = torch.equal(s, g)
            md = (s - g).abs().max().item()
            print(f"  {key_part}: equal={eq}, max_diff={md:.8f}")

    # Router layer 0
    for key_part in ["gate.weight"]:
        std_key = [k for k in std_init if key_part in k and ".0." in k]
        glb_key = [k for k in glb_init if key_part in k and ".0." in k]
        if std_key and glb_key:
            s, g = std_init[std_key[0]], glb_init[glb_key[0]]
            print(f"  Layer 0 router: std shape={s.shape}, glb shape={g.shape}")
            if s.shape == g.shape:
                eq = torch.equal(s, g)
                md = (s - g).abs().max().item()
                print(f"    equal={eq}, max_diff={md:.8f}")
            else:
                print(f"    Different shapes (expected: std has {s.shape[0]} experts, glb has {g.shape[0]})")

    # Expert weights
    std_expert_keys = [k for k in std_init if "gate_up_proj" in k]
    glb_expert_keys = [k for k in glb_init if "gate_up_proj" in k]
    print(f"\n  Expert param tensors: standard has {len(std_expert_keys)}, global has {len(glb_expert_keys)}")
    for k in std_expert_keys[:2]:
        print(f"    Standard: {k} shape={std_init[k].shape}")
    for k in glb_expert_keys[:2]:
        print(f"    Global:   {k} shape={glb_init[k].shape}")

    # === 2. Per-step losses ===
    print("\n" + "=" * 80)
    print("2. PER-STEP LOSS COMPARISON (100 steps)")
    print("=" * 80)

    names = list(all_losses.keys())
    print(f"\n{'Step':>5} {'Standard':>12} {'Global':>12} {'Diff':>10}")
    print("-" * 42)
    for i in range(STEPS):
        s = all_losses["Standard"][i]
        g = all_losses["Global"][i]
        d = abs(s - g)
        if i < 20 or i % 10 == 9:
            print(f"{i+1:>5} {s:>12.4f} {g:>12.4f} {d:>10.4f}")

    diffs = [abs(all_losses["Standard"][i] - all_losses["Global"][i]) for i in range(STEPS)]
    print(f"\n  Mean diff: {sum(diffs)/len(diffs):.4f}")
    print(f"  Max diff:  {max(diffs):.4f}")
    print(f"  Mean diff (first 10): {sum(diffs[:10])/10:.4f}")
    print(f"  Mean diff (last 10):  {sum(diffs[-10:])/10:.4f}")

    # === 3. Compare final weights ===
    print("\n" + "=" * 80)
    print("3. FINAL WEIGHT COMPARISON (after 100 steps)")
    print("=" * 80)

    std_final = all_final["Standard"]
    glb_final = all_final["Global"]

    # How much did each model's weights change?
    print("\n  Weight change magnitude (L2 norm of delta):")
    for name, init, final in [("Standard", std_init, std_final), ("Global", glb_init, glb_final)]:
        attn_delta = expert_delta = router_delta = embed_delta = 0.0
        for k in init:
            d = (final[k] - init[k]).float().norm().item()
            if any(x in k for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                attn_delta += d
            elif "gate_up_proj" in k or "down_proj" in k:
                expert_delta += d
            elif "gate.weight" in k:
                router_delta += d
            elif "embed" in k or "lm_head" in k:
                embed_delta += d
        print(f"  {name:>10}: attn={attn_delta:.2f}, expert={expert_delta:.2f}, router={router_delta:.2f}, embed={embed_delta:.2f}")

    # Are the final weights the same between models?
    print("\n  Are final attention weights same between standard and global?")
    for key_part in ["self_attn.q_proj.weight", "self_attn.k_proj.weight"]:
        std_key = [k for k in std_final if key_part in k and ".0." in k]
        glb_key = [k for k in glb_final if key_part in k and ".0." in k]
        if std_key and glb_key:
            s, g = std_final[std_key[0]], glb_final[glb_key[0]]
            md = (s - g).abs().max().item()
            mn = (s - g).float().norm().item()
            print(f"    Layer 0 {key_part}: max_diff={md:.6f}, norm_diff={mn:.4f}")

    print("\n  Are final embedding weights same?")
    std_key = [k for k in std_final if "embed_tokens" in k]
    glb_key = [k for k in glb_final if "embed_tokens" in k]
    if std_key and glb_key:
        s, g = std_final[std_key[0]], glb_final[glb_key[0]]
        md = (s - g).abs().max().item()
        mn = (s - g).float().norm().item()
        print(f"    embed_tokens: max_diff={md:.6f}, norm_diff={mn:.4f}")


if __name__ == "__main__":
    main()
