"""
Check if hidden states are nearly identical between standard and global models
during forward pass. If they are, that explains why losses match despite
different init.

Also: compare hidden states between TWO RUNS of the SAME model with different
seeds on the SAME data.
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

DEVICE = "cuda"


def get_hidden_states(model, input_ids):
    """Hook into every decoder layer to capture hidden states."""
    states = {}

    def make_hook(name):
        def hook(module, input, output):
            # Decoder layers return hidden_states tensor
            if isinstance(output, torch.Tensor):
                states[name] = output.detach().float()
            elif isinstance(output, tuple) and len(output) > 0:
                states[name] = output[0].detach().float() if isinstance(output[0], torch.Tensor) else None
        return hook

    hooks = []
    for name, module in model.named_modules():
        if "layers." in name and name.count(".") == 2 and "layers" in name:
            # e.g. "model.layers.0"
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Also hook the embedding
    def embed_hook(module, input, output):
        states["embedding"] = output.detach().float()
    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, output_router_logits=True)

    for h in hooks:
        h.remove()

    return states, out


def cosine_sim(a, b):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def main():
    # Get same input data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    cfg = DataConfig(data_dir="./data/parquet", text_column="text", seq_len=1024, tokenizer_name="Qwen/Qwen3-0.6B")
    ds = StatefulParquetDataset(config=cfg, tokenizer=tokenizer, rank=0, world_size=1, seed=42)
    dl = DataLoader(ds, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    input_ids = batch["input_ids"].to(DEVICE)

    # === Run 1: Standard seed=42 vs Global seed=42 (diff arch, same seed) ===
    set_seed(42)
    model_std, _ = build_model(load_config("configs/scaling/xs_deepseek_standard.yaml"))
    model_std = model_std.to(DEVICE).eval()

    set_seed(42)
    model_glb, _ = build_model(load_config("configs/scaling/xs_deepseek_global.yaml"))
    model_glb = model_glb.to(DEVICE).eval()

    states_std, out_std = get_hidden_states(model_std, input_ids)
    states_glb, out_glb = get_hidden_states(model_glb, input_ids)

    # === Run 2: Standard seed=42 vs Standard seed=123 (same arch, diff seed) ===
    set_seed(123)
    model_std2, _ = build_model(load_config("configs/scaling/xs_deepseek_standard.yaml"))
    model_std2 = model_std2.to(DEVICE).eval()

    states_std2, out_std2 = get_hidden_states(model_std2, input_ids)

    # === Compare hidden states layer by layer ===
    print("=" * 90)
    print("Hidden state comparison at each layer")
    print("=" * 90)
    print(f"{'Layer':<25} {'Std42 vs Glb42':>18} {'Std42 vs Std123':>18} {'Which closer?':>15}")
    print(f"{'':25} {'(diff arch)':>18} {'(diff seed)':>18}")
    print("-" * 90)

    # Compare embeddings first
    if "embedding" in states_std and "embedding" in states_glb and "embedding" in states_std2:
        cos_arch = cosine_sim(states_std["embedding"], states_glb["embedding"])
        cos_seed = cosine_sim(states_std["embedding"], states_std2["embedding"])
        l2_arch = (states_std["embedding"] - states_glb["embedding"]).norm().item()
        l2_seed = (states_std["embedding"] - states_std2["embedding"]).norm().item()
        closer = "arch" if l2_arch < l2_seed else "seed"
        print(f"{'embedding':<25} cos={cos_arch:.6f} L2={l2_arch:>8.1f}  cos={cos_seed:.6f} L2={l2_seed:>8.1f}  {closer}")

    # Compare each decoder layer
    common_layers = sorted(set(states_std.keys()) & set(states_glb.keys()) & set(states_std2.keys()))
    for layer_name in common_layers:
        s1 = states_std[layer_name]
        s2 = states_glb[layer_name]
        s3 = states_std2[layer_name]
        if s1 is not None and s2 is not None and s3 is not None:
            cos_arch = cosine_sim(s1, s2)
            cos_seed = cosine_sim(s1, s3)
            l2_arch = (s1 - s2).norm().item()
            l2_seed = (s1 - s3).norm().item()
            norm = s1.norm().item()
            closer = "arch" if l2_arch < l2_seed else "seed"
            print(f"{layer_name:<25} cos={cos_arch:.6f} L2={l2_arch:>8.1f}  cos={cos_seed:.6f} L2={l2_seed:>8.1f}  {closer}  (norm={norm:.0f})")

    # Compare final logits/loss
    print(f"\n{'Final loss':<25} std42={out_std.loss.item():.6f}  glb42={out_glb.loss.item():.6f}  std123={out_std2.loss.item():.6f}")
    print(f"{'':25} arch_diff={abs(out_std.loss.item()-out_glb.loss.item()):.6f}  seed_diff={abs(out_std.loss.item()-out_std2.loss.item()):.6f}")

    # === Key question: are the MoE layer outputs similar? ===
    print("\n" + "=" * 90)
    print("MoE FFN output analysis")
    print("=" * 90)

    # Hook into MLP specifically
    mlp_outs_std = []
    mlp_outs_glb = []
    mlp_outs_std2 = []

    def make_mlp_hook(storage):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage.append(output.detach().float())
        return hook

    for model, storage in [(model_std, mlp_outs_std), (model_glb, mlp_outs_glb), (model_std2, mlp_outs_std2)]:
        hooks = []
        for name, module in model.named_modules():
            if name.endswith(".mlp") and ("layers." in name):
                hooks.append(module.register_forward_hook(make_mlp_hook(storage)))
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=input_ids, output_router_logits=True)
        for h in hooks:
            h.remove()

    print(f"{'Layer':<10} {'MoE out: Std42 vs Glb42':>30} {'MoE out: Std42 vs Std123':>30}")
    print("-" * 75)
    for i in range(min(len(mlp_outs_std), len(mlp_outs_glb), len(mlp_outs_std2))):
        s1, s2, s3 = mlp_outs_std[i], mlp_outs_glb[i], mlp_outs_std2[i]
        cos_arch = cosine_sim(s1, s2)
        cos_seed = cosine_sim(s1, s3)
        l2_arch = (s1 - s2).norm().item()
        l2_seed = (s1 - s3).norm().item()
        print(f"Layer {i:<4} cos={cos_arch:.6f} L2={l2_arch:>8.1f}    cos={cos_seed:.6f} L2={l2_seed:>8.1f}")


if __name__ == "__main__":
    main()
