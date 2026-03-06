"""
Diagnose why standard vs global MoE produce near-identical loss curves.

Tests:
1. Are expert parameters actually different across models?
2. Does the MoE FFN output differ across architectures?
3. What fraction of the total loss comes from attention vs FFN?
4. Are router gradients flowing properly?
5. Are expert weights actually being updated?
"""
import sys
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from src.models import (
    Qwen3MoeConfig,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    DeepSeekGlobalMoEForCausalLM,
)
from src.models.router import DeepSeekRouter

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_standard(cfg):
    mcfg = cfg["model"]
    common = dict(
        vocab_size=mcfg["vocab_size"], hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"], head_dim=mcfg["head_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        moe_intermediate_size=mcfg["moe_intermediate_size"],
        intermediate_size=mcfg.get("intermediate_size", 3072),
        max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
        rope_theta=mcfg.get("rope_theta", 1e6), rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        router_aux_loss_coef=0.0, norm_topk_prob=True,
        num_experts_per_tok=mcfg["num_experts_per_tok"], output_router_logits=True,
    )
    config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
    config.topk_scaling_factor = mcfg.get("topk_scaling_factor")
    config.num_groups = mcfg.get("num_groups")
    config.group_topk = mcfg.get("group_topk")
    return DeepSeekStandardMoEModel(config), config


def build_global(cfg):
    mcfg = cfg["model"]
    common = dict(
        vocab_size=mcfg["vocab_size"], hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"], head_dim=mcfg["head_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        moe_intermediate_size=mcfg["moe_intermediate_size"],
        intermediate_size=mcfg.get("intermediate_size", 3072),
        max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
        rope_theta=mcfg.get("rope_theta", 1e6), rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        router_aux_loss_coef=0.0, norm_topk_prob=True,
        num_experts_per_tok=mcfg["num_experts_per_tok"], output_router_logits=True,
    )
    config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
    config.topk_scaling_factor = mcfg.get("topk_scaling_factor")
    config.num_groups = mcfg.get("num_groups")
    config.group_topk = mcfg.get("group_topk")
    return DeepSeekGlobalMoEForCausalLM(config), config


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for n, p in model.named_parameters()
                 if "gate_up_proj" in n or "down_proj" in n)
    attn = sum(p.numel() for n, p in model.named_parameters()
               if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj"]))
    router = sum(p.numel() for n, p in model.named_parameters()
                 if "gate.weight" in n)
    embed = sum(p.numel() for n, p in model.named_parameters()
                if "embed_tokens" in n or "lm_head" in n)
    return dict(total=total, expert=expert, attn=attn, router=router, embed=embed)


def analyze_expert_output(model, input_ids, model_name):
    """Hook into the MoE layers to measure expert output magnitudes."""
    moe_outputs = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            moe_outputs.append(output.detach())

    hooks = []
    for name, module in model.named_modules():
        if "mlp" in name and hasattr(module, "gate"):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)

    for h in hooks:
        h.remove()

    if moe_outputs:
        norms = [o.float().norm().item() for o in moe_outputs]
        print(f"  [{model_name}] MoE output norms per layer: "
              f"mean={sum(norms)/len(norms):.4f}, min={min(norms):.4f}, max={max(norms):.4f}")
    return output


def analyze_routing(model, input_ids, model_name):
    """Check what the routers are actually doing."""
    router_data = []

    def hook_fn(module, input, output):
        scores, weights, indices = output
        router_data.append({
            "scores": scores.detach(),
            "weights": weights.detach(),
            "indices": indices.detach(),
        })

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, DeepSeekRouter):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_ids=input_ids, output_router_logits=True)

    for h in hooks:
        h.remove()

    print(f"\n  [{model_name}] Router analysis ({len(router_data)} layers):")
    all_indices = []
    for i, rd in enumerate(router_data):
        scores = rd["scores"]
        weights = rd["weights"]
        indices = rd["indices"]
        unique_experts = indices.unique().numel()
        total_experts = scores.shape[1]

        print(f"    Layer {i}: scores range=[{scores.min():.4f}, {scores.max():.4f}], "
              f"score_std={scores.std():.6f}, "
              f"weight range=[{weights.min():.4f}, {weights.max():.4f}], "
              f"unique_experts_used={unique_experts}/{total_experts}")
        all_indices.append(indices)

    return router_data


def check_gradient_flow(model, input_ids, model_name):
    """Check if gradients actually flow to expert parameters."""
    model.zero_grad()
    output = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    loss = output.loss
    loss.backward()

    print(f"\n  [{model_name}] Gradient analysis:")
    print(f"    Total loss: {loss.item():.6f}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "gate_up_proj" in name or "down_proj" in name:
                print(f"    {name}: grad_norm={grad_norm:.6f}, param_norm={param.data.norm().item():.6f}")
                break  # just show one

    # Check router weight gradients
    for name, param in model.named_parameters():
        if "gate.weight" in name and param.grad is not None:
            print(f"    {name}: grad_norm={param.grad.norm().item():.6f}")
            break

    return loss.item()


def measure_attention_vs_ffn(model, input_ids, model_name):
    """Measure how much the FFN/MoE layer changes the hidden states."""
    residuals_before_ffn = []
    residuals_after_ffn = []

    # For standard model, hook into the decoder layer
    ffn_input_norms = []
    ffn_output_norms = []

    def pre_hook(module, input):
        if isinstance(input, tuple) and len(input) > 0:
            ffn_input_norms.append(input[0].detach().float().norm().item())

    def post_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            ffn_output_norms.append(output.detach().float().norm().item())

    hooks = []
    for name, module in model.named_modules():
        if "mlp" in name and hasattr(module, "gate"):
            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(post_hook))

    with torch.no_grad():
        model(input_ids=input_ids, output_router_logits=True)

    for h in hooks:
        h.remove()

    if ffn_input_norms and ffn_output_norms:
        ratios = [o / (i + 1e-8) for i, o in zip(ffn_input_norms, ffn_output_norms)]
        print(f"\n  [{model_name}] FFN output/input norm ratios per layer:")
        print(f"    mean={sum(ratios)/len(ratios):.4f}, "
              f"min={min(ratios):.4f}, max={max(ratios):.4f}")
        print(f"    FFN input norms: mean={sum(ffn_input_norms)/len(ffn_input_norms):.2f}")
        print(f"    FFN output norms: mean={sum(ffn_output_norms)/len(ffn_output_norms):.2f}")


def simulate_training(model, input_ids, steps=20, lr=1e-3, model_name=""):
    """Run a few training steps and track loss + expert param changes."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    # Snapshot initial expert params
    initial_expert_params = {}
    for name, param in model.named_parameters():
        if "gate_up_proj" in name or "down_proj" in name:
            initial_expert_params[name] = param.data.clone()
            if len(initial_expert_params) >= 2:
                break

    initial_router_params = {}
    for name, param in model.named_parameters():
        if "gate.weight" in name:
            initial_router_params[name] = param.data.clone()
            if len(initial_router_params) >= 2:
                break

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    print(f"\n  [{model_name}] Training {steps} steps:")
    print(f"    Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:.4f})")

    # Check how much expert params changed
    for name, init_p in initial_expert_params.items():
        curr_p = dict(model.named_parameters())[name]
        delta = (curr_p.data - init_p).norm().item()
        print(f"    Expert param delta ({name}): {delta:.6f}")

    for name, init_p in initial_router_params.items():
        curr_p = dict(model.named_parameters())[name]
        delta = (curr_p.data - init_p).norm().item()
        print(f"    Router param delta ({name}): {delta:.6f}")

    return losses


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Why are standard vs global MoE losses identical?")
    print("=" * 70)

    cfg_standard = load_cfg("configs/scaling/xs_deepseek_standard.yaml")
    cfg_global = load_cfg("configs/scaling/xs_deepseek_global.yaml")

    # Create fake input
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use real tokenized text for a more realistic test
    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    # ===== Test 1: Parameter counts =====
    print("\n" + "=" * 70)
    print("TEST 1: Parameter counts")
    print("=" * 70)

    torch.manual_seed(42)
    model_std, cfg_std = build_standard(cfg_standard)
    params_std = count_params(model_std)

    torch.manual_seed(42)
    model_glb, cfg_glb = build_global(cfg_global)
    params_glb = count_params(model_glb)

    for k in params_std:
        print(f"  {k:>10s}: standard={params_std[k]:>12,d}  global={params_glb[k]:>12,d}  "
              f"{'SAME' if params_std[k] == params_glb[k] else 'DIFF'}")

    # ===== Test 2: Forward pass comparison =====
    print("\n" + "=" * 70)
    print("TEST 2: Forward pass loss comparison (same input, different models)")
    print("=" * 70)

    model_std = model_std.to(device).eval()
    model_glb = model_glb.to(device).eval()

    with torch.no_grad():
        out_std = model_std(input_ids=input_ids, labels=input_ids, output_router_logits=True)
        out_glb = model_glb(input_ids=input_ids, labels=input_ids, output_router_logits=True)

    print(f"  Standard loss: {out_std.loss.item():.6f}")
    print(f"  Global loss:   {out_glb.loss.item():.6f}")
    print(f"  Difference:    {abs(out_std.loss.item() - out_glb.loss.item()):.6f}")

    # ===== Test 3: Router analysis =====
    print("\n" + "=" * 70)
    print("TEST 3: Router analysis — are routers actually routing differently?")
    print("=" * 70)

    rd_std = analyze_routing(model_std, input_ids, "standard")
    rd_glb = analyze_routing(model_glb, input_ids, "global")

    # ===== Test 4: MoE output magnitude =====
    print("\n" + "=" * 70)
    print("TEST 4: MoE output magnitude")
    print("=" * 70)

    analyze_expert_output(model_std, input_ids, "standard")
    analyze_expert_output(model_glb, input_ids, "global")

    # ===== Test 5: FFN contribution =====
    print("\n" + "=" * 70)
    print("TEST 5: FFN contribution relative to residual stream")
    print("=" * 70)

    measure_attention_vs_ffn(model_std, input_ids, "standard")
    measure_attention_vs_ffn(model_glb, input_ids, "global")

    # ===== Test 6: Gradient flow =====
    print("\n" + "=" * 70)
    print("TEST 6: Gradient flow through experts")
    print("=" * 70)

    model_std.train()
    model_glb.train()
    loss_std = check_gradient_flow(model_std, input_ids, "standard")
    loss_glb = check_gradient_flow(model_glb, input_ids, "global")

    # ===== Test 7: Short training run =====
    print("\n" + "=" * 70)
    print("TEST 7: Short training (20 steps) — do losses diverge?")
    print("=" * 70)

    torch.manual_seed(42)
    model_std2, _ = build_standard(cfg_standard)
    model_std2 = model_std2.to(device)

    torch.manual_seed(42)
    model_glb2, _ = build_global(cfg_global)
    model_glb2 = model_glb2.to(device)

    # Use bigger batch for more realistic training
    big_input = torch.randint(0, 151936, (4, 256), dtype=torch.long, device=device)

    losses_std = simulate_training(model_std2, big_input, steps=20, model_name="standard")
    losses_glb = simulate_training(model_glb2, big_input, steps=20, model_name="global")

    print("\n  Step-by-step loss comparison:")
    print(f"  {'Step':>5s}  {'Standard':>10s}  {'Global':>10s}  {'Diff':>10s}")
    for i, (ls, lg) in enumerate(zip(losses_std, losses_glb)):
        print(f"  {i:>5d}  {ls:>10.4f}  {lg:>10.4f}  {abs(ls-lg):>10.4f}")

    # ===== Test 8: Check if attention dominates =====
    print("\n" + "=" * 70)
    print("TEST 8: Parameter ratio analysis")
    print("=" * 70)

    p = params_std
    non_expert = p["total"] - p["expert"]
    print(f"  Standard model:")
    print(f"    Total params:     {p['total']:>12,d}")
    print(f"    Expert params:    {p['expert']:>12,d} ({100*p['expert']/p['total']:.1f}%)")
    print(f"    Attention params: {p['attn']:>12,d} ({100*p['attn']/p['total']:.1f}%)")
    print(f"    Router params:    {p['router']:>12,d} ({100*p['router']/p['total']:.1f}%)")
    print(f"    Embed params:     {p['embed']:>12,d} ({100*p['embed']/p['total']:.1f}%)")
    print(f"    Non-expert:       {non_expert:>12,d} ({100*non_expert/p['total']:.1f}%)")

    p = params_glb
    non_expert = p["total"] - p["expert"]
    print(f"  Global model:")
    print(f"    Total params:     {p['total']:>12,d}")
    print(f"    Expert params:    {p['expert']:>12,d} ({100*p['expert']/p['total']:.1f}%)")
    print(f"    Attention params: {p['attn']:>12,d} ({100*p['attn']/p['total']:.1f}%)")
    print(f"    Router params:    {p['router']:>12,d} ({100*p['router']/p['total']:.1f}%)")
    print(f"    Embed params:     {p['embed']:>12,d} ({100*p['embed']/p['total']:.1f}%)")
    print(f"    Non-expert:       {non_expert:>12,d} ({100*non_expert/p['total']:.1f}%)")


if __name__ == "__main__":
    main()
