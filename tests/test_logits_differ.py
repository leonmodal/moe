"""
Verify: are the parameters initialized the same? Are the logits actually different?
Trace through layer by layer to see where outputs diverge (if they do).
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM
from src.models.global_moe import GlobalMoEModel


def old_standard_config():
    return Qwen3MoeConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
    )

def old_global_config():
    return GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
    )


def test_param_init_comparison():
    """Check if both models get the same parameter values with the same seed."""
    print("=" * 65)
    print("TEST 1: Parameter initialization comparison")
    print("=" * 65)

    torch.manual_seed(0)
    std = StandardMoEModel(old_standard_config())

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config())

    # Compare shared parameters (embeddings, attention, norms)
    std_params = dict(std.named_parameters())
    glb_params = dict(glb.named_parameters())

    print(f"\n  Standard params: {len(std_params)}")
    print(f"  Global params:   {len(glb_params)}")

    # Check embedding
    std_emb = std_params["model.embed_tokens.weight"]
    glb_emb = glb_params["model.embed_tokens.weight"]
    emb_match = torch.equal(std_emb, glb_emb)
    emb_diff = (std_emb - glb_emb).abs().max().item()
    print(f"\n  Embeddings identical: {emb_match}  (max diff: {emb_diff:.2e})")

    # Check lm_head
    std_head = std_params["lm_head.weight"]
    glb_head = glb_params["lm_head.weight"]
    head_match = torch.equal(std_head, glb_head)
    head_diff = (std_head - glb_head).abs().max().item()
    print(f"  LM head identical:   {head_match}  (max diff: {head_diff:.2e})")

    # Check attention weights layer by layer
    print(f"\n  Layer-by-layer attention comparison:")
    for i in range(8):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            std_key = f"model.layers.{i}.self_attn.{proj}.weight"
            glb_key = f"model.layers.{i}.self_attn.{proj}.weight"
            if std_key in std_params and glb_key in glb_params:
                match = torch.equal(std_params[std_key], glb_params[glb_key])
                diff = (std_params[std_key] - glb_params[glb_key]).abs().max().item()
                if not match:
                    print(f"    Layer {i} {proj}: DIFFER (max diff: {diff:.2e})")
                    break
        else:
            continue
        break
    else:
        print(f"    All 8 layers attention weights: IDENTICAL")

    # Check layer norms
    norms_match = True
    for i in range(8):
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            std_key = f"model.layers.{i}.{norm}.weight"
            glb_key = f"model.layers.{i}.{norm}.weight"
            if std_key in std_params and glb_key in glb_params:
                if not torch.equal(std_params[std_key], glb_params[glb_key]):
                    norms_match = False
                    diff = (std_params[std_key] - glb_params[glb_key]).abs().max().item()
                    print(f"    Layer {i} {norm}: DIFFER (max diff: {diff:.2e})")
    if norms_match:
        print(f"    All layer norms: IDENTICAL")

    # Check expert weights
    print(f"\n  Expert weight comparison:")
    # Standard: model.layers.{i}.mlp.experts.gate_up_proj [16, H, 2*I]
    # Global: model.global_experts.gate_up_proj [128, H, 2*I]
    for i in range(8):
        std_key = f"model.layers.{i}.mlp.experts.gate_up_proj"
        if std_key in std_params:
            std_exp = std_params[std_key]  # [16, 1024, 1536]
            print(f"    Standard layer {i} experts shape: {std_exp.shape}")
            break

    glb_exp = glb_params.get("model.global_experts.gate_up_proj")
    if glb_exp is not None:
        print(f"    Global experts shape: {glb_exp.shape}")

    # Check router weights
    print(f"\n  Router weight comparison:")
    for i in range(min(2, 8)):
        std_key = f"model.layers.{i}.mlp.gate.weight"
        glb_key = f"model.layers.{i}.mlp.gate.weight"
        if std_key in std_params and glb_key in glb_params:
            std_r = std_params[std_key]
            glb_r = glb_params[glb_key]
            print(f"    Layer {i} router: std={std_r.shape} glb={glb_r.shape}")
            if std_r.shape == glb_r.shape:
                match = torch.equal(std_r, glb_r)
                print(f"      Same shape, identical: {match}")
            else:
                print(f"      Different shapes (expected: std=16xH, glb=128xH)")

    return std, glb


def test_logits_differ():
    """Run same input through both models and compare logits."""
    print("\n" + "=" * 65)
    print("TEST 2: Logit comparison (same input, same seed init)")
    print("=" * 65)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    std = StandardMoEModel(old_standard_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (2, 64), device=device)

    with torch.no_grad():
        std_out = std(input_ids=ids, output_router_logits=True)
        glb_out = glb(input_ids=ids, output_router_logits=True)

    std_logits = std_out.logits
    glb_logits = glb_out.logits

    logits_match = torch.equal(std_logits, glb_logits)
    logits_diff = (std_logits.float() - glb_logits.float()).abs()
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    rel_diff = (logits_diff / (std_logits.float().abs() + 1e-8)).mean().item()

    print(f"\n  Logits identical:     {logits_match}")
    print(f"  Max absolute diff:    {max_diff:.6f}")
    print(f"  Mean absolute diff:   {mean_diff:.6f}")
    print(f"  Mean relative diff:   {rel_diff:.6f} ({rel_diff*100:.4f}%)")

    # Check predictions (argmax)
    std_preds = std_logits.argmax(dim=-1)
    glb_preds = glb_logits.argmax(dim=-1)
    pred_match_rate = (std_preds == glb_preds).float().mean().item()
    print(f"  Prediction match rate: {pred_match_rate:.4f} ({pred_match_rate*100:.1f}%)")

    # Compare router logits
    print(f"\n  Router logit comparison:")
    for i, (sr, gr) in enumerate(zip(std_out.router_logits, glb_out.router_logits)):
        print(f"    Layer {i}: std shape={sr.shape}, glb shape={gr.shape}")

    # Check if routing decisions overlap
    print(f"\n  Routing decision comparison (top-4 experts selected):")
    for i, (sr, gr) in enumerate(zip(std_out.router_logits, glb_out.router_logits)):
        std_top4 = sr.topk(4, dim=-1).indices  # [T, 4] from [T, 16]
        glb_top4 = gr.topk(4, dim=-1).indices  # [T, 4] from [T, 128]
        print(f"    Layer {i}:")
        print(f"      Std top-4 experts (first 5 tokens): {std_top4[:5].tolist()}")
        print(f"      Glb top-4 experts (first 5 tokens): {glb_top4[:5].tolist()}")

    del std, glb
    torch.cuda.empty_cache()


def test_intermediate_activations():
    """Hook into both models to compare activations at each layer."""
    print("\n" + "=" * 65)
    print("TEST 3: Layer-by-layer activation comparison")
    print("=" * 65)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    std = StandardMoEModel(old_standard_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (2, 64), device=device)

    # Capture hidden states after each layer
    std_hidden = {}
    glb_hidden = {}

    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name] = output.detach().clone()
            elif isinstance(output, tuple):
                storage[name] = output[0].detach().clone()
        return hook

    for i in range(8):
        std.model.layers[i].register_forward_hook(make_hook(std_hidden, f"layer_{i}"))
        glb.model.layers[i].register_forward_hook(make_hook(glb_hidden, f"layer_{i}"))

    with torch.no_grad():
        std_out = std(input_ids=ids, output_router_logits=True)
        glb_out = glb(input_ids=ids, output_router_logits=True)

    print(f"\n  {'Layer':>8}  {'Max Diff':>12}  {'Mean Diff':>12}  {'Rel Diff%':>12}  {'Cosine Sim':>12}")
    print(f"  {'-'*60}")

    for i in range(8):
        key = f"layer_{i}"
        if key in std_hidden and key in glb_hidden:
            s = std_hidden[key].float()
            g = glb_hidden[key].float()
            diff = (s - g).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            # Cosine similarity
            s_flat = s.reshape(-1)
            g_flat = g.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(s_flat.unsqueeze(0),
                                                              g_flat.unsqueeze(0)).item()
            rel_d = mean_d / (s.abs().mean().item() + 1e-8) * 100
            print(f"  Layer {i:2d}  {max_d:12.6f}  {mean_d:12.6f}  {rel_d:12.4f}%  {cos_sim:12.8f}")

    # Also compare embeddings
    std_emb = std.model.embed_tokens(ids).float()
    glb_emb = glb.model.embed_tokens(ids).float()
    emb_diff = (std_emb - glb_emb).abs().max().item()
    print(f"\n  Embedding output max diff: {emb_diff:.2e}")

    del std, glb
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_param_init_comparison()
    test_logits_differ()
    test_intermediate_activations()
    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
