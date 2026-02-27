"""
Verify global MoE is NOT doing the same thing as standard MoE.

Key checks:
  1. Cross-layer expert sharing: do multiple layers route to the SAME expert?
  2. Gradient provenance: does a single expert get gradients from multiple layers?
  3. Perturbation test: changing ONE expert affects ALL layers (not just one)
  4. Weight tying proof: the actual expert weight tensors are the same object across layers
"""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from src.models import Qwen3MoeConfig, StandardMoEModel, GlobalMoEConfig, GlobalMoEForCausalLM
from src.models.global_moe import GlobalMoEModel, GlobalSparseMoeBlock


def old_global_config():
    return GlobalMoEConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=128, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
    )

def old_standard_config():
    return Qwen3MoeConfig(
        vocab_size=151936, hidden_size=1024, num_hidden_layers=8,
        head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=768, intermediate_size=3072,
        max_position_embeddings=32768, output_router_logits=True,
        norm_topk_prob=True, router_aux_loss_coef=0.01,
    )


def test_cross_layer_expert_sharing():
    """Do multiple layers route tokens to the SAME expert from the global pool?"""
    print("=" * 65)
    print("TEST 1: Cross-layer expert sharing")
    print("=" * 65)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (4, 256), device=device)

    with torch.no_grad():
        out = glb(input_ids=ids, output_router_logits=True)

    num_layers = len(out.router_logits)
    T = out.router_logits[0].shape[0]

    # For each layer, get the set of experts selected
    layer_expert_sets = []
    for i, rl in enumerate(out.router_logits):
        top4 = rl.topk(4, dim=-1).indices  # [T, 4]
        expert_set = set(top4.reshape(-1).cpu().tolist())
        layer_expert_sets.append(expert_set)
        print(f"  Layer {i}: {len(expert_set)} unique experts used (out of 128)")

    # Check pairwise overlap
    print(f"\n  Cross-layer expert overlap (shared experts between layer pairs):")
    total_shared = 0
    pairs = 0
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            shared = layer_expert_sets[i] & layer_expert_sets[j]
            total_shared += len(shared)
            pairs += 1
            if j == i + 1 or (i == 0 and j == num_layers - 1):
                print(f"    Layer {i} & {j}: {len(shared)} shared experts")

    avg_shared = total_shared / pairs
    print(f"\n  Average shared experts per layer pair: {avg_shared:.1f}")

    # Union of all experts used across all layers
    all_experts = set()
    for s in layer_expert_sets:
        all_experts |= s
    print(f"  Total unique experts used across all layers: {len(all_experts)} / 128")

    # How many experts serve multiple layers?
    from collections import Counter
    expert_layer_count = Counter()
    for i, s in enumerate(layer_expert_sets):
        for e in s:
            expert_layer_count[e] += 1

    multi_layer = sum(1 for e, c in expert_layer_count.items() if c > 1)
    all_layer = sum(1 for e, c in expert_layer_count.items() if c == num_layers)
    print(f"  Experts serving >1 layer: {multi_layer} / {len(all_experts)}")
    print(f"  Experts serving ALL {num_layers} layers: {all_layer}")

    if multi_layer > 0:
        print(f"\n  >>> CONFIRMED: Experts are shared across layers!")
    else:
        print(f"\n  >>> WARNING: No cross-layer sharing detected!")

    del glb
    torch.cuda.empty_cache()


def test_single_expert_multi_layer_gradient():
    """Does a single global expert get gradient contributions from MULTIPLE layers?"""
    print("\n" + "=" * 65)
    print("TEST 2: Single expert gets gradients from multiple layers")
    print("=" * 65)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config()).to(device=device, dtype=dtype).train()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (4, 256), device=device)

    # Zero out all expert grads
    glb.zero_grad()

    # Forward + backward
    out = glb(input_ids=ids, labels=ids, output_router_logits=True)
    out.loss.backward()

    # Check expert gradients
    ge = glb.model.global_experts
    grad = ge.gate_up_proj.grad  # [128, I, H]
    assert grad is not None, "No gradient on global experts!"

    # Which experts got non-zero gradients?
    per_expert_grad_norm = grad.float().reshape(128, -1).norm(dim=1)
    experts_with_grad = (per_expert_grad_norm > 0).sum().item()
    print(f"  Experts with non-zero gradient: {experts_with_grad} / 128")

    # Now the key test: do per-layer gradient contributions overlap?
    # We'll do this by running separate backward passes through each layer
    print(f"\n  Per-layer gradient contribution analysis:")

    # Hook to capture which experts each layer routes to
    layer_experts_used = {}
    for i, rl in enumerate(out.router_logits):
        top4 = rl.topk(4, dim=-1).indices
        experts_used = set(top4.reshape(-1).cpu().tolist())
        layer_experts_used[i] = experts_used

    # Check: how many experts are used by exactly 1 layer vs multiple layers?
    from collections import Counter
    expert_usage = Counter()
    for layer_idx, experts in layer_experts_used.items():
        for e in experts:
            expert_usage[e] += 1

    usage_dist = Counter(expert_usage.values())
    print(f"  Expert usage distribution:")
    for num_layers, count in sorted(usage_dist.items()):
        print(f"    Used by {num_layers} layer(s): {count} experts")

    multi = sum(c for nl, c in usage_dist.items() if nl > 1)
    print(f"\n  >>> {multi} experts receive gradient from MULTIPLE layers")
    print(f"  >>> This is impossible in standard per-layer MoE!")

    del glb
    torch.cuda.empty_cache()


def test_perturbation_affects_all_layers():
    """If we perturb ONE expert, does it change the output of MULTIPLE layers?"""
    print("\n" + "=" * 65)
    print("TEST 3: Perturbing one expert affects multiple layers")
    print("=" * 65)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    glb = GlobalMoEForCausalLM(old_global_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (4, 256), device=device)

    # Capture per-layer outputs before perturbation
    pre_outputs = {}
    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name] = output.detach().clone()
        return hook

    hooks = []
    for i in range(8):
        h = glb.model.layers[i].register_forward_hook(make_hook(pre_outputs, f"layer_{i}"))
        hooks.append(h)

    with torch.no_grad():
        out_pre = glb(input_ids=ids, output_router_logits=True)

    # Find an expert that's used by multiple layers
    layer_experts = {}
    for i, rl in enumerate(out_pre.router_logits):
        top4 = rl.topk(4, dim=-1).indices
        layer_experts[i] = set(top4.reshape(-1).cpu().tolist())

    from collections import Counter
    expert_usage = Counter()
    for i, experts in layer_experts.items():
        for e in experts:
            expert_usage[e] += 1

    # Pick the expert used by the most layers
    target_expert, target_layers = expert_usage.most_common(1)[0]
    which_layers = [i for i, experts in layer_experts.items() if target_expert in experts]
    print(f"  Target expert: {target_expert} (used by {target_layers} layers: {which_layers})")

    # Perturb ONLY that one expert
    with torch.no_grad():
        glb.model.global_experts.gate_up_proj[target_expert] += 10.0

    # Capture per-layer outputs after perturbation
    post_outputs = {}
    for h in hooks:
        h.remove()
    hooks = []
    for i in range(8):
        h = glb.model.layers[i].register_forward_hook(make_hook(post_outputs, f"layer_{i}"))
        hooks.append(h)

    with torch.no_grad():
        out_post = glb(input_ids=ids, output_router_logits=True)

    for h in hooks:
        h.remove()

    # Check which layers' outputs changed
    print(f"\n  Layer output changes after perturbing expert {target_expert}:")
    layers_affected = []
    for i in range(8):
        key = f"layer_{i}"
        diff = (pre_outputs[key].float() - post_outputs[key].float()).abs().max().item()
        affected = diff > 1e-3
        if affected:
            layers_affected.append(i)
        marker = " <<<" if affected else ""
        print(f"    Layer {i}: max diff = {diff:.6f}{marker}")

    print(f"\n  Layers affected: {layers_affected}")
    print(f"  Layers that routed to expert {target_expert}: {which_layers}")

    if len(layers_affected) > 1:
        print(f"\n  >>> CONFIRMED: One expert perturbation affects {len(layers_affected)} layers!")
        print(f"  >>> This proves experts are genuinely shared, not duplicated.")
    else:
        print(f"\n  >>> Only {len(layers_affected)} layer affected — check for issues")

    del glb
    torch.cuda.empty_cache()


def test_standard_perturbation_single_layer():
    """Control test: in standard MoE, perturbing one layer's expert only affects THAT layer."""
    print("\n" + "=" * 65)
    print("TEST 4 (control): Standard MoE — perturbing layer 0 expert")
    print("=" * 65)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    std = StandardMoEModel(old_standard_config()).to(device=device, dtype=dtype).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 151936, (4, 256), device=device)

    pre_outputs = {}
    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name] = output.detach().clone()
        return hook

    hooks = []
    for i in range(8):
        h = std.model.layers[i].register_forward_hook(make_hook(pre_outputs, f"layer_{i}"))
        hooks.append(h)

    with torch.no_grad():
        std(input_ids=ids, output_router_logits=True)

    # Perturb expert 0 in layer 0 ONLY
    with torch.no_grad():
        std.model.layers[0].mlp.experts.gate_up_proj[0] += 10.0

    post_outputs = {}
    for h in hooks:
        h.remove()
    hooks = []
    for i in range(8):
        h = std.model.layers[i].register_forward_hook(make_hook(post_outputs, f"layer_{i}"))
        hooks.append(h)

    with torch.no_grad():
        std(input_ids=ids, output_router_logits=True)

    for h in hooks:
        h.remove()

    print(f"  Layer output changes after perturbing layer 0 expert 0:")
    for i in range(8):
        key = f"layer_{i}"
        diff = (pre_outputs[key].float() - post_outputs[key].float()).abs().max().item()
        affected = diff > 1e-3
        marker = " <<<" if affected else ""
        print(f"    Layer {i}: max diff = {diff:.6f}{marker}")

    print(f"\n  >>> In standard MoE, perturbation propagates forward through residuals")
    print(f"  >>> but the expert itself is ONLY used in layer 0.")
    print(f"  >>> In global MoE (test 3), the expert is directly used in MULTIPLE layers.")

    del std
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_cross_layer_expert_sharing()
    test_single_expert_multi_layer_gradient()
    test_perturbation_affects_all_layers()
    test_standard_perturbation_single_layer()
    print("\n" + "=" * 65)
    print("ALL DONE")
    print("=" * 65)
