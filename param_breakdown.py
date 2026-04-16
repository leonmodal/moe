"""Instantiate each model from depth_matched_fp32_no_liger configs and print parameter breakdown."""

import os
import sys
import yaml
import torch

sys.path.insert(0, "/tmp/moe")
os.chdir("/tmp/moe")

from src.models import (
    Qwen3MoeConfig,
    StandardMoEModel,
    DeepSeekStandardMoEModel,
    GlobalMoEConfig,
    GlobalMoEForCausalLM,
    DeepSeekGlobalMoEForCausalLM,
    MoEverythingConfig,
    MoEverythingForCausalLM,
)


def build_model(cfg: dict):
    mtype = cfg["model"]["type"]
    mcfg = cfg["model"]

    common = dict(
        vocab_size=mcfg["vocab_size"],
        hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        head_dim=mcfg["head_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        moe_intermediate_size=mcfg["moe_intermediate_size"],
        intermediate_size=mcfg.get("intermediate_size", mcfg["moe_intermediate_size"] * 4),
        max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
        rope_theta=mcfg.get("rope_theta", 1_000_000.0),
        rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        router_aux_loss_coef=mcfg.get("router_aux_loss_coef", 0.001),
        seq_aux_loss_coef=mcfg.get("seq_aux_loss_coef", 0.0),
        norm_topk_prob=mcfg.get("norm_topk_prob", True),
        num_experts_per_tok=mcfg["num_experts_per_tok"],
        output_router_logits=True,
    )

    def _set_deepseek_router_params(config, mcfg):
        config.topk_scaling_factor = mcfg.get("topk_scaling_factor", None)
        config.num_groups = mcfg.get("num_groups", None)
        config.group_topk = mcfg.get("group_topk", None)

    use_deepseek = mcfg.get("router_type") == "deepseek" or mcfg.get("use_deepseek_routing", False)

    if mtype == "standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        if use_deepseek:
            _set_deepseek_router_params(config, mcfg)
            model = DeepSeekStandardMoEModel(config)
        else:
            model = StandardMoEModel(config)
    elif mtype == "global_moe":
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        if use_deepseek:
            _set_deepseek_router_params(config, mcfg)
            model = DeepSeekGlobalMoEForCausalLM(config)
        else:
            model = GlobalMoEForCausalLM(config)
    elif mtype == "moe_everything":
        config = MoEverythingConfig(
            num_experts=mcfg["num_experts"],
            num_attn_experts=mcfg.get("num_attn_experts", 4),
            num_attn_experts_per_tok=mcfg.get("num_attn_experts_per_tok", 1),
            attn_expert_mode=mcfg.get("attn_expert_mode", "bundled"),
            branch_router_aux_loss_coef=mcfg.get("branch_router_aux_loss_coef", 0.0),
            use_deepseek_routing=mcfg.get("use_deepseek_routing", False),
            topk_scaling_factor=mcfg.get("topk_scaling_factor", None),
            num_groups=mcfg.get("num_groups", None),
            group_topk=mcfg.get("group_topk", None),
            per_layer_router=mcfg.get("per_layer_router", False),
            per_layer_mlp_router=mcfg.get("per_layer_mlp_router", False),
            per_layer_attn_router=mcfg.get("per_layer_attn_router", False),
            routed_norm=mcfg.get("routed_norm", False),
            per_layer_norm=mcfg.get("per_layer_norm", False),
            per_layer_qk_norm=mcfg.get("per_layer_qk_norm", False),
            post_norm=mcfg.get("post_norm", False),
            dynamic_depth_min=mcfg.get("dynamic_depth_min", 1.0),
            dynamic_depth_max=mcfg.get("dynamic_depth_max", 1.0),
            depthwise_attention=mcfg.get("depthwise_attention", False),
            depthwise_block_size=mcfg.get("depthwise_block_size", 0),
            per_head_compute_mode=mcfg.get("per_head_compute_mode", "auto"),
            per_head_dense_fraction_threshold=mcfg.get(
                "per_head_dense_fraction_threshold", 0.75
            ),
            sanity_check_mode=mcfg.get("sanity_check_mode"),
            **common,
        )
        model = MoEverythingForCausalLM(config)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    return model


def categorize_param(name):
    """Categorize a parameter name into one of the defined categories."""
    # Order matters: more specific checks first

    # Embedding
    if "embed" in name:
        return "Embedding"

    # MLP: gate_up_proj, down_proj, gate_proj, up_proj (but not router gate)
    if any(k in name for k in ["gate_up_proj", "down_proj", "gate_proj", "up_proj"]):
        return "MLP"

    # Attention: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
    if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]):
        return "Attention"

    # Router: "router" or "gate" (but NOT gate_up_proj/gate_proj which are MLP - already handled above)
    # Also include "branch" router
    if "router" in name or "gate" in name or "branch" in name:
        return "Router"

    # Norm: "norm" in name (but not q_norm/k_norm which are attention - already handled above)
    if "norm" in name:
        return "Norm"

    return "Other"


def analyze_model(model, config_name):
    """Analyze and print parameter breakdown for a model."""
    categories = {}
    category_details = {}  # track individual param names per category

    for name, param in model.named_parameters():
        cat = categorize_param(name)
        count = param.numel()
        categories[cat] = categories.get(cat, 0) + count
        if cat not in category_details:
            category_details[cat] = []
        category_details[cat].append((name, count))

    total = sum(categories.values())

    print(f"\n{'='*90}")
    print(f"  {config_name}")
    print(f"{'='*90}")
    print(f"  Total parameters: {total:,}")
    print(f"  Total parameters (M): {total/1e6:.2f}M")
    print(f"{'─'*90}")
    print(f"  {'Category':<15} {'Count':>15} {'Percentage':>12}")
    print(f"  {'─'*42}")

    # Print in fixed order
    for cat in ["Embedding", "Attention", "MLP", "Router", "Norm", "Other"]:
        count = categories.get(cat, 0)
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {cat:<15} {count:>15,} {pct:>11.2f}%")

    print(f"  {'─'*42}")

    # Print "Other" details if any
    if "Other" in category_details and category_details["Other"]:
        print(f"\n  Other params detail:")
        for pname, pcount in sorted(category_details["Other"], key=lambda x: -x[1]):
            print(f"    {pname}: {pcount:,}")

    print()
    return categories, total


def main():
    base = "/tmp/moe/configs/depth_matched_fp32_no_liger"
    layer_dirs = ["4_layers", "8_layers", "16_layers"]
    config_files = [
        "standard_moe.yaml",
        "global_moe.yaml",
        "moe_everything_per_head_independent_perlayer_prenorm.yaml",
        "moe_everything_per_head_precompute_kv_perlayer_prenorm.yaml",
        "moe_everything_per_head_precompute_kv_sanity.yaml",
    ]

    # Collect summary data for comparison table
    all_results = []

    for layer_dir in layer_dirs:
        print(f"\n{'#'*90}")
        print(f"#  {layer_dir.upper()}")
        print(f"{'#'*90}")

        for config_file in config_files:
            config_path = os.path.join(base, layer_dir, config_file)
            if not os.path.exists(config_path):
                print(f"\n  [SKIP] {config_path} not found")
                continue

            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            short_name = config_file.replace(".yaml", "")
            full_name = f"{layer_dir}/{short_name}"

            try:
                with torch.no_grad():
                    model = build_model(cfg)
                categories, total = analyze_model(model, full_name)
                all_results.append((full_name, categories, total))
                del model  # free memory
            except Exception as e:
                print(f"\n  [ERROR] {full_name}: {e}")
                import traceback
                traceback.print_exc()

    # Print comparison summary table
    print(f"\n\n{'='*130}")
    print(f"  COMPARISON SUMMARY TABLE")
    print(f"{'='*130}")

    cats = ["Embedding", "Attention", "MLP", "Router", "Norm", "Other"]
    header = f"  {'Config':<65} {'Total(M)':>8}"
    for c in cats:
        header += f" {c:>10}"
    print(header)
    print(f"  {'─'*125}")

    for name, categories, total in all_results:
        row = f"  {name:<65} {total/1e6:>7.2f}M"
        for c in cats:
            count = categories.get(c, 0)
            pct = 100.0 * count / total if total > 0 else 0
            row += f" {pct:>9.1f}%"
        print(row)

    print()


if __name__ == "__main__":
    main()
