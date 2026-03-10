"""
Tiny diagnostic for the current question:

- Standard DeepSeek MoE: per-layer private experts
- Global DeepSeek MoE: one shared expert pool across layers
- Global nointerp: same shared pool, different bias-update rule

Keeps total expert count matched so we isolate sharing topology, not capacity.
Runs a short repeated-batch training loop with the real build_model() and
update_expert_biases() logic from train.py.
"""
import copy

import torch

torch.backends.cuda.preferred_blas_library("cublaslt")

from train import build_model, update_expert_biases, bias_alpha_schedule
from src.models.router import DeepSeekRouter


class FakeAccelerator:
    num_processes = 1

    def unwrap_model(self, model):
        return model


def make_cfg(model_type: str, *, num_layers: int, per_layer_experts: int, bias_interpolation: bool = False) -> dict:
    total_experts = num_layers * per_layer_experts
    if model_type == "deepseek_standard_moe":
        num_experts = per_layer_experts
    elif model_type == "deepseek_global_moe":
        num_experts = total_experts
    else:
        raise ValueError(model_type)

    cfg = {
        "experiment_name": model_type,
        "model": {
            "type": model_type,
            "vocab_size": 256,
            "hidden_size": 128,
            "num_hidden_layers": num_layers,
            "head_dim": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "intermediate_size": 256,
            "norm_topk_prob": True,
            "router_aux_loss_coef": 0.0,
            "seq_aux_loss_coef": 0.0001,
            "bias_update_rate": 0.001,
            "topk_scaling_factor": 2.5,
            "num_groups": 4,
            "group_topk": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
            "tie_word_embeddings": True,
            "output_router_logits": True,
        },
        "training": {
            "learning_rate": 1.0e-3,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
        },
    }
    if model_type == "deepseek_global_moe" and bias_interpolation:
        cfg["model"]["bias_interpolation"] = True
    return cfg


def make_repeated_batches(*, batch_size: int, seq_len: int, num_batches: int, vocab_subset: int, device: str):
    torch.manual_seed(123)
    batches = []
    for _ in range(num_batches):
        ids = torch.randint(0, vocab_subset, (batch_size, seq_len), device=device)
        batches.append(ids)
    return batches


def summarize_model(name: str, cfg: dict, model) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "gate_up_proj" in n or "down_proj" in n
    )
    routers = [m for m in model.modules() if isinstance(m, DeepSeekRouter)]
    print(f"\n=== {name} ===")
    print(f"type={cfg['model']['type']}")
    print(f"layers={cfg['model']['num_hidden_layers']}")
    print(f"num_experts field={cfg['model']['num_experts']}")
    print(f"total_params={total_params:,}")
    print(f"expert_params={expert_params:,}")
    print(f"routers={len(routers)}")
    print(f"bias_interpolation={cfg['model'].get('bias_interpolation', False)}")


def collect_layer_expert_sets(model, input_ids: torch.Tensor) -> list[set[int]]:
    with torch.no_grad():
        out = model(input_ids=input_ids, output_router_logits=True)
    expert_sets = []
    for gate_logits in out.router_logits:
        topk = gate_logits.topk(k=2, dim=-1).indices
        expert_sets.append(set(topk.reshape(-1).tolist()))
    return expert_sets


def print_routing_overlap(name: str, layer_sets: list[set[int]]) -> None:
    print(f"\n{name} routing overlap")
    for idx, expert_set in enumerate(layer_sets):
        print(f"layer {idx}: {len(expert_set)} unique experts -> {sorted(expert_set)}")
    shared = set.intersection(*layer_sets) if layer_sets else set()
    print(f"experts used by every layer: {sorted(shared)}")


def train_short(name: str, cfg: dict, model, batches, steps: int) -> list[dict]:
    device = next(model.parameters()).device
    seq_aux_loss_coef = cfg["model"].get("seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef > 0:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
    )
    model.train()
    accelerator = FakeAccelerator()
    is_global = cfg["model"]["type"] == "deepseek_global_moe"
    bias_update_rate = cfg["model"].get("bias_update_rate", 0.0)
    bias_interpolation = cfg["model"].get("bias_interpolation", False)

    rows = []
    print(f"\n{name} short training")
    print(f"{'step':>4} {'loss':>10} {'alpha':>10} {'bias_std':>10}")
    for step in range(steps):
        input_ids = batches[step % len(batches)].to(device)
        labels = input_ids
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, output_router_logits=True)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if bias_update_rate > 0:
            alpha = bias_alpha_schedule(step) if (is_global and bias_interpolation) else 0.0
            bias_stats = update_expert_biases(
                model,
                bias_update_rate,
                accelerator,
                is_global=is_global,
                alpha=alpha,
            )
        else:
            alpha = None
            bias_stats = {}

        row = {
            "step": step,
            "loss": loss.item(),
            "alpha": alpha,
            "bias_std": bias_stats.get("routing/expert_bias_std", 0.0),
        }
        rows.append(row)
        if step in (0, 1, 2, 4, 7, steps - 1):
            alpha_text = "n/a" if alpha is None else f"{alpha:.6f}"
            print(f"{step:4d} {row['loss']:10.6f} {alpha_text:>10} {row['bias_std']:10.6f}")
    return rows


def print_pairwise(rows_by_name: dict[str, list[dict]]) -> None:
    names = list(rows_by_name)
    print("\npairwise loss diffs")
    print(f"{'step':>4} {'std-glb':>10} {'std-noi':>10} {'glb-noi':>10}")
    steps = len(next(iter(rows_by_name.values())))
    for step in range(steps):
        s = rows_by_name[names[0]][step]["loss"]
        g = rows_by_name[names[1]][step]["loss"]
        n = rows_by_name[names[2]][step]["loss"]
        if step in (0, 1, 2, 4, 7, steps - 1):
            print(f"{step:4d} {abs(s-g):10.6f} {abs(s-n):10.6f} {abs(g-n):10.6f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = "cuda"
    dtype = torch.bfloat16
    num_layers = 4
    per_layer_experts = 4

    cfg_standard = make_cfg("deepseek_standard_moe", num_layers=num_layers, per_layer_experts=per_layer_experts)
    cfg_global = make_cfg(
        "deepseek_global_moe",
        num_layers=num_layers,
        per_layer_experts=per_layer_experts,
        bias_interpolation=True,
    )
    cfg_global_nointerp = make_cfg(
        "deepseek_global_moe",
        num_layers=num_layers,
        per_layer_experts=per_layer_experts,
        bias_interpolation=False,
    )

    configs = {
        "standard": cfg_standard,
        "global": cfg_global,
        "global_nointerp": cfg_global_nointerp,
    }

    torch.manual_seed(0)
    probe_ids = torch.randint(0, 32, (1, 24), device=device)
    batches = make_repeated_batches(batch_size=4, seq_len=32, num_batches=4, vocab_subset=32, device=device)

    models = {}
    for name, cfg in configs.items():
        torch.manual_seed(0)
        model, _ = build_model(copy.deepcopy(cfg))
        model = model.to(device=device, dtype=dtype)
        if cfg["model"].get("seq_aux_loss_coef", 0.0) > 0:
            model._seq_aux_loss_coef = cfg["model"]["seq_aux_loss_coef"]
        summarize_model(name, cfg, model)
        models[name] = model

    print("\nrouting structure probe")
    for name, model in models.items():
        layer_sets = collect_layer_expert_sets(model.eval(), probe_ids)
        print_routing_overlap(name, layer_sets)

    rows_by_name = {}
    for name, model in models.items():
        rows_by_name[name] = train_short(name, configs[name], model, batches, steps=10)

    print_pairwise(rows_by_name)


if __name__ == "__main__":
    main()
