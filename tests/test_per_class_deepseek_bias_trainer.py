"""Round 29 review Finding 3: a yaml with nested
`model.mlp_router.balancing: deepseek_bias` AND a per-class
`bias_update_rate` (and NO top-level `training.bias_update_rate`)
must drive the trainer's post-step bias update — the rate has to
flow from the per-class block to the trainer's actual call path.

Earlier rounds stamped the per-class rate onto config under
`effective_bias_update_rate`, but `trainer_post_optimizer_bias_update`
gated solely on `train_cfg.bias_update_rate`. With the migrator
removing the top-level rate, the trainer fell silent — bias never
updated.

Round 30 reads the `effective_*` knobs as a fallback when the
top-level fields are zero/unset, so the trainer's real call path
honors per-class deepseek configuration.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _bypass_training_init():
    import types
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
        pkg = types.ModuleType("src.training")
        pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = pkg

    def _load(modname, relpath):
        spec = importlib.util.spec_from_file_location(modname, str(repo / relpath))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("src.training.config", "src/training/config.py")
    factory_mod = _load("src.training.model_factory", "src/training/model_factory.py")
    routing_mod = _load("src.training.routing", "src/training/routing.py")
    return cfg_mod, factory_mod, routing_mod


_CFG_MOD, _FACTORY_MOD, _ROUTING_MOD = _bypass_training_init()


def _build_nested_deepseek():
    """Build a tiny standard_moe model with NESTED-only deepseek_bias
    configuration: top-level `training.bias_update_rate` is absent,
    `model.mlp_router.balancing == "deepseek_bias"` with
    `bias_update_rate=0.001`. The migrator emits this exact shape.
    """
    cfg = {
        "model": {
            "type": "standard_moe",
            "router_type": "deepseek",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "head_dim": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "intermediate_size": 32,
            "norm_topk_prob": True,
            "topk_scaling_factor": 2.5,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1.0e-06,
            "rope_theta": 10000.0,
            "max_position_embeddings": 32,
            "tie_word_embeddings": True,
            "output_router_logits": False,
            "attn_implementation": "eager",
            "mlp_router": {
                "balancing": "deepseek_bias",
                "bias_update_rate": 0.001,
                "bias_update_zero_sum": True,
            },
        },
        "training": {
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "warmup_steps": 0,
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "mixed_precision": "",
            "output_dir": "/tmp",
            # NO top-level bias_update_rate.
        },
    }
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    train_cfg = _CFG_MOD.build_training_config(cfg)
    return model, train_cfg, cfg


def test_per_class_deepseek_bias_drives_trainer_bias_update():
    """Build a nested-only deepseek_bias config, run one step's
    forward + backward + optimizer.step + post-step bias-update,
    and assert at least one balancing owner's `expert_bias`
    changed. With the per-class wiring, the bias update must fire
    even though `train_cfg.bias_update_rate == 0`.
    """
    model, train_cfg, cfg = _build_nested_deepseek()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Snapshot initial expert_bias values across every owner.
    initial: dict[str, torch.Tensor] = {}
    for owner, label in model.get_all_balancing_owners():
        initial[id(owner)] = owner.expert_bias.detach().clone()

    # Trainer's exact post-step path.
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    _ROUTING_MOD.trainer_optimizer_step_and_bias_update(
        model, optimizer, scheduler=type("S", (), {"step": lambda _: None})(),
        train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    # At least one owner's expert_bias should have moved.
    moved = False
    for owner, _label in model.get_all_balancing_owners():
        if not torch.allclose(owner.expert_bias.detach(), initial[id(owner)]):
            moved = True
            break
    assert moved, (
        "nested-only deepseek_bias config (no top-level "
        "training.bias_update_rate) failed to drive any expert_bias "
        "change. The per-class rate must flow through "
        "trainer_post_optimizer_bias_update."
    )


def test_per_class_deepseek_bias_train_cfg_zero_rate_is_no_longer_blocking():
    """Belt-and-suspenders sanity: the trainer's gating used to
    short-circuit when `train_cfg.bias_update_rate == 0`. The fix
    falls back to `config.effective_bias_update_rate` when that's
    set and positive. Probe the helper directly."""
    model, train_cfg, cfg = _build_nested_deepseek()
    # Snapshot.
    initial: dict[int, torch.Tensor] = {}
    for owner, _ in model.get_all_balancing_owners():
        initial[id(owner)] = owner.expert_bias.detach().clone()

    # Force a non-trivial counts state on the MLP owner so the bias
    # update has something to act on.
    for owner, label in model.get_all_balancing_owners():
        if label == "mlp":
            owner.local_tokens_per_expert.zero_()
            owner.local_tokens_per_expert[0] = 100.0  # heavy expert 0
            owner.local_tokens_per_expert[1:] = 1.0
            break

    _ROUTING_MOD.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )
    moved = any(
        not torch.allclose(owner.expert_bias.detach(), initial[id(owner)])
        for owner, _ in model.get_all_balancing_owners()
    )
    assert moved, (
        "trainer_post_optimizer_bias_update returned without updating "
        "expert_bias under nested-only deepseek_bias; effective_* "
        "fallback is not wired correctly."
    )


if __name__ == "__main__":
    print("Run via pytest")
