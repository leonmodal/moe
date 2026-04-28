"""Round 29 review Finding 2: standard_moe / global_moe non-aux paths
must not leak Switch-aux gradient through `output.loss`. Codex's probe
showed `grad_diff_norm = 8.6e-5` between method='none' (claimed
no-aux) and a true no-aux baseline; the bug was that `old_aux.detach()`
cancelled only the scalar VALUE, leaving the autograd graph path
`coef * old_aux -> router_weights` alive.

These tests pin the contract: under non-aux methods the gradient on
router weights from the loss must be IDENTICAL to a build that
forces `output_router_logits=False` (and thus computes no aux at
all).
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
    return cfg_mod, factory_mod


_CFG_MOD, _FACTORY_MOD = _bypass_training_init()


def _build_minimal_moe(family: str, *, method_via_top_level: bool):
    """Build a tiny standard_moe or global_moe model.

    `method_via_top_level=True` sets the top-level `load_balancing_method`
    (legacy shape).  `method_via_top_level=False` uses the nested
    `mlp_router.balancing` (post-migration nested-only shape). Both
    must produce the same gradients on router weights — the
    non-aux gradient leak fix proves the two are equivalent.
    """
    base = {
        "model": {
            "type": family,
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
            "output_router_logits": True,
            "attn_implementation": "eager",
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
        },
    }
    if method_via_top_level:
        base["training"]["load_balancing_method"] = "none"
    else:
        base["model"]["mlp_router"] = {"balancing": "none"}
        base["model"]["attn_router"] = {"balancing": "none"}
        base["model"]["branch_router"] = {"balancing": "none"}
    # normalize_balancing_config + validate run on load_config; use
    # build_model directly with the dict here.
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(base)
    return model


def _gradient_norms(model, *, seed: int = 11111):
    """Run forward + backward on a fixed seed; return summed
    `param.grad.norm()` over every router-bearing parameter so the
    test compares only the routing-affecting gradient surface."""
    model.train()
    torch.manual_seed(seed)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    out.loss.backward()
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "router" in name or "gate" in name:
            total += float(p.grad.norm().item())
    return total, float(out.loss.item())


@pytest.mark.parametrize("family", ["standard_moe", "global_moe"])
def test_nested_method_none_does_not_leak_switch_aux_gradient(family):
    """The probe Codex ran: with method='none' (via the nested
    schema, mlp_router.balancing='none'), the router-weight
    gradient must be IDENTICAL to the top-level `method='none'`
    legacy path. Earlier rounds showed a 8.6e-5 grad delta because
    `old_aux.detach()` cancelled scalar value but not the autograd
    graph. Round 30 uses graph-bearing old_aux subtraction so the
    two builds match.
    """
    model_top = _build_minimal_moe(family, method_via_top_level=True)
    model_nested = _build_minimal_moe(family, method_via_top_level=False)
    grad_top, loss_top = _gradient_norms(model_top)
    grad_nested, loss_nested = _gradient_norms(model_nested)
    # Loss values should match exactly (both build to same model).
    assert abs(loss_top - loss_nested) < 1e-9, (
        f"{family}: loss diverged between top-level and nested "
        f"method=none builds: top={loss_top}, nested={loss_nested}"
    )
    # Router-weight gradient sums must match within fp32 tolerance.
    assert abs(grad_top - grad_nested) < 1e-9, (
        f"{family}: router-weight gradient diverged between "
        f"top-level method=none ({grad_top}) and nested "
        f"mlp_router.balancing=none ({grad_nested}). The Switch-aux "
        f"gradient leak fix should make these identical."
    )


@pytest.mark.parametrize("family", ["standard_moe", "global_moe"])
def test_method_none_router_weight_gradient_matches_no_router_logits_baseline(family):
    """A stronger contract: under method=none, the router-weight
    gradient should equal the baseline that NEVER computed aux at
    all (i.e. `output_router_logits=False`). Earlier rounds had a
    1e-4-scale residual; Round 30 should drive it to bit-equality.
    """
    model_none = _build_minimal_moe(family, method_via_top_level=False)
    base = {
        "model": {
            "type": family,
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
            "mlp_router": {"balancing": "none"},
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
        },
    }
    torch.manual_seed(20260428)
    model_baseline, _ = _FACTORY_MOD.build_model(base)
    grad_none, _ = _gradient_norms(model_none)
    grad_baseline, _ = _gradient_norms(model_baseline)
    # Tolerance: 1e-7 is plenty for fp32. The Round 29 probe showed
    # 8.6e-5 leak, far above this floor.
    assert abs(grad_none - grad_baseline) < 1e-7, (
        f"{family}: method=none router gradient ({grad_none}) "
        f"differs from no-router-logits baseline ({grad_baseline}) "
        f"by {abs(grad_none - grad_baseline):.3e}; expected near-zero."
    )


if __name__ == "__main__":
    print("Run via pytest")
