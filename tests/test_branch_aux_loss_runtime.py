"""Round 32 review Finding 4 (partial): BranchRouter now accepts
`aux_loss` and `seq_aux_loss` and the model's forward adds the
branch contribution to the loss via the same load-balancing
helpers used for MLP / attention routers.

`deepseek_bias` and `quantile` on the branch router still need
owner-state plumbing and are deliberately out of scope here —
they remain rejected at validator construction time.
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


_BASE_YAML = """experiment_name: branch_aux_runtime
model:
  type: moe_everything
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 1
  head_dim: 8
  num_attention_heads: 2
  num_key_value_heads: 2
  num_experts: 4
  num_experts_per_tok: 2
  moe_intermediate_size: 32
  intermediate_size: 32
  norm_topk_prob: true
  router_exploration_rate: 0.0
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_fully_independent
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
  per_head_compute_mode: dense
  use_deepseek_routing: true
  branch_deepseek: false
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 32
  tie_word_embeddings: true
  output_router_logits: true
  attn_implementation: eager
  mlp_router:
    balancing: none
  attn_router:
    balancing: none
  branch_router:
    balancing: %BAL%
    router_aux_loss_coef: 0.5
    seq_aux_loss_coef: 0.5
training:
  learning_rate: 1.0e-3
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 0
  max_steps: 1
  batch_size: 1
  gradient_accumulation: 1
  mixed_precision: ""
  output_dir: /tmp
"""


def _build_with_branch_method(method: str, tmp_path):
    yaml_text = _BASE_YAML.replace("%BAL%", method)
    p = tmp_path / f"branch_{method}.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    return model


def test_branch_aux_loss_drives_branch_gate_gradient(tmp_path):
    """Build a moe_everything with branch_router.balancing=aux_loss
    and a non-zero per-class coef. After backward, the branch
    router's gate weight MUST receive non-zero gradient that
    came from the aux contribution (branch logits flow into
    load_balancing_loss_func via _all_branch_probs / _all_branch_selected_experts).
    """
    model = _build_with_branch_method("aux_loss", tmp_path)
    model.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    out.loss.backward()
    branch = model.model.branch_router
    assert branch.gate.weight.grad is not None
    grad_norm = float(branch.gate.weight.grad.norm().item())
    assert grad_norm > 1e-12, (
        f"branch gate gradient is too small under aux_loss method: "
        f"{grad_norm:.3e}; expected > 1e-12"
    )


def test_branch_seq_aux_loss_drives_branch_gate_gradient(tmp_path):
    """Same contract under seq_aux_loss."""
    model = _build_with_branch_method("seq_aux_loss", tmp_path)
    model.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    out.loss.backward()
    branch = model.model.branch_router
    assert branch.gate.weight.grad is not None
    assert float(branch.gate.weight.grad.norm().item()) > 1e-12


def test_branch_balancing_none_produces_no_extra_loss_term(tmp_path):
    """Negative companion: with branch_router.balancing=none, the
    branch contribution to total loss is zero; loss == ce_loss
    when MLP and attn routers are also `none`.
    """
    yaml_text = _BASE_YAML.replace("%BAL%", "none").replace(
        "    router_aux_loss_coef: 0.5\n    seq_aux_loss_coef: 0.5\n", "",
    )
    p = tmp_path / "branch_none.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    delta = float((out.loss - out.ce_loss).item())
    assert abs(delta) < 1e-6, (
        f"branch_router.balancing=none should leave loss == ce_loss "
        f"when MLP/attn are also `none`; got delta={delta:.3e}"
    )


def test_branch_router_rejects_deepseek_bias_at_construction():
    """`deepseek_bias` on branch_router still requires owner-state
    plumbing — rejected at BranchRouter constructor."""
    from src.models.routing.routers import BranchRouter
    with pytest.raises(ValueError, match="BranchRouter balancing must be"):
        BranchRouter(hidden_size=16, balancing="deepseek_bias")


def test_branch_router_rejects_quantile_at_construction():
    from src.models.routing.routers import BranchRouter
    with pytest.raises(ValueError, match="BranchRouter balancing must be"):
        BranchRouter(hidden_size=16, balancing="quantile")


if __name__ == "__main__":
    print("Run via pytest")
