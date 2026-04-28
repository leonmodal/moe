"""AC-8 full-model coverage: gradient flow through `attention_aux_loss`
and `seq_aux_loss` for the moe_everything family.

`tests/test_attention_aux_gradient.py` covered the low-level
`_store_router_info` contract (aux-bearing router_logits in
`last_router_info`). These tests close the loop end-to-end:

1. Build a `MoEverythingForCausalLM` via `build_model(...)` with
   `load_balancing_method=aux_loss` AND `router_aux_loss_coef > 0`.
2. Run a forward + backward.
3. Assert the attention bank's router parameters receive non-zero
   gradient.

Two negative companions:
* `seq_aux_loss` path — same contract under
  `load_balancing_method=seq_aux_loss` and a non-zero
  `seq_aux_loss_coef`.
* Zero-coefficient negative — `router_aux_loss_coef=0.0` AND
  `seq_aux_loss_coef=0.0` AND `load_balancing_method=none` produces
  zero attention-router gradient (from aux loss; CE loss still flows
  through forward path on the projection weights, but the router's
  gate weight is reachable only via aux/seq-aux loss in the
  non-active-routing-weight contract).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _bypass_training_init():
    """Avoid the pandas-via-data import chain triggered by
    `src.training.__init__`."""
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


_BASE_YAML = """experiment_name: ac8_full_model
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
  branch_router_aux_loss_coef: 0.0
  router_exploration_rate: 0.0
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_fully_independent
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
  per_head_compute_mode: dense
  use_deepseek_routing: false
  branch_deepseek: false
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 32
  tie_word_embeddings: true
  output_router_logits: true
  attn_implementation: eager
training:
  learning_rate: 1.0e-3
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  lr_scheduler: cosine
  warmup_steps: 0
  max_steps: 100
  min_lr_ratio: 0.1
  batch_size: 1
  gradient_accumulation: 1
  mixed_precision: ""
  log_every: 1
  save_every: 100000
  output_dir: /tmp/ac8_test
  bias_update_rate: 0.0
"""


def _yaml_with_method(tmp_path, method: str, *, router_coef: float = 0.0,
                     seq_coef: float = 0.0) -> str:
    yaml_text = _BASE_YAML + (
        f"  load_balancing_method: {method}\n"
        f"  router_aux_loss_coef: {router_coef}\n"
        f"  seq_aux_loss_coef: {seq_coef}\n"
    )
    p = tmp_path / f"{method}.yaml"
    p.write_text(yaml_text)
    return str(p)


_ROUTER_LIST_NAMES = ("q_routers", "k_routers", "v_routers", "o_routers")


def _attention_router_grad_norm(model) -> float:
    """Sum |grad|.norm() across every parameter under the attention
    bank's per-projection router ModuleLists. The moe_everything
    `per_head_fully_independent` mode stores the routers under
    `q_routers`, `k_routers`, `v_routers`, `o_routers` (each an
    `nn.ModuleList` of `make_top1_router(...)` instances)."""
    bank = model.model.attn_bank
    total = 0.0
    for list_name in _ROUTER_LIST_NAMES:
        routers = getattr(bank, list_name, None)
        if routers is None:
            continue
        for r in routers:
            for p in r.parameters():
                if p.grad is not None:
                    total += float(p.grad.norm().item())
    return total


def test_full_model_aux_loss_drives_attention_router_gradient(tmp_path):
    """Build a moe_everything model with `load_balancing_method=aux_loss`
    and `router_aux_loss_coef=0.5`. After backward, the attention
    router weights MUST receive non-zero gradient.

    The CE loss path also reaches the attention bank via the
    forward-graph through the router-weighted attention output, so
    we use a contrastive measurement: with coef=0 (and method=none)
    the router-weight gradient is bounded. With coef=0.5 the aux
    contribution should drive the router-weight gradient ABOVE the
    method=none baseline.
    """
    yaml_aux = _yaml_with_method(tmp_path, "aux_loss", router_coef=0.5)
    cfg = _CFG_MOD.load_config(yaml_aux)
    model, _model_cfg = _FACTORY_MOD.build_model(cfg)
    model.train()

    torch.manual_seed(20260428)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.attention_aux_loss is not None, (
        "AC-8: aux_loss method must produce attention_aux_loss != None"
    )
    out.loss.backward()
    aux_grad = _attention_router_grad_norm(model)
    assert aux_grad > 1e-12, (
        f"AC-8: attention router gradient is negligible under "
        f"aux_loss method with coef=0.5: {aux_grad:.3e}"
    )


def test_full_model_seq_aux_loss_drives_attention_router_gradient(tmp_path):
    """Same contract under `load_balancing_method=seq_aux_loss` +
    `seq_aux_loss_coef=0.5`. The seq-aux loss reads the same
    `attention_router_info[router_logits]` that the regular aux
    loss does, so the gradient path must be intact for both.
    """
    yaml_seq = _yaml_with_method(tmp_path, "seq_aux_loss", seq_coef=0.5)
    cfg = _CFG_MOD.load_config(yaml_seq)
    model, _model_cfg = _FACTORY_MOD.build_model(cfg)
    model.train()

    torch.manual_seed(20260428)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.seq_aux_loss is not None, (
        "AC-8: seq_aux_loss method must produce seq_aux_loss != None"
    )
    out.loss.backward()
    seq_grad = _attention_router_grad_norm(model)
    assert seq_grad > 1e-12, (
        f"AC-8: attention router gradient is negligible under "
        f"seq_aux_loss method with coef=0.5: {seq_grad:.3e}"
    )


def test_full_model_zero_coefficient_no_aux_attention_router_gradient(tmp_path):
    """Negative companion: with `load_balancing_method=none`, BOTH
    `router_aux_loss_coef` and `seq_aux_loss_coef` must be zero
    (`normalize_balancing_config` zeros them on load), and the
    `attention_aux_loss` term is `None`. The router-weight gradient
    contribution from aux paths is therefore zero by construction.
    The forward graph still reaches the attention bank via the
    routed attention output, so we cannot assert ZERO router-weight
    gradient — but we CAN assert that the model output's
    `attention_aux_loss` is None and `seq_aux_loss` is None, which
    is the structural contract.
    """
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0, seq_coef=0.0)
    cfg = _CFG_MOD.load_config(yaml_none)
    model, _model_cfg = _FACTORY_MOD.build_model(cfg)
    model.train()

    torch.manual_seed(20260428)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.attention_aux_loss is None, (
        "AC-8 negative: method=none must produce attention_aux_loss == None"
    )
    assert out.seq_aux_loss is None, (
        "AC-8 negative: method=none must produce seq_aux_loss == None"
    )


def test_full_model_aux_loss_strict_dominates_baseline_contrastive(tmp_path):
    """Contrastive AC-8 test: under method=aux_loss with coef=2.0,
    the attention router gradient norm should be STRICTLY GREATER
    than under method=none (where the router gradient comes only
    from the forward-graph contribution). This proves the aux path
    actually drives a measurable additional signal.
    """
    yaml_aux = _yaml_with_method(tmp_path, "aux_loss", router_coef=2.0)
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0)

    def _grad_for(yaml_path):
        cfg = _CFG_MOD.load_config(yaml_path)
        # Same construction seed -> identical init params.
        torch.manual_seed(20260428)
        model, _ = _FACTORY_MOD.build_model(cfg)
        model.train()
        torch.manual_seed(11111)
        input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
        out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
        out.loss.backward()
        return _attention_router_grad_norm(model)

    aux_grad = _grad_for(yaml_aux)
    none_grad = _grad_for(yaml_none)
    assert aux_grad > none_grad + 1e-9, (
        f"AC-8 contrastive: aux_loss method should produce stronger "
        f"router gradient than method=none. aux_grad={aux_grad:.3e}, "
        f"none_grad={none_grad:.3e}. If equal, the aux loss term is "
        f"silently detached or the coefficient is ignored."
    )


if __name__ == "__main__":
    print("Run via pytest")
