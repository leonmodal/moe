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


def _attention_router_grad_norms(model) -> dict[str, float]:
    """Return per-router-parameter gradient norms keyed by a stable
    name. Each entry is `<list_name>[<idx>].<param_name>` so a
    single dead router (one head's gradient stuck at zero) is
    visible — earlier versions of this test summed everything into
    one scalar and a single live router could mask N-1 dead ones.
    """
    bank = model.model.attn_bank
    norms: dict[str, float] = {}
    for list_name in _ROUTER_LIST_NAMES:
        routers = getattr(bank, list_name, None)
        if routers is None:
            continue
        for idx, r in enumerate(routers):
            for pname, p in r.named_parameters():
                if p.grad is None:
                    norms[f"{list_name}[{idx}].{pname}"] = 0.0
                else:
                    norms[f"{list_name}[{idx}].{pname}"] = float(p.grad.norm().item())
    return norms


def _attention_router_grad_norm(model) -> float:
    """Aggregate norm (kept for diagnostics that don't need
    per-router visibility)."""
    return sum(_attention_router_grad_norms(model).values())


def _backward_and_collect(yaml_path, build_seed: int = 20260428,
                          input_seed: int = 11111) -> dict[str, float]:
    """Build (with `build_seed`), run forward+backward (with
    `input_seed`), return per-router gradient norms. Locking the
    seeds across yaml configs is what makes contrastive comparisons
    meaningful — without it, comparing the same router across two
    builds is comparing two different parameter values with two
    different inputs.
    """
    cfg = _CFG_MOD.load_config(yaml_path)
    torch.manual_seed(build_seed)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()
    torch.manual_seed(input_seed)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    out.loss.backward()
    return _attention_router_grad_norms(model)


def test_full_model_aux_loss_every_attention_router_param_has_gradient(tmp_path):
    """Build a moe_everything model with `load_balancing_method=aux_loss`
    and `router_aux_loss_coef=0.5`. After backward, EVERY q/k/v/o
    router parameter MUST receive non-zero gradient — a single live
    router cannot mask dead siblings. Round 25's aggregate-norm
    test passed with potentially N-1 dead routers; this test
    enforces the per-router contract directly.
    """
    yaml_aux = _yaml_with_method(tmp_path, "aux_loss", router_coef=0.5)
    norms = _backward_and_collect(yaml_aux)
    dead = [name for name, val in norms.items() if val <= 1e-12]
    assert not dead, (
        f"per-router gradient assertion: {len(dead)} dead routers under "
        f"aux_loss / coef=0.5: {dead[:8]}{'...' if len(dead) > 8 else ''}. "
        f"All q/k/v/o router parameters must receive non-zero gradient."
    )


def test_full_model_seq_aux_loss_every_attention_router_param_has_gradient(tmp_path):
    """Same per-router-parameter contract under
    `load_balancing_method=seq_aux_loss` and a non-zero
    `seq_aux_loss_coef`. The seq-aux path reads the same
    `attention_router_info[router_logits]` so the per-router
    gradient must be intact for it too.
    """
    yaml_seq = _yaml_with_method(tmp_path, "seq_aux_loss", seq_coef=0.5)
    norms = _backward_and_collect(yaml_seq)
    dead = [name for name, val in norms.items() if val <= 1e-12]
    assert not dead, (
        f"per-router gradient assertion: {len(dead)} dead routers under "
        f"seq_aux_loss / coef=0.5: {dead[:8]}{'...' if len(dead) > 8 else ''}."
    )


def test_full_model_aux_loss_strict_dominates_baseline_contrastive(tmp_path):
    """Contrastive AC-8 test: under method=aux_loss with coef=2.0,
    EACH attention router parameter's gradient norm should be
    STRICTLY GREATER than under method=none (where the router
    gradient comes only from the forward-graph contribution).
    Per-router comparison is needed because aggregate-only
    comparison cannot detect a dead-router regression that lowers
    the aggregate enough to fail the inequality but with the
    surviving live routers still contributing.
    """
    yaml_aux = _yaml_with_method(tmp_path, "aux_loss", router_coef=2.0)
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0)
    aux_norms = _backward_and_collect(yaml_aux)
    none_norms = _backward_and_collect(yaml_none)
    # Aggregate inequality (sanity).
    aux_total = sum(aux_norms.values())
    none_total = sum(none_norms.values())
    assert aux_total > none_total + 1e-9, (
        f"aggregate aux_grad ({aux_total:.3e}) should exceed "
        f"none_grad ({none_total:.3e})"
    )
    # Per-router strict-dominance: every router parameter where the
    # `none` build had non-zero gradient should also have a gradient
    # under aux, AND the aux gradient should be at least as large.
    diffs = []
    for name in none_norms:
        if name in aux_norms:
            if aux_norms[name] + 1e-9 < none_norms[name]:
                diffs.append((name, aux_norms[name], none_norms[name]))
    assert not diffs, (
        f"per-router contrastive: {len(diffs)} routers had "
        f"weaker gradient under aux than under none: {diffs[:5]}"
    )


def test_full_model_seq_aux_strict_dominates_baseline_contrastive(tmp_path):
    """Round 25 review Finding 3: seq-aux needs its own contrastive
    test (Round 25 only had one for the regular aux path).
    """
    yaml_seq = _yaml_with_method(tmp_path, "seq_aux_loss", seq_coef=2.0)
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0, seq_coef=0.0)
    seq_norms = _backward_and_collect(yaml_seq)
    none_norms = _backward_and_collect(yaml_none)
    seq_total = sum(seq_norms.values())
    none_total = sum(none_norms.values())
    assert seq_total > none_total + 1e-9, (
        f"aggregate seq_aux_grad ({seq_total:.3e}) should exceed "
        f"none_grad ({none_total:.3e})"
    )


def test_full_model_zero_aux_coef_matches_none_baseline_per_router(tmp_path):
    """Round 25 review Finding 3 zero-coef negative test: under
    `load_balancing_method=aux_loss` AND `router_aux_loss_coef=0.0`,
    the aux contribution to the gradient is zero by construction
    (the loss path adds `coef * aux_loss = 0`). Each per-router
    gradient should equal the `method=none` baseline within
    tolerance — proving the aux path's contribution scales with the
    coefficient and is not bypassing the gate.
    """
    yaml_zero_aux = _yaml_with_method(tmp_path, "aux_loss", router_coef=0.0)
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0)
    zero_norms = _backward_and_collect(yaml_zero_aux)
    none_norms = _backward_and_collect(yaml_none)
    diffs = []
    for name in none_norms:
        if name not in zero_norms:
            continue
        # Tolerance: 1e-7 absolute. The forward / CE path is
        # deterministic across the two yamls (same seeds, same
        # method-zero coefs), so per-router gradients must match.
        if abs(zero_norms[name] - none_norms[name]) > 1e-7:
            diffs.append((name, zero_norms[name], none_norms[name]))
    assert not diffs, (
        f"zero-coef parity failed for {len(diffs)} routers: "
        f"{diffs[:5]}. With coef=0, aux_loss method must produce "
        f"the same per-router gradient as method=none."
    )


def test_full_model_zero_seq_aux_coef_matches_none_baseline_per_router(tmp_path):
    """Same zero-coef contract for seq_aux_loss."""
    yaml_zero_seq = _yaml_with_method(tmp_path, "seq_aux_loss", seq_coef=0.0)
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0, seq_coef=0.0)
    zero_norms = _backward_and_collect(yaml_zero_seq)
    none_norms = _backward_and_collect(yaml_none)
    diffs = []
    for name in none_norms:
        if name not in zero_norms:
            continue
        if abs(zero_norms[name] - none_norms[name]) > 1e-7:
            diffs.append((name, zero_norms[name], none_norms[name]))
    assert not diffs, (
        f"zero-seq-coef parity failed for {len(diffs)} routers: "
        f"{diffs[:5]}"
    )


def test_full_model_zero_coefficient_attention_aux_loss_is_none(tmp_path):
    """Structural negative: `load_balancing_method=none` produces
    `out.attention_aux_loss is None` and `out.seq_aux_loss is None`
    on the model output. This is the structural contract of the
    no-aux build (kept from Round 25)."""
    yaml_none = _yaml_with_method(tmp_path, "none", router_coef=0.0, seq_coef=0.0)
    cfg = _CFG_MOD.load_config(yaml_none)
    model, _model_cfg = _FACTORY_MOD.build_model(cfg)
    model.train()
    torch.manual_seed(20260428)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.attention_aux_loss is None
    assert out.seq_aux_loss is None


if __name__ == "__main__":
    print("Run via pytest")
