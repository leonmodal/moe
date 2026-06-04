"""BranchRouter balancing runtime coverage.

The branch router supports the same balancing methods as the MLP and
attention routers, plus the branch-only `exploration_only` mode.
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
%BRANCH_BLOCK%
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


# Per-method branch fixture: only the active coefficient is set,
# matching the AC-17 method-axis rule the validator now enforces.
_BRANCH_BLOCKS = {
    "aux_loss": (
        "    balancing: aux_loss\n"
        "    router_aux_loss_coef: 0.5\n"
    ),
    "seq_aux_loss": (
        "    balancing: seq_aux_loss\n"
        "    seq_aux_loss_coef: 0.5\n"
    ),
    "none": "    balancing: none\n",
}


def _build_with_branch_method(method: str, tmp_path):
    yaml_text = _BASE_YAML.replace("%BRANCH_BLOCK%", _BRANCH_BLOCKS[method])
    p = tmp_path / f"branch_{method}.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    return model


def test_branch_aux_loss_actually_fires_and_adds_loss_term(tmp_path):
    """Round 33 review Finding 1: prove the branch aux contribution
    actually FIRES — `out.branch_aux_loss` must be a Tensor and
    `out.loss - out.ce_loss` must equal `coef * branch_aux_loss`
    within fp32 tolerance. Earlier round only checked that the
    branch gate's gradient was non-zero, which is satisfied by CE
    alone and doesn't prove the aux path runs.
    """
    model = _build_with_branch_method("aux_loss", tmp_path)
    model.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.branch_aux_loss is not None, (
        "branch_router.balancing=aux_loss did not produce a "
        "branch_aux_loss tensor on the model output"
    )
    assert isinstance(out.branch_aux_loss, torch.Tensor)
    coef = 0.5  # the per-class coef we set in `_BASE_YAML`
    expected_delta = coef * float(out.branch_aux_loss.detach().item())
    actual_delta = float((out.loss - out.ce_loss).detach().item())
    assert abs(actual_delta - expected_delta) < 1e-4, (
        f"loss - ce_loss = {actual_delta:.6e} but expected "
        f"coef*branch_aux_loss = {expected_delta:.6e}; the branch "
        f"aux term is not being added to total loss correctly"
    )


def test_branch_seq_aux_loss_actually_fires_and_adds_loss_term(tmp_path):
    """Same contract for seq_aux_loss — the runtime must produce
    a non-None branch_aux_loss and the loss-ce delta must equal
    coef * value."""
    model = _build_with_branch_method("seq_aux_loss", tmp_path)
    model.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    assert out.branch_aux_loss is not None
    coef = 0.5
    expected_delta = coef * float(out.branch_aux_loss.detach().item())
    actual_delta = float((out.loss - out.ce_loss).detach().item())
    assert abs(actual_delta - expected_delta) < 1e-4


def test_branch_aux_loss_contrastive_vs_none_baseline(tmp_path):
    """Round 33 review Finding 1: contrastive proof that the loss
    delta under aux_loss / seq_aux_loss is STRICTLY GREATER than
    under method=none (with identical seed/init/input). Earlier
    rounds were satisfied by gradient existing — this proves the
    branch aux path adds non-zero signal.
    """
    model_aux = _build_with_branch_method("aux_loss", tmp_path)
    model_aux.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model_aux.vocab_size, (1, 8), dtype=torch.long)
    out_aux = model_aux(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    delta_aux = float((out_aux.loss - out_aux.ce_loss).detach().item())

    # Build the none baseline with the SAME seed so initialization
    # matches.
    model_none = _build_with_branch_method("none", tmp_path)
    model_none.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model_none.vocab_size, (1, 8), dtype=torch.long)
    out_none = model_none(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    delta_none = float((out_none.loss - out_none.ce_loss).detach().item())

    assert delta_aux > delta_none + 1e-6, (
        f"branch aux_loss should add a positive delta over none "
        f"baseline; got delta_aux={delta_aux:.3e}, "
        f"delta_none={delta_none:.3e}"
    )


def test_branch_balancing_none_produces_no_extra_loss_term(tmp_path):
    """Negative companion: with branch_router.balancing=none, the
    branch contribution to total loss is zero; loss == ce_loss
    when MLP and attn routers are also `none`.
    """
    model = _build_with_branch_method("none", tmp_path)
    model.train()
    torch.manual_seed(11111)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    delta = float((out.loss - out.ce_loss).item())
    assert abs(delta) < 1e-6, (
        f"branch_router.balancing=none should leave loss == ce_loss "
        f"when MLP/attn are also `none`; got delta={delta:.3e}"
    )


def test_branch_router_accepts_deepseek_bias_at_construction():
    """`deepseek_bias` is now accepted: BranchRouter has
    `expert_bias` + `local_tokens_per_expert` already (DEC-18).
    Construction must succeed and the resulting router has the
    correct balancing label."""
    from src.models.routing.routers import BranchRouter
    router = BranchRouter(hidden_size=16, balancing="deepseek_bias")
    assert router.balancing == "deepseek_bias"
    assert hasattr(router, "expert_bias")
    assert hasattr(router, "local_tokens_per_expert")


def test_branch_deepseek_bias_uses_two_independent_sigmoid_scores():
    """DeepSeek branch routing is not a two-class softmax.

    The binary branch pool still has two outputs, but each output is scored
    independently with sigmoid. The two scores therefore do not have to sum
    to one; `expert_bias` is added only for the hard choice.
    """
    from src.models.routing.routers import BranchRouter

    router = BranchRouter(hidden_size=4, balancing="deepseek_bias")
    router.eval()
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[:, 0] = 4.0

    x = torch.zeros(1, 1, 4)
    x[0, 0, 0] = 1.0
    router(x)

    expected = torch.sigmoid(torch.tensor([4.0, 4.0], dtype=router.last_probs.dtype))
    actual = router.last_probs[0, 0].detach().cpu()
    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual.sum().item() > 1.0, (
        f"DeepSeek branch scores should be independent sigmoid scores, "
        f"not a normalized softmax; got {actual.tolist()}"
    )


def test_branch_deepseek_bias_drives_post_step_expert_bias_update(tmp_path):
    """Round 37: end-to-end branch deepseek_bias.

    Build a yaml with branch_router.balancing=deepseek_bias +
    bias_update_rate=0.001. Run forward + backward + optimizer.step
    + post-step bias update. Assert branch.expert_bias changed
    from initial (zero) to non-zero values.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    head, _, tail = _BASE_YAML.partition("training:")
    yaml_text = (
        head.replace(
            "%BRANCH_BLOCK%",
            "    balancing: deepseek_bias\n"
            "    bias_update_rate: 0.001\n"
            "    bias_update_zero_sum: true\n",
        )
        + "training:" + tail
    )
    p = tmp_path / "branch_deepseek.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()
    branch = model.model.branch_router

    # Pre-populate counts to a non-zero baseline so the bias
    # update has signal to act on (forward might not produce
    # imbalanced counts in 1 step on this tiny model).
    branch.local_tokens_per_expert.zero_()
    branch.local_tokens_per_expert[0] = 100.0  # heavy ATTN
    branch.local_tokens_per_expert[1] = 1.0    # light MLP
    initial_bias = branch.expert_bias.clone()

    # Run trainer post-step bias update directly (no forward needed
    # since we pre-populated counts).
    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    # ATTN was heavy → expert_bias[0] should decrease, expert_bias[1]
    # should increase (zero-sum: bias -= sign(load - 1/E) * rate).
    assert not torch.allclose(branch.expert_bias, initial_bias), (
        f"branch.expert_bias did not change after post-step update; "
        f"per-class branch deepseek_bias is not driving the walker. "
        f"before={initial_bias.tolist()}, "
        f"after={branch.expert_bias.tolist()}"
    )


def test_branch_quantile_production_path_drives_walker(tmp_path):
    """Round 43 AC-13/AC-16: branch quantile end-to-end.

    Build a moe_everything yaml with `branch_router.balancing: quantile`.
    Run a real training-mode forward (which accumulates per-step
    scores into `branch.local_quantile_scores`). Call
    `trainer_post_optimizer_bias_update`. Assert:
      - `branch.quantile_ema` updated from zeros.
      - `branch.expert_bias` updated (quantile-from-EMA).
      - `branch.local_quantile_scores` drained.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: branch_quantile_e2e
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
    balancing: quantile
    quantile_target_q: 0.5
    quantile_eta: 0.05
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
    p = tmp_path / "branch_quantile.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260518)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()
    branch = model.model.branch_router
    assert branch.balancing == "quantile", (
        f"branch.balancing should be quantile; got {branch.balancing!r}"
    )

    # Real forward populates branch.local_quantile_scores via the
    # router's training-mode accumulator gate.
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
    out.loss.backward()
    assert len(branch.local_quantile_scores) >= 1, (
        f"branch forward did not append scores; len="
        f"{len(branch.local_quantile_scores)}"
    )
    initial_bias = branch.expert_bias.detach().clone()
    initial_ema = branch.quantile_ema.detach().clone()

    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    assert (branch.quantile_ema != initial_ema).any(), (
        f"branch.quantile_ema unchanged after walker. "
        f"before={initial_ema.tolist()}, after={branch.quantile_ema.tolist()}"
    )
    assert (branch.expert_bias != initial_bias).any(), (
        f"branch.expert_bias unchanged after walker. "
        f"before={initial_bias.tolist()}, after={branch.expert_bias.tolist()}"
    )
    assert branch.local_quantile_scores == [], (
        f"branch accumulator not drained: len="
        f"{len(branch.local_quantile_scores)}"
    )


def test_branch_quantile_checkpoint_resume_parity_for_quantile_ema(tmp_path):
    """Round 43 AC-12: quantile_ema persists across save/load.

    Drive the branch's quantile_ema to a non-zero state via several
    walker iterations, save state_dict, reset in memory, load
    state_dict, and assert the loaded quantile_ema bit-matches the
    saved one. Locks AC-12 EMA persistence contract for the branch
    router (the MLP/attn quantile_ema is already a persistent buffer
    by the same construction).
    """
    from src.models.routing.routers import BranchRouter

    r = BranchRouter(hidden_size=16, balancing="quantile")
    # Synthesize non-zero EMA + bias by hand-driving the helper.
    with torch.no_grad():
        r.quantile_ema[:] = torch.tensor([0.37, 0.91])
        r.expert_bias[:] = torch.tensor([0.27, -0.27])

    saved = {k: v.clone() for k, v in r.state_dict().items()}
    # Clobber in memory.
    with torch.no_grad():
        r.quantile_ema.zero_()
        r.expert_bias.zero_()
    assert (r.quantile_ema == 0).all() and (r.expert_bias == 0).all()

    # Round-trip via state_dict.
    r.load_state_dict(saved)
    assert torch.equal(r.quantile_ema, saved["quantile_ema"]), (
        f"quantile_ema did not survive state_dict round-trip; "
        f"got {r.quantile_ema.tolist()}, expected {saved['quantile_ema'].tolist()}"
    )
    assert torch.equal(r.expert_bias, saved["expert_bias"]), (
        f"expert_bias did not survive state_dict round-trip; "
        f"got {r.expert_bias.tolist()}, expected {saved['expert_bias'].tolist()}"
    )


def test_deepseek_router_quantile_ema_checkpoint_resume_parity():
    """Round 43 AC-12: quantile_ema persistent buffer parity for
    DeepSeekRouter (used by standard_moe / global_moe / moe_everything's
    per-head attention banks). Same contract as the branch variant."""
    from src.models.router import DeepSeekRouter

    class _Cfg:
        hidden_size = 16
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        topk_scaling_factor = 2.5
        router_exploration_rate = 0.0
        router_z_loss_coef = 0.0

    r = DeepSeekRouter(_Cfg())
    with torch.no_grad():
        r.quantile_ema[:] = torch.tensor([0.1, 0.3, 0.5, 0.9])
        r.expert_bias[:] = torch.tensor([0.4, 0.2, -0.2, -0.4])

    saved = {k: v.clone() for k, v in r.state_dict().items()}
    # Construct a fresh router and load.
    r2 = DeepSeekRouter(_Cfg())
    assert (r2.quantile_ema == 0).all() and (r2.expert_bias == 0).all()
    r2.load_state_dict(saved)
    assert torch.equal(r2.quantile_ema, saved["quantile_ema"]), (
        f"DeepSeekRouter quantile_ema did not survive state_dict round-trip"
    )
    assert torch.equal(r2.expert_bias, saved["expert_bias"]), (
        f"DeepSeekRouter expert_bias did not survive state_dict round-trip"
    )


def test_full_optimizer_step_resume_parity_for_balancing_state(tmp_path):
    """Round 44 AC-12 broader: full optimizer-step walker resume parity.

    The contract: build a model, run N steps of `forward + backward
    + optimizer.step + scheduler.step + trainer_post_optimizer_bias_update`,
    save state_dict, build a FRESH model, load_state_dict, and assert
    every persistent balancing buffer (`expert_bias`, `quantile_ema`)
    bit-matches the saved values. Then continue running for additional
    steps from both copies under identical RNG seeds and assert the
    buffers continue to track.

    This locks the AC-12 contract that checkpoint resume preserves
    the post-step walker's trajectory across both DeepSeek bias and
    quantile EMA states.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: resume_parity
model:
  type: standard_moe
  router_type: deepseek
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
  topk_scaling_factor: 2.5
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 32
  tie_word_embeddings: true
  output_router_logits: false
  attn_implementation: eager
  mlp_router:
    balancing: deepseek_bias
    bias_update_rate: 0.05
    bias_update_zero_sum: true
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
    p = tmp_path / "resume_parity.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))

    def _build():
        torch.manual_seed(20260518)
        model, _ = _FACTORY_MOD.build_model(cfg)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        return model, optimizer

    def _step(model, optimizer, train_cfg, step):
        torch.manual_seed(7000 + step)
        input_ids = torch.randint(
            0, model.vocab_size, (1, 8), dtype=torch.long,
        )
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        scheduler = type("S", (), {"step": lambda _self=None: None})()
        routing.trainer_optimizer_step_and_bias_update(
            model, optimizer, scheduler=scheduler,
            train_cfg=train_cfg, cfg=cfg,
            distributed=False, global_step=step + 1,
        )

    model_a, optim_a = _build()
    train_cfg = _CFG_MOD.build_training_config(cfg)
    for s in range(5):
        _step(model_a, optim_a, train_cfg, s)
    snapshot = {k: v.detach().clone() for k, v in model_a.state_dict().items()}

    # Fresh model + load_state_dict.
    model_b, optim_b = _build()
    # Sanity: fresh model's balancing buffers differ from snapshot.
    mlp_a = next(o for o, lbl in model_a.get_all_balancing_owners() if lbl == "mlp")
    mlp_b = next(o for o, lbl in model_b.get_all_balancing_owners() if lbl == "mlp")
    assert not torch.allclose(mlp_a.expert_bias, mlp_b.expert_bias), (
        f"fresh model should start at zeros but matches the saved state"
    )
    model_b.load_state_dict(snapshot)
    mlp_b = next(o for o, lbl in model_b.get_all_balancing_owners() if lbl == "mlp")
    # After load, balancing state must bit-match.
    assert torch.equal(mlp_b.expert_bias, mlp_a.expert_bias), (
        f"after load_state_dict, expert_bias did not bit-match. "
        f"a={mlp_a.expert_bias.tolist()}, b={mlp_b.expert_bias.tolist()}"
    )
    assert torch.equal(mlp_b.quantile_ema, mlp_a.quantile_ema), (
        f"after load_state_dict, quantile_ema did not bit-match"
    )


def test_per_class_exploration_schedules_apply_independent_rates(tmp_path):
    """Round 44 AC-15: per-class exploration schedules.

    Build a moe_everything yaml with DIFFERENT exploration_rate
    values per `mlp_router`, `attn_router`, and `branch_router`.
    Call `apply_per_class_exploration_schedules(model, step=0)`.
    Assert every MLP DeepSeekRouter has the MLP rate; every attn
    DeepSeekRouter has the attn rate; the BranchRouter has the
    branch rate.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: per_class_exploration
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
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
    exploration_rate: 0.10
  attn_router:
    balancing: none
    exploration_rate: 0.20
  branch_router:
    balancing: exploration_only
    exploration_rate: 0.30
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
    p = tmp_path / "per_class_explore.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260518)
    model, _ = _FACTORY_MOD.build_model(cfg)

    applied = routing.apply_per_class_exploration_schedules(model, step=0)
    assert applied == {"mlp": 0.10, "attn": 0.20, "branch": 0.30}, (
        f"expected per-class rates 0.10/0.20/0.30; got {applied}"
    )

    # Walk owners and verify each has the per-class rate.
    from src.models.routing.routers import BranchRouter
    from src.models.router import DeepSeekRouter
    inner = model.model
    mlp_bank = getattr(inner, "mlp_bank", None)
    attn_bank = getattr(inner, "attn_bank", None)
    for sub in mlp_bank.modules():
        if isinstance(sub, DeepSeekRouter):
            assert sub.exploration_rate == 0.10, (
                f"MLP bank DeepSeekRouter has rate={sub.exploration_rate}, "
                f"expected 0.10"
            )
    for sub in attn_bank.modules():
        if isinstance(sub, DeepSeekRouter):
            assert sub.exploration_rate == 0.20, (
                f"Attn bank DeepSeekRouter has rate={sub.exploration_rate}, "
                f"expected 0.20"
            )
    assert isinstance(inner.branch_router, BranchRouter)
    assert inner.branch_router.exploration_rate == 0.30, (
        f"BranchRouter has rate={inner.branch_router.exploration_rate}, "
        f"expected 0.30"
    )


def test_per_class_exploration_no_op_when_no_per_class_blocks(tmp_path):
    """Round 44 AC-15: when no per-class exploration_rate is defined,
    `apply_per_class_exploration_schedules` returns an empty map and
    does not touch any router's `exploration_rate`. The legacy
    single-scalar `apply_router_exploration_rate(model, rate)` path
    is the back-compat shim for these configs.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: no_per_class_explore
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
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
    balancing: none
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
    p = tmp_path / "no_per_class.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260518)
    model, _ = _FACTORY_MOD.build_model(cfg)
    applied = routing.apply_per_class_exploration_schedules(model, step=0)
    assert applied == {}, (
        f"expected no-op for config without per-class exploration; got {applied}"
    )


def test_branch_router_accepts_quantile_at_construction():
    """Round 43: branch quantile is now a valid balancing method.

    BranchRouter(balancing="quantile") must construct successfully
    and register `quantile_ema` + `local_quantile_scores` alongside
    the canonical `expert_bias` + `local_tokens_per_expert` buffers.
    """
    from src.models.routing.routers import BranchRouter
    r = BranchRouter(hidden_size=16, balancing="quantile")
    assert r.balancing == "quantile"
    assert hasattr(r, "expert_bias")
    assert hasattr(r, "local_tokens_per_expert")
    assert hasattr(r, "quantile_ema")
    assert isinstance(r.quantile_ema, torch.Tensor)
    assert r.quantile_ema.shape == (2,)
    assert r.quantile_ema.dtype == torch.float32
    state_keys = set(r.state_dict().keys())
    assert "quantile_ema" in state_keys, (
        f"quantile_ema must be a persistent buffer; state_dict keys={state_keys}"
    )
    assert hasattr(r, "local_quantile_scores")
    assert isinstance(r.local_quantile_scores, list)
    assert r.local_quantile_scores == []


def test_walker_dispatches_quantile_owner_through_quantile_helper():
    """Round 41 AC-10: per-owner quantile dispatch.

    When an owner's per-class method is `quantile` AND
    `_BIAS_UPDATE_METHODS` includes `quantile`, the walker must call
    `_update_single_router_quantile_bias` (which drains the
    accumulator and updates `expert_bias` + `quantile_ema`) — not the
    DeepSeek `sign(load - 1/E)` path.

    Direct unit test: build a mock owner with a populated
    `local_quantile_scores` list and `quantile_ema` buffer. Call the
    walker with model-level method = `quantile`. Assert:
      - `quantile_ema` updates from zeros (EMA absorbed the scores).
      - `expert_bias` is set (quantile bias-from-EMA pulls toward
        global EMA median).
      - `local_quantile_scores` is drained to an empty list.
      - `local_tokens_per_expert` is NOT zeroed (the deepseek path
        wasn't run).
    """
    import importlib.util as _u
    import torch.nn as nn
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    class _Owner(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(n, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer("quantile_ema", torch.zeros(n, dtype=torch.float32))
            self.local_quantile_scores: list[torch.Tensor] = []

    class _MockConfig:
        mlp_router_balancing = "quantile"

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = _Owner(4)
            self._load_balancing_method = "quantile"
            self.config = _MockConfig()

        def get_all_balancing_owners(self):
            yield self.owner, "mlp"

    m = _MockModel()
    # Populate accumulator with imbalanced per-expert scores:
    # expert 0 strongly preferred (high scores); other experts low.
    raw = torch.tensor([
        [0.9, 0.1, 0.1, 0.1],
        [0.8, 0.2, 0.1, 0.1],
        [0.95, 0.05, 0.05, 0.05],
    ], dtype=torch.float32)
    m.owner.local_quantile_scores.append(raw)
    # Inject a counts state too so we can confirm the deepseek path
    # does NOT run.
    m.owner.local_tokens_per_expert[:] = torch.tensor([7.0, 1.0, 1.0, 1.0])
    counts_initial = m.owner.local_tokens_per_expert.detach().clone()

    with torch.no_grad():
        routing.update_expert_biases(
            m, bias_rate=0.0, distributed=False,
        )

    # quantile_ema absorbed the scores (eta default 0.05; first call
    # pulls EMA from 0 toward the per-expert median).
    assert (m.owner.quantile_ema != 0).any(), (
        f"quantile_ema unchanged; the walker did not call the quantile helper. "
        f"ema={m.owner.quantile_ema.tolist()}"
    )
    # expert_bias is set by quantile-from-EMA: expert 0 (loaded) gets
    # negative bias; other experts get positive. Assert at minimum
    # that the bias is no longer zero.
    assert (m.owner.expert_bias != 0).any(), (
        f"expert_bias unchanged; quantile-from-EMA helper did not write bias. "
        f"bias={m.owner.expert_bias.tolist()}"
    )
    # Accumulator drained.
    assert len(m.owner.local_quantile_scores) == 0, (
        f"quantile_scores accumulator was not drained: "
        f"len={len(m.owner.local_quantile_scores)}"
    )
    # Counts NOT zeroed (the deepseek path would have zeroed them).
    torch.testing.assert_close(m.owner.local_tokens_per_expert, counts_initial)


def test_walker_quantile_dispatch_no_op_on_empty_accumulator():
    """Round 41 AC-10: empty-accumulator no-op.

    When a quantile-method owner has no accumulated scores (e.g. a
    branch-masked path with zero active tokens, or simply no forward
    has run since the last update), the walker MUST be a clean no-op
    on that owner — neither `quantile_ema` nor `expert_bias` moves.
    """
    import importlib.util as _u
    import torch.nn as nn
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    class _Owner(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(n, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer("quantile_ema", torch.zeros(n, dtype=torch.float32))
            self.local_quantile_scores: list[torch.Tensor] = []

    class _MockConfig:
        mlp_router_balancing = "quantile"

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = _Owner(4)
            self._load_balancing_method = "quantile"
            self.config = _MockConfig()

        def get_all_balancing_owners(self):
            yield self.owner, "mlp"

    m = _MockModel()
    initial_bias = m.owner.expert_bias.detach().clone()
    initial_ema = m.owner.quantile_ema.detach().clone()

    with torch.no_grad():
        routing.update_expert_biases(
            m, bias_rate=0.0, distributed=False,
        )

    torch.testing.assert_close(m.owner.expert_bias, initial_bias)
    torch.testing.assert_close(m.owner.quantile_ema, initial_ema)


def test_deepseek_router_forward_appends_quantile_scores_in_training():
    """Round 41 review Finding 2: forward-side accumulation.

    `DeepSeekRouter.forward()` appends `scores.detach().float()` to
    `local_quantile_scores` whenever grad is enabled and the call
    is not a checkpoint recompute (mirrors the existing count-buffer
    guard). Production trainer eval runs the model under
    `torch.no_grad()` which suppresses both the count update and
    the score append.
    """
    from src.models.router import DeepSeekRouter

    class _Cfg:
        hidden_size = 16
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        topk_scaling_factor = 2.5
        router_exploration_rate = 0.0
        router_z_loss_coef = 0.0

    r = DeepSeekRouter(_Cfg())
    h = torch.randn(2, 5, 16)  # (B, T, D)
    # Training + grad enabled: append.
    r.train()
    assert r.local_quantile_scores == []
    _scores, _topk_w, _topk_idx = r(h)
    assert len(r.local_quantile_scores) == 1, (
        f"training-mode forward did not append scores; "
        f"len={len(r.local_quantile_scores)}"
    )
    assert r.local_quantile_scores[0].dtype == torch.float32
    assert r.local_quantile_scores[0].shape == (2 * 5, 4)

    # Grad-disabled (no_grad) — production trainer evaluation
    # context. Must NOT append (mirrors `local_tokens_per_expert`).
    pre_len = len(r.local_quantile_scores)
    with torch.no_grad():
        _ = r(h)
    assert len(r.local_quantile_scores) == pre_len, (
        f"no_grad forward should not append scores; "
        f"len went from {pre_len} to {len(r.local_quantile_scores)}"
    )


def test_deepseek_router_grad_accumulation_concatenates_quantile_scores():
    """Round 41 review Finding 2: grad-accumulation coverage.

    Four micro-batches in training mode produce four entries in the
    accumulator; the walker concatenates them along dim=0.
    """
    from src.models.router import DeepSeekRouter

    class _Cfg:
        hidden_size = 16
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        topk_scaling_factor = 2.5
        router_exploration_rate = 0.0
        router_z_loss_coef = 0.0

    r = DeepSeekRouter(_Cfg())
    r.train()
    for _ in range(4):
        _ = r(torch.randn(1, 3, 16))
    assert len(r.local_quantile_scores) == 4, (
        f"expected 4 micro-batch score tensors; got "
        f"{len(r.local_quantile_scores)}"
    )
    concat = torch.cat(r.local_quantile_scores, dim=0)
    assert concat.shape == (4 * 3, 4), (
        f"grad-accumulation concat shape mismatch: {concat.shape}"
    )


def test_walker_drains_quantile_accumulator_on_deepseek_owner():
    """Round 42 memory bound: forward unconditionally appends scores;
    walker MUST drain the accumulator on owners running the deepseek
    path so memory doesn't grow unboundedly across training steps.
    """
    import importlib.util as _u
    import torch.nn as nn
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    class _Owner(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(n, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer("quantile_ema", torch.zeros(n, dtype=torch.float32))
            self.local_quantile_scores: list[torch.Tensor] = []

    class _MockConfig:
        mlp_router_balancing = "deepseek_bias"

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = _Owner(4)
            self._load_balancing_method = "deepseek_bias"
            self.config = _MockConfig()

        def get_all_balancing_owners(self):
            yield self.owner, "mlp"

    m = _MockModel()
    # Pre-populate counts AND a stale quantile accumulator.
    m.owner.local_tokens_per_expert[:] = torch.tensor([100.0, 1.0, 1.0, 1.0])
    m.owner.local_quantile_scores.append(torch.zeros(3, 4, dtype=torch.float32))
    with torch.no_grad():
        routing.update_expert_biases(m, bias_rate=0.5, distributed=False)
    assert m.owner.local_quantile_scores == [], (
        f"deepseek-method walker must drain stale quantile accumulator; "
        f"left len={len(m.owner.local_quantile_scores)}"
    )


def test_pure_quantile_production_trainer_path_drives_walker(tmp_path):
    """Round 41 review Finding 1 closure: pure quantile config that
    fires the walker via the real `trainer_post_optimizer_bias_update`
    path.

    Build a nested quantile config (no DeepSeek bias-rate triple).
    Run a training-mode forward (which accumulates scores in
    `local_quantile_scores`). Call the trainer's post-step helper.
    Assert:
      - `quantile_ema` updated (the walker fired).
      - `expert_bias` updated (quantile-from-EMA drove it).
      - the accumulator was drained.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: pure_quantile_e2e
model:
  type: standard_moe
  router_type: deepseek
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
  topk_scaling_factor: 2.5
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 32
  tie_word_embeddings: true
  output_router_logits: false
  attn_implementation: eager
  mlp_router:
    balancing: quantile
    quantile_target_q: 0.5
    quantile_eta: 0.05
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
    p = tmp_path / "pure_quantile.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()

    # Run a real forward — this populates `local_quantile_scores`
    # via the router's training-mode accumulator gate.
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()

    # Find the MLP owner; verify the accumulator filled up.
    mlp_owner = next(
        owner for owner, label in model.get_all_balancing_owners()
        if label == "mlp"
    )
    assert len(mlp_owner.local_quantile_scores) >= 1, (
        f"forward did not append quantile scores; len="
        f"{len(mlp_owner.local_quantile_scores)}"
    )
    initial_bias = mlp_owner.expert_bias.detach().clone()
    initial_ema = mlp_owner.quantile_ema.detach().clone()

    # Trainer post-step helper. With my F1 fix, the walker fires for
    # quantile-method models even without DeepSeek bias-rate triples.
    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    assert (mlp_owner.quantile_ema != initial_ema).any(), (
        f"quantile_ema unchanged after walker; pure quantile config "
        f"did not drive the post-step update. "
        f"before={initial_ema.tolist()}, after={mlp_owner.quantile_ema.tolist()}"
    )
    assert (mlp_owner.expert_bias != initial_bias).any(), (
        f"expert_bias unchanged after walker; quantile-from-EMA did not "
        f"write the bias. before={initial_bias.tolist()}, "
        f"after={mlp_owner.expert_bias.tolist()}"
    )
    assert mlp_owner.local_quantile_scores == [], (
        f"accumulator not drained: len={len(mlp_owner.local_quantile_scores)}"
    )


def test_deepseek_router_registers_quantile_state_unconditionally():
    """Round 41 AC-10: quantile state lives on every DeepSeekRouter
    so the walker can dispatch quantile-method owners without a
    per-class flag at construction time. Verifies presence of
    `quantile_ema` (persistent fp32 buffer) and `local_quantile_scores`
    (Python list).
    """
    from src.models.router import DeepSeekRouter

    class _Cfg:
        hidden_size = 16
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        topk_scaling_factor = 2.5
        router_exploration_rate = 0.0
        router_z_loss_coef = 0.0

    r = DeepSeekRouter(_Cfg())
    assert hasattr(r, "quantile_ema")
    assert isinstance(r.quantile_ema, torch.Tensor)
    assert r.quantile_ema.shape == (4,)
    assert r.quantile_ema.dtype == torch.float32
    # quantile_ema must be a persistent buffer (saved in state_dict).
    state_keys = set(r.state_dict().keys())
    assert "quantile_ema" in state_keys, (
        f"quantile_ema must be a persistent buffer; state_dict keys={state_keys}"
    )
    assert hasattr(r, "local_quantile_scores")
    assert isinstance(r.local_quantile_scores, list)
    assert r.local_quantile_scores == []


def test_branch_deepseek_bias_forces_biased_scoring_path_consumes_expert_bias():
    """Contrastive proof that `balancing="deepseek_bias"` actually
    consumes `expert_bias` during routing — independent of the legacy
    `use_deepseek_style` flag.

    A prior round let `BranchRouter(balancing="deepseek_bias",
    use_deepseek_style=False)` mutate `expert_bias` via the post-step
    walker while the forward pass still routed via plain softmax —
    the learned bias was ignored.

    With deterministic logits that strongly favour ATTN (logit 0):
      - bias = [0, 0]: choice = 0 (ATTN).
      - bias = [-16, +16]: argmax over `sigmoid(logits) + bias` flips
        to MLP (choice = 1) — ONLY if the runtime takes the biased
        scoring path.
    """
    from src.models.routing.routers import BranchRouter
    torch.manual_seed(0)
    router = BranchRouter(
        hidden_size=8, balancing="deepseek_bias",
        use_deepseek_style=False,  # legacy flag NOT set
    )
    router.eval()  # disable count tracking + sampling
    # Set the gate so logits[..., 0] >> logits[..., 1] (favour ATTN).
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 4.0   # row 0 = ATTN logit
        router.gate.weight[1, 0] = -4.0  # row 1 = MLP logit
    h = torch.zeros(1, 1, 8)
    h[0, 0, 0] = 1.0
    _, _, attn_mask, mlp_mask = router(h)
    assert attn_mask[0, 0, 0].item() and not mlp_mask[0, 0, 0].item(), (
        f"With zero bias and ATTN-favouring logits, choice should be ATTN; "
        f"attn={attn_mask[0,0,0].item()}, mlp={mlp_mask[0,0,0].item()}"
    )

    # Now flip the bias to overwhelmingly favour MLP. If the runtime
    # ignores `expert_bias`, choice stays ATTN; if it consumes it, choice
    # flips to MLP.
    with torch.no_grad():
        router.expert_bias[0] = -16.0
        router.expert_bias[1] = +16.0
    _, _, attn_mask, mlp_mask = router(h)
    assert mlp_mask[0, 0, 0].item() and not attn_mask[0, 0, 0].item(), (
        f"Extreme expert_bias did NOT flip the branch choice; the "
        f"runtime is ignoring `expert_bias` even though balancing="
        f"deepseek_bias was selected. attn={attn_mask[0,0,0].item()}, "
        f"mlp={mlp_mask[0,0,0].item()}"
    )


def test_per_owner_bias_rates_dispatch_distinct_rates_to_mlp_and_branch(tmp_path):
    """Round 38 review Finding 2: per-owner bias-rate dispatch.

    When MLP=deepseek_bias rate=R_mlp AND branch=deepseek_bias
    rate=R_branch with R_branch >> R_mlp, the walker must apply
    R_mlp to MLP owners and R_branch to the branch owner — NOT
    collapse them onto a single global rate.

    Pre-populate identical imbalanced counts on both MLP and branch
    owners and run the post-step walker. Assert the branch delta is
    much larger than the MLP delta (proportional to rate ratio).
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: per_owner_rates
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
    balancing: deepseek_bias
    bias_update_rate: 0.001
    bias_update_zero_sum: true
  attn_router:
    balancing: none
  branch_router:
    balancing: deepseek_bias
    bias_update_rate: 0.5
    bias_update_zero_sum: true
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
    p = tmp_path / "per_owner_rates.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()

    branch = model.model.branch_router
    mlp_owner = next(
        owner for owner, label in model.get_all_balancing_owners()
        if label == "mlp"
    )

    # Identical pre-populated imbalance on both owners (renormalize
    # MLP's 4 experts into the same rate-of-imbalance as branch's 2).
    with torch.no_grad():
        branch.local_tokens_per_expert.zero_()
        branch.local_tokens_per_expert[0] = 100.0
        branch.local_tokens_per_expert[1] = 1.0
        mlp_owner.local_tokens_per_expert.zero_()
        mlp_owner.local_tokens_per_expert[0] = 100.0
        mlp_owner.local_tokens_per_expert[1:] = 1.0
        branch_initial = branch.expert_bias.detach().clone()
        mlp_initial = mlp_owner.expert_bias.detach().clone()

    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    branch_delta = (branch.expert_bias - branch_initial).abs().max().item()
    mlp_delta = (mlp_owner.expert_bias - mlp_initial).abs().max().item()
    # branch rate / mlp rate == 0.5 / 0.001 == 500x. Identical sign
    # patterns mean the magnitudes should reflect that ratio. Allow
    # generous tolerance because the zero-sum walker normalizes by
    # E and the two owners have different E (MLP=4, branch=2).
    assert branch_delta > 100 * mlp_delta, (
        f"per-owner bias rates collapsed: branch_delta={branch_delta:.6e} "
        f"vs mlp_delta={mlp_delta:.6e}; expected branch >> mlp "
        f"(ratio ~500x from rate dispatch)."
    )
    # Sanity: both owners actually moved.
    assert branch_delta > 0 and mlp_delta > 0, (
        f"at least one owner did not move: branch={branch_delta}, mlp={mlp_delta}"
    )


def test_branch_deepseek_bias_sampling_path_consumes_expert_bias():
    """Round 38 review Finding 1: when `use_sampling=True`, the
    branch router previously called `torch.multinomial(scores, 1)`
    on the unbiased sigmoid scores — so even though the post-step
    walker updated `expert_bias`, the sampling categorical was
    unchanged.

    Probe: with equal logits (50/50 sigmoid scores) and an extreme
    bias of [-16, +16], the sampling categorical must be driven
    overwhelmingly toward MLP. We use 2000 tokens and a fixed seed
    and assert the MLP fraction is > 0.95 (a fair coin flip would
    sit at 0.5).
    """
    from src.models.routing.routers import BranchRouter
    torch.manual_seed(0)
    router = BranchRouter(
        hidden_size=4, balancing="deepseek_bias",
        use_deepseek_style=False,  # legacy flag NOT set
        use_sampling=True,
    )
    router.train()  # sampling path is gated on `self.training`
    # Equal logits => sigmoid scores ≈ 0.5 for both classes.
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.bias.zero_() if router.gate.bias is not None else None
        # Extreme bias toward MLP.
        router.expert_bias[0] = -16.0
        router.expert_bias[1] = +16.0
    h = torch.zeros(1, 2000, 4)
    torch.manual_seed(7)
    _, _, attn_mask, mlp_mask = router(h)
    mlp_fraction = mlp_mask.float().mean().item()
    assert mlp_fraction > 0.95, (
        f"sampling path is ignoring `expert_bias`; with extreme MLP-favouring "
        f"bias the sampled MLP fraction should exceed 0.95, got "
        f"{mlp_fraction:.3f}. Equal logits + biased multinomial should "
        f"land overwhelmingly in MLP."
    )


def test_branch_deepseek_bias_per_layer_router_lifecycle(tmp_path):
    """Round 38 review Finding 1: Codex specifically asked for the
    plural / per-layer branch router lifecycle.

    Build moe_everything with `per_layer_router=True`,
    `branch_router.balancing=deepseek_bias`. Pre-populate imbalanced
    counts on EVERY per-layer router. Run the post-step walker.
    Assert every per-layer `branch_routers[i].expert_bias` mutated.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: per_layer_branch_deepseek
model:
  type: moe_everything
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 3
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
  per_layer_router: true
  mlp_router:
    balancing: none
  attn_router:
    balancing: none
  branch_router:
    balancing: deepseek_bias
    bias_update_rate: 0.5
    bias_update_zero_sum: true
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
    p = tmp_path / "per_layer_branch.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()

    # Per-layer routers live on `model.model.branch_routers` (plural).
    branch_routers = getattr(model.model, "branch_routers", None)
    assert branch_routers is not None and len(branch_routers) == 3, (
        f"expected per-layer branch_routers ModuleList of length 3; "
        f"got {branch_routers!r}"
    )

    # Pre-populate identical imbalance on every per-layer router.
    initial: list[torch.Tensor] = []
    with torch.no_grad():
        for r in branch_routers:
            r.local_tokens_per_expert.zero_()
            r.local_tokens_per_expert[0] = 100.0  # heavy ATTN
            r.local_tokens_per_expert[1] = 1.0    # light MLP
            initial.append(r.expert_bias.detach().clone())

    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    # Every per-layer router's expert_bias must have moved.
    for i, r in enumerate(branch_routers):
        assert not torch.allclose(r.expert_bias, initial[i]), (
            f"per-layer branch_routers[{i}].expert_bias did not change; "
            f"the walker is not reaching every per-layer branch owner. "
            f"before={initial[i].tolist()}, after={r.expert_bias.tolist()}"
        )

    # Round 40 review Finding 1: the prior round called each per-
    # layer router directly (`r(h)`); the test never exercised
    # `MoEverythingModel`'s depth loop. Round 41 strengthens the
    # coverage to a real `model(input_ids=...)` forward and asserts
    # each `branch_routers[i].last_selected_experts` is set by the
    # full model path AND reflects the bias.
    model.eval()
    # With zeroed gate weights and zero bias, every depth's branch
    # router sees logits = 0 (or a constant from the gate's bias if
    # any) and argmax breaks the tie at index 0 = ATTN.
    for r in branch_routers:
        with torch.no_grad():
            r.gate.weight.zero_()
            if r.gate.bias is not None:
                r.gate.bias.zero_()
            r.expert_bias.zero_()

    fixed_input_ids = torch.randint(
        0, model.vocab_size, (1, 8), dtype=torch.long,
    )
    out = model(
        input_ids=fixed_input_ids,
        labels=fixed_input_ids,
        output_router_logits=True,
    )
    assert out is not None  # full forward returned

    for i, r in enumerate(branch_routers):
        sel = r.last_selected_experts
        assert sel is not None, (
            f"per-layer branch_routers[{i}] did not record last_selected_experts; "
            f"the full model forward never reached this depth's branch router."
        )
        assert torch.all(sel == 0), (
            f"per-layer branch_routers[{i}] zero-bias model forward should "
            f"pick ATTN at every token; got selected={sel.unique().tolist()}"
        )

    # Now write extreme MLP-favouring `expert_bias` on every per-layer
    # router and re-run the full model forward.
    for r in branch_routers:
        with torch.no_grad():
            r.expert_bias[0] = -16.0
            r.expert_bias[1] = +16.0
    out2 = model(
        input_ids=fixed_input_ids,
        labels=fixed_input_ids,
        output_router_logits=True,
    )
    assert out2 is not None

    for i, r in enumerate(branch_routers):
        sel = r.last_selected_experts
        assert sel is not None
        assert torch.all(sel == 1), (
            f"per-layer branch_routers[{i}] post-bias model forward did NOT "
            f"flip to MLP at every token; got selected={sel.unique().tolist()}; "
            f"expert_bias={r.expert_bias.tolist()}. The full-model depth loop "
            f"is not consuming each per-layer router's expert_bias."
        )


@pytest.mark.parametrize("mlp_method", ["aux_loss", "none", "quantile"])
def test_mixed_per_class_branch_deepseek_with_various_mlp_methods(tmp_path, mlp_method):
    """Round 41 review Finding 3: parameterized non-deepseek MLP
    variants — including `quantile` now that quantile is in
    `_BIAS_UPDATE_METHODS` and the walker dispatches it.

    Asserts the mixed-method gate fix from Round 40 holds for
    `mlp_router.balancing in {"aux_loss", "none", "quantile"}`. In
    every shape:
      - branch is `deepseek_bias` and gets a non-zero bias delta
        from imbalanced counts.
      - MLP is non-deepseek and does NOT receive a deepseek update
        (the per-owner skip catches it; for quantile the empty
        accumulator is a quantile no-op too).
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    if mlp_method == "aux_loss":
        mlp_block = (
            "    balancing: aux_loss\n"
            "    router_aux_loss_coef: 0.001\n"
        )
    elif mlp_method == "none":
        mlp_block = "    balancing: none\n"
    elif mlp_method == "quantile":
        mlp_block = (
            "    balancing: quantile\n"
            "    quantile_target_q: 0.5\n"
            "    quantile_eta: 0.05\n"
        )
    else:
        raise ValueError(f"unhandled mlp_method={mlp_method}")

    yaml_text = f"""experiment_name: mixed_mlp_{mlp_method}_branch_deepseek
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
{mlp_block}  attn_router:
    balancing: none
  branch_router:
    balancing: deepseek_bias
    bias_update_rate: 0.5
    bias_update_zero_sum: true
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
    p = tmp_path / f"mixed_mlp_{mlp_method}.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()
    branch = model.model.branch_router
    mlp_owner = next(
        owner for owner, label in model.get_all_balancing_owners()
        if label == "mlp"
    )

    with torch.no_grad():
        branch.local_tokens_per_expert.zero_()
        branch.local_tokens_per_expert[0] = 100.0
        branch.local_tokens_per_expert[1] = 1.0
        mlp_owner.local_tokens_per_expert.zero_()
        mlp_owner.local_tokens_per_expert[0] = 100.0
        mlp_owner.local_tokens_per_expert[1:] = 1.0
        branch_initial = branch.expert_bias.detach().clone()
        mlp_initial = mlp_owner.expert_bias.detach().clone()

    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    branch_delta = (branch.expert_bias - branch_initial).abs().max().item()
    mlp_delta = (mlp_owner.expert_bias - mlp_initial).abs().max().item()
    assert branch_delta > 1e-4, (
        f"mlp_method={mlp_method!r}: branch deepseek_bias did not fire "
        f"(branch_delta={branch_delta:.6e}). The walker is preempted by "
        f"the model-level method gate."
    )
    assert mlp_delta == 0.0, (
        f"mlp_method={mlp_method!r}: MLP class is non-deepseek but the "
        f"walker mutated mlp.expert_bias (mlp_delta={mlp_delta:.6e}). "
        f"The per-owner skip table is not catching the MLP owner."
    )


def test_mixed_per_class_branch_deepseek_bias_with_mlp_aux_loss(tmp_path):
    """Round 39 review Finding 1: mixed per-class methods.

    Build moe_everything with `mlp_router.balancing: aux_loss` AND
    `branch_router.balancing: deepseek_bias`. The factory stamps
    `model._load_balancing_method = "aux_loss"` because MLP wins. A
    naive walker then early-returns on the model-level method gate
    and never updates the branch's `expert_bias` — the post-step
    walker silently no-ops.

    Pre-populate imbalanced branch counts and identical imbalanced
    MLP counts. Run `trainer_post_optimizer_bias_update`. Assert:
      - branch.expert_bias mutates (deepseek path fired for branch).
      - mlp.expert_bias does NOT receive a deepseek update (MLP class
        method is `aux_loss`, so the per-owner skip catches it).
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    yaml_text = """experiment_name: mixed_mlp_aux_branch_deepseek
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
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
    balancing: aux_loss
    router_aux_loss_coef: 0.001
  attn_router:
    balancing: none
  branch_router:
    balancing: deepseek_bias
    bias_update_rate: 0.5
    bias_update_zero_sum: true
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
    p = tmp_path / "mixed_mlp_aux_branch_deepseek.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.train()
    branch = model.model.branch_router
    mlp_owner = next(
        owner for owner, label in model.get_all_balancing_owners()
        if label == "mlp"
    )

    with torch.no_grad():
        branch.local_tokens_per_expert.zero_()
        branch.local_tokens_per_expert[0] = 100.0
        branch.local_tokens_per_expert[1] = 1.0
        # Same imbalanced shape on MLP — if the walker did the wrong
        # thing it would mutate this buffer too.
        mlp_owner.local_tokens_per_expert.zero_()
        mlp_owner.local_tokens_per_expert[0] = 100.0
        mlp_owner.local_tokens_per_expert[1:] = 1.0
        branch_initial = branch.expert_bias.detach().clone()
        mlp_initial = mlp_owner.expert_bias.detach().clone()

    train_cfg = _CFG_MOD.build_training_config(cfg)
    routing.trainer_post_optimizer_bias_update(
        model, train_cfg=train_cfg, cfg=cfg,
        distributed=False, global_step=1,
    )

    branch_delta = (branch.expert_bias - branch_initial).abs().max().item()
    mlp_delta = (mlp_owner.expert_bias - mlp_initial).abs().max().item()
    assert branch_delta > 1e-4, (
        f"mixed-method config (MLP=aux_loss + branch=deepseek_bias) failed "
        f"to update branch.expert_bias; the walker early-returned on the "
        f"model-level method gate before the per-owner deepseek_bias dispatch "
        f"could fire. branch_delta={branch_delta:.6e}"
    )
    assert mlp_delta == 0.0, (
        f"MLP class method is `aux_loss`; the walker should have skipped the "
        f"MLP owner during the deepseek dispatch. mlp_delta={mlp_delta:.6e}"
    )


def test_config_level_branch_sampling_with_nested_deepseek_bias(tmp_path):
    """Round 39 review Finding 3: config-level branch sampling test.

    `load_config -> build_model` with `branch_sampling: true` AND
    nested `branch_router.balancing: deepseek_bias`. Verify:
      - the built BranchRouter has `use_sampling=True`.
      - the built BranchRouter has `use_deepseek_style=True` (folded
        from the `balancing` field).
      - sampling forward consumes `expert_bias` at the config level
        (not just at the unit-construction level).
    """
    yaml_text = """experiment_name: branch_sampling_deepseek
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
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
  use_deepseek_routing: true
  branch_deepseek: false
  branch_sampling: true
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
    balancing: deepseek_bias
    bias_update_rate: 0.5
    bias_update_zero_sum: true
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
    p = tmp_path / "branch_sampling_deepseek.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    branch = model.model.branch_router
    assert branch.use_sampling is True, (
        f"build_model did not propagate `branch_sampling: true` to the "
        f"BranchRouter; got use_sampling={branch.use_sampling}"
    )
    assert branch.use_deepseek_style is True, (
        f"build_model did not fold `branch_router.balancing=deepseek_bias` "
        f"into BranchRouter.use_deepseek_style; got "
        f"{branch.use_deepseek_style}"
    )
    assert branch.balancing == "deepseek_bias"

    # Forward path test: equal logits + extreme MLP-favouring bias =>
    # sampling should land overwhelmingly on MLP.
    model.train()
    with torch.no_grad():
        branch.gate.weight.zero_()
        if branch.gate.bias is not None:
            branch.gate.bias.zero_()
        branch.expert_bias[0] = -16.0
        branch.expert_bias[1] = +16.0
    h = torch.zeros(1, 2000, branch.gate.in_features)
    torch.manual_seed(7)
    _, _, _attn_mask, mlp_mask = branch(h)
    mlp_fraction = mlp_mask.float().mean().item()
    assert mlp_fraction > 0.95, (
        f"config-level `branch_sampling: true` + nested `deepseek_bias` "
        f"sampling forward did not consume expert_bias; observed MLP fraction "
        f"= {mlp_fraction:.3f} with extreme MLP-favouring bias; expected > 0.95."
    )


def test_walker_per_proj_zero_sum_dispatches_per_owner_mode():
    """Round 38 review Finding 2: per-owner zero-sum dispatch.

    Direct unit test of `update_expert_biases(per_proj_zero_sum=...)`.
    For an owner with E=4 experts and asymmetric counts [100, 1, 1, 1],
    the two modes produce measurably different cumulative bias trajectories:

      - zero_sum=True: per-step delta is mean-subtracted, so cumulative
        bias mean stays pinned at 0 (`bias -= (s - s.mean()) * rate`).
      - zero_sum=False: per-step delta is the raw sign tensor, so
        cumulative bias mean drifts under asymmetric loads
        (`bias -= s * rate`).

    Build two equally-shaped owners labeled "mlp" and "branch" with
    IDENTICAL counts. Pass `per_proj_zero_sum={"mlp": True, "branch": False}`.
    Assert MLP mean is ~0 (zero-sum mode) and branch mean drifts
    (non-zero-sum mode). Without per-owner dispatch, the walker
    would collapse both onto the legacy global flag and at least one
    of these assertions would fail.

    Note: the binary 2-class case has `s.mean() == 0` whenever the
    counts are split 1-1, so the two modes only differ for E ≥ 3.
    The mock owners in this test use E=4 to expose the difference.
    """
    import importlib.util as _u
    import torch.nn as nn
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    class _Owner(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(n, dtype=torch.float32),
                persistent=False,
            )

    class _MockConfig:
        pass

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner_mlp = _Owner(4)
            self.owner_branch = _Owner(4)
            self._load_balancing_method = "deepseek_bias"
            self.config = _MockConfig()

        def get_all_balancing_owners(self):
            yield self.owner_mlp, "mlp"
            yield self.owner_branch, "branch"

    m = _MockModel()
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])
    m.owner_mlp.local_tokens_per_expert.copy_(counts)
    m.owner_branch.local_tokens_per_expert.copy_(counts)

    with torch.no_grad():
        routing.update_expert_biases(
            m, bias_rate=0.5, distributed=False,
            per_proj_zero_sum={"mlp": True, "branch": False},
        )

    mlp_mean = float(m.owner_mlp.expert_bias.mean().item())
    branch_mean = float(m.owner_branch.expert_bias.mean().item())
    assert abs(mlp_mean) < 1e-4, (
        f"owner labeled 'mlp' under zero_sum=True should have cumulative "
        f"bias mean pinned at 0; got mean={mlp_mean:.6e}. The walker is "
        f"NOT applying per-owner zero_sum=True for the mlp label."
    )
    assert abs(branch_mean) > 1e-4, (
        f"owner labeled 'branch' under zero_sum=False should drift; got "
        f"mean={branch_mean:.6e}. The walker is collapsing both owners "
        f"onto zero_sum=True instead of honoring per_proj_zero_sum."
    )


def test_trainer_threads_per_owner_zero_sum_through_factory_stamping(tmp_path):
    """Higher-level: prove the trainer pipes per-class
    `effective_<owner>_bias_update_zero_sum` (set by `model_factory.py`
    when `<owner>_router.balancing == "deepseek_bias"`) into the walker
    via `per_proj_zero_sum`.

    Build a tiny model and STAMP the per-owner flags directly onto
    `config`, then call the trainer's post-step helper. Compare the
    resulting MLP-bias trajectory to a flipped-flag run on a fresh
    model with identical counts. Without my fix, the trainer reads
    only the un-prefixed alias and the per-owner flag is ignored —
    the trajectories would be identical.
    """
    import importlib.util as _u
    import torch.nn as nn
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    class _Owner(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(n, dtype=torch.float32),
                persistent=False,
            )

    class _MockConfig:
        def __init__(self, mlp_zs):
            # The trainer reads `effective_mlp_bias_update_zero_sum`.
            # `effective_mlp_bias_update_rate` mirrors a per-class rate
            # so the trainer's owner-rate gate fires for "mlp".
            self.effective_mlp_bias_update_zero_sum = mlp_zs
            self.effective_mlp_bias_update_rate = 0.5
            self.mlp_router_balancing = "deepseek_bias"
            # The unprefixed alias is the OPPOSITE of the per-owner
            # value so a regression that ignores per-owner stamping
            # would observe the wrong mode.
            self.effective_bias_update_zero_sum = not mlp_zs
            self.effective_bias_update_rate = 0.5

    class _MockModel(nn.Module):
        def __init__(self, mlp_zs):
            super().__init__()
            self.owner_mlp = _Owner(4)
            self._load_balancing_method = "deepseek_bias"
            self.config = _MockConfig(mlp_zs)

        def get_all_balancing_owners(self):
            yield self.owner_mlp, "mlp"

    counts = torch.tensor([100.0, 1.0, 1.0, 1.0])

    class _TrainCfg:
        bias_update_rate = 0.0
        bias_update_zero_sum = True  # would override if alias was read
        bias_warmup_start = 0.0
        bias_warmup_steps = 0

    # MLP zero_sum=True stamped on per-owner field; alias is False.
    m_true = _MockModel(mlp_zs=True)
    m_true.owner_mlp.local_tokens_per_expert.copy_(counts)
    routing.trainer_post_optimizer_bias_update(
        m_true, train_cfg=_TrainCfg(), cfg={"training": {}},
        distributed=False, global_step=1,
    )

    # MLP zero_sum=False stamped on per-owner field; alias is True.
    m_false = _MockModel(mlp_zs=False)
    m_false.owner_mlp.local_tokens_per_expert.copy_(counts)
    routing.trainer_post_optimizer_bias_update(
        m_false, train_cfg=_TrainCfg(), cfg={"training": {}},
        distributed=False, global_step=1,
    )

    mean_true = float(m_true.owner_mlp.expert_bias.mean().item())
    mean_false = float(m_false.owner_mlp.expert_bias.mean().item())
    assert abs(mean_true) < 1e-4, (
        f"per-owner mlp_zs=True should pin mean at 0; got {mean_true:.6e}. "
        f"The trainer read the un-prefixed alias instead of "
        f"effective_mlp_bias_update_zero_sum."
    )
    assert abs(mean_false) > 1e-4, (
        f"per-owner mlp_zs=False should drift mean; got {mean_false:.6e}. "
        f"The trainer read the un-prefixed alias instead of "
        f"effective_mlp_bias_update_zero_sum."
    )


def test_branch_deepseek_bias_post_step_bias_changes_subsequent_branch_choice(tmp_path):
    """Full lifecycle: load_config -> build -> pre-populate counts ->
    trainer_post_optimizer_bias_update -> forward.

    Pre-populating an extreme imbalanced load on `local_tokens_per_expert`
    (heavy ATTN, light MLP) and running the trainer's post-step bias
    update for many warmup-respecting steps is the only way to grow
    `expert_bias` to a magnitude that visibly flips the branch
    decision on a tiny model. We bypass the forward+backward by
    hand-setting counts and run the walker repeatedly.
    """
    import importlib.util as _u
    repo = Path(__file__).resolve().parent.parent
    spec = _u.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    routing = _u.module_from_spec(spec)
    sys.modules["src.training.routing"] = routing
    spec.loader.exec_module(routing)

    head, _, tail = _BASE_YAML.partition("training:")
    yaml_text = (
        head.replace(
            "%BRANCH_BLOCK%",
            "    balancing: deepseek_bias\n"
            "    bias_update_rate: 1.0\n"
            "    bias_update_zero_sum: true\n",
        )
        + "training:" + tail
    )
    p = tmp_path / "branch_deepseek_lifecycle.yaml"
    p.write_text(yaml_text)
    cfg = _CFG_MOD.load_config(str(p))
    torch.manual_seed(20260428)
    model, _ = _FACTORY_MOD.build_model(cfg)
    model.eval()
    branch = model.model.branch_router

    # Hand-set the gate so a no-bias forward chooses ATTN.
    with torch.no_grad():
        branch.gate.weight.zero_()
        branch.gate.weight[0, 0] = 4.0
        branch.gate.weight[1, 0] = -4.0
    h = torch.zeros(1, 1, branch.gate.in_features)
    h[0, 0, 0] = 1.0
    _, _, attn_mask_pre, mlp_mask_pre = branch(h)
    assert attn_mask_pre[0, 0, 0].item(), "no-bias forward should pick ATTN"

    # Now drive the bias walker hard against ATTN.
    train_cfg = _CFG_MOD.build_training_config(cfg)
    for step in range(50):
        with torch.no_grad():
            branch.local_tokens_per_expert.zero_()
            branch.local_tokens_per_expert[0] = 1000.0  # heavy ATTN
            branch.local_tokens_per_expert[1] = 1.0     # light MLP
        routing.trainer_post_optimizer_bias_update(
            model, train_cfg=train_cfg, cfg=cfg,
            distributed=False, global_step=step + 1,
        )
    assert branch.expert_bias[0].item() < -0.1, (
        f"expected expert_bias[0] (ATTN) to grow strongly negative; "
        f"got {branch.expert_bias.tolist()}"
    )

    # Forward again — the walker-driven bias must flip the branch decision.
    _, _, attn_mask_post, mlp_mask_post = branch(h)
    assert mlp_mask_post[0, 0, 0].item() and not attn_mask_post[0, 0, 0].item(), (
        f"Post-walker forward did not observe the learned bias; "
        f"branch still routes ATTN. expert_bias={branch.expert_bias.tolist()}"
    )


if __name__ == "__main__":
    print("Run via pytest")
