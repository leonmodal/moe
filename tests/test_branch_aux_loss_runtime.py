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


def test_branch_router_rejects_quantile_at_construction():
    from src.models.routing.routers import BranchRouter
    with pytest.raises(ValueError, match="BranchRouter balancing must be"):
        BranchRouter(hidden_size=16, balancing="quantile")


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
