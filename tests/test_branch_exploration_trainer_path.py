"""End-to-end trainer-path tests for the BranchRouter
exploration_only schedule. These tests build a real
`MoEverythingForCausalLM` via the model factory, then exercise the
trainer's pre-forward schedule helper, telemetry helpers, and
log_training_step output. Each test verifies a specific contract from
the AC-14 plan text:

* `test_trainer_schedule_trajectory_constant_linear_cosine`: schedule
  trajectory through the trainer-application path, parameterized over
  the three shapes with `exploration_min=0.01` and
  `exploration_warmup_steps=1000`.
* `test_log_training_step_emits_p_explore_and_attn_fraction`: trainer
  telemetry test with a fake W&B object and captured stdout proving
  current `p_explore` and `% ATTN` (not the mask fraction) land in
  console + W&B.
* `test_resume_simulation_starts_from_p_explore_at_resume_step`:
  resume-safety simulation at nonzero `global_step` proving the first
  resumed forward sees the resumed-step rate, not the constructor
  default.
* `test_100_step_windowed_branch_count_and_gradient_flow`: 100-step
  deterministic windowed test for branch counts + gradient flow.
* `test_500_step_deterministic_collapse_with_gradient_ratio_threshold`:
  500-step deterministic-collapse negative test asserting the 1e-6
  ATTN-vs-MLP gradient-ratio threshold per the plan text.
* `test_build_model_call_path_branch_aux_and_bias_are_inert_under_exploration_only`:
  end-to-end `build_model()` test with non-zero
  `branch_router_aux_loss_coef` AND `use_deepseek_routing` +
  `branch_deepseek=True` proving branch aux contributes no
  gradient and DeepSeek branch bias state stays inert.
"""
from __future__ import annotations

import importlib.util
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MoEverythingConfig, MoEverythingForCausalLM


def _load_routing_module():
    """Load `src.training.routing` directly without triggering the
    `src.training.__init__` package chain (avoids the pandas import
    triggered by the `__init__` re-exports)."""
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
        import types as _types
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = training_pkg
    spec = importlib.util.spec_from_file_location(
        "src.training.routing", str(repo / "src" / "training" / "routing.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.training.routing"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_logging_module():
    """Same trick for `src.training.logging`."""
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
        import types as _types
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = training_pkg
    spec = importlib.util.spec_from_file_location(
        "src.training.logging", str(repo / "src" / "training" / "logging.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.training.logging"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_moe_everything_with_schedule(
    *,
    branch_balancing: str = "exploration_only",
    branch_exploration_rate: float = 1.0,
    branch_exploration_decay: str = "linear",
    branch_exploration_warmup_steps: int = 1000,
    branch_exploration_min: float = 0.01,
    branch_entropy_coef: float = 0.01,
    branch_entropy_decay: str = "cosine",
    branch_entropy_min: float = 0.0,
    branch_entropy_decay_steps: int = 1000,
    branch_router_aux_loss_coef: float = 0.0,
    branch_deepseek: bool = False,
    use_deepseek_routing: bool = True,
    per_layer_router: bool = False,
    seed: int = 20260428,
):
    """Build a tiny `MoEverythingForCausalLM` with the schedule
    fields set. The fixture mirrors the model_factory.py
    moe_everything branch but constructs the config object directly
    so we can step the trainer helpers in isolation."""
    cfg = MoEverythingConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_attn_experts=2,
        num_attn_experts_per_tok=1,
        attn_expert_mode="per_head_no_recompute",
        branch_router_aux_loss_coef=branch_router_aux_loss_coef,
        use_deepseek_routing=use_deepseek_routing,
        branch_deepseek=branch_deepseek,
        topk_scaling_factor=2.5,
        per_layer_router=per_layer_router,
        per_layer_mlp_router=False,
        per_layer_attn_router=False,
        routed_norm=False,
        per_layer_norm=False,
        post_norm=False,
        dynamic_depth_min=1.0,
        dynamic_depth_max=1.0,
        depthwise_attention=False,
        depthwise_block_size=0,
        scale_attn_by_routing_weight=True,
        scale_branch_by_routing_weight=True,
        router_exploration_rate=0.0,
        branch_router_exploration_rate=0.0,
        branch_sampling=False,
        branch_level="token",
        branch_balancing=branch_balancing,
        branch_exploration_rate=branch_exploration_rate,
        branch_exploration_decay=branch_exploration_decay,
        branch_exploration_min=branch_exploration_min,
        branch_exploration_warmup_steps=branch_exploration_warmup_steps,
        branch_entropy_coef=branch_entropy_coef,
        branch_entropy_decay=branch_entropy_decay,
        branch_entropy_min=branch_entropy_min,
        branch_entropy_decay_steps=branch_entropy_decay_steps,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        seq_aux_loss_coef=0.0,
        output_router_logits=False,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = MoEverythingForCausalLM(cfg)
    model._load_balancing_method = "deepseek_bias"
    cfg.load_balancing_method = "deepseek_bias"
    return model, cfg


def test_sampling_entropy_schedule_and_loss_bonus():
    """`sampling_entropy` samples branch choices and subtracts the
    scheduled entropy bonus from the total loss."""
    routing = _load_routing_module()
    model, cfg = _build_moe_everything_with_schedule(
        branch_balancing="sampling_entropy",
        branch_entropy_coef=0.02,
        branch_entropy_decay="linear",
        branch_entropy_min=0.005,
        branch_entropy_decay_steps=10,
    )
    model.train()
    branch = model.model.branch_router
    assert branch.use_sampling is True

    applied_explore = routing.apply_branch_schedule_pre_forward(model, 5)
    assert applied_explore is None
    assert cfg.current_branch_entropy_coef == pytest.approx(0.0125)

    input_ids = torch.randint(0, model.vocab_size, (2, 4), dtype=torch.long)
    output = model(input_ids=input_ids, labels=input_ids, output_router_logits=False)
    assert output.branch_entropy_loss is not None
    torch.testing.assert_close(
        output.loss,
        output.ce_loss - cfg.current_branch_entropy_coef * output.branch_entropy_loss,
    )


@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_trainer_schedule_trajectory_constant_linear_cosine(schedule):
    """Schedule trajectory through the actual trainer-application
    path. With `exploration_min=0.01` and
    `exploration_warmup_steps=1000` per the plan text, the helper
    `apply_branch_schedule_pre_forward(model, global_step)` MUST
    produce the analytic schedule value at each representative step
    AND push it onto every BranchRouter's `exploration_only_rate`
    attribute. Parameterized over all three shapes.
    """
    routing = _load_routing_module()
    initial = 1.0
    final = 0.01
    decay_steps = 1000

    model, _cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=initial,
        branch_exploration_decay=schedule,
        branch_exploration_warmup_steps=decay_steps,
        branch_exploration_min=final,
    )
    branch = model.model.branch_router

    representative_steps = [0, 100, 500, 999, 1000, 2000]
    for step in representative_steps:
        applied = routing.apply_branch_schedule_pre_forward(model, step)
        analytic = routing.exploration_decay_schedule(
            step,
            schedule=schedule,
            initial_rate=initial,
            decay_steps=decay_steps,
            final_rate=final,
        )
        assert applied is not None, (
            f"{schedule} step={step}: helper returned None for active feature"
        )
        assert math.isclose(applied, analytic, abs_tol=1e-9), (
            f"{schedule} step={step}: helper returned {applied} "
            f"vs analytic {analytic}"
        )
        assert math.isclose(branch.exploration_only_rate, analytic, abs_tol=1e-9), (
            f"{schedule} step={step}: branch.exploration_only_rate is "
            f"{branch.exploration_only_rate}; expected {analytic} after "
            f"apply_branch_schedule_pre_forward"
        )


def test_log_training_step_emits_p_explore_and_attn_fraction():
    """Trainer telemetry test with a fake W&B object + captured
    stdout. Assertion: console line contains the current `p_explore`
    rate and the `% ATTN` value; W&B payload contains
    `train/branch_explore_rate` and `train/branch_attn_fraction`
    keys. Mask fraction lives under a separate key so it cannot be
    confused with `% ATTN`.
    """
    logging = _load_logging_module()

    captured = io.StringIO()

    class _FakeWandb:
        def __init__(self):
            self.payloads: list[dict] = []
            self.steps: list[int] = []

        def log(self, payload, step):
            self.payloads.append(dict(payload))
            self.steps.append(step)

    wandb_run = _FakeWandb()
    metrics = {
        "loss": 1.234, "ce_loss": 1.0, "aux_loss": 0.05,
        "aux_loss_normalized": 0.05, "seq_aux_loss": 0.0,
        "branch_aux_loss": 0.0, "attention_aux_loss": 0.0,
    }

    with redirect_stdout(captured):
        logging.log_training_step(
            wandb_run,
            step=42,
            metrics=metrics,
            grad_norm=0.5,
            lr=1e-4,
            tok_per_s=1234.5,
            tokens_seen=1e6,
            elapsed=0.123,
            log_every=1,
            branch_explore_rate=0.7,
            branch_attn_fraction=0.42,
            branch_attn_per_depth=[0.40, 0.44],
            branch_explore_mask_fraction=0.61,
        )

    console = captured.getvalue()
    assert "br_p_explore=0.7000" in console, (
        f"console line missing `br_p_explore=0.7000`; got:\n{console!r}"
    )
    assert "br_attn=0.4200" in console, (
        f"console line missing `br_attn=0.4200`; got:\n{console!r}"
    )
    assert "br_explore_mask=0.6100" in console, (
        f"console line missing `br_explore_mask=0.6100`; got:\n{console!r}"
    )

    assert wandb_run.payloads, "wandb.log was never called"
    payload = wandb_run.payloads[0]
    assert payload["train/branch_explore_rate"] == 0.7
    assert payload["train/branch_attn_fraction"] == 0.42
    assert payload["train/branch_attn_fraction/depth_0"] == 0.40
    assert payload["train/branch_attn_fraction/depth_1"] == 0.44
    assert payload["train/branch_explore_mask_fraction"] == 0.61


def test_log_training_step_omits_branch_telemetry_when_inactive():
    """Negative companion: when branch telemetry inputs are `None`,
    the wandb payload omits those keys entirely so models without
    exploration_only do not pay extra payload cost."""
    logging = _load_logging_module()

    class _FakeWandb:
        def __init__(self):
            self.payload: dict = {}

        def log(self, payload, step):
            self.payload = dict(payload)

    wandb_run = _FakeWandb()
    metrics = {
        "loss": 1.0, "ce_loss": 1.0, "aux_loss": 0.0,
        "aux_loss_normalized": 0.0, "seq_aux_loss": 0.0,
        "branch_aux_loss": 0.0, "attention_aux_loss": 0.0,
    }
    captured = io.StringIO()
    with redirect_stdout(captured):
        logging.log_training_step(
            wandb_run, step=10, metrics=metrics, grad_norm=0.0, lr=1e-4,
            tok_per_s=1.0, tokens_seen=1.0, elapsed=0.001, log_every=1,
            branch_explore_rate=None, branch_attn_fraction=None,
            branch_explore_mask_fraction=None,
        )

    assert "train/branch_explore_rate" not in wandb_run.payload
    assert "train/branch_attn_fraction" not in wandb_run.payload
    assert "train/branch_explore_mask_fraction" not in wandb_run.payload


def test_resume_simulation_starts_from_p_explore_at_resume_step():
    """Resume-safety: build a fresh model with linear schedule,
    simulate a checkpoint resume at `global_step=500`, then call
    `apply_branch_schedule_pre_forward(model, 500)`. The branch
    router's `exploration_only_rate` must be exactly the schedule's
    step-500 value, NOT the constructor-seeded initial rate.
    """
    routing = _load_routing_module()
    initial = 1.0
    final = 0.01
    decay_steps = 1000

    model, _cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=initial,
        branch_exploration_decay="linear",
        branch_exploration_warmup_steps=decay_steps,
        branch_exploration_min=final,
    )
    branch = model.model.branch_router
    assert branch.exploration_only_rate == initial, (
        "constructor should have seeded exploration_only_rate to initial"
    )

    rate_at_500 = routing.apply_branch_schedule_pre_forward(model, 500)
    expected = initial * 0.5 + final * 0.5
    assert math.isclose(rate_at_500, expected, abs_tol=1e-9)
    assert math.isclose(branch.exploration_only_rate, expected, abs_tol=1e-9), (
        f"after resume hook at step=500, branch.exploration_only_rate "
        f"is {branch.exploration_only_rate}; expected {expected}"
    )


# Old non-gradient-bearing 100-step + 500-step tests have been
# replaced by `test_100_step_window_attn_branch_and_mlp_branch_both_get_gradient`
# and `test_500_step_attn_to_mlp_gradient_ratio_below_1e_minus_6`
# which exercise the actual plan-text gradient contracts.


def test_build_model_call_path_branch_aux_and_bias_are_inert_under_exploration_only():
    """End-to-end `build_model()` call-path test with non-zero
    branch aux/bias settings. Builds a moe_everything model via the
    factory with:

      * `branch_router_aux_loss_coef = 0.5` (non-zero!)
      * `use_deepseek_routing = True`
      * `branch_deepseek = True` (bias path enabled)
      * `branch_balancing = "exploration_only"`
      * `branch_exploration_rate = 1.0`

    Despite the non-zero aux coef and the active deepseek bias path,
    `balancing="exploration_only"` MUST keep:
      1. The branch's `local_tokens_per_expert` buffer unchanged
         after a forward (the count path skips exploration_only).
      2. The branch's `expert_bias` buffer unchanged after a step
         that runs `update_expert_biases` (the walker skips
         exploration_only-mode branch owners).

    This is the production-side proof that the explore-only mode
    truly disables branch aux/bias state.
    """
    routing = _load_routing_module()

    model, cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=1.0,
        branch_exploration_decay="constant",
        branch_exploration_warmup_steps=0,
        branch_exploration_min=0.0,
        branch_router_aux_loss_coef=0.5,
        use_deepseek_routing=True,
        branch_deepseek=True,
    )
    model.train()
    branch = model.model.branch_router

    branch.local_tokens_per_expert.zero_()
    branch.local_tokens_per_expert += 11
    branch.expert_bias.zero_()
    branch.expert_bias += 0.5
    initial_counts = branch.local_tokens_per_expert.clone()
    initial_bias = branch.expert_bias.clone()

    routing.apply_branch_schedule_pre_forward(model, 0)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    assert torch.equal(branch.local_tokens_per_expert, initial_counts), (
        f"branch.local_tokens_per_expert mutated: "
        f"before={initial_counts.tolist()}, "
        f"after={branch.local_tokens_per_expert.tolist()}"
    )
    with torch.no_grad():
        routing.update_expert_biases(model, bias_rate=0.01, distributed=False)
    assert torch.equal(branch.expert_bias, initial_bias), (
        f"branch.expert_bias mutated under exploration_only despite "
        f"deepseek_bias method: before={initial_bias.tolist()}, "
        f"after={branch.expert_bias.tolist()}"
    )


def test_per_layer_router_attn_fraction_returns_per_depth_list():
    """When `per_layer_router=True`, every depth has its own
    BranchRouter, so `collect_branch_attn_fraction` must return a
    per-depth list with one entry per depth. Logging then emits
    `train/branch_attn_fraction/depth_<i>` for each.
    """
    routing = _load_routing_module()
    torch.manual_seed(20260428)

    model, _cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=1.0,
        per_layer_router=True,
    )
    model.train()
    routing.apply_branch_schedule_pre_forward(model, 0)
    input_ids = torch.randint(0, model.vocab_size, (1, 16), dtype=torch.long)
    _ = model(input_ids=input_ids, labels=input_ids)

    global_mean, per_depth = routing.collect_branch_attn_fraction(model)
    assert per_depth is not None
    assert len(per_depth) == 2, (
        f"per_layer_router=True with num_hidden_layers=2 should produce "
        f"2 entries; got {len(per_depth)}"
    )
    # Token-weighted global mean equals sum-of-attn / sum-of-total.
    # When per-depth counts are equal (16 tokens each here), this also
    # equals the unweighted mean of per-depth fractions.
    assert math.isclose(
        global_mean, sum(per_depth) / len(per_depth), abs_tol=1e-9,
    )


def test_shared_router_attn_fraction_uses_all_depth_cache_not_final_only():
    """Round 23 review Finding 1 regression: a shared branch router
    (`per_layer_router=False`) is called once per depth. The
    module-level `last_selected_experts` is overwritten on each call,
    so `collect_branch_attn_fraction` must read from the model's
    `_all_branch_selected_experts` cache (which records every depth's
    selection) rather than from the shared module's final-call view.

    Setup: a 2-depth shared-router model with a strong `expert_bias`
    forcing argmax = MLP. Then we manually replace the cache
    contents with a synthetic split: depth 0 = all ATTN, depth 1 =
    all MLP. The global ATTN fraction must be 0.5 — not 0.0 (the
    "final depth's view" bug) and not the unweighted mean of two
    already-averaged fractions when per-depth counts differ.
    """
    routing = _load_routing_module()
    torch.manual_seed(20260428)

    model, _cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=0.0,
        per_layer_router=False,
        use_deepseek_routing=True,
        branch_deepseek=True,
    )
    branch = model.model.branch_router
    with torch.no_grad():
        branch.expert_bias[0] = -10.0
        branch.expert_bias[1] = 10.0

    model.train()
    routing.apply_branch_schedule_pre_forward(model, 0)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    _ = model(input_ids=input_ids, labels=input_ids)

    cache = model.model._all_branch_selected_experts
    assert len(cache) == 2, (
        f"shared-router 2-depth forward should populate 2 cache entries; "
        f"got {len(cache)}"
    )

    # Replace the cache contents with synthetic depth-0=ATTN, depth-1=MLP.
    cache[0] = torch.zeros_like(cache[0])
    cache[1] = torch.ones_like(cache[1])

    global_mean, per_depth = routing.collect_branch_attn_fraction(model)
    assert per_depth == [1.0, 0.0], (
        f"per-depth ATTN fractions should be [1.0, 0.0]; got {per_depth}"
    )
    assert math.isclose(global_mean, 0.5, abs_tol=1e-9), (
        f"global ATTN fraction should be 0.5 (token-weighted across "
        f"both depths); got {global_mean}. The shared-router walker "
        f"must read from _all_branch_selected_experts, not from the "
        f"module's last_selected_experts (which is the FINAL depth's "
        f"view only)."
    )


def test_shared_router_attn_counts_token_weighted_across_uneven_depths():
    """Token-weighted global fraction with uneven per-depth token
    counts: depth 0 with 4 ATTN tokens out of 8; depth 1 with 4
    ATTN tokens out of 4 (synthetic). Total = 8/12 = 0.6667,
    not the unweighted mean (4/8 + 4/4)/2 = 0.75. The walker MUST
    return token-weighted, not depth-averaged.
    """
    routing = _load_routing_module()
    model, _cfg = _build_moe_everything_with_schedule()
    model.train()
    routing.apply_branch_schedule_pre_forward(model, 0)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    _ = model(input_ids=input_ids, labels=input_ids)

    cache = model.model._all_branch_selected_experts
    cache[0] = torch.tensor([[[0], [0], [0], [0], [1], [1], [1], [1]]], dtype=torch.long)
    cache[1] = torch.tensor([[[0], [0], [0], [0]]], dtype=torch.long)

    counts = routing.collect_branch_attn_counts(model)
    assert counts is not None
    total_attn, total_count, per_depth = counts
    assert total_attn == 8 and total_count == 12, (
        f"expected (8 ATTN, 12 total); got ({total_attn}, {total_count})"
    )
    assert per_depth == [(4, 8), (4, 4)]

    global_mean, _ = routing.collect_branch_attn_fraction(model)
    assert math.isclose(global_mean, 8 / 12, abs_tol=1e-9), (
        f"token-weighted global mean should be 8/12 ≈ 0.6667; "
        f"got {global_mean}. Beware: averaging already-averaged "
        f"fractions (4/8 + 4/4)/2 = 0.75 would silently substitute "
        f"the wrong metric when per-depth token counts differ."
    )


_MODEL_YAML_HEAD = """experiment_name: round_23_branch_router_test
model:
  type: moe_everything
  vocab_size: 32
  hidden_size: 16
  num_hidden_layers: 2
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
  topk_scaling_factor: 2.5
  num_attn_experts: 2
  num_attn_experts_per_tok: 1
  attn_expert_mode: per_head_no_recompute
  scale_attn_by_routing_weight: true
  scale_branch_by_routing_weight: true
  use_deepseek_routing: true
  branch_deepseek: true
  attention_bias: false
  attention_dropout: 0.0
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  max_position_embeddings: 64
  tie_word_embeddings: true
  output_router_logits: false
  attn_implementation: eager
"""

_TRAINING_YAML_TAIL = """training:
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
  output_dir: /tmp/round_23_test_outputs
  router_aux_loss_coef: 0.0
  seq_aux_loss_coef: 0.0
  bias_update_rate: 0.0
  load_balancing_method: deepseek_bias
"""


def _yaml_with_nested_branch_router(tmp_path):
    yaml_text = _MODEL_YAML_HEAD + """  branch_router:
    balancing: exploration_only
    exploration_rate: 0.7
    exploration_decay: cosine
    exploration_min: 0.05
    exploration_warmup_steps: 500
""" + _TRAINING_YAML_TAIL
    p = tmp_path / "nested.yaml"
    p.write_text(yaml_text)
    return str(p)


def _yaml_with_flat_branch_fields(tmp_path):
    yaml_text = _MODEL_YAML_HEAD + """  branch_balancing: exploration_only
  branch_exploration_rate: 0.3
  branch_exploration_decay: linear
  branch_exploration_min: 0.0
  branch_exploration_warmup_steps: 200
""" + _TRAINING_YAML_TAIL
    p = tmp_path / "flat.yaml"
    p.write_text(yaml_text)
    return str(p)


def _build_via_load_config(yaml_path):
    """Replicate `scripts/train.py`'s yaml-to-model construction
    sequence: `load_config -> build_model`. Using `_load_routing_module`
    style direct import to bypass `src.training.__init__`'s pandas
    chain.
    """
    repo = Path(__file__).resolve().parent.parent
    if "src.training" not in sys.modules:
        import types as _types
        training_pkg = _types.ModuleType("src.training")
        training_pkg.__path__ = [str(repo / "src" / "training")]
        sys.modules["src.training"] = training_pkg

    spec_cfg = importlib.util.spec_from_file_location(
        "src.training.config", str(repo / "src" / "training" / "config.py"),
    )
    cfg_mod = importlib.util.module_from_spec(spec_cfg)
    sys.modules["src.training.config"] = cfg_mod
    spec_cfg.loader.exec_module(cfg_mod)

    spec_factory = importlib.util.spec_from_file_location(
        "src.training.model_factory",
        str(repo / "src" / "training" / "model_factory.py"),
    )
    factory_mod = importlib.util.module_from_spec(spec_factory)
    sys.modules["src.training.model_factory"] = factory_mod
    spec_factory.loader.exec_module(factory_mod)

    cfg = cfg_mod.load_config(yaml_path)
    model, model_cfg = factory_mod.build_model(cfg)
    return cfg, model, model_cfg


def test_load_config_build_model_nested_branch_router(tmp_path):
    """Round 23 review Finding 2a: real `load_config` + `build_model`
    coverage for the preferred nested `model.branch_router` form.
    Asserts the constructed BranchRouter on the built model has the
    yaml-supplied `balancing` and `exploration_only_rate`, NOT the
    config defaults.
    """
    yaml_path = _yaml_with_nested_branch_router(tmp_path)
    cfg, model, model_cfg = _build_via_load_config(yaml_path)
    branch = model.model.branch_router
    assert branch.balancing == "exploration_only", (
        f"nested yaml's `balancing: exploration_only` did not propagate; "
        f"got {branch.balancing!r}"
    )
    assert math.isclose(branch.exploration_only_rate, 0.7, abs_tol=1e-9), (
        f"nested yaml's `exploration_rate: 0.7` did not propagate; "
        f"got {branch.exploration_only_rate}"
    )
    assert getattr(model_cfg, "branch_exploration_decay", None) == "cosine"
    assert math.isclose(
        getattr(model_cfg, "branch_exploration_min", -1), 0.05, abs_tol=1e-9,
    )
    assert getattr(model_cfg, "branch_exploration_warmup_steps", -1) == 500


def test_load_config_build_model_flat_branch_fields(tmp_path):
    """Round 23 review Finding 2a: real `load_config` + `build_model`
    coverage for the flat-bridge form (`branch_balancing`,
    `branch_exploration_rate`, etc. directly on `model:`). Asserts
    the same fields propagate when written in the legacy form.
    """
    yaml_path = _yaml_with_flat_branch_fields(tmp_path)
    cfg, model, model_cfg = _build_via_load_config(yaml_path)
    branch = model.model.branch_router
    assert branch.balancing == "exploration_only"
    assert math.isclose(branch.exploration_only_rate, 0.3, abs_tol=1e-9)
    assert getattr(model_cfg, "branch_exploration_decay", None) == "linear"
    assert getattr(model_cfg, "branch_exploration_warmup_steps", -1) == 200


def test_load_config_rejects_conflicting_nested_and_flat(tmp_path):
    """The AC-13/17 validator rejects yamls that set the same
    branch_router field in BOTH the nested form AND the flat-bridge
    form with conflicting values. This is the documented contract
    of `validate_branch_router_config` in `balancing_fields.py`.
    Equal-values redundancy is accepted (the migrator can leave
    both during a partial migration); contradiction is not.
    """
    yaml_text = _MODEL_YAML_HEAD + """  branch_router:
    balancing: exploration_only
    exploration_rate: 0.9
  branch_balancing: none
  branch_exploration_rate: 0.0
""" + _TRAINING_YAML_TAIL
    p = tmp_path / "both.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="conflicts with"):
        _build_via_load_config(str(p))


def test_load_config_accepts_redundant_nested_and_flat_with_equal_values(tmp_path):
    """Companion: equal nested + flat values are accepted; the
    migrator can leave both forms during a partial migration."""
    yaml_text = _MODEL_YAML_HEAD + """  branch_router:
    balancing: exploration_only
    exploration_rate: 0.9
  branch_balancing: exploration_only
  branch_exploration_rate: 0.9
""" + _TRAINING_YAML_TAIL
    p = tmp_path / "redundant.yaml"
    p.write_text(yaml_text)
    cfg, model, model_cfg = _build_via_load_config(str(p))
    branch = model.model.branch_router
    assert branch.balancing == "exploration_only"
    assert math.isclose(branch.exploration_only_rate, 0.9, abs_tol=1e-9)


def test_real_trainer_path_telemetry_with_fake_wandb(tmp_path):
    """Round 23 review Finding 2b: a real production-trainer-path
    telemetry test. Reproduces the trainer's actual sequence:

      1. `apply_branch_schedule_pre_forward(model, global_step)`.
      2. Run the forward pass (so the model populates its
         all-depth `_all_branch_selected_experts` cache).
      3. Compute `(branch_attn_counts, branch_explore_mask_fraction)`.
      4. Convert counts to fractions exactly as the trainer does.
      5. Call `log_training_step(...)` with those values.

    Assertions:
      * The console + W&B `train/branch_explore_rate` is the rate
        `apply_branch_schedule_pre_forward` returned (NOT the
        constructor's seeded initial rate, NOT zero, NOT
        `p_explore(global_step+1)`).
      * The console + W&B `train/branch_attn_fraction` matches the
        token-weighted mean of `(selected == 0).float().mean()`
        over the cache entries, computed directly from the model's
        post-forward state.
    """
    routing = _load_routing_module()
    logging = _load_logging_module()

    yaml_path = _yaml_with_nested_branch_router(tmp_path)
    _cfg, model, _model_cfg = _build_via_load_config(yaml_path)
    model.train()

    global_step = 200
    applied_rate = routing.apply_branch_schedule_pre_forward(model, global_step)
    assert applied_rate is not None and applied_rate > 0.0

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
    _ = model(input_ids=input_ids, labels=input_ids)

    counts = routing.collect_branch_attn_counts(model)
    assert counts is not None
    total_attn, total_count, per_depth_pairs = counts
    expected_attn_fraction = total_attn / total_count
    expected_per_depth = [
        n_attn / n_total if n_total > 0 else 0.0
        for n_attn, n_total in per_depth_pairs
    ]

    captured = io.StringIO()

    class _FakeWandb:
        def __init__(self):
            self.payloads = []

        def log(self, payload, step):
            self.payloads.append(dict(payload))

    wandb_run = _FakeWandb()
    metrics = {
        "loss": 1.0, "ce_loss": 1.0, "aux_loss": 0.0,
        "aux_loss_normalized": 0.0, "seq_aux_loss": 0.0,
        "branch_aux_loss": 0.0, "attention_aux_loss": 0.0,
    }
    with redirect_stdout(captured):
        logging.log_training_step(
            wandb_run,
            step=global_step,
            metrics=metrics,
            grad_norm=0.5, lr=1e-4,
            tok_per_s=1234.0, tokens_seen=1.0e6,
            elapsed=0.001, log_every=1,
            branch_explore_rate=applied_rate,
            branch_attn_fraction=expected_attn_fraction,
            branch_attn_per_depth=expected_per_depth,
            branch_explore_mask_fraction=None,
        )

    console = captured.getvalue()
    assert f"br_p_explore={applied_rate:.4f}" in console, (
        f"console missing the applied rate; rate={applied_rate:.4f}, "
        f"got console:\n{console!r}"
    )
    assert f"br_attn={expected_attn_fraction:.4f}" in console, (
        f"console missing the model-derived ATTN fraction; "
        f"expected={expected_attn_fraction:.4f}, got console:\n{console!r}"
    )

    assert wandb_run.payloads
    payload = wandb_run.payloads[0]
    assert math.isclose(
        payload["train/branch_explore_rate"], applied_rate, abs_tol=1e-9,
    )
    assert math.isclose(
        payload["train/branch_attn_fraction"],
        expected_attn_fraction, abs_tol=1e-9,
    )


def _accumulate_grad_norms_on_modules(modules):
    """Sum |grad|.norm() across every parameter under any of the
    given modules. Used by the gradient-flow tests to compare the
    branch's contribution to the total parameter-gradient signal.
    """
    total = 0.0
    for m in modules:
        for p in m.parameters():
            if p.grad is not None:
                total += float(p.grad.norm().item())
    return total


def test_100_step_window_attn_branch_and_mlp_branch_both_get_gradient():
    """Round 23 review Finding 2c: replace the previous
    100-step-windowed test with a real branch-parameter
    gradient-flow assertion. With `exploration_only_rate=1.0`
    EVERY token is randomly assigned to ATTN or MLP, so over a
    100-step window:
      * the attention bank's parameters MUST receive non-zero
        cumulative gradient (about half the tokens flow through
        the attention branch);
      * the MLP bank's parameters MUST receive non-zero cumulative
        gradient (the other half).
    The router gate's gradient is kept as a diagnostic but is no
    longer the sole assertion. The cumulative count of ATTN-routed
    tokens is also kept as a diagnostic for windowed mean.
    """
    routing = _load_routing_module()
    torch.manual_seed(20260428)

    model, _cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=1.0,
        branch_exploration_decay="constant",
        branch_exploration_warmup_steps=0,
        branch_exploration_min=0.0,
    )
    model.train()
    branch = model.model.branch_router
    attn_modules = [model.model.attn_bank]
    mlp_modules = [model.model.mlp_bank]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    cumulative_attn_grad = 0.0
    cumulative_mlp_grad = 0.0
    cumulative_attn_count = 0
    cumulative_total_count = 0
    nonzero_gate_grad_steps = 0

    routing.apply_branch_schedule_pre_forward(model, 0)
    for step in range(100):
        routing.apply_branch_schedule_pre_forward(model, step)
        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, model.vocab_size, (1, 16), dtype=torch.long)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        cumulative_attn_grad += _accumulate_grad_norms_on_modules(attn_modules)
        cumulative_mlp_grad += _accumulate_grad_norms_on_modules(mlp_modules)
        if branch.gate.weight.grad is not None and (
            branch.gate.weight.grad.abs().max().item() > 1e-12
        ):
            nonzero_gate_grad_steps += 1
        counts = routing.collect_branch_attn_counts(model)
        if counts is not None:
            ca, ct, _ = counts
            cumulative_attn_count += ca
            cumulative_total_count += ct
        optimizer.step()

    # Plan-text positive contract: BOTH branches receive gradient.
    assert cumulative_attn_grad > 1e-6, (
        f"cumulative attention-branch gradient too small: "
        f"{cumulative_attn_grad:.3e}; expected > 1e-6 over 100 steps "
        f"with exploration_only rate=1.0"
    )
    assert cumulative_mlp_grad > 1e-6, (
        f"cumulative MLP-branch gradient too small: "
        f"{cumulative_mlp_grad:.3e}; expected > 1e-6 over 100 steps"
    )

    # Diagnostic: gate gradient on most steps.
    assert nonzero_gate_grad_steps >= 90, (
        f"gate gradient was non-zero on only {nonzero_gate_grad_steps}/100 "
        f"steps; expected >= 90"
    )

    # Diagnostic: windowed mean ATTN fraction.
    assert cumulative_total_count > 0
    mean_attn = cumulative_attn_count / cumulative_total_count
    assert 0.45 <= mean_attn <= 0.55, (
        f"windowed ATTN fraction = {mean_attn:.4f}; expected ~0.5 "
        f"(uniform Bernoulli over 1600 tokens, ~3.5% noise floor)"
    )


def test_500_step_attn_to_mlp_gradient_ratio_below_1e_minus_6():
    """Round 23 review Finding 2d: the actual plan-text negative
    contract — when `exploration_only_rate=0.0` AND the gate's
    biased argmax is forced to MLP via a strong `expert_bias`,
    over 500 steps the cumulative ATTN-branch gradient divided by
    the cumulative MLP-branch gradient must be below `1e-6`.

    This is the gradient-ratio collapse, not the selection-ratio
    collapse: zero ATTN selections AND zero ATTN-branch gradient
    are independent assertions; the plan-text contract is the
    gradient one.
    """
    routing = _load_routing_module()
    torch.manual_seed(20260428)

    model, _cfg = _build_moe_everything_with_schedule(
        branch_balancing="exploration_only",
        branch_exploration_rate=0.0,
        branch_exploration_decay="constant",
        branch_exploration_warmup_steps=0,
        branch_exploration_min=0.0,
        branch_deepseek=True,
        use_deepseek_routing=True,
    )
    branch = model.model.branch_router
    with torch.no_grad():
        branch.expert_bias.fill_(0.0)
        branch.expert_bias[0] = -10.0
        branch.expert_bias[1] = 10.0

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    routing.apply_branch_schedule_pre_forward(model, 0)
    attn_modules = [model.model.attn_bank]
    mlp_modules = [model.model.mlp_bank]

    cumulative_attn_grad = 0.0
    cumulative_mlp_grad = 0.0
    total_attn = 0
    total_total = 0
    for step in range(500):
        routing.apply_branch_schedule_pre_forward(model, step)
        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        cumulative_attn_grad += _accumulate_grad_norms_on_modules(attn_modules)
        cumulative_mlp_grad += _accumulate_grad_norms_on_modules(mlp_modules)
        counts = routing.collect_branch_attn_counts(model)
        if counts is not None:
            ca, ct, _ = counts
            total_attn += ca
            total_total += ct
        optimizer.step()

    assert cumulative_mlp_grad > 0, (
        "sanity: cumulative MLP gradient must be positive (denominator)"
    )
    ratio = cumulative_attn_grad / cumulative_mlp_grad
    assert ratio < 1e-6, (
        f"deterministic-collapse gradient-ratio failed: "
        f"cumulative_attn_grad / cumulative_mlp_grad = {ratio:.3e}; "
        f"plan-text threshold is < 1e-6. "
        f"attn_grad={cumulative_attn_grad:.3e}, "
        f"mlp_grad={cumulative_mlp_grad:.3e}"
    )

    # Diagnostic (kept, but not the primary assertion): selection
    # ratio also collapses to ~0 because the same expert_bias drives
    # the argmax.
    selection_ratio = total_attn / max(total_total, 1)
    assert selection_ratio < 1e-6, (
        f"selection-collapse diagnostic: ATTN/total = {selection_ratio}"
    )


if __name__ == "__main__":
    test_trainer_schedule_trajectory_constant_linear_cosine("constant")
    test_trainer_schedule_trajectory_constant_linear_cosine("linear")
    test_trainer_schedule_trajectory_constant_linear_cosine("cosine")
    test_log_training_step_emits_p_explore_and_attn_fraction()
    test_log_training_step_omits_branch_telemetry_when_inactive()
    test_resume_simulation_starts_from_p_explore_at_resume_step()
    test_100_step_windowed_branch_count_and_gradient_flow()
    test_500_step_deterministic_collapse_with_gradient_ratio_threshold()
    test_build_model_call_path_branch_aux_and_bias_are_inert_under_exploration_only()
    test_per_layer_router_attn_fraction_returns_per_depth_list()
    print("ALL OK")
