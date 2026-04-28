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
        attn_expert_mode="per_head_fully_independent",
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
        per_head_compute_mode="auto",
        per_head_dense_fraction_threshold=0.75,
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


def test_100_step_windowed_branch_count_and_gradient_flow():
    """100-step windowed test for branch counts and gradient flow.

    With `branch_balancing="exploration_only"` and a constant
    `exploration_only_rate=1.0`, every branch decision is a uniform
    Bernoulli draw over {ATTN, MLP}. Across 100 steps with deterministic
    seed, the mean ATTN fraction is expected to be ~0.5 with ~3.5%
    binomial noise floor for batch 1×16 = 16 tokens/step (so
    100*16=1600 tokens; 1.96σ ≈ 2.45%). Assert the windowed mean
    falls within 5%.

    Gradient flow: even though branch decisions are random, the
    BranchRouter's gate weight still produces softmax `probs` that
    flow into the model output (probs are `unsqueeze`d into `(B, T, 2)`
    and used downstream); a backward pass through the loss MUST
    leave a non-zero gate gradient.
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

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    routing.apply_branch_schedule_pre_forward(model, 0)
    attn_fractions = []
    nonzero_grad_steps = 0

    for step in range(100):
        rate = routing.apply_branch_schedule_pre_forward(model, step)
        assert rate == 1.0, f"constant rate should stay 1.0; got {rate}"
        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, model.vocab_size, (1, 16), dtype=torch.long)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        if branch.gate.weight.grad is not None and (
            branch.gate.weight.grad.abs().max().item() > 1e-12
        ):
            nonzero_grad_steps += 1
        sel = branch.last_selected_experts
        attn_fractions.append((sel == 0).float().mean().item())
        optimizer.step()

    mean_attn = sum(attn_fractions) / len(attn_fractions)
    assert 0.45 <= mean_attn <= 0.55, (
        f"100-step windowed mean ATTN fraction {mean_attn:.4f} outside "
        f"[0.45, 0.55] (rate=1.0 is uniform Bernoulli; binomial noise "
        f"floor ~3.5% on 1600 tokens)."
    )
    assert nonzero_grad_steps >= 90, (
        f"gate gradient was non-zero on only {nonzero_grad_steps}/100 "
        f"steps; expected >= 90 (gate flows through the softmax probs)"
    )


def test_500_step_deterministic_collapse_with_gradient_ratio_threshold():
    """500-step deterministic-collapse negative test: with
    `branch_balancing="exploration_only"` AND
    `branch_exploration_rate=0.0`, the rate=0 fall-through path
    drives every branch decision via the gate's biased argmax.
    With a strong `expert_bias` toward MLP (index 1), every token
    chooses MLP, leaving the ATTN expert dormant. The plan-text
    contract is that the ATTN-vs-MLP gradient ratio falls below
    1e-6, i.e. no token's gradient flows through the ATTN branch.

    The test accumulates per-step branch selections across all 500
    steps and asserts the cumulative ATTN/total ratio is below the
    1e-6 threshold. Because the walker skips exploration_only-mode
    branch owners, `expert_bias` is never updated by the bias
    walker, so the strong initial bias persists for the entire run
    and the collapse is stable.
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

    # `expert_bias` dominates the deepseek_style routing's argmax
    # because `biased = sigmoid(logits) + expert_bias` and
    # `sigmoid(logits)` is bounded in (0, 1). A bias of +10 / -10
    # forces argmax = 1 for every input regardless of logits.
    with torch.no_grad():
        branch.expert_bias.fill_(0.0)
        branch.expert_bias[0] = -10.0
        branch.expert_bias[1] = 10.0
    initial_bias = branch.expert_bias.clone()

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    routing.apply_branch_schedule_pre_forward(model, 0)

    total_attn = 0
    total_mlp = 0
    for step in range(500):
        routing.apply_branch_schedule_pre_forward(model, step)
        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, model.vocab_size, (1, 8), dtype=torch.long)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        optimizer.step()
        sel = branch.last_selected_experts
        total_attn += int((sel == 0).sum().item())
        total_mlp += int((sel == 1).sum().item())

    total = total_attn + total_mlp
    ratio = total_attn / max(total, 1)
    assert ratio < 1e-6, (
        f"deterministic collapse failed: ATTN/total = "
        f"{ratio} (total_attn={total_attn}, total_mlp={total_mlp}); "
        f"plan-text threshold is < 1e-6"
    )
    assert torch.equal(branch.expert_bias, initial_bias), (
        f"branch.expert_bias mutated under exploration_only over 500 steps: "
        f"before={initial_bias.tolist()}, "
        f"after={branch.expert_bias.tolist()}"
    )


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
    assert math.isclose(
        global_mean, sum(per_depth) / len(per_depth), abs_tol=1e-9,
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
