import torch

from src.training.model_factory import build_model
from src.training.config import build_training_config
from src.training.eval import run_validation
from src.training.metrics import compute_output_metrics
from src.training.recurrence import (
    mean_recurrence_for_step,
    recurrence_step_for_data_step,
    sample_hrm_recurrence_step,
)
from src.training.routing import trainer_post_optimizer_bias_update, update_expert_biases
from src.utils.load_balance_artifacts import build_load_balance_artifacts, save_load_balance_artifacts
from src.utils.recurrent_diagnostics import (
    build_recurrent_diagnostics,
    save_recurrent_diagnostics_artifacts,
)


def _tiny_cfg(
    model_type="recurrent_standard_moe",
    *,
    core_attention=True,
    core_attention_layers=None,
    recurrent_sandwich_norm=False,
    recurrent_history_attention=False,
):
    num_experts = 8 if model_type == "recurrent_global_moe" else 4
    recurrent_layers = 8 if model_type == "hrm_recurrent_standard_moe" else 2
    cfg = {
        "model": {
            "type": model_type,
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 1 + recurrent_layers + 1,
            "prelude_layers": 1,
            "recurrent_layers": recurrent_layers,
            "coda_layers": 1,
            "core_attention": core_attention,
            "recurrent_sandwich_norm": recurrent_sandwich_norm,
            "recurrent_history_attention": recurrent_history_attention,
            "head_dim": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "intermediate_size": 64,
            "norm_topk_prob": True,
            "router_type": "deepseek",
            "use_deepseek_routing": True,
            "topk_scaling_factor": 2.5,
            "num_groups": 2,
            "group_topk": 1,
            "tie_word_embeddings": True,
            "use_fused_linear_ce": False,
            "mlp_router": {
                "balancing": "deepseek_bias",
                "bias_update_rate": 0.001,
                "bias_update_zero_sum": True,
            },
        },
        "recurrence": {
            "mean_recurrence_schedule": {
                "turn_on": True,
                "warmup": 0.25,
                "warmup_type": "linear",
                "max_mean_rec": 32,
            },
            "mean_backprop_depth": 2,
            "eval_recurrence": 3,
        },
        "training": {},
    }
    if core_attention_layers is not None:
        cfg["model"]["core_attention_layers"] = core_attention_layers
    if model_type == "hrm_recurrent_standard_moe":
        cfg["model"]["hrm_l_layers"] = 4
        cfg["model"]["hrm_h_layers"] = 4
        cfg["model"]["hrm_history_attention"] = True
    if model_type == "recurrent_global_moe":
        cfg["model"]["global_router_update"] = True
        cfg["model"]["prelude_coda"] = {
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
        }
    return cfg


def test_hrm_recurrence_sampler_is_deterministic_and_splits_l_h_cycles():
    first = sample_hrm_recurrence_step(
        data_step=123,
        mean_recurrence=32,
        mean_backprop_depth=8,
    )
    second = sample_hrm_recurrence_step(
        data_step=123,
        mean_recurrence=32,
        mean_backprop_depth=8,
    )

    assert first == second
    h_cycles = first.num_steps_no_grad + first.num_steps_with_grad
    assert h_cycles == len(first.extra_steps)
    assert h_cycles > 0
    assert first.num_steps_with_grad > 0
    assert all(l_count >= 1 for l_count in first.extra_steps)
    assert sum(first.extra_steps) >= h_cycles


def test_recurrent_standard_moe_forward_backward():
    model, _ = build_model(_tiny_cfg())
    model.gradient_checkpointing_enable()
    assert all(block.gradient_checkpointing for block in model.model.recurrent_blocks)
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))

    out = model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([1, 2]),
    )

    assert out.loss is not None
    counts_before = [
        owner.local_tokens_per_expert.detach().clone()
        for owner, label in model.get_all_balancing_owners()
        if label == "mlp"
    ]
    out.loss.backward()
    counts_after = [
        owner.local_tokens_per_expert.detach().clone()
        for owner, label in model.get_all_balancing_owners()
        if label == "mlp"
    ]
    assert len(counts_before) == len(counts_after)
    for before, after in zip(counts_before, counts_after):
        torch.testing.assert_close(after, before)
    assert out.num_steps_no_grad == 1
    assert out.num_steps_with_grad == 2
    assert out.selected_experts is not None


def test_flat_recurrent_sandwich_norm_runs_on_recurrent_blocks_only():
    model, _ = build_model(_tiny_cfg(recurrent_sandwich_norm=True))
    model.train()
    inner = model.model
    assert len(inner.recurrent_post_attention_norms) == len(inner.recurrent_blocks)
    assert len(inner.recurrent_post_mlp_norms) == len(inner.recurrent_blocks)

    calls = {"attn": 0, "mlp": 0}

    def _count(kind):
        def _hook(_module, _args, _out):
            calls[kind] += 1

        return _hook

    handles = []
    for norm in inner.recurrent_post_attention_norms:
        handles.append(norm.register_forward_hook(_count("attn")))
    for norm in inner.recurrent_post_mlp_norms:
        handles.append(norm.register_forward_hook(_count("mlp")))
    try:
        input_ids = torch.randint(0, 64, (2, 8))
        out = model(
            input_ids=input_ids,
            labels=input_ids,
            num_steps=torch.tensor([0, 2]),
        )
        out.loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    expected = len(inner.recurrent_blocks) * 2
    assert calls == {"attn": expected, "mlp": expected}


def test_flat_recurrent_history_attention_sees_all_loop_outputs():
    model, _ = build_model(_tiny_cfg(recurrent_history_attention=True))
    model.train()
    seen_history_lengths = []

    def _capture_history(_module, args):
        seen_history_lengths.append(len(args[1]))

    handle = model.model.flat_history_attention.register_forward_pre_hook(_capture_history)
    try:
        input_ids = torch.randint(0, 64, (2, 8))
        out = model(
            input_ids=input_ids,
            labels=input_ids,
            num_steps=torch.tensor([1, 2]),
        )
        out.loss.backward()
    finally:
        handle.remove()

    assert seen_history_lengths == [3]


def test_hrm_recurrent_standard_moe_runs_l_h_modules_and_history_attention(tmp_path):
    model, _ = build_model(_tiny_cfg("hrm_recurrent_standard_moe"))
    model.train()
    inner = model.model
    assert inner.config.recurrent_loop == "hrm"
    assert inner.h_history_attention is not None

    input_ids = torch.randint(0, 64, (2, 8))
    out = model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([1, 2, 2, 1, 3]),
        collect_recurrence_diagnostics=True,
    )

    assert out.loss is not None
    assert out.hrm_h_cycles == 3
    assert out.hrm_l_cycles == 6
    out.loss.backward()

    assert inner.h_history_attention.o_proj.weight.grad is not None
    assert next(inner.recurrent_blocks[0].parameters()).grad is not None
    assert next(inner.recurrent_blocks[-1].parameters()).grad is not None

    payload = build_load_balance_artifacts(model, step=1)
    recurrent_rows = payload["pools"]["mlp_recurrent"]["per_depth"]
    assert {row["_hrm_module"] for row in recurrent_rows} == {"L", "H"}
    assert max(row["_loop"] for row in recurrent_rows) == 2

    diagnostics = build_recurrent_diagnostics(model, step=1, input_ids=input_ids)
    assert diagnostics is not None
    assert diagnostics["routing_rows"] == []
    hrm_l_rows = diagnostics["hrm_l_routing_rows"]
    assert len(hrm_l_rows) == 12
    assert len(diagnostics["hrm_h_routing_rows"]) == 8
    assert len(diagnostics["flat_residual_rows"]) == 36
    routing_similarity_rows = diagnostics["routing_similarity_rows"]
    assert len(routing_similarity_rows) == 28
    assert sorted({row["block"] for row in routing_similarity_rows}) == list(range(8))
    assert sorted({row["transition_index"] for row in hrm_l_rows}) == [0, 1, 2]
    assert sorted({row["block"] for row in hrm_l_rows}) == [0, 1, 2, 3]
    assert all(row["cycle"] == 0 for row in hrm_l_rows if row["transition_index"] == 0)
    assert all(row["cycle"] == 2 for row in hrm_l_rows if row["transition_index"] in {1, 2})
    assert any(
        row["cycle"] == 0
        and row["l_index"] == 0
        and row["next_l_index"] == 1
        and row["block"] == 0
        for row in hrm_l_rows
    )
    assert any(
        row["cycle"] == 2
        and row["l_index"] == 1
        and row["next_l_index"] == 2
        and row["block"] == 3
        for row in hrm_l_rows
    )
    assert 0.0 <= diagnostics["summary"]["hrm_l_routing_jaccard_distance_mean"] <= 1.0
    assert diagnostics["summary"]["flat_layer_relative_residual_rms_mean"] > 0.0
    assert 0.0 <= diagnostics["summary"]["routing_similarity_jaccard_mean"] <= 1.0
    assert save_recurrent_diagnostics_artifacts(
        model,
        step_dir=str(tmp_path),
        step=1,
        input_ids=input_ids,
    )
    out_dir = tmp_path / "recurrent_diagnostics"
    assert (out_dir / "hrm_l_routing_transitions.csv").exists()
    assert (out_dir / "hrm_h_routing_transitions.csv").exists()
    assert (out_dir / "flattened_residuals.csv").exists()
    assert (out_dir / "flattened_residuals.png").exists()
    assert (out_dir / "routing_similarity_by_block.csv").exists()
    assert (out_dir / "routing_similarity_by_block.png").exists()


def test_recurrent_core_no_attention_skips_recurrent_attention():
    for model_type in ("recurrent_standard_moe", "recurrent_global_moe"):
        model, _ = build_model(_tiny_cfg(model_type, core_attention=False))
        model.train()
        called = {"value": False}

        def _mark_called(_module, _args):
            called["value"] = True

        handles = [
            block.self_attn.register_forward_pre_hook(_mark_called)
            for block in model.model.recurrent_blocks
        ]
        try:
            input_ids = torch.randint(0, 64, (2, 8))
            model(
                input_ids=input_ids,
                labels=input_ids,
                num_steps=torch.tensor([0, 2]),
            )
        finally:
            for handle in handles:
                handle.remove()

        assert called["value"] is False


def test_recurrent_core_first_attention_layer_only():
    for model_type in ("recurrent_standard_moe", "recurrent_global_moe"):
        model, _ = build_model(_tiny_cfg(model_type, core_attention=False, core_attention_layers=[0]))
        model.train()
        calls = [0 for _ in model.model.recurrent_blocks]

        handles = []
        for idx, block in enumerate(model.model.recurrent_blocks):
            def _mark_called(_module, _args, *, block_idx=idx):
                calls[block_idx] += 1

            handles.append(block.self_attn.register_forward_pre_hook(_mark_called))
        try:
            input_ids = torch.randint(0, 64, (2, 8))
            model(
                input_ids=input_ids,
                labels=input_ids,
                num_steps=torch.tensor([0, 2]),
            )
        finally:
            for handle in handles:
                handle.remove()

        assert calls[0] == 2
        assert calls[1:] == [0 for _ in calls[1:]]


def test_recurrent_adapter_keeps_prelude_gradients_with_truncated_bptt():
    for model_type in ("recurrent_standard_moe", "recurrent_global_moe"):
        model, _ = build_model(_tiny_cfg(model_type))
        model.train()
        assert model.model.recurrent_adapter.up_proj.in_features == 64
        assert model.model.recurrent_adapter.up_proj.out_features == 64
        assert model.model.recurrent_adapter.down_proj.in_features == 64
        assert model.model.recurrent_adapter.down_proj.out_features == 32
        input_ids = torch.randint(0, 64, (2, 8))

        out = model(
            input_ids=input_ids,
            labels=input_ids,
            num_steps=torch.tensor([2, 1]),
        )
        out.loss.backward()

        prelude_grad = model.model.prelude_blocks[0].input_layernorm.weight.grad
        adapter_grad = model.model.recurrent_adapter.down_proj.weight.grad
        assert prelude_grad is not None
        assert adapter_grad is not None
        assert prelude_grad.abs().sum().item() > 0
        assert adapter_grad.abs().sum().item() > 0


def test_recurrent_global_pool_has_shared_bias_owner_and_counts_no_grad_loops():
    model, _ = build_model(_tiny_cfg("recurrent_global_moe"))
    model.train()
    inner = model.model
    owner = inner.shared_recurrent_bias
    assert owner is not None

    gates = [block.mlp.gate for block in inner.recurrent_blocks]
    assert gates
    for gate in gates:
        assert gate.expert_bias.data_ptr() == owner.expert_bias.data_ptr()
        assert gate.local_tokens_per_expert.data_ptr() == owner.local_tokens_per_expert.data_ptr()

    owners = list(model.get_all_balancing_owners())
    assert sum(candidate is owner for candidate, label in owners if label == "mlp") == 1

    owner.local_tokens_per_expert.zero_()
    input_ids = torch.randint(0, 64, (2, 8))
    model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([2, 0]),
    )
    assert owner.local_tokens_per_expert.sum().item() > 0
    assert owner.local_tokens_per_expert.std().item() > 0
    payload = build_load_balance_artifacts(model, step=1)
    assert payload is not None
    assert sorted(payload["pools"]) == ["mlp_boundary", "mlp_recurrent"]
    assert sorted(payload["bias_pools"]) == ["mlp_boundary", "mlp_recurrent"]
    recurrent_bias_rows = payload["bias_pools"]["mlp_recurrent"]["rows"]
    assert {row["router"] for row in recurrent_bias_rows} == {"shared"}

    initial_bias = owner.expert_bias.detach().clone()
    with torch.no_grad():
        update_expert_biases(model, bias_rate=0.1, distributed=False)
    assert owner.local_tokens_per_expert.sum().item() == 0
    assert not torch.equal(owner.expert_bias, initial_bias)
    for gate in gates:
        assert torch.equal(gate.expert_bias, owner.expert_bias)


def test_recurrent_global_pool_shared_bias_survives_module_apply():
    model, _ = build_model(_tiny_cfg("recurrent_global_moe"))
    model.train()
    inner = model.model
    owner = inner.shared_recurrent_bias
    assert owner is not None

    inner._apply(lambda tensor: tensor.clone())
    gates = [block.mlp.gate for block in inner.recurrent_blocks]
    for gate in gates:
        assert gate.expert_bias.data_ptr() == owner.expert_bias.data_ptr()
        assert gate.local_tokens_per_expert.data_ptr() == owner.local_tokens_per_expert.data_ptr()

    owner.local_tokens_per_expert.zero_()
    input_ids = torch.randint(0, 64, (2, 8), device=owner.expert_bias.device)
    model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([2, 0], device=owner.expert_bias.device),
    )
    assert owner.local_tokens_per_expert.sum().item() > 0


def test_recurrent_global_pool_trainer_update_moves_shared_bias_owner():
    cfg = _tiny_cfg("recurrent_global_moe")
    model, _ = build_model(cfg)
    train_cfg = build_training_config(cfg)
    model.train()

    owner = model.model.shared_recurrent_bias
    assert owner is not None
    owner.local_tokens_per_expert.zero_()
    owner.local_tokens_per_expert[0] = 100.0
    owner.local_tokens_per_expert[1:] = 1.0
    initial_bias = owner.expert_bias.detach().clone()

    trainer_post_optimizer_bias_update(
        model,
        train_cfg,
        cfg,
        distributed=False,
        global_step=1,
    )

    assert owner.local_tokens_per_expert.sum().item() == 0
    assert not torch.equal(owner.expert_bias, initial_bias)
    assert owner.expert_bias[0].item() < initial_bias[0].item()


def test_recurrent_global_pool_bias_moves_with_gradient_checkpointing():
    cfg = _tiny_cfg("recurrent_global_moe")
    model, _ = build_model(cfg)
    model.gradient_checkpointing_enable()
    model.train()

    owner = model.model.shared_recurrent_bias
    assert owner is not None
    owner.local_tokens_per_expert.zero_()
    input_ids = torch.randint(0, 64, (2, 8))
    out = model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([2, 2]),
    )
    out.loss.backward()

    assert owner.local_tokens_per_expert.sum().item() > 0
    assert owner.local_tokens_per_expert.std().item() > 0
    initial_bias = owner.expert_bias.detach().clone()
    with torch.no_grad():
        update_expert_biases(model, bias_rate=0.1, distributed=False)

    assert owner.local_tokens_per_expert.sum().item() == 0
    assert not torch.equal(owner.expert_bias, initial_bias)


def test_recurrent_load_balance_artifacts_track_loop_and_block(tmp_path):
    model, _ = build_model(_tiny_cfg("recurrent_global_moe"))
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))

    model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([2, 2]),
    )
    payload = build_load_balance_artifacts(model, step=1)

    recurrent_rows = payload["pools"]["mlp_recurrent"]["per_depth"]
    assert len(recurrent_rows) == 8
    assert [(row["_loop"], row["_block"]) for row in recurrent_rows] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]

    assert save_load_balance_artifacts(model, step_dir=str(tmp_path), step=1)
    recurrent_dir = tmp_path / "load_balancing" / "mlp_recurrent"
    assert (recurrent_dir / "per_layer_histograms.png").exists()
    assert (recurrent_dir / "heatmap_block_00.png").exists()
    assert (recurrent_dir / "heatmap_block_01.png").exists()


def test_recurrent_diagnostics_track_residuals_and_routing_changes(tmp_path):
    for model_type in ("recurrent_standard_moe", "recurrent_global_moe"):
        model, _ = build_model(_tiny_cfg(model_type))
        model.train()
        input_ids = torch.randint(0, 64, (2, 8))

        model(
            input_ids=input_ids,
            labels=input_ids,
            num_steps=torch.tensor([1, 2]),
            collect_recurrence_diagnostics=True,
        )
        payload = build_recurrent_diagnostics(model, step=50, input_ids=input_ids)

        assert payload is not None
        assert len(payload["loop_rows"]) == 3
        assert len(payload["routing_rows"]) == 4
        assert len(payload["flat_residual_rows"]) == 6
        assert len(payload["routing_similarity_rows"]) == 4
        summary = payload["summary"]
        assert summary["loop_count"] == 3.0
        assert summary["relative_residual_rms_mean"] > 0.0
        assert -1.0 <= summary["cosine_mean"] <= 1.0
        assert 0.0 <= summary["routing_jaccard_distance_mean"] <= 1.0
        assert 0.0 <= summary["routing_last_token_jaccard_distance_mean"] <= 1.0
        assert summary["flat_layer_relative_residual_rms_mean"] > 0.0
        assert 0.0 <= summary["routing_similarity_jaccard_mean"] <= 1.0

        assert save_recurrent_diagnostics_artifacts(
            model,
            step_dir=str(tmp_path / model_type),
            step=50,
            input_ids=input_ids,
        )
        out_dir = tmp_path / model_type / "recurrent_diagnostics"
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "loop_residuals.csv").exists()
        assert (out_dir / "routing_changes.csv").exists()
        assert (out_dir / "flattened_residuals.csv").exists()
        assert (out_dir / "routing_similarity_by_block.csv").exists()
        assert (out_dir / "loop_residuals.png").exists()
        assert (out_dir / "routing_changes.png").exists()
        assert (out_dir / "flattened_residuals.png").exists()
        assert (out_dir / "routing_similarity_by_block.png").exists()


def test_eval_recurrence_sweep_logs_metrics_and_artifacts(tmp_path):
    model, _ = build_model(_tiny_cfg("recurrent_standard_moe"))
    input_ids = torch.randint(0, 64, (2, 8))
    eval_batches = [{"input_ids": input_ids}]

    metrics = run_validation(
        model=model,
        model_cfg=_tiny_cfg("recurrent_standard_moe")["model"],
        eval_dataloader=eval_batches,
        max_batches=1,
        is_dense=False,
        seq_aux_loss_coef=0.0,
        device=torch.device("cpu"),
        step=250,
        output_dir=str(tmp_path),
        recurrence_sweep=[1, 2],
    )

    assert "eval/ce_loss" in metrics
    assert "eval/recurrence_001/ce_loss" in metrics
    assert "eval/recurrence_002/ce_loss" in metrics
    # eval_recurrence=3 is added to preserve the historical eval/* keys.
    assert "eval/recurrence_003/ce_loss" in metrics
    assert "eval/recurrence_002/recurrent/relative_residual_rms_last" in metrics
    assert "eval/recurrence_002/load_balance/mlp_recurrent/cv" in metrics

    eval_dir = tmp_path / "eval_logs" / "step_00000250"
    assert (eval_dir / "recurrence_001" / "load_balancing" / "summary.csv").exists()
    assert (
        eval_dir
        / "recurrence_002"
        / "recurrent_diagnostics"
        / "routing_changes.png"
    ).exists()
    assert (eval_dir / "load_balancing" / "summary.csv").exists()


def test_recurrent_global_metrics_handle_boundary_and_recurrent_expert_widths():
    model, model_cfg = build_model(_tiny_cfg("recurrent_global_moe"))
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))

    out = model(
        input_ids=input_ids,
        labels=input_ids,
        num_steps=torch.tensor([1, 2]),
        output_router_logits=False,
    )
    metrics, _, _ = compute_output_metrics(
        out,
        model,
        model_cfg,
        input_ids,
        seq_aux_loss_coef=0.0,
    )

    assert metrics["aux_loss_normalized"] > 0


def test_recurrence_schedule_and_sampler_match_retrofitting_contract():
    cfg = _tiny_cfg()
    assert mean_recurrence_for_step(cfg, global_step=0, max_steps=100) == 1
    assert mean_recurrence_for_step(cfg, global_step=25, max_steps=100) == 32
    assert mean_recurrence_for_step(cfg, global_step=13, max_steps=100) == 17

    first = recurrence_step_for_data_step(
        cfg,
        data_step=7,
        global_step=0,
        max_steps=100,
    )
    second = recurrence_step_for_data_step(
        cfg,
        data_step=7,
        global_step=0,
        max_steps=100,
    )
    assert first == second
    assert first.mean_recurrence == 1
    assert first.mean_backprop_depth == 1
    assert first.num_steps_with_grad <= 1
