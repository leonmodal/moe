"""Unified training loop using shared modules.

This is the core training loop used by the CLI entrypoint. It supports all
model types (dense, standard_moe, global_moe, moe_everything) with DDP/FSDP.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils.training import (
    build_lr_scheduler,
    build_muon_optimizer,
    build_optimizer,
    count_parameters,
)
from src.utils.recurrent_diagnostics import build_recurrent_diagnostics

from .checkpoint import (
    cleanup_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainingConfig, resolve_initialization_spec
from .data import build_train_dataset, build_eval_dataset
from .distributed import (
    barrier,
    cleanup_distributed,
    describe_wrapper,
    dist_world_size,
    infer_dtype,
    is_distributed,
    is_main_process,
    reduce_scalar,
    reduce_scalar_dict,
    seed_everything,
    setup_distributed,
    unwrap_model,
    wrap_model,
)
from .eval import run_validation
from .logging import (
    log_eval_metrics,
    log_training_step,
    save_routing_plots,
    setup_wandb,
)
from .metrics import compute_output_metrics, output_get, output_set
from .model_factory import build_model, configure_liger_kernels
from .recurrence import is_recurrent_model_type, recurrence_step_for_data_step
from .routing import (
    apply_branch_schedule_pre_forward,
    apply_router_exploration_rate,
    collect_branch_attn_counts,
    collect_branch_explore_mask_fraction,
    collect_router_z_loss,
    exploration_rate_schedule,
    get_bias_rate,
    trainer_optimizer_step_and_bias_update,
    trainer_post_optimizer_bias_update,
    update_expert_biases,
)


def _stateful_dataloader_workers(dataset, requested: int, *, role: str, verbose: bool) -> int:
    """Return the safe `num_workers` for a DataLoader wrapping `dataset`.

    `StatefulParquetDataset.get_state()` reads live attributes that are only
    mutated inside `__iter__` on the dataset copy that is actually iterating.
    When `num_workers > 0` that copy lives in a subprocess, so the checkpointed
    state pulled from the main-process object is stale and resume lands at the
    wrong position. To keep deterministic resume authoritative, any
    dataset that exposes `get_state`/`set_state` is iterated in-process.
    """
    is_stateful = hasattr(dataset, "get_state") and hasattr(dataset, "set_state")
    if is_stateful and requested != 0:
        if verbose:
            print(
                f"[data] {role}: forcing num_workers=0 for stateful dataset "
                f"{type(dataset).__name__} (config requested {requested}); "
                "in-process iteration keeps checkpoint state authoritative.",
                flush=True,
            )
        return 0
    return max(0, int(requested))


def run_training(cfg: dict, train_cfg: TrainingConfig, args) -> None:
    """Main training loop for all supported model types."""
    initialization_spec = resolve_initialization_spec(
        cfg,
        config_path=args.config,
        cli_source_config=getattr(args, "init_from_config", None),
        cli_strategy=getattr(args, "init_strategy", None),
    )

    liger_mode = configure_liger_kernels(cfg)
    if is_main_process():
        print(f"Liger kernels: {liger_mode} for model type '{cfg['model']['type']}'", flush=True)

    rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed + rank)

    # Apply CLI overrides
    dcfg_dict = cfg.get("data", {})
    if getattr(args, "data_dir", None):
        dcfg_dict["data_dir"] = args.data_dir
    if getattr(args, "output_dir", None):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)
    if getattr(args, "max_steps", None) is not None:
        train_cfg = replace(train_cfg, max_steps=args.max_steps)

    strategy = getattr(args, "dist_strategy", "ddp")
    fsdp_sharding_strategy = (
        getattr(args, "fsdp_sharding_strategy", None)
        or train_cfg.fsdp_sharding_strategy
    )

    # Build model
    model, model_cfg = build_model(cfg)
    base_model = model
    params = count_parameters(base_model)

    if initialization_spec is not None:
        from .config import load_config
        source_cfg = load_config(initialization_spec["source_config"])
        source_model, _ = build_model(source_cfg)
        strategy_name = initialization_spec["strategy"]
        if strategy_name == "global_to_alternating_sanity":
            from src.models.init_mapping import copy_global_to_alternating_sanity
            copy_global_to_alternating_sanity(source_model, model)
        else:
            raise ValueError(f"Unknown init strategy: {strategy_name}")

    model_type = cfg.get("model", {}).get("type")
    if train_cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            if is_main_process():
                print("Gradient checkpointing enabled.", flush=True)

    model.to(device)
    # Per the canonical-block resolver: seq_aux_loss_coef lives under `training:` (canonical). The
    # resolver falls back to `model:` with a deprecation warning for any
    # unmigrated yamls.
    from .balancing_fields import _resolve_balancing_field
    seq_aux_loss_coef = _resolve_balancing_field(cfg, "seq_aux_loss_coef", 0.0)
    if seq_aux_loss_coef:
        model._seq_aux_loss_coef = seq_aux_loss_coef

    # Stamp the resolved `load_balancing_method` onto the model so each
    # family's `forward` gates aux / seq-aux additions explicitly, not
    # just by coefficient values. `normalize_balancing_config` (called
    # from `load_config`) has already auto-zeroed conflicting coefficients,
    # so this is belt-and-suspenders against future regressions where a
    # non-zero default coefficient leaks into a non-active method.
    load_balancing_method_resolved = _resolve_balancing_field(cfg, "load_balancing_method", None)
    if load_balancing_method_resolved is not None:
        model._load_balancing_method = load_balancing_method_resolved

    # Resolve `output_router_logits` once and reuse for every per-step
    # forward. Aux methods need gradient-bearing router scores in the
    # model output; non-aux methods don't (the routing-decision state
    # lives in router-internal buffers).
    from .balancing_fields import output_router_logits_for_method
    output_router_logits = bool(
        getattr(
            model_cfg,
            "output_router_logits",
            output_router_logits_for_method(load_balancing_method_resolved),
        )
    )

    if train_cfg.torch_compile:
        compile_mode = train_cfg.torch_compile_mode
        if compile_mode is True:
            compile_mode = "default"
        if is_main_process():
            print(f"Compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(model, mode=compile_mode, dynamic=False)

    model = wrap_model(
        model,
        strategy=strategy,
        local_rank=local_rank,
        mixed_precision_name=train_cfg.mixed_precision,
        model_type=model_type,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
    )
    raw_model = base_model

    # Build optimizer
    if train_cfg.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model,
            train_cfg,
            muon_lr=train_cfg.muon_lr,
            muon_weight_decay=train_cfg.muon_weight_decay,
            adam_lr=train_cfg.adam_lr,
        )
    else:
        optimizer = build_optimizer(model, train_cfg)
    scheduler = build_lr_scheduler(optimizer, train_cfg)

    is_dense = cfg["model"]["type"] == "dense"
    is_recurrent = is_recurrent_model_type(model_type)
    if is_main_process():
        print("=" * 60, flush=True)
        print(f"  Model     : {cfg['model']['type']}", flush=True)
        print(f"  Params    : {params['total']/1e9:.3f}B total", flush=True)
        print(f"  Strategy  : {strategy}", flush=True)
        print(f"  Wrapper   : {describe_wrapper(model)}", flush=True)
        print(f"  Precision : {train_cfg.mixed_precision}", flush=True)
        print("=" * 60, flush=True)

    # Build datasets
    from .data import is_synthetic_format

    if is_synthetic_format(dcfg_dict):
        # Synthetic tasks emit integer tokens directly; no tokenizer needed.
        # `build_train_dataset` / `build_eval_dataset` accept `tokenizer=None`
        # for the synthetic path.
        if is_main_process():
            print(
                f"Synthetic data format ({dcfg_dict.get('format')}): skipping tokenizer load",
                flush=True,
            )
        tokenizer = None
    else:
        tokenizer_name = dcfg_dict.get("tokenizer_name", "gpt2")
        if is_main_process():
            print(f"Loading tokenizer {tokenizer_name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    barrier()

    dataset = build_train_dataset(cfg, tokenizer=tokenizer, rank=rank, world_size=world_size)
    data_num_workers = dcfg_dict.get("num_workers", 4)
    train_workers = _stateful_dataloader_workers(
        dataset, data_num_workers, role="train", verbose=is_main_process()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=train_workers,
        pin_memory=True,
        prefetch_factor=2 if train_workers > 0 else None,
        persistent_workers=train_workers > 0,
    )

    eval_dataset = build_eval_dataset(cfg, tokenizer=tokenizer, rank=rank, world_size=world_size)
    # Attention-pattern eval targets are derived once from the (synthetic)
    # eval dataset and reused across eval steps. `None` for parquet datasets.
    _attention_eval_targets = None
    if eval_dataset is not None:
        try:
            from .attention_eval import build_ground_truth_targets

            _attention_eval_targets = build_ground_truth_targets(eval_dataset)
        except Exception as exc:  # noqa: BLE001
            if is_main_process():
                print(f"[attn_eval] Failed to build ground-truth targets: {exc}", flush=True)
            _attention_eval_targets = None
        if is_main_process() and _attention_eval_targets is not None:
            t, m = _attention_eval_targets
            print(
                f"[attn_eval] Targets enabled: shape={t.shape}, learnable rows={int(m.sum())}",
                flush=True,
            )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_cfg = cfg.get("eval", {})
        # Eval DataLoader must also run in-process for parquet-backed eval
        # sets. `StatefulParquetDataset` is an `IterableDataset` whose
        # `__iter__` shards only on `rank`/`world_size` and does not consult
        # `torch.utils.data.get_worker_info()`. With `num_workers > 0` each
        # DataLoader worker process would iterate the same file subset and
        # emit every validation example once per worker, inflating step
        # counts and corrupting eval metrics. Clamping to 0 here is a
        # correctness gate, not a resume-safety one — it can be lifted once
        # the dataset gains per-worker sub-sharding.
        eval_workers = _stateful_dataloader_workers(
            eval_dataset, data_num_workers, role="eval", verbose=is_main_process()
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=int(eval_cfg.get("batch_size", train_cfg.batch_size)),
            num_workers=eval_workers,
            pin_memory=True,
            prefetch_factor=2 if eval_workers > 0 else None,
            persistent_workers=eval_workers > 0,
        )

    # Resume from checkpoint
    global_step = 0
    tokens_seen = 0.0
    resume_from = getattr(args, "resume", None) or cfg.get("checkpoint", {}).get("resume_from")
    if getattr(args, "auto_resume", False) and not resume_from:
        resume_from = find_latest_checkpoint(train_cfg.output_dir)
    wandb_run_id = None
    if resume_from:
        global_step, dataset_state, tokens_seen = load_checkpoint(model, optimizer, scheduler, resume_from)
        if dataset_state:
            dataset.set_state(dataset_state)
        if is_main_process():
            print(f"Resumed from step {global_step}", flush=True)
        # Try to recover wandb run id
        meta_path = os.path.join(resume_from, "meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            wandb_run_id = meta.get("wandb_run_id")

    wandb_run = setup_wandb(
        project=train_cfg.wandb_project,
        run_name=train_cfg.wandb_run_name,
        config=cfg,
        output_dir=train_cfg.output_dir,
        resume_run_id=wandb_run_id,
    )

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    data_iter = iter(dataloader)
    autocast_dtype = infer_dtype(train_cfg.mixed_precision)
    autocast_enabled = autocast_dtype is not None
    model.train()

    if is_main_process():
        print(f"Starting training from step {global_step}", flush=True)

    eval_cfg = cfg.get("eval", {})
    train_logging_cfg = cfg.get("training", {})
    heatmap_every = int(
        train_logging_cfg.get(
            "routing_log_every",
            train_logging_cfg.get("heatmap_every", eval_cfg.get("every", 0)),
        )
        or 0
    )
    # Dense-early windows: log every step up to <until>, then sparsify.
    # 0 keeps legacy behavior (no dense early window). See `log_training_step`
    # and the heatmap gate below.
    log_dense_until = int(train_logging_cfg.get("log_dense_until", 0) or 0)
    routing_log_dense_until = int(
        train_logging_cfg.get("routing_log_dense_until", 0) or 0
    )
    distributed = is_distributed()

    # Read the model-side target exploration rate once; the trainer schedules
    # the effective rate between warmup_start and this target when warmup_steps > 0.
    router_exploration_target = float(
        cfg.get("model", {}).get("router_exploration_rate", 0.0) or 0.0
    )
    router_exploration_enabled = (
        train_cfg.router_exploration_warmup_steps > 0
        or router_exploration_target > 0.0
    )
    # Cache the last applied rate so we skip the module-tree walk once the
    # schedule has plateaued. Float inequality is safe during warmup because
    # `exploration_rate_schedule` returns `target` verbatim once past
    # `warmup_steps`; we additionally short-circuit the call entirely after
    # `warmup_steps` to avoid the Python-side schedule arithmetic on the hot
    # path. Using `None` as the sentinel forces an apply on the first step.
    last_applied_exploration_rate: float | None = None
    exploration_plateau_applied = False

    # Seed branch pre-forward schedules once before the main loop so
    # step-0 forwards (and the first forward after a checkpoint resume
    # at step=N) see the scheduled branch settings rather than the
    # constructor defaults. The intra-loop call below applies the
    # current branch settings on every subsequent step.
    current_branch_explore_rate = apply_branch_schedule_pre_forward(
        model, global_step
    )

    while global_step < train_cfg.max_steps:
        step_start = time.perf_counter()
        if router_exploration_enabled and not exploration_plateau_applied:
            past_warmup = (
                train_cfg.router_exploration_warmup_steps <= 0
                or global_step >= train_cfg.router_exploration_warmup_steps
            )
            if past_warmup:
                current_rate = router_exploration_target
            else:
                current_rate = exploration_rate_schedule(
                    global_step,
                    target=router_exploration_target,
                    warmup_start=train_cfg.router_exploration_warmup_start,
                    warmup_steps=train_cfg.router_exploration_warmup_steps,
                )
            if current_rate != last_applied_exploration_rate:
                apply_router_exploration_rate(model, current_rate)
                last_applied_exploration_rate = current_rate
            if past_warmup:
                # Rate is now constant for the rest of training; no further
                # schedule evaluation or module-tree walks are needed.
                exploration_plateau_applied = True

        # Apply branch pre-forward schedules for the CURRENT step
        # before the forward pass runs. The returned exploration-only
        # rate remains the source of truth for that legacy logging
        # field; sampling-entropy exposes its value through the loss
        # metric instead.
        current_branch_explore_rate = apply_branch_schedule_pre_forward(
            model, global_step
        )
        next_global_step = global_step + 1
        collect_recurrent_diagnostics_step = (
            cfg["model"]["type"] in {
                "recurrent_standard_moe",
                "recurrent_global_moe",
                "hrm_recurrent_standard_moe",
            }
            and heatmap_every > 0
            and next_global_step > 0
            and (
                next_global_step < routing_log_dense_until
                or next_global_step % heatmap_every == 0
            )
        )

        optimizer.zero_grad(set_to_none=True)
        window_metrics = {
            "loss": 0.0, "ce_loss": 0.0, "aux_loss": 0.0,
            "aux_loss_normalized": 0.0, "seq_aux_loss": 0.0,
            "branch_aux_loss": 0.0, "branch_entropy_loss": 0.0,
            "attention_aux_loss": 0.0,
        }
        local_tokens_in_step = 0
        grad_norm = 0.0
        recurrence_no_grad_sum = 0.0
        recurrence_with_grad_sum = 0.0
        recurrence_mean_value = None
        recurrence_backprop_value = None
        recurrence_micro_count = 0
        hrm_h_cycles_sum = 0.0
        hrm_l_cycles_sum = 0.0
        hrm_micro_count = 0

        for micro_idx in range(train_cfg.gradient_accumulation):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            input_ids = batch["input_ids"]
            # Synthetic datasets emit explicit labels (with -100 masking on
            # positions the model cannot predict, e.g. the initial random
            # state in linear_map / cellular_automata). For parquet, the
            # dataset doesn't emit labels and we default to input_ids — the
            # model handles the shift internally either way.
            labels = batch.get("labels", input_ids)
            sync_context = (
                nullcontext()
                if micro_idx == train_cfg.gradient_accumulation - 1 or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    model_kwargs = {} if is_dense else {"output_router_logits": output_router_logits}
                    if cfg["model"]["type"] in {"moe_everything", "recurrent_moe_everything"}:
                        model_kwargs["return_logits"] = False
                    if is_recurrent:
                        recurrence_info = recurrence_step_for_data_step(
                            cfg,
                            data_step=global_step * train_cfg.gradient_accumulation + micro_idx + 1,
                            global_step=global_step,
                            max_steps=train_cfg.max_steps,
                        )
                        model_kwargs["num_steps"] = recurrence_info.as_tensor(device)
                        recurrence_no_grad_sum += recurrence_info.num_steps_no_grad
                        recurrence_with_grad_sum += recurrence_info.num_steps_with_grad
                        recurrence_mean_value = recurrence_info.mean_recurrence
                        recurrence_backprop_value = recurrence_info.mean_backprop_depth
                        recurrence_micro_count += 1
                        if (
                            collect_recurrent_diagnostics_step
                            and micro_idx == train_cfg.gradient_accumulation - 1
                        ):
                            model_kwargs["collect_recurrence_diagnostics"] = True
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        **model_kwargs,
                    )
                # Router z-loss (if any router has `router_z_loss_coef > 0`)
                # is accumulated per-router during forward and summed here so
                # it rides the same gradient-accumulation scaling as the base
                # loss. No cost when every router has the feature disabled —
                # `collect_router_z_loss` returns None in that case.
                if not is_dense:
                    z_loss = collect_router_z_loss(model)
                    if z_loss is not None:
                        output_set(output, "loss", output_get(output, "loss") + z_loss)
                loss = output_get(output, "loss") / train_cfg.gradient_accumulation
                loss.backward()

            output_hrm_h = output_get(output, "hrm_h_cycles", None)
            output_hrm_l = output_get(output, "hrm_l_cycles", None)
            if output_hrm_h is not None and output_hrm_l is not None:
                hrm_h_cycles_sum += float(output_hrm_h)
                hrm_l_cycles_sum += float(output_hrm_l)
                hrm_micro_count += 1

            metrics, _, _ = compute_output_metrics(
                output, raw_model, model_cfg, input_ids,
                seq_aux_loss_coef=seq_aux_loss_coef,
            )
            for key in window_metrics:
                window_metrics[key] += metrics[key]
            local_tokens_in_step += input_ids.numel()

        if train_cfg.max_grad_norm > 0:
            # `clip_grad_norm_` returns the pre-clip total norm, so there's no
            # need for a separate `get_grad_norm(...)` traversal — that used to
            # double the per-step parameter walk (and do one CPU↔GPU sync per
            # parameter via `.item()` inside the utility).
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.max_grad_norm,
            )
            grad_norm = grad_norm_t.item() if isinstance(grad_norm_t, torch.Tensor) else float(grad_norm_t)
        # Optimizer-step + scheduler-step + post-step bias update,
        # extracted into a single helper so the production call
        # sequence is testable end-to-end. The helper locks the
        # order: optimizer.step → scheduler.step → bias update,
        # and wraps the bias update in `torch.no_grad()` so the
        # function's misuse guard is satisfied.
        global_step += 1
        trainer_optimizer_step_and_bias_update(
            model, optimizer, scheduler, train_cfg, cfg,
            distributed=distributed, global_step=global_step,
        )

        # Momentum warmup for Muon — applied AFTER the helper so the
        # next forward picks up the new momentum. Order vs the bias
        # update is irrelevant (muon mutates optimizer state, bias
        # update mutates expert_bias buffers — disjoint state).
        if train_cfg.optimizer == "muon":
            from src.utils.muon import get_muon_momentum
            new_mom = get_muon_momentum(global_step, warmup_steps=train_cfg.momentum_warmup_steps)
            for group in optimizer.param_groups:
                if group.get("is_muon", False):
                    group["momentum"] = new_mom

        tokens_seen += local_tokens_in_step * world_size
        elapsed = time.perf_counter() - step_start
        step_tokens = local_tokens_in_step * world_size
        tok_per_s = step_tokens / max(elapsed, 1e-9)

        # Cross-rank reduction only happens on log steps — non-log steps keep
        # the per-rank values locally. One batched `all_reduce` replaces the
        # previous 7–8 separate `reduce_scalar` calls per step (7 window
        # metrics + grad_norm), which cut out one CPU↔GPU sync per scalar
        # and collapsed the NCCL collectives into a single launch.
        is_log_step = (global_step % train_cfg.log_every == 0)
        if is_log_step:
            bundle = {
                key: value / train_cfg.gradient_accumulation
                for key, value in window_metrics.items()
            }
            bundle["__grad_norm"] = grad_norm
            reduced_bundle = reduce_scalar_dict(bundle, device=device)
            reduced_grad = reduced_bundle.pop("__grad_norm")
            reduced = reduced_bundle
        else:
            reduced = {
                key: value / train_cfg.gradient_accumulation
                for key, value in window_metrics.items()
            }
            reduced_grad = grad_norm

        # Branch routing telemetry: per-step `p_explore` (the rate
        # applied before this step's forward) and the measured fraction
        # of branch tokens that chose ATTN on this step's forward. Both
        # are `None` when the feature is inactive on this model, in
        # which case the logging path skips emitting the extra fields.
        # On distributed runs, the global mean is reduced across ranks
        # so rank-0 logs see the cluster-wide branch behavior. The mask
        # fraction is exposed under a separate diagnostic key
        # (`branch_explore_mask_fraction`) so the trainer's `% ATTN`
        # value never gets confused with how often the random override
        # was taken.
        branch_explore_rate = current_branch_explore_rate
        branch_counts = collect_branch_attn_counts(model)
        branch_explore_mask_fraction = collect_branch_explore_mask_fraction(model)
        branch_attn_fraction = None
        branch_attn_per_depth = None
        if branch_counts is not None:
            local_attn, local_total, local_depth_pairs = branch_counts
            if is_log_step and distributed:
                # Token-weighted reduction: SUM the (numerator,
                # denominator) pairs across ranks, then divide. Avoids
                # the bias that arises from averaging already-averaged
                # fractions when ranks have different per-rank token
                # counts.
                bundle = {
                    "_attn_num": float(local_attn),
                    "_attn_den": float(local_total),
                }
                for idx, (n_attn, n_total) in enumerate(local_depth_pairs):
                    bundle[f"_attn_num_d{idx}"] = float(n_attn)
                    bundle[f"_attn_den_d{idx}"] = float(n_total)
                if branch_explore_mask_fraction is not None:
                    bundle["_mask"] = branch_explore_mask_fraction
                reduced_branch = reduce_scalar_dict(
                    bundle, reduction="sum", device=device,
                )
                summed_attn = reduced_branch["_attn_num"]
                summed_total = reduced_branch["_attn_den"]
                branch_attn_fraction = (
                    summed_attn / summed_total if summed_total > 0 else 0.0
                )
                branch_attn_per_depth = []
                for idx in range(len(local_depth_pairs)):
                    n_attn = reduced_branch[f"_attn_num_d{idx}"]
                    n_total = reduced_branch[f"_attn_den_d{idx}"]
                    branch_attn_per_depth.append(
                        n_attn / n_total if n_total > 0 else 0.0,
                    )
                if branch_explore_mask_fraction is not None:
                    # Mask fraction reduction is mean-of-rank-means
                    # (we don't track per-rank mask token counts);
                    # convert SUM-reduce back to mean explicitly.
                    branch_explore_mask_fraction = (
                        reduced_branch["_mask"] / max(1, dist_world_size())
                    )
            else:
                branch_attn_fraction = (
                    local_attn / local_total if local_total > 0 else 0.0
                )
                branch_attn_per_depth = [
                    n_attn / n_total if n_total > 0 else 0.0
                    for n_attn, n_total in local_depth_pairs
                ]
        recurrence_diag_summary = None
        if collect_recurrent_diagnostics_step:
            recurrence_diag_payload = build_recurrent_diagnostics(
                raw_model,
                step=global_step,
                input_ids=input_ids,
            )
            if recurrence_diag_payload is not None:
                recurrence_diag_summary = recurrence_diag_payload.get("summary") or None
                if recurrence_diag_summary is not None and distributed:
                    recurrence_diag_summary = reduce_scalar_dict(
                        recurrence_diag_summary,
                        device=device,
                    )
        log_training_step(
            wandb_run,
            step=global_step,
            metrics=reduced,
            grad_norm=reduced_grad,
            lr=scheduler.get_last_lr()[0],
            tok_per_s=tok_per_s,
            tokens_seen=tokens_seen,
            elapsed=elapsed,
            log_every=train_cfg.log_every,
            log_dense_until=log_dense_until,
            branch_explore_rate=branch_explore_rate,
            branch_attn_fraction=branch_attn_fraction,
            branch_attn_per_depth=branch_attn_per_depth,
            branch_explore_mask_fraction=branch_explore_mask_fraction,
            recurrence_mean=recurrence_mean_value,
            recurrence_backprop_depth=recurrence_backprop_value,
            recurrence_num_steps_no_grad=(
                recurrence_no_grad_sum / recurrence_micro_count
                if recurrence_micro_count else None
            ),
            recurrence_num_steps_with_grad=(
                recurrence_with_grad_sum / recurrence_micro_count
                if recurrence_micro_count else None
            ),
            hrm_h_cycles=(
                hrm_h_cycles_sum / hrm_micro_count
                if hrm_micro_count else None
            ),
            hrm_l_cycles=(
                hrm_l_cycles_sum / hrm_micro_count
                if hrm_micro_count else None
            ),
            recurrence_diagnostics=recurrence_diag_summary,
        )

        # Routing heatmaps. Dense early window mirrors `log_dense_until`
        # for the per-step training-stat log: when global_step is below
        # routing_log_dense_until (and > 0), save plots every step.
        # Past that window, fall back to the sparse `heatmap_every` cadence.
        if heatmap_every > 0 and global_step > 0:
            in_dense_routing = global_step < routing_log_dense_until
            if in_dense_routing or global_step % heatmap_every == 0:
                save_routing_plots(
                    model,
                    output_dir=train_cfg.output_dir,
                    step=global_step,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                )

        # Eval
        if eval_dataloader is not None and int(eval_cfg.get("every", 0)) > 0:
            if global_step % int(eval_cfg["every"]) == 0:
                # Attention-pattern eval against known ground-truth (synthetic
                # tasks only). Targets are derived once from the eval dataset
                # below; heatmaps are saved at checkpoint cadence to avoid
                # exploding disk usage at the eval cadence.
                attn_targets = _attention_eval_targets
                save_heatmaps_this_step = (
                    attn_targets is not None
                    and global_step > 0
                    and global_step % train_cfg.save_every == 0
                )
                eval_metrics = run_validation(
                    model=model,
                    model_cfg=model_cfg,
                    eval_dataloader=eval_dataloader,
                    max_batches=int(eval_cfg.get("max_batches", 0)),
                    is_dense=is_dense,
                    seq_aux_loss_coef=seq_aux_loss_coef,
                    device=device,
                    step=global_step,
                    output_dir=train_cfg.output_dir,
                    tokenizer=tokenizer,
                    recurrence_sweep=(
                        eval_cfg.get("recurrence_sweep")
                        if is_recurrent
                        else None
                    ),
                    attention_eval_targets=attn_targets,
                    attention_eval_save_heatmaps=save_heatmaps_this_step,
                )
                log_eval_metrics(
                    wandb_run,
                    step=global_step,
                    eval_metrics=eval_metrics,
                    output_dir=train_cfg.output_dir,
                )

        # Save checkpoint
        if global_step % train_cfg.save_every == 0:
            dataset_state = getattr(dataset, "get_state", lambda: None)()
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                step=global_step, output_dir=train_cfg.output_dir,
                dataset_state=dataset_state,
                wandb_run_id=getattr(wandb_run, "id", None),
                tokens_seen=tokens_seen,
            )
            cleanup_checkpoints(train_cfg.output_dir, train_cfg.max_checkpoints)

    # Final checkpoint
    dataset_state = getattr(dataset, "get_state", lambda: None)()
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler,
        step=global_step, output_dir=train_cfg.output_dir,
        dataset_state=dataset_state,
        wandb_run_id=getattr(wandb_run, "id", None),
        tokens_seen=tokens_seen,
    )
    cleanup_checkpoints(train_cfg.output_dir, train_cfg.max_checkpoints)

    if wandb_run is not None:
        wandb_run.finish()
    if is_main_process():
        print("Training complete.", flush=True)
    cleanup_distributed()
