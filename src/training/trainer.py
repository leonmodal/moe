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
from .metrics import compute_output_metrics
from .model_factory import build_model, configure_liger_kernels
from .routing import (
    apply_router_exploration_rate,
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

    if train_cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            if is_main_process():
                print("Gradient checkpointing enabled.", flush=True)

    model.to(device)
    # Per the canonical-block resolver rule: seq_aux_loss_coef lives under `training:` (canonical). The
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
    output_router_logits = output_router_logits_for_method(load_balancing_method_resolved)

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
        model_type=cfg.get("model", {}).get("type"),
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
    if is_main_process():
        print("=" * 60, flush=True)
        print(f"  Model     : {cfg['model']['type']}", flush=True)
        print(f"  Params    : {params['total']/1e9:.3f}B total", flush=True)
        print(f"  Strategy  : {strategy}", flush=True)
        print(f"  Wrapper   : {describe_wrapper(model)}", flush=True)
        print(f"  Precision : {train_cfg.mixed_precision}", flush=True)
        print("=" * 60, flush=True)

    # Build datasets
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
    heatmap_every = int(cfg.get("training", {}).get("heatmap_every", eval_cfg.get("every", 0)))
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
        optimizer.zero_grad(set_to_none=True)
        window_metrics = {
            "loss": 0.0, "ce_loss": 0.0, "aux_loss": 0.0,
            "aux_loss_normalized": 0.0, "seq_aux_loss": 0.0,
            "branch_aux_loss": 0.0, "attention_aux_loss": 0.0,
        }
        local_tokens_in_step = 0
        grad_norm = 0.0

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
            labels = input_ids  # Model handles the label shift internally
            sync_context = (
                nullcontext()
                if micro_idx == train_cfg.gradient_accumulation - 1 or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        **({} if is_dense else {"output_router_logits": output_router_logits}),
                    )
                # Router z-loss (if any router has `router_z_loss_coef > 0`)
                # is accumulated per-router during forward and summed here so
                # it rides the same gradient-accumulation scaling as the base
                # loss. No cost when every router has the feature disabled —
                # `collect_router_z_loss` returns None in that case.
                if not is_dense:
                    z_loss = collect_router_z_loss(model)
                    if z_loss is not None:
                        output.loss = output.loss + z_loss
                loss = output.loss / train_cfg.gradient_accumulation
                loss.backward()

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
        )

        # Routing heatmaps
        if heatmap_every > 0 and global_step > 0 and global_step % heatmap_every == 0:
            save_routing_plots(model, output_dir=train_cfg.output_dir, step=global_step)

        # Eval
        if eval_dataloader is not None and int(eval_cfg.get("every", 0)) > 0:
            if global_step % int(eval_cfg["every"]) == 0:
                eval_metrics = run_validation(
                    model=model,
                    model_cfg=model_cfg,
                    eval_dataloader=eval_dataloader,
                    max_batches=int(eval_cfg.get("max_batches", 0)),
                    is_dense=is_dense,
                    seq_aux_loss_coef=seq_aux_loss_coef,
                    device=device,
                    step=global_step,
                )
                log_eval_metrics(wandb_run, step=global_step, eval_metrics=eval_metrics)

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
