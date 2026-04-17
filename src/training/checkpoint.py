"""Checkpoint IO: separate model/optimizer/data saves, resume, safetensors conversion."""

from __future__ import annotations

import json
import os
import re
import shutil

import torch

from .distributed import dist_rank, dist_world_size, is_main_process, barrier, unwrap_model

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )
except Exception:
    FSDP = None
    FullOptimStateDictConfig = None
    FullStateDictConfig = None
    StateDictType = None


def _save_optimizer_state(optimizer, optim_state: dict, ckpt_dir: str) -> None:
    """Save optimizer state, splitting Muon hybrid optimizer into separate files."""
    try:
        from src.utils.muon import Muon
        if isinstance(optimizer, Muon):
            # Muon optimizer has both muon and adam param groups
            # Save the full state dict but mark it as muon-type
            torch.save(
                {"type": "muon", "state_dict": optim_state},
                os.path.join(ckpt_dir, "optimizer_muon.pt"),
            )
            return
    except ImportError:
        pass

    # Standard optimizer (AdamW or similar)
    torch.save(
        {"type": "adam", "state_dict": optim_state},
        os.path.join(ckpt_dir, "optimizer_adam.pt"),
    )


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: str,
    dataset_state: dict | None = None,
    wandb_run_id: str | None = None,
    tokens_seen: float = 0.0,
) -> None:
    """Save checkpoint with separate model/optimizer/training/data state files.

    Writes go into `checkpoint-{step}.tmp/` and the directory is atomically
    renamed to `checkpoint-{step}/` once every rank has finished flushing.
    `find_latest_checkpoint` only sees the renamed directory, so a crash
    mid-save leaves a `.tmp` sibling behind rather than a torn checkpoint
    that resume would try to use.
    """
    final_dir = os.path.join(output_dir, f"checkpoint-{step}")
    tmp_dir = os.path.join(output_dir, f"checkpoint-{step}.tmp")
    if is_main_process():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    # Every rank writes into `tmp_dir`; main must create it first.
    barrier()

    if FSDP is not None and isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy, optim_policy):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = unwrap_model(model).state_dict()
        optim_state = optimizer.state_dict()

    if is_main_process():
        # Save model weights separately
        torch.save(model_state, os.path.join(tmp_dir, "model.pt"))

        # Save optimizer state — detect Muon hybrid and save separately
        _save_optimizer_state(optimizer, optim_state, tmp_dir)

        # Save training state (scheduler, step, etc.)
        training_state = {
            "scheduler": scheduler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
        }
        if wandb_run_id:
            training_state["wandb_run_id"] = wandb_run_id
        torch.save(training_state, os.path.join(tmp_dir, "training_state.pt"))

        # Save metadata as JSON for easy inspection
        meta = {"step": step, "tokens_seen": tokens_seen}
        if wandb_run_id:
            meta["wandb_run_id"] = wandb_run_id
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    # Data state is per-rank — the parquet dataset shards files by
    # `i % world_size == rank`, so rank 0's state is meaningless for other
    # ranks. Every rank writes its own `data_state_rank{r}.pt`; rank 0 also
    # writes the legacy unsuffixed `data_state.pt` so single-process and
    # pre-fix multi-rank checkpoints stay resumable (via the fallback in
    # `_load_data_state`).
    if dataset_state:
        rank = dist_rank()
        torch.save(dataset_state, os.path.join(tmp_dir, f"data_state_rank{rank}.pt"))
        if is_main_process():
            torch.save(dataset_state, os.path.join(tmp_dir, "data_state.pt"))

    # All files flushed; barrier then atomic rename on rank 0.
    barrier()
    if is_main_process():
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
        print(f"Saved checkpoint to {final_dir}", flush=True)
    barrier()


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    resume_from: str,
) -> tuple[int, dict | None, float]:
    """Load checkpoint from directory.

    Supports:
    - New separate format: model.pt + optimizer_adam.pt/optimizer_muon.pt + training_state.pt
    - Safetensors format: model.safetensors (for model weights after conversion)
    - Legacy monolithic format: trainer.pt
    """
    # Try new format first (separate files)
    model_path = os.path.join(resume_from, "model.pt")
    if os.path.exists(model_path):
        return _load_separate_checkpoint(model, optimizer, scheduler, resume_from)

    # Try safetensors model weights
    safetensors_path = os.path.join(resume_from, "model.safetensors")
    if os.path.exists(safetensors_path):
        return _load_safetensors_checkpoint(model, optimizer, scheduler, resume_from)

    # Fall back to legacy monolithic format
    trainer_path = os.path.join(resume_from, "trainer.pt")
    if os.path.exists(trainer_path):
        return _load_legacy_checkpoint(model, optimizer, scheduler, resume_from)

    raise FileNotFoundError(f"No checkpoint found in {resume_from}")


def _load_safetensors_checkpoint(
    model,
    optimizer,
    scheduler,
    resume_from: str,
) -> tuple[int, dict | None, float]:
    """Load checkpoint from safetensors model weights + other state files."""
    from safetensors.torch import load_file

    model_state = load_file(os.path.join(resume_from, "model.safetensors"))

    if FSDP is not None and isinstance(model, FSDP):
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(model_state)
    else:
        unwrap_model(model).load_state_dict(model_state)

    # Load optimizer if available
    try:
        optim_state = _load_optimizer_state(resume_from)
        if FSDP is not None and isinstance(model, FSDP):
            optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, optim_policy=optim_policy):
                loaded_optim = FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
                optimizer.load_state_dict(loaded_optim)
        else:
            optimizer.load_state_dict(optim_state)
    except FileNotFoundError:
        print("No optimizer state found in safetensors checkpoint, starting fresh", flush=True)

    # Load training state
    training_state_path = os.path.join(resume_from, "training_state.pt")
    step, tokens_seen = 0, 0.0
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        scheduler.load_state_dict(training_state["scheduler"])
        step = training_state.get("step", 0)
        tokens_seen = training_state.get("tokens_seen", 0.0)

    data_state = _load_data_state(resume_from)

    return step, data_state, tokens_seen


def _load_data_state(resume_from: str) -> dict | None:
    """Load this rank's data state, preferring per-rank file over the legacy
    unsuffixed one.

    Parquet sharding gives each rank a distinct set of files; pre-fix
    checkpoints only saved rank 0's state, so when a non-zero rank sees
    only `data_state.pt` it falls back to that shared state (the old
    behaviour). New checkpoints write `data_state_rank{r}.pt` per rank.
    """
    rank = dist_rank()
    per_rank_path = os.path.join(resume_from, f"data_state_rank{rank}.pt")
    if os.path.exists(per_rank_path):
        return torch.load(per_rank_path, map_location="cpu")
    shared_path = os.path.join(resume_from, "data_state.pt")
    if os.path.exists(shared_path):
        return torch.load(shared_path, map_location="cpu")
    return None


def _load_optimizer_state(resume_from: str) -> dict:
    """Load optimizer state from checkpoint directory, supporting all formats."""
    # Try typed format: optimizer_muon.pt or optimizer_adam.pt
    for name in ("optimizer_muon.pt", "optimizer_adam.pt"):
        path = os.path.join(resume_from, name)
        if os.path.exists(path):
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                return payload["state_dict"]
            return payload

    # Fall back to legacy single optimizer.pt
    path = os.path.join(resume_from, "optimizer.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")

    raise FileNotFoundError(f"No optimizer state found in {resume_from}")


def _load_separate_checkpoint(
    model,
    optimizer,
    scheduler,
    resume_from: str,
) -> tuple[int, dict | None, float]:
    """Load checkpoint from separate files."""
    model_state = torch.load(os.path.join(resume_from, "model.pt"), map_location="cpu")

    # Load optimizer state — try new typed format first, then legacy
    optim_state = _load_optimizer_state(resume_from)

    if FSDP is not None and isinstance(model, FSDP):
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy, optim_policy):
            model.load_state_dict(model_state)
            loaded_optim = FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
            optimizer.load_state_dict(loaded_optim)
    else:
        unwrap_model(model).load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

    training_state_path = os.path.join(resume_from, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        scheduler.load_state_dict(training_state["scheduler"])
        step = training_state.get("step", 0)
        tokens_seen = training_state.get("tokens_seen", 0.0)
    else:
        step = 0
        tokens_seen = 0.0

    data_state = _load_data_state(resume_from)

    return step, data_state, tokens_seen


def _load_legacy_checkpoint(
    model,
    optimizer,
    scheduler,
    resume_from: str,
) -> tuple[int, dict | None, float]:
    """Load checkpoint from legacy monolithic trainer.pt format."""
    trainer_path = os.path.join(resume_from, "trainer.pt")
    payload = torch.load(trainer_path, map_location="cpu")

    if FSDP is not None and isinstance(model, FSDP):
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy, optim_policy):
            model.load_state_dict(payload["model"])
            loaded_optim = FSDP.optim_state_dict_to_load(model, optimizer, payload["optimizer"])
            optimizer.load_state_dict(loaded_optim)
    else:
        unwrap_model(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])

    scheduler.load_state_dict(payload["scheduler"])

    meta_path = os.path.join(resume_from, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("step", 0), meta.get("dataset_state"), meta.get("tokens_seen", 0.0)
    return 0, None, 0.0


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint directory by step number."""
    if not os.path.isdir(output_dir):
        return None
    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    for entry in os.listdir(output_dir):
        m = pattern.match(entry)
        if m:
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path):
                checkpoints.append((int(m.group(1)), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def cleanup_checkpoints(output_dir: str, max_keep: int) -> None:
    """Remove old checkpoints, keeping only the most recent `max_keep`."""
    if max_keep <= 0 or not is_main_process():
        return
    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    for entry in os.listdir(output_dir):
        m = pattern.match(entry)
        if m:
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path):
                checkpoints.append((int(m.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_keep:
        _, path = checkpoints.pop(0)
        shutil.rmtree(path)


def convert_to_safetensors(checkpoint_dir: str, output_path: str | None = None) -> str:
    """Convert a model checkpoint to safetensors format.

    Args:
        checkpoint_dir: Directory containing model.pt (or trainer.pt for legacy format)
        output_path: Output path for the safetensors file. Defaults to model.safetensors in checkpoint_dir.

    Returns:
        Path to the saved safetensors file.
    """
    from safetensors.torch import save_file

    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
    else:
        trainer_path = os.path.join(checkpoint_dir, "trainer.pt")
        if not os.path.exists(trainer_path):
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
        payload = torch.load(trainer_path, map_location="cpu")
        state_dict = payload["model"]

    if output_path is None:
        output_path = os.path.join(checkpoint_dir, "model.safetensors")

    save_file(state_dict, output_path)
    print(f"Saved safetensors to {output_path}", flush=True)
    return output_path
