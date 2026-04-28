"""Training configuration and config file loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .balancing_fields import _resolve_balancing_field


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    # Schedule
    lr_scheduler: str = "cosine"  # cosine | linear | constant | stable_decay
    warmup_steps: int = 2000
    max_steps: int = 100_000
    min_lr_ratio: float = 0.1
    cooldown_frac: float = 0.45
    # Batch
    batch_size: int = 4
    gradient_accumulation: int = 4
    # Mixed precision / memory
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    # Logging
    log_every: int = 10
    save_every: int = 5000
    output_dir: str = "./outputs"
    wandb_project: str | None = "moe-experiments"
    wandb_run_name: str | None = None
    # Optimizer type
    optimizer: str = "adamw"  # adamw | muon
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.0
    adam_lr: float = 3e-4
    momentum_warmup_steps: int = 300
    # Misc
    max_checkpoints: int = 0
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    disable_liger: bool = False
    # Bias update (DeepSeek-style)
    bias_update_rate: float = 0.0
    bias_warmup_start: float = 0.0
    bias_warmup_steps: int = 0
    # Router-exploration warmup (AC-10, from docs/research/external_moe_techniques.md).
    # The "target" rate is the model-side `router_exploration_rate`. When
    # `router_exploration_warmup_steps > 0` the trainer lineartly schedules the
    # effective rate from `router_exploration_warmup_start` at step 0 to the
    # model's target at step `router_exploration_warmup_steps`, then holds at
    # the target. Default (steps=0) is a no-op and preserves existing configs.
    router_exploration_warmup_start: float = 0.0
    router_exploration_warmup_steps: int = 0


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_training_config(cfg: dict) -> TrainingConfig:
    """Build a TrainingConfig from the raw config dict."""
    tcfg = cfg.get("training", {})
    return TrainingConfig(
        learning_rate=tcfg.get("learning_rate", 3e-4),
        weight_decay=tcfg.get("weight_decay", 0.1),
        beta1=tcfg.get("beta1", 0.9),
        beta2=tcfg.get("beta2", 0.95),
        eps=tcfg.get("eps", 1e-8),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        lr_scheduler=tcfg.get("lr_scheduler", "cosine"),
        warmup_steps=tcfg.get("warmup_steps", 2000),
        max_steps=tcfg.get("max_steps", 100_000),
        min_lr_ratio=tcfg.get("min_lr_ratio", 0.1),
        cooldown_frac=tcfg.get("cooldown_frac", 0.45),
        batch_size=tcfg.get("batch_size", 4),
        gradient_accumulation=tcfg.get("gradient_accumulation", 4),
        mixed_precision=tcfg.get("mixed_precision", "bf16"),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", False),
        log_every=tcfg.get("log_every", 10),
        save_every=tcfg.get("save_every", 5000),
        output_dir=tcfg.get("output_dir", "./outputs"),
        wandb_project=tcfg.get("wandb_project"),
        wandb_run_name=tcfg.get("wandb_run_name"),
        optimizer=tcfg.get("optimizer", "adamw"),
        muon_lr=tcfg.get("muon_lr", 0.02),
        muon_weight_decay=tcfg.get("muon_weight_decay", 0.0),
        adam_lr=tcfg.get("adam_lr", 3e-4),
        momentum_warmup_steps=tcfg.get("momentum_warmup_steps", 300),
        max_checkpoints=tcfg.get("max_checkpoints", 0),
        torch_compile=tcfg.get("torch_compile", False),
        torch_compile_mode=tcfg.get("torch_compile_mode", "default"),
        disable_liger=tcfg.get("disable_liger", False),
        # Per DEC-3b (AC-3): the canonical block for these balancing fields is
        # `training:`. The resolver falls back to `model:` with a deprecation
        # warning so unmigrated yamls still produce the correct effective rate
        # instead of silently zeroing it (the production-trainer regression
        # Codex's Round 1 review identified). Round 2's resolver was only
        # wired into `model_factory.py`; Round 3 wires it through here so the
        # same fallback applies to the trainer-side reads as well.
        bias_update_rate=_resolve_balancing_field(cfg, "bias_update_rate", 0.0),
        bias_warmup_start=_resolve_balancing_field(cfg, "bias_warmup_start", 0.0),
        bias_warmup_steps=_resolve_balancing_field(cfg, "bias_warmup_steps", 0),
        router_exploration_warmup_start=tcfg.get("router_exploration_warmup_start", 0.0),
        router_exploration_warmup_steps=tcfg.get("router_exploration_warmup_steps", 0),
    )


def resolve_related_config_path(config_path: str, related_path: str) -> str:
    """Resolve a related config path relative to the given config file."""
    resolved = Path(related_path)
    if resolved.is_absolute():
        return str(resolved)
    return str((Path(config_path).resolve().parent / resolved).resolve())


def resolve_initialization_spec(
    cfg: dict,
    *,
    config_path: str,
    cli_source_config: str | None = None,
    cli_strategy: str | None = None,
) -> dict | None:
    """Resolve initialization specification from config and CLI overrides."""
    spec = dict(cfg.get("initialization") or {})
    if cli_source_config is not None:
        spec["source_config"] = cli_source_config
    if cli_strategy is not None:
        spec["strategy"] = cli_strategy
    if not spec:
        return None
    if "source_config" not in spec or "strategy" not in spec:
        raise ValueError("initialization requires both 'source_config' and 'strategy'")
    spec["source_config"] = resolve_related_config_path(config_path, spec["source_config"])
    return spec
