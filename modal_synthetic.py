"""Modal launcher for the synthetic-task sweep (Zhao et al. NeurIPS 2026 §3.1).

Six configs in `configs/synthetic/`:
  * 1L linear-map × {top1_explore_decay, fixed_alternating, sampling_entropy}
  * 4L cellular-automata × {top1_explore_decay, fixed_alternating, sampling_entropy}

Each config is ≤100M params on tiny sequences (seq_len 32 or 256), so we run
single-GPU per job on H100 (override with MOE_SYNTHETIC_GPU_TYPE). The
default `main` entrypoint spawns all 6 in parallel; pass `--config <yaml>`
to launch one.

Self-contained: image, volumes, and app are defined here (no import from
modal_train.py, since the container image only mounts the explicit dirs).
The checkpoint volume `moe-checkpoints` is shared with the prod sweep.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

GPU_TYPE = os.environ.get("MOE_SYNTHETIC_GPU_TYPE", "H100")  # H100 | A10G | L4 | L40S | A100
TIMEOUT_HOURS = int(os.environ.get("MOE_SYNTHETIC_TIMEOUT_HOURS", "3"))

# --------------------------------------------------------------------------- #
#  Image (mirrors modal_train.py but standalone)                              #
# --------------------------------------------------------------------------- #

moe_dir = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "git", "vim", "htop")
    .pip_install(
        "torch>=2.8.0",
        "safetensors>=0.4.0",
        "transformers>=5.0.0",
        "datasets>=2.19.0",
        "pyarrow>=15.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "wandb>=0.17.0",
        "einops>=0.7.0",
        "tokenizers>=0.19.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "liger-kernel>=0.7.0",
        "pydantic>=2.0.0",
        "huggingface-hub>=0.20.0",
        "matplotlib>=3.8.0",
    )
    .add_local_dir(str(moe_dir / "scripts"), remote_path="/root/moe/scripts")
    .add_local_dir(str(moe_dir / "src"), remote_path="/root/moe/src")
    .add_local_dir(str(moe_dir / "configs"), remote_path="/root/moe/configs")
)

env_file = moe_dir / ".env"
if env_file.exists():
    image = image.add_local_file(str(env_file), remote_path="/root/moe/.env")

# --------------------------------------------------------------------------- #
#  App                                                                         #
# --------------------------------------------------------------------------- #

data_volume = modal.Volume.from_name("moe-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("moe-checkpoints", create_if_missing=True)

app = modal.App(
    "moe-synthetic",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/data": data_volume,
        "/checkpoints": ckpt_volume,
    },
)

CHECKPOINT_ROOT = "/checkpoints"

ALL_CONFIGS = [
    "configs/synthetic/1L_linearmap_branch_top1_explore_decay.yaml",
    "configs/synthetic/1L_linearmap_branch_fixed_alternating.yaml",
    "configs/synthetic/1L_linearmap_branch_sampling_entropy.yaml",
    "configs/synthetic/4L_cellular_branch_top1_explore_decay.yaml",
    "configs/synthetic/4L_cellular_branch_fixed_alternating.yaml",
    "configs/synthetic/4L_cellular_branch_sampling_entropy.yaml",
]


def _output_dir_for(config: str) -> str:
    name = Path(config).stem
    return f"{CHECKPOINT_ROOT}/synthetic/{name}"


@app.function(
    gpu=GPU_TYPE,
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(max_retries=2),
)
def train_synthetic(config: str) -> str:
    """Single-GPU training for one synthetic config."""
    import subprocess
    import yaml

    remote_config = f"/root/moe/{config}"
    with open(remote_config) as f:
        cfg = yaml.safe_load(f)
    experiment_name = cfg.get("experiment_name", Path(config).stem)
    output_dir = _output_dir_for(config)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        "python",
        "/root/moe/scripts/train.py",
        "--config", remote_config,
        "--auto_resume",
        "--dist-strategy", "none",
        "--output_dir", output_dir,
        "--max_checkpoints", "5",
    ]
    print(f"[{experiment_name}] GPU={GPU_TYPE} out={output_dir}", flush=True)
    print(f"[{experiment_name}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    ckpt_volume.commit()
    return experiment_name


@app.function(
    gpu=GPU_TYPE,
    timeout=60 * 30,
    retries=modal.Retries(max_retries=1),
)
def regen_heatmaps(config: str, step: int | None = None) -> str:
    """Re-run attention eval against a saved checkpoint and write heatmap PNGs.

    Used to produce attention-pattern visualizations for dense / standard_moe
    runs whose attention layers don't have the moe_everything `capture_attention_maps`
    hook. Picks the latest checkpoint under the experiment's output dir
    unless `step` is given.
    """
    import subprocess

    remote_config = f"/root/moe/{config}"
    name = Path(config).stem
    exp_dir = f"{CHECKPOINT_ROOT}/synthetic/{name}"
    ckpt_root = Path(exp_dir)
    if not ckpt_root.exists():
        raise RuntimeError(f"experiment dir not found: {ckpt_root}")
    ckpts = sorted(ckpt_root.glob("checkpoint-*"))
    if not ckpts:
        raise RuntimeError(f"no checkpoints under {ckpt_root}")
    if step is not None:
        ckpt = ckpt_root / f"checkpoint-{step}"
        if not ckpt.exists():
            raise RuntimeError(f"requested step {step} not present (have: {[c.name for c in ckpts]})")
    else:
        # Last checkpoint by step number.
        ckpt = max(ckpts, key=lambda p: int(p.name.split("-")[-1]))
    step_n = int(ckpt.name.split("-")[-1])
    out_dir = ckpt_root / "attn_eval" / f"step_{step_n:08d}"
    print(f"[{name}] regenerating heatmaps from {ckpt} -> {out_dir}", flush=True)
    cmd = [
        "python",
        "/root/moe/scripts/extract_heatmaps_via_hooks.py",
        "--config",
        remote_config,
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(out_dir),
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    ckpt_volume.commit()
    return name


@app.local_entrypoint()
def regen(config: str):
    """Spawn a regen_heatmaps job for one config."""
    call = regen_heatmaps.spawn(config=config)
    print(f"Spawned regen for {config}: {call.object_id}")


@app.local_entrypoint()
def main(
    config: str | None = None,
    background: bool = True,
):
    """Launch one or all synthetic configs.

    Default: spawn all 6 in parallel. Pass `--config <yaml>` to launch a
    single config. Use with `modal run --detach` so the spawned function
    calls survive the local entrypoint exit.
    """
    if config is not None:
        if background:
            call = train_synthetic.spawn(config=config)
            print(f"Spawned {config}: {call.object_id}")
        else:
            train_synthetic.remote(config=config)
        return

    print(f"Launching {len(ALL_CONFIGS)} synthetic configs on {GPU_TYPE} (parallel)...")
    for cfg in ALL_CONFIGS:
        call = train_synthetic.spawn(config=cfg)
        print(f"  spawned {cfg}: {call.object_id}")
    print()
    print("Track them in the Modal dashboard. W&B project: moe-synthetic")
