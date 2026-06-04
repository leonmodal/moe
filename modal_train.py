"""
Modal multi-node training launcher for MoE pretraining.

Features:
  1. Multi-node distributed training via torchrun
  2. Auto-resume from latest checkpoint on Modal Volume
  3. Fault tolerance via checkpoint resume
  4. Unified trainer for all supported models (dense, standard_moe, global_moe, moe_everything)

Configuration:
  Edit N_NODES, GPUS_PER_NODE, GPU_TYPE at the top of this file.

Usage:
  # Download data to Modal volume (first time only)
  modal run modal_train.py::download_data --max-shards 64

  # Launch training (uses CONFIG_FILE by default)
  modal run modal_train.py

  # Launch with a specific config
  modal run modal_train.py --config configs/scaling/m_standard.yaml
"""
import os
from pathlib import Path

import modal
import modal.experimental

# --------------------------------------------------------------------------- #
#  Cluster configuration — edit these before launching                         #
# --------------------------------------------------------------------------- #

N_NODES = int(os.environ.get("MOE_MODAL_N_NODES", "1"))              # number of containers in the cluster
GPUS_PER_NODE = int(os.environ.get("MOE_MODAL_GPUS_PER_NODE", "8"))  # GPUs per container
GPU_TYPE = os.environ.get("MOE_MODAL_GPU_TYPE", "B200")              # B200, H200, or H100
EFA_ENABLED = os.environ.get("MOE_MODAL_EFA_ENABLED", "1").lower() not in {
    "0", "false", "no", "off",
}
MODAL_CLOUD = os.environ.get("MOE_MODAL_CLOUD") or None
TIMEOUT_HOURS = 24       # max wall-clock time
MAX_CHECKPOINTS = 3      # checkpoints to keep on volume (0 = unlimited)
DIST_STRATEGY = os.environ.get("MOE_MODAL_DIST_STRATEGY", "ddp")  # fsdp | ddp | none
FSDP_SHARDING_STRATEGY = os.environ.get(
    "MOE_MODAL_FSDP_SHARDING_STRATEGY", "auto"
)  # auto | no_shard | hybrid_shard | full_shard | shard_grad_op
NCCL_DEBUG_LEVEL = os.environ.get("MOE_MODAL_NCCL_DEBUG", "WARN")
NCCL_DEBUG_SUBSYS = os.environ.get("MOE_MODAL_NCCL_DEBUG_SUBSYS")

CONFIG_FILE = "configs/scaling/xs_standard.yaml"  # default training config

# --------------------------------------------------------------------------- #
#  Modal image                                                                 #
# --------------------------------------------------------------------------- #

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

moe_dir = Path(__file__).parent

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "curl", "git", "vim", "htop",
        # Required for NCCL to use RDMA/InfiniBand instead of TCP sockets
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc15",
        "libnl-route-3-200",
    )
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
    .run_commands(
        'python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
    )
    .add_local_dir(str(moe_dir / "scripts"), remote_path="/root/moe/scripts")
    .add_local_dir(str(moe_dir / "src"), remote_path="/root/moe/src")
    .add_local_dir(str(moe_dir / "configs"), remote_path="/root/moe/configs")
    .add_local_python_source("torchrun_util")
)

env_file = moe_dir / ".env"
if env_file.exists():
    image = image.add_local_file(str(env_file), remote_path="/root/moe/.env")

# --------------------------------------------------------------------------- #
#  Modal resources                                                             #
# --------------------------------------------------------------------------- #

data_volume = modal.Volume.from_name("moe-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("moe-checkpoints", create_if_missing=True)

app = modal.App(
    "moe-training",
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


TRAINING_SCRIPT = "/root/moe/scripts/train.py"
REMOTE_DATA_DIR = "/data/parquet"
CHECKPOINT_ROOT = "/checkpoints"


def build_train_script_args(
    config_path: str,
    *,
    data_dir: str = REMOTE_DATA_DIR,
    output_dir: str,
    max_checkpoints: int,
    auto_resume: bool = True,
    dist_strategy: str = DIST_STRATEGY,
    fsdp_sharding_strategy: str | None = FSDP_SHARDING_STRATEGY,
    max_steps: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    save_every: int | None = None,
    disable_wandb: bool = False,
) -> list[str]:
    """Assemble the `scripts/train.py` argv for the Modal launcher.

    Kept as a pure function so tests can pin the exact command shape without
    importing the Modal decorators. Mirrors what `modal_train.train()` passes
    to `torchrun_util.torchrun.run(..., training_script_args=...)`.
    """
    args = ["--config", config_path]
    if auto_resume:
        args.append("--auto_resume")
    args += ["--dist-strategy", dist_strategy]
    if dist_strategy == "fsdp" and fsdp_sharding_strategy:
        args += ["--fsdp-sharding-strategy", fsdp_sharding_strategy]
    if max_steps is not None:
        args += ["--max-steps", str(max_steps)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if gradient_accumulation is not None:
        args += ["--gradient-accumulation", str(gradient_accumulation)]
    if save_every is not None:
        args += ["--save-every", str(save_every)]
    if disable_wandb:
        args.append("--disable-wandb")
    args += [
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--max_checkpoints", str(max_checkpoints),
    ]
    return args


def resolve_output_dir(cfg: dict, checkpoint_root: str = CHECKPOINT_ROOT) -> str:
    """Derive the checkpoint output directory from the parsed YAML config."""
    experiment_name = cfg.get("experiment_name", "default")
    return f"{checkpoint_root}/{experiment_name}"


def build_torchrun_invocation(
    config_path: str,
    *,
    node_rank: int,
    master_addr: str,
    nnodes: int,
    nproc_per_node: int,
    max_checkpoints: int,
    master_port: int = 1234,
    data_dir: str = REMOTE_DATA_DIR,
    checkpoint_root: str = CHECKPOINT_ROOT,
    training_script: str = TRAINING_SCRIPT,
    auto_resume: bool = True,
    dist_strategy: str = DIST_STRATEGY,
    fsdp_sharding_strategy: str | None = FSDP_SHARDING_STRATEGY,
    max_steps: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    save_every: int | None = None,
    disable_wandb: bool = False,
    output_suffix: str | None = None,
) -> dict:
    """Assemble the full kwargs passed to `torchrun_util.torchrun.run(...)`.

    Reads the YAML config at `config_path` to derive `output_dir`, then
    wires together the training script, script args, and cluster coordinates.
    Pure function: same inputs → same kwargs. `modal_train.train()` calls
    this internally so tests can exercise the Modal launcher's full command
    assembly without needing to import Modal decorators.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = resolve_output_dir(cfg, checkpoint_root)
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"
    script_args = build_train_script_args(
        config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        max_checkpoints=max_checkpoints,
        auto_resume=auto_resume,
        dist_strategy=dist_strategy,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        save_every=save_every,
        disable_wandb=disable_wandb,
    )
    return {
        "node_rank": node_rank,
        "master_addr": master_addr,
        "master_port": master_port,
        "nnodes": str(nnodes),
        "nproc_per_node": str(nproc_per_node),
        "training_script": training_script,
        "training_script_args": script_args,
    }


# --------------------------------------------------------------------------- #
#  Training function                                                           #
# --------------------------------------------------------------------------- #

@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    experimental_options={"efa_enabled": EFA_ENABLED},
    cloud=MODAL_CLOUD,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train(
    config: str = CONFIG_FILE,
    max_steps: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    save_every: int | None = None,
    disable_wandb: bool = False,
    output_suffix: str | None = None,
    dist_strategy: str | None = None,
    fsdp_sharding_strategy: str | None = None,
    disable_grouped_mm: bool = False,
    disable_attn_grouped_mm: bool = False,
    disable_mlp_grouped_mm: bool = False,
    force_eager_attention: bool = False,
):
    import torch
    import yaml
    from torchrun_util import torchrun

    cluster_info = modal.experimental.get_cluster_info()
    cluster_n_nodes = max(1, len(cluster_info.container_ips))
    visible_gpus = max(1, torch.cuda.device_count())

    remote_config_path = f"/root/moe/{config}"
    with open(remote_config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = resolve_output_dir(cfg)
    display_output_dir = f"{output_dir}_{output_suffix}" if output_suffix else output_dir
    experiment_name = cfg.get("experiment_name", "default")
    effective_dist_strategy = dist_strategy or DIST_STRATEGY
    effective_fsdp_sharding_strategy = fsdp_sharding_strategy or FSDP_SHARDING_STRATEGY

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_DEBUG"] = NCCL_DEBUG_LEVEL
    if NCCL_DEBUG_SUBSYS:
        os.environ["NCCL_DEBUG_SUBSYS"] = NCCL_DEBUG_SUBSYS
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["MOE_EVERYTHING_DISABLE_GROUPED_MM"] = "1" if disable_grouped_mm else "0"
    os.environ["MOE_EVERYTHING_DISABLE_ATTN_GROUPED_MM"] = "1" if disable_attn_grouped_mm else "0"
    os.environ["MOE_EVERYTHING_DISABLE_MLP_GROUPED_MM"] = "1" if disable_mlp_grouped_mm else "0"
    if force_eager_attention:
        os.environ["MOE_EVERYTHING_FORCE_ATTN_IMPL"] = "eager"

    print(f"[Node {cluster_info.rank}/{cluster_n_nodes}] Starting MoE training")
    print(f"  Config     : {config}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Output     : {display_output_dir}")
    print(f"  Data       : {REMOTE_DATA_DIR}")
    print(f"  Master     : {cluster_info.container_ips[0]}")
    print(f"  GPUs       : {cluster_n_nodes} x {visible_gpus} = {cluster_n_nodes * visible_gpus}")
    print(f"  RDMA       : clustered=True efa_enabled={EFA_ENABLED} cloud={MODAL_CLOUD or 'default'}")
    if effective_dist_strategy == "fsdp":
        print(f"  Dist       : {effective_dist_strategy} ({effective_fsdp_sharding_strategy})")
    else:
        print(f"  Dist       : {effective_dist_strategy}")
    if batch_size is not None or gradient_accumulation is not None:
        print(f"  Batch      : batch_size={batch_size} grad_accum={gradient_accumulation}")
    if save_every is not None:
        print(f"  Save every : {save_every} steps")
    if disable_wandb:
        print("  WandB      : disabled", flush=True)
    if disable_grouped_mm or disable_attn_grouped_mm or disable_mlp_grouped_mm or force_eager_attention:
        print(
            "  Debug      : "
            f"disable_grouped_mm={disable_grouped_mm} "
            f"disable_attn_grouped_mm={disable_attn_grouped_mm} "
            f"disable_mlp_grouped_mm={disable_mlp_grouped_mm} "
            f"force_eager_attention={force_eager_attention}",
            flush=True,
        )

    invocation = build_torchrun_invocation(
        remote_config_path,
        node_rank=cluster_info.rank,
        master_addr=cluster_info.container_ips[0],
        nnodes=cluster_n_nodes,
        nproc_per_node=visible_gpus,
        max_checkpoints=MAX_CHECKPOINTS,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        save_every=save_every,
        disable_wandb=disable_wandb,
        output_suffix=output_suffix,
        dist_strategy=effective_dist_strategy,
        fsdp_sharding_strategy=effective_fsdp_sharding_strategy,
    )
    torchrun.run(**invocation)


# --------------------------------------------------------------------------- #
#  Data download helper                                                        #
# --------------------------------------------------------------------------- #

@app.function(
    timeout=60 * 60 * 4,  # 4 hours for large downloads
    volumes={"/data": data_volume},
)
def download_data(max_shards: int = None, workers: int = 16):
    """Download training data parquet shards to the Modal volume."""
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm

    repo_id = "leonli66/latent-cot-finewebedu"
    out_dir = Path("/data/parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set. Add huggingface-secret to Modal.")

    api = HfApi(token=token)
    all_files = sorted(
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.endswith(".parquet")
    )

    if max_shards is not None:
        all_files = all_files[:max_shards]

    already = sum(1 for f in all_files if (out_dir / f).exists())
    todo = [f for f in all_files if not (out_dir / f).exists()]
    print(f"Shards: {len(all_files)} total, {already} present, {len(todo)} to fetch")

    if not todo:
        print("All shards already present.")
        return

    def dl(filename):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(out_dir),
        )
        return filename

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(dl, f): f for f in todo}
        with tqdm(total=len(todo), unit="shard") as bar:
            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((fname, str(e)))
                    tqdm.write(f"ERROR {fname}: {e}")
                bar.update(1)

    data_volume.commit()
    print(f"Done. {len(todo) - len(errors)} downloaded, {len(errors)} failed.")
    if errors:
        for f, e in errors:
            print(f"  FAILED: {f}: {e}")


# --------------------------------------------------------------------------- #
#  Local entrypoint                                                            #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(
    config: str = CONFIG_FILE,
    background: bool = False,
    max_steps: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    save_every: int | None = None,
    disable_wandb: bool = False,
    output_suffix: str | None = None,
    dist_strategy: str | None = None,
    fsdp_sharding_strategy: str | None = None,
    disable_grouped_mm: bool = False,
    disable_attn_grouped_mm: bool = False,
    disable_mlp_grouped_mm: bool = False,
    force_eager_attention: bool = False,
):
    if background:
        call = train.spawn(
            config=config,
            max_steps=max_steps,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            save_every=save_every,
            disable_wandb=disable_wandb,
            output_suffix=output_suffix,
            dist_strategy=dist_strategy,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            disable_grouped_mm=disable_grouped_mm,
            disable_attn_grouped_mm=disable_attn_grouped_mm,
            disable_mlp_grouped_mm=disable_mlp_grouped_mm,
            force_eager_attention=force_eager_attention,
        )
        print(
            f"Spawned training call for {config} "
            f"(max_steps={max_steps}, batch_size={batch_size}, "
            f"gradient_accumulation={gradient_accumulation}, "
            f"save_every={save_every}, "
            f"disable_wandb={disable_wandb}, output_suffix={output_suffix}, "
            f"dist_strategy={dist_strategy}, "
            f"fsdp_sharding_strategy={fsdp_sharding_strategy}, "
            f"disable_grouped_mm={disable_grouped_mm}, "
            f"disable_attn_grouped_mm={disable_attn_grouped_mm}, "
            f"disable_mlp_grouped_mm={disable_mlp_grouped_mm}, "
            f"force_eager_attention={force_eager_attention}): {call}"
        )
        return
    train.remote(
        config=config,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        save_every=save_every,
        disable_wandb=disable_wandb,
        output_suffix=output_suffix,
        dist_strategy=dist_strategy,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
        disable_grouped_mm=disable_grouped_mm,
        disable_attn_grouped_mm=disable_attn_grouped_mm,
        disable_mlp_grouped_mm=disable_mlp_grouped_mm,
        force_eager_attention=force_eager_attention,
    )
