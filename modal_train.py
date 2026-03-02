"""
Modal multi-node training launcher for MoE pretraining.

Features:
  1. Multi-node distributed training via torchrun + Accelerate
  2. Auto-resume from latest checkpoint on Modal Volume
  3. Fault tolerance via Modal retries + checkpoint resume

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

N_NODES = 2              # number of containers in the cluster
GPUS_PER_NODE = 8        # GPUs per container
GPU_TYPE = "B200"        # B200, H200, or H100
TIMEOUT_HOURS = 24       # max wall-clock time
MAX_CHECKPOINTS = 3      # checkpoints to keep on volume (0 = unlimited)

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
        "accelerate>=0.30.0",
        "transformers>=4.40.0",
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
    .add_local_file(str(moe_dir / "train.py"), remote_path="/root/moe/train.py")
    .add_local_file(str(moe_dir / ".env"), remote_path="/root/moe/.env")
    .add_local_dir(str(moe_dir / "src"), remote_path="/root/moe/src")
    .add_local_dir(str(moe_dir / "configs"), remote_path="/root/moe/configs")
    .add_local_python_source("torchrun_util")
)

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


# --------------------------------------------------------------------------- #
#  Training function                                                           #
# --------------------------------------------------------------------------- #

@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 60 * TIMEOUT_HOURS,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train(config: str = CONFIG_FILE):
    import yaml
    from torchrun_util import torchrun

    cluster_info = modal.experimental.get_cluster_info()

    # Read experiment name from config for checkpoint directory
    with open(f"/root/moe/{config}") as f:
        cfg = yaml.safe_load(f)
    experiment_name = cfg.get("experiment_name", "default")
    output_dir = f"/checkpoints/{experiment_name}"

    # NCCL config
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    print(f"[Node {cluster_info.rank}/{N_NODES}] Starting MoE training")
    print(f"  Config     : {config}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Output     : {output_dir}")
    print(f"  Data       : /data/parquet")
    print(f"  Master     : {cluster_info.container_ips[0]}")
    print(f"  GPUs       : {N_NODES} x {GPUS_PER_NODE} = {N_NODES * GPUS_PER_NODE}")

    torchrun.run(
        node_rank=cluster_info.rank,
        master_addr=cluster_info.container_ips[0],
        master_port=1234,
        nnodes=str(N_NODES),
        nproc_per_node=str(GPUS_PER_NODE),
        training_script="/root/moe/train.py",
        training_script_args=[
            "--config", f"/root/moe/{config}",
            "--auto_resume",
            "--data_dir", "/data/parquet",
            "--output_dir", output_dir,
            "--max_checkpoints", str(MAX_CHECKPOINTS),
        ],
    )


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
def main(config: str = CONFIG_FILE):
    train.remote(config=config)
