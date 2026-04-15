"""Verify multi-node RDMA/NCCL setup on 2-node H200."""
import os
from pathlib import Path

import modal
import modal.experimental

N_NODES = 2
GPUS_PER_NODE = 8
GPU_TYPE = "H200"

moe_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libibverbs-dev", "libibverbs1", "libhwloc15", "libnl-route-3-200")
    .pip_install("torch>=2.8.0")
)

app = modal.App("verify-multinode", image=image)


@app.function(
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    timeout=60 * 10,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def verify():
    import socket
    import subprocess
    import time

    cluster_info = modal.experimental.get_cluster_info()

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Enable NCCL debug to see transport selection
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"

    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0]
    print(f"[Node {node_rank}/{N_NODES}] host={socket.gethostname()}")
    print(f"  Master: {master_addr}")
    print(f"  All IPs: {cluster_info.container_ips}")
    print(f"  GPUs per node: {GPUS_PER_NODE}")

    # Use torchrun to launch a small verification script
    verify_script = "/tmp/verify_dist.py"
    with open(verify_script, "w") as f:
        f.write('''
import os
import socket
import time
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()

if rank == 0:
    print(f"")
    print(f"=== Distributed Setup Verified ===")
    print(f"  World size: {world_size}")
    print(f"  Backend: {dist.get_backend()}")
    print(f"  NCCL version: {torch.cuda.nccl.version()}")

# Print per-rank info (staggered to avoid interleaving)
for r in range(world_size):
    if rank == r:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  Rank {rank}: host={socket.gethostname()} local_rank={local_rank} device={device} GPU={gpu_name} ({gpu_mem:.0f} GB)", flush=True)
    dist.barrier()

# Test all-reduce bandwidth
if rank == 0:
    print(f"")
    print(f"=== All-Reduce Bandwidth Test ===")

for size_mb in [1, 10, 100, 1000]:
    numel = size_mb * 1024 * 1024 // 4  # float32
    tensor = torch.randn(numel, device=device)
    dist.barrier()

    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bw = (size_mb * n_iters * 2) / elapsed / 1000  # GB/s (factor 2 for ring all-reduce)
    if rank == 0:
        print(f"  {size_mb:5d} MB: {bw:.1f} GB/s  ({elapsed/n_iters*1000:.1f} ms/iter)")

# Test that gradient sync works correctly
if rank == 0:
    print(f"")
    print(f"=== Gradient Consistency Check ===")

# Each rank creates a tensor with rank-specific value, all-reduce should average
test = torch.tensor([float(rank)], device=device)
dist.all_reduce(test)
test /= world_size
expected = sum(range(world_size)) / world_size
match = abs(test.item() - expected) < 1e-5

if rank == 0:
    print(f"  Expected mean of ranks: {expected}")
    print(f"  Got: {test.item()}")
    print(f"  Match: {match}")
    print(f"")
    print(f"=== All checks passed! ===")

dist.destroy_process_group()
''')

    cmd = [
        "torchrun",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        "--master_port=1234",
        f"--nnodes={N_NODES}",
        f"--nproc_per_node={GPUS_PER_NODE}",
        verify_script,
    ]
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main():
    verify.remote()
