"""Run focused pytest targets on Modal GPU infrastructure.

Usage:
  modal run modal_pytest.py
  modal run modal_pytest.py --test-args "tests/test_branch_patterns.py -q"
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


moe_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.8.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "pytest>=8.0",
        "pyyaml>=6.0",
        "transformers>=5.0.0",
        "pyarrow>=15.0.0",
        "pandas>=2.0.0",
        "einops>=0.7.0",
        "tokenizers>=0.19.0",
        "pydantic>=2.0.0",
        "liger-kernel>=0.7.0",
    )
    .add_local_dir(str(moe_dir / "src"), remote_path="/root/moe/src", copy=True)
    .add_local_dir(str(moe_dir / "tests"), remote_path="/root/moe/tests", copy=True)
    .add_local_dir(str(moe_dir / "scripts"), remote_path="/root/moe/scripts", copy=True)
    .add_local_dir(str(moe_dir / "configs"), remote_path="/root/moe/configs", copy=True)
    .add_local_dir(str(moe_dir / "torchrun_util"), remote_path="/root/moe/torchrun_util", copy=True)
    .add_local_file(str(moe_dir / "modal_train.py"), remote_path="/root/moe/modal_train.py", copy=True)
)

app = modal.App("moe-pytest", image=image)


DEFAULT_TEST_ARGS = "tests/test_branch_patterns.py tests/test_routing_plots.py tests/test_moe_everything_fused_ce.py tests/test_causal_lm_next_token_loss.py tests/test_gpu_runtime_paths.py"


@app.function(gpu="H100", timeout=60 * 20, cpu=4, memory=16 * 1024)
def run_pytest(test_args: str = DEFAULT_TEST_ARGS, env_vars: str = "") -> int:
    workdir = "/root/moe"
    os.chdir(workdir)
    env = os.environ.copy()
    env["PYTHONPATH"] = workdir
    for assignment in shlex.split(env_vars):
        if "=" not in assignment:
            raise ValueError(f"Invalid env assignment {assignment!r}; expected KEY=VALUE")
        key, value = assignment.split("=", 1)
        env[key] = value
    cmd = ["python", "-m", "pytest", *shlex.split(test_args)]
    print(f">>> command: {' '.join(cmd)}", flush=True)
    if env_vars:
        print(f">>> env: {env_vars}", flush=True)
    proc = subprocess.run(cmd, cwd=workdir, env=env, text=True)
    print(f">>> pytest returncode: {proc.returncode}", flush=True)
    return int(proc.returncode)


@app.local_entrypoint()
def main(test_args: str = DEFAULT_TEST_ARGS, env_vars: str = "") -> None:
    rc = run_pytest.remote(test_args, env_vars)
    raise SystemExit(rc)
