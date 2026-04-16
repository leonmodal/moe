"""AC-9 unit test: pin the `modal_train.py` torchrun argv shape.

`modal_train.train()` is a Modal decorator stack and cannot be invoked in a
plain pytest process. We instead expose the argv assembly as a pure helper
(`build_train_script_args`) and pin its output here so a future refactor
cannot silently drop one of the required flags (`--config`, `--auto_resume`,
`--data_dir`, `--output_dir`, `--max_checkpoints`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

modal_train = pytest.importorskip("modal_train")


def test_build_train_script_args_pins_required_flags():
    args = modal_train.build_train_script_args(
        "/root/moe/configs/scaling/m_standard.yaml",
        data_dir="/data/parquet",
        output_dir="/checkpoints/m_standard",
        max_checkpoints=3,
    )
    assert args == [
        "--config", "/root/moe/configs/scaling/m_standard.yaml",
        "--auto_resume",
        "--data_dir", "/data/parquet",
        "--output_dir", "/checkpoints/m_standard",
        "--max_checkpoints", "3",
    ]


def test_build_train_script_args_can_disable_auto_resume():
    args = modal_train.build_train_script_args(
        "/cfg.yaml", data_dir="/data", output_dir="/ckpt",
        max_checkpoints=0, auto_resume=False,
    )
    assert "--auto_resume" not in args
    assert args[:2] == ["--config", "/cfg.yaml"]
    assert args[-6:] == [
        "--data_dir", "/data",
        "--output_dir", "/ckpt",
        "--max_checkpoints", "0",
    ]


def test_resolve_output_dir_uses_experiment_name():
    assert modal_train.resolve_output_dir({"experiment_name": "my_exp"}) == "/checkpoints/my_exp"


def test_resolve_output_dir_defaults_when_experiment_name_missing():
    assert modal_train.resolve_output_dir({}) == "/checkpoints/default"


def test_resolve_output_dir_honors_custom_root():
    assert modal_train.resolve_output_dir(
        {"experiment_name": "x"}, checkpoint_root="/tmp/ckpts"
    ) == "/tmp/ckpts/x"


def test_training_script_constant_points_at_unified_entrypoint():
    # Pin the constant so a rename of scripts/train.py surfaces here.
    assert modal_train.TRAINING_SCRIPT == "/root/moe/scripts/train.py"


def _write_minimal_modal_config(tmp_path) -> str:
    """Write a YAML config of the shape `modal_train.train()` would read."""
    path = tmp_path / "modal_cfg.yaml"
    path.write_text(
        "experiment_name: smoke_standard_moe\n"
        "model:\n"
        "  type: standard_moe\n"
        "  vocab_size: 256\n"
        "  hidden_size: 64\n"
        "  num_hidden_layers: 2\n"
        "  head_dim: 16\n"
        "  num_attention_heads: 4\n"
        "  num_key_value_heads: 2\n"
        "  num_experts: 4\n"
        "  num_experts_per_tok: 2\n"
        "  moe_intermediate_size: 32\n"
        "  intermediate_size: 128\n"
        "  max_position_embeddings: 128\n"
        "training:\n"
        "  learning_rate: 1.0e-3\n"
        "  weight_decay: 0.0\n"
        "  max_grad_norm: 1.0\n"
        "  lr_scheduler: constant\n"
        "  warmup_steps: 0\n"
        "  max_steps: 2\n"
        "  batch_size: 1\n"
        "  gradient_accumulation: 1\n"
        "  mixed_precision: bf16\n"
        "  output_dir: /checkpoints/smoke_standard_moe\n"
        "data:\n"
        "  data_dir: /data/parquet\n"
        "  text_column: text\n"
        "  seq_len: 32\n"
        "  tokenizer_name: gpt2\n"
        "eval:\n"
        "  enabled: false\n"
        "checkpoint:\n"
        "  resume_from: null\n"
    )
    return str(path)


def test_build_torchrun_invocation_pins_full_kwargs(tmp_path):
    """`modal_train.train()` calls `build_torchrun_invocation` then passes the
    result verbatim to `torchrun_util.torchrun.run(**invocation)`. This test
    drives that helper end-to-end on a real (tiny) config file so the Modal
    launcher path is covered beyond just argv shape.
    """
    config_path = _write_minimal_modal_config(tmp_path)

    invocation = modal_train.build_torchrun_invocation(
        config_path,
        node_rank=0,
        master_addr="10.0.0.1",
        nnodes=2,
        nproc_per_node=8,
        max_checkpoints=3,
    )

    assert invocation["node_rank"] == 0
    assert invocation["master_addr"] == "10.0.0.1"
    assert invocation["master_port"] == 1234
    assert invocation["nnodes"] == "2"  # torchrun expects str
    assert invocation["nproc_per_node"] == "8"
    assert invocation["training_script"] == modal_train.TRAINING_SCRIPT

    args = invocation["training_script_args"]
    assert args == [
        "--config", config_path,
        "--auto_resume",
        "--data_dir", "/data/parquet",
        "--output_dir", "/checkpoints/smoke_standard_moe",
        "--max_checkpoints", "3",
    ]


def test_build_torchrun_invocation_uses_experiment_name_from_config(tmp_path):
    # Rebuild the config with a different experiment_name and verify the
    # output_dir is resolved from it.
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "experiment_name: xyz\n"
        "model: {type: dense, vocab_size: 128, hidden_size: 32, num_hidden_layers: 1, "
        "head_dim: 8, num_attention_heads: 2, num_key_value_heads: 1, "
        "max_position_embeddings: 64}\n"
        "training: {learning_rate: 1.0e-3, weight_decay: 0.0, max_grad_norm: 1.0, "
        "lr_scheduler: constant, warmup_steps: 0, max_steps: 1, batch_size: 1, "
        "gradient_accumulation: 1, mixed_precision: bf16, output_dir: /tmp/out}\n"
        "data: {data_dir: /data, text_column: text, seq_len: 16, tokenizer_name: gpt2}\n"
        "eval: {enabled: false}\n"
        "checkpoint: {}\n"
    )
    invocation = modal_train.build_torchrun_invocation(
        str(path), node_rank=0, master_addr="127.0.0.1",
        nnodes=1, nproc_per_node=1, max_checkpoints=0,
    )
    assert "--output_dir" in invocation["training_script_args"]
    idx = invocation["training_script_args"].index("--output_dir") + 1
    assert invocation["training_script_args"][idx] == "/checkpoints/xyz"


def test_build_torchrun_invocation_defaults_experiment_name_when_missing(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model: {type: dense, vocab_size: 128, hidden_size: 32, num_hidden_layers: 1, "
        "head_dim: 8, num_attention_heads: 2, num_key_value_heads: 1, "
        "max_position_embeddings: 64}\n"
        "training: {learning_rate: 1.0e-3, weight_decay: 0.0, max_grad_norm: 1.0, "
        "lr_scheduler: constant, warmup_steps: 0, max_steps: 1, batch_size: 1, "
        "gradient_accumulation: 1, mixed_precision: bf16, output_dir: /tmp/out}\n"
        "data: {data_dir: /data, text_column: text, seq_len: 16, tokenizer_name: gpt2}\n"
    )
    invocation = modal_train.build_torchrun_invocation(
        str(path), node_rank=1, master_addr="10.0.0.2",
        nnodes=4, nproc_per_node=8, max_checkpoints=5,
    )
    idx = invocation["training_script_args"].index("--output_dir") + 1
    assert invocation["training_script_args"][idx] == "/checkpoints/default"


def test_modal_train_raw_function_forwards_expected_torchrun_kwargs(monkeypatch, tmp_path):
    """Execute `modal_train.train`'s real body (not just scan its source).

    The Modal decorators (`@app.function`, `@modal.experimental.clustered`)
    wrap `train` into a `modal.Function`; `train.get_raw_f()` returns the
    underlying Python callable. We patch the three external dependencies
    (`modal.experimental.get_cluster_info` for cluster coordinates, the
    `open` the function uses to read the YAML config, and
    `torchrun_util.torchrun.run` to capture the forwarded kwargs), invoke
    the raw function, and assert the exact command shape passed to
    `torchrun.run(**invocation)`.
    """
    # Build a real YAML config on disk — the raw function opens its config
    # via f"/root/moe/{config}", so we stage the file there via monkeypatched
    # open rather than actually touching /root/moe.
    yaml_payload = (
        "experiment_name: raw_launcher_smoke\n"
        "model:\n"
        "  type: dense\n"
        "  vocab_size: 128\n"
        "  hidden_size: 32\n"
        "  num_hidden_layers: 1\n"
        "  head_dim: 8\n"
        "  num_attention_heads: 2\n"
        "  num_key_value_heads: 1\n"
        "  intermediate_size: 64\n"
        "  max_position_embeddings: 64\n"
        "training:\n"
        "  learning_rate: 1.0e-3\n"
        "  weight_decay: 0.0\n"
        "  max_grad_norm: 1.0\n"
        "  lr_scheduler: constant\n"
        "  warmup_steps: 0\n"
        "  max_steps: 2\n"
        "  batch_size: 1\n"
        "  gradient_accumulation: 1\n"
        "  mixed_precision: bf16\n"
        "  output_dir: /checkpoints/raw_launcher_smoke\n"
        "data:\n"
        "  data_dir: /data/parquet\n"
        "  text_column: text\n"
        "  seq_len: 16\n"
        "  tokenizer_name: gpt2\n"
        "eval:\n"
        "  enabled: false\n"
    )

    expected_config_path = f"/root/moe/configs/{tmp_path.name}/raw.yaml"
    real_open = open

    def patched_open(path, *args, **kwargs):
        if str(path) == expected_config_path:
            from io import StringIO
            return StringIO(yaml_payload)
        return real_open(path, *args, **kwargs)

    class _FakeClusterInfo:
        def __init__(self):
            self.rank = 0
            self.container_ips = ["10.0.0.7"]

    captured_kwargs: dict = {}

    def _fake_torchrun_run(**kwargs):
        captured_kwargs.update(kwargs)

    # modal.experimental.get_cluster_info is looked up dynamically inside
    # train() via `modal.experimental.get_cluster_info()`, so we patch the
    # attribute on the module modal_train already has imported.
    import modal.experimental as modal_experimental  # noqa: E402
    monkeypatch.setattr(modal_experimental, "get_cluster_info",
                        lambda: _FakeClusterInfo())

    # torchrun_util.torchrun.run is imported lazily inside train(). Pre-inject
    # a stub into sys.modules so the inner `from torchrun_util import torchrun`
    # binds to our fake.
    import sys
    import types
    fake_torchrun_util = types.ModuleType("torchrun_util")
    fake_torchrun = types.SimpleNamespace(run=_fake_torchrun_run)
    fake_torchrun_util.torchrun = fake_torchrun
    monkeypatch.setitem(sys.modules, "torchrun_util", fake_torchrun_util)

    # The raw function opens f"/root/moe/{config}" as a string; intercept
    # builtins.open so we can return our in-memory YAML without touching disk.
    import builtins
    monkeypatch.setattr(builtins, "open", patched_open)

    raw_train = modal_train.train.get_raw_f()
    relative_config = expected_config_path.replace("/root/moe/", "")
    raw_train(config=relative_config)

    # Capture must contain exactly the kwargs build_torchrun_invocation produces.
    assert set(captured_kwargs.keys()) == {
        "node_rank", "master_addr", "master_port",
        "nnodes", "nproc_per_node",
        "training_script", "training_script_args",
    }, f"Unexpected kwargs forwarded to torchrun.run: {captured_kwargs.keys()}"
    assert captured_kwargs["node_rank"] == 0
    assert captured_kwargs["master_addr"] == "10.0.0.7"
    assert captured_kwargs["master_port"] == 1234
    # Modal-side N_NODES / GPUS_PER_NODE come from module globals; pin them.
    assert captured_kwargs["nnodes"] == str(modal_train.N_NODES)
    assert captured_kwargs["nproc_per_node"] == str(modal_train.GPUS_PER_NODE)
    assert captured_kwargs["training_script"] == modal_train.TRAINING_SCRIPT
    assert captured_kwargs["training_script_args"] == [
        "--config", expected_config_path,
        "--auto_resume",
        "--data_dir", modal_train.REMOTE_DATA_DIR,
        "--output_dir", "/checkpoints/raw_launcher_smoke",
        "--max_checkpoints", str(modal_train.MAX_CHECKPOINTS),
    ]
