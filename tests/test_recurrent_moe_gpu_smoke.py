from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import yaml

from src.training.config import load_config
from src.training.model_factory import build_model


DEFAULT_CONFIGS = [
    "configs/16_layers/recurrent_standard_moe_4_8_4_deepseek_bias.yaml",
    "configs/16_layers/recurrent_standard_moe_4_8_4_core_no_attention_deepseek_bias.yaml",
    "configs/16_layers/recurrent_global_moe_4_8_4_deepseek_bias.yaml",
    "configs/16_layers/recurrent_global_moe_4_8_4_core_no_attention_deepseek_bias.yaml",
]
CONFIGS = [
    item.strip()
    for item in os.environ.get(
        "MOE_RECURRENT_SMOKE_CONFIGS",
        ",".join(DEFAULT_CONFIGS),
    ).split(",")
    if item.strip()
]


def _smoke_shape() -> tuple[int, int, int, int]:
    return (
        int(os.environ.get("MOE_RECURRENT_SMOKE_BATCH", "4")),
        int(os.environ.get("MOE_RECURRENT_SMOKE_SEQ_LEN", "512")),
        int(os.environ.get("MOE_RECURRENT_SMOKE_NO_GRAD", "1")),
        int(os.environ.get("MOE_RECURRENT_SMOKE_WITH_GRAD", "1")),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for recurrent MoE smoke")
@pytest.mark.parametrize("config_path", CONFIGS)
def test_recurrent_moe_configs_cuda_forward_backward_no_oom(config_path: str):
    pytest.importorskip("liger_kernel.transformers")

    batch_size, seq_len, no_grad_steps, grad_steps = _smoke_shape()
    cfg = load_config(config_path)
    cfg["model"]["use_fused_linear_ce"] = True

    torch.manual_seed(1234)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, _ = build_model(cfg)
    model.gradient_checkpointing_enable()
    model.train().to("cuda")

    vocab_size = int(cfg["model"]["vocab_size"])
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    num_steps = torch.tensor([no_grad_steps, grad_steps], device="cuda", dtype=torch.long)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            labels=input_ids,
            num_steps=num_steps,
            return_logits=False,
        )
    assert output.loss is not None
    assert torch.isfinite(output.loss.detach())
    output.loss.backward()

    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(
        f"{Path(config_path).name}: batch={batch_size} seq={seq_len} "
        f"num_steps=({no_grad_steps},{grad_steps}) peak={peak_gib:.2f}GiB/"
        f"{total_gib:.1f}GiB",
        flush=True,
    )

    del output, input_ids, model
    torch.cuda.empty_cache()
