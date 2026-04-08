import pytest
import torch
import torch.distributed as dist

from src.models.speedrun_moe_gpt import SpeedrunMoEGPT


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only tests")


@pytest.fixture(scope="module", autouse=True)
def _dist_group():
    if dist.is_available() and not dist.is_initialized():
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29631",
            rank=0,
            world_size=1,
        )
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["per_head_precompute_kv", "per_head_fully_independent"])
def test_speedrun_moe_uses_24_effective_depths_and_runs(mode):
    model = SpeedrunMoEGPT(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        model_dim=768,
        head_dim=128,
        max_seq_len=128,
        mode=mode,
        num_attn_experts=66,
        num_mlp_experts=12,
    ).cuda().eval()

    assert model.num_depths == 24

    input_ids = torch.randint(0, 50257, (2, 32), device="cuda")
    with torch.no_grad():
        loss = model(input_ids, input_ids.clone())

    assert torch.isfinite(loss)


@pytest.mark.parametrize("mode", ["per_head_precompute_kv", "per_head_fully_independent"])
def test_speedrun_moe_computes_seq_aux_loss(mode):
    model = SpeedrunMoEGPT(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        model_dim=768,
        head_dim=128,
        max_seq_len=128,
        mode=mode,
        num_attn_experts=66,
        num_mlp_experts=12,
    ).cuda().train()

    input_ids = torch.randint(0, 50257, (2, 32), device="cuda")
    loss = model(input_ids, input_ids.clone())

    assert torch.isfinite(loss)
    assert torch.isfinite(model._seq_aux_loss)


@pytest.mark.parametrize("mode", ["per_head_precompute_kv", "per_head_fully_independent"])
def test_speedrun_moe_runs_under_bf16_autocast(mode):
    model = SpeedrunMoEGPT(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        model_dim=768,
        head_dim=128,
        max_seq_len=128,
        mode=mode,
        num_attn_experts=66,
        num_mlp_experts=12,
    ).cuda().train()

    input_ids = torch.randint(0, 50257, (2, 32), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids, input_ids.clone())

    assert torch.isfinite(loss)
