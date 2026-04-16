import pytest
import torch
import torch.distributed as dist

from src.models.speedrun_moe_gpt import (
    AttentionExpertBank,
    MLPExpertBank,
    RoutedAttentionFullyIndependent,
    RoutedAttentionPrecomputeKV,
    SpeedrunMoEGPT,
)


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


def test_speedrun_mlp_sparse_mask_routes_only_selected_tokens():
    bank = MLPExpertBank(num_experts=8, dim=64).cuda().train()
    x = torch.randn(12, 64, device="cuda")
    token_mask = torch.zeros(3, 4, 1, device="cuda", dtype=torch.bool)
    token_mask[:, :2] = True

    _ = bank(x, token_mask=token_mask)

    assert int(bank.router.local_tokens_per_expert.sum().item()) == int(token_mask.sum().item())


def test_speedrun_precompute_sparse_mask_routes_only_selected_queries():
    bank = AttentionExpertBank(
        num_experts=8,
        dim=64,
        num_heads=4,
        head_dim=16,
        mode="per_head_precompute_kv",
    ).cuda().train()
    attn = RoutedAttentionPrecomputeKV(bank, num_heads=4, head_dim=16, max_seq_len=16).cuda().train()

    x = torch.randn(2, 8, 64, device="cuda")
    token_mask = torch.zeros(2, 8, 1, device="cuda", dtype=torch.bool)
    token_mask[:, :3] = True
    lambdas = torch.tensor([1.0, 0.0], device="cuda")

    out = attn(x, None, lambdas, token_mask=token_mask)

    assert torch.isfinite(out).all()
    expected = int(token_mask.sum().item())
    for router in bank.routers:
        assert int(router.local_tokens_per_expert.sum().item()) == expected


def test_speedrun_fully_independent_sparse_mask_routes_q_and_o_on_selected_queries():
    bank = AttentionExpertBank(
        num_experts=8,
        dim=64,
        num_heads=4,
        head_dim=16,
        mode="per_head_fully_independent",
    ).cuda().train()
    attn = RoutedAttentionFullyIndependent(bank, num_heads=4, head_dim=16, max_seq_len=16).cuda().train()

    x = torch.randn(2, 8, 64, device="cuda")
    token_mask = torch.zeros(2, 8, 1, device="cuda", dtype=torch.bool)
    token_mask[:, :3] = True
    lambdas = torch.tensor([1.0, 0.0], device="cuda")

    out = attn(x, None, lambdas, token_mask=token_mask)

    assert torch.isfinite(out).all()
    selected = int(token_mask.sum().item())
    total = x.shape[0] * x.shape[1]
    for router in bank.q_routers:
        assert int(router.local_tokens_per_expert.sum().item()) == selected
    for router in bank.o_routers:
        assert int(router.local_tokens_per_expert.sum().item()) == selected
    for router in bank.k_routers:
        assert int(router.local_tokens_per_expert.sum().item()) == total
    for router in bank.v_routers:
        assert int(router.local_tokens_per_expert.sum().item()) == total


def test_fully_independent_sparse_and_dense_query_paths_match_in_fp32():
    bank = AttentionExpertBank(
        num_experts=8,
        dim=64,
        num_heads=4,
        head_dim=16,
        mode="per_head_fully_independent",
    ).cuda().eval()
    attn = RoutedAttentionFullyIndependent(bank, num_heads=4, head_dim=16, max_seq_len=16).cuda().eval()

    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)
    token_mask = torch.zeros(2, 8, 1, device="cuda", dtype=torch.bool)
    token_mask[:, :3] = True
    lambdas = torch.tensor([1.0, 0.0], device="cuda", dtype=torch.float32)

    attn.query_sparse_fraction_threshold = 1.1
    with torch.no_grad():
        sparse_out = attn(x, None, lambdas, token_mask=token_mask)
    attn.query_sparse_fraction_threshold = 0.0
    with torch.no_grad():
        dense_out = attn(x, None, lambdas, token_mask=token_mask)

    assert torch.allclose(sparse_out, dense_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sparse_out.square().mean(), dense_out.square().mean(), atol=1e-6, rtol=1e-5)


def test_precompute_sparse_and_dense_query_paths_match_in_fp32():
    bank = AttentionExpertBank(
        num_experts=8,
        dim=64,
        num_heads=4,
        head_dim=16,
        mode="per_head_precompute_kv",
    ).cuda().eval()
    attn = RoutedAttentionPrecomputeKV(bank, num_heads=4, head_dim=16, max_seq_len=16).cuda().eval()

    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)
    token_mask = torch.zeros(2, 8, 1, device="cuda", dtype=torch.bool)
    token_mask[:, :3] = True
    lambdas = torch.tensor([1.0, 0.0], device="cuda", dtype=torch.float32)

    attn.query_sparse_fraction_threshold = 1.1
    with torch.no_grad():
        sparse_out = attn(x, None, lambdas, token_mask=token_mask)
    attn.query_sparse_fraction_threshold = 0.0
    with torch.no_grad():
        dense_out = attn(x, None, lambdas, token_mask=token_mask)

    assert torch.allclose(sparse_out, dense_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(sparse_out.square().mean(), dense_out.square().mean(), atol=1e-6, rtol=1e-5)
