from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton grouped GEMM")
def test_triton_grouped_gemm_matches_grouped_matmul():
    from src.models.triton_grouped_gemm import triton_grouped_gemm

    torch.manual_seed(0)
    counts = torch.tensor([5, 7, 4], device="cuda", dtype=torch.int64)
    unique_experts = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int64)
    total = int(counts.sum().item())
    hidden_in = 64
    hidden_out = 32
    num_experts = 4

    a = torch.randn(total, hidden_in, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(num_experts, hidden_in, hidden_out, device="cuda", dtype=torch.bfloat16)

    got = triton_grouped_gemm(a, weight, unique_experts, counts)

    ref = torch.empty_like(got)
    start = 0
    for expert, count in zip(unique_experts.tolist(), counts.tolist()):
        end = start + count
        ref[start:end] = a[start:end] @ weight[expert]
        start = end

    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton grouped GEMM")
def test_triton_grouped_gemm_output_input_matches_linear_layout():
    from src.models.triton_grouped_gemm import triton_grouped_gemm_output_input

    torch.manual_seed(0)
    counts = torch.tensor([6, 3, 5], device="cuda", dtype=torch.int64)
    unique_experts = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int64)
    total = int(counts.sum().item())
    hidden_in = 64
    hidden_out = 48
    num_experts = 4

    a = torch.randn(total, hidden_in, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(num_experts, hidden_out, hidden_in, device="cuda", dtype=torch.bfloat16)

    got = triton_grouped_gemm_output_input(a, weight, unique_experts, counts)

    ref = torch.empty_like(got)
    start = 0
    for expert, count in zip(unique_experts.tolist(), counts.tolist()):
        end = start + count
        ref[start:end] = torch.nn.functional.linear(a[start:end], weight[expert])
        start = end

    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton grouped GEMM")
def test_triton_grouped_gemm_output_input_backward_matches_linear_layout():
    from src.models.triton_grouped_gemm import triton_grouped_gemm_output_input

    torch.manual_seed(0)
    counts = torch.tensor([6, 3, 5], device="cuda", dtype=torch.int64)
    unique_experts = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int64)
    total = int(counts.sum().item())
    hidden_in = 64
    hidden_out = 48
    num_experts = 4

    a = torch.randn(total, hidden_in, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        num_experts, hidden_out, hidden_in,
        device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    ref_a = a.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)

    got = triton_grouped_gemm_output_input(a, weight, unique_experts, counts)

    refs = []
    start = 0
    for expert, count in zip(unique_experts.tolist(), counts.tolist()):
        end = start + count
        refs.append(torch.nn.functional.linear(ref_a[start:end], ref_weight[expert]))
        start = end
    ref = torch.cat(refs, dim=0)

    grad = torch.randn_like(got)
    got.backward(grad)
    ref.backward(grad)

    torch.testing.assert_close(a.grad.float(), ref_a.grad.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(weight.grad.float(), ref_weight.grad.float(), atol=3e-2, rtol=3e-2)
