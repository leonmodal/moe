import torch
import triton
import triton.language as tl

@triton.jit
def _grouped_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_stride_0, a_stride_1,
    b_stride_0, b_stride_1, b_stride_2,
    c_stride_0, c_stride_1,
    offsets_ptr, experts_ptr,
    K, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m_n = tl.program_id(0)
    expert_idx = tl.program_id(1)

    start_row = tl.load(offsets_ptr + expert_idx)
    end_row = tl.load(offsets_ptr + expert_idx + 1)
    M = end_row - start_row
    
    if M <= 0:
        return

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_m_n // num_pid_n
    pid_n = pid_m_n % num_pid_n

    if pid_m * BLOCK_M >= M:
        return

    real_expert = tl.load(experts_ptr + expert_idx)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (start_row + offs_m[:, None]) * a_stride_0 + offs_k[None, :] * a_stride_1
    b_ptrs = b_ptr + real_expert * b_stride_0 + offs_k[:, None] * b_stride_1 + offs_n[None, :] * b_stride_2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        mask_b = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        
        a_ptrs += BLOCK_K * a_stride_1
        b_ptrs += BLOCK_K * b_stride_1

    c_ptrs = c_ptr + (start_row + offs_m[:, None]) * c_stride_0 + offs_n[None, :] * c_stride_1
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=mask_c)

class TritonGroupedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight_bank, unique_experts, counts):
        ctx.save_for_backward(a, weight_bank, unique_experts, counts)
        
        M_total, K = a.shape
        E, K_w, N = weight_bank.shape
        assert K == K_w
        
        c = torch.empty((M_total, N), device=a.device, dtype=a.dtype)
        
        if M_total == 0:
            return c
            
        offsets = torch.zeros(counts.shape[0] + 1, device=a.device, dtype=torch.int32)
        offsets[1:] = torch.cumsum(counts, dim=0)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        
        max_m = counts.max().item()
        grid = (
            triton.cdiv(max_m, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            unique_experts.shape[0]
        )
        
        _grouped_gemm_kernel[grid](
            a, weight_bank, c,
            a.stride(0), a.stride(1),
            weight_bank.stride(0), weight_bank.stride(1), weight_bank.stride(2),
            c.stride(0), c.stride(1),
            offsets, unique_experts,
            K, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        return c

    @staticmethod
    def backward(ctx, grad_c):
        a, weight_bank, unique_experts, counts = ctx.saved_tensors
        
        # grad_a = grad_c @ weight_bank^T
        # We can reuse the same kernel for grad_a by passing weight_bank transposed
        weight_bank_t = weight_bank.transpose(1, 2).contiguous()
        grad_a = TritonGroupedGemmFunction.apply(grad_c, weight_bank_t, unique_experts, counts)
        
        # grad_weight_bank[e] = a[start:end]^T @ grad_c[start:end]
        # For simplicity and exact fp32 accumulation, we use a PyTorch loop over unique experts.
        # This loop runs once per layer backward, and for max 256 iterations it is very fast.
        grad_weight_bank = torch.zeros_like(weight_bank, dtype=torch.float32)
        
        start = 0
        a_f32 = a.float()
        grad_c_f32 = grad_c.float()
        
        for idx, (expert, count) in enumerate(zip(unique_experts.tolist(), counts.tolist())):
            end = start + count
            if count > 0:
                grad_weight_bank[expert] = a_f32[start:end].T @ grad_c_f32[start:end]
            start = end
            
        grad_weight_bank = grad_weight_bank.to(weight_bank.dtype)
        return grad_a, grad_weight_bank, None, None

def triton_grouped_gemm(a, weight_bank, unique_experts, counts):
    return TritonGroupedGemmFunction.apply(a, weight_bank, unique_experts, counts)


@triton.jit
def _grouped_gemm_output_input_kernel(
    a_ptr, b_ptr, c_ptr,
    a_stride_0, a_stride_1,
    b_stride_0, b_stride_1, b_stride_2,
    c_stride_0, c_stride_1,
    offsets_ptr, experts_ptr,
    K, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m_n = tl.program_id(0)
    expert_idx = tl.program_id(1)

    start_row = tl.load(offsets_ptr + expert_idx)
    end_row = tl.load(offsets_ptr + expert_idx + 1)
    M = end_row - start_row

    if M <= 0:
        return

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_m_n // num_pid_n
    pid_n = pid_m_n % num_pid_n

    if pid_m * BLOCK_M >= M:
        return

    real_expert = tl.load(experts_ptr + expert_idx)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (start_row + offs_m[:, None]) * a_stride_0 + offs_k[None, :] * a_stride_1
    b_ptrs = b_ptr + real_expert * b_stride_0 + offs_n[None, :] * b_stride_1 + offs_k[:, None] * b_stride_2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offs_n[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        accumulator += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * a_stride_1
        b_ptrs += BLOCK_K * b_stride_2

    c_ptrs = c_ptr + (start_row + offs_m[:, None]) * c_stride_0 + offs_n[None, :] * c_stride_1
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=mask_c)


class TritonGroupedGemmOutputInputFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight_bank, unique_experts, counts):
        ctx.save_for_backward(a, weight_bank, unique_experts, counts)

        M_total, K = a.shape
        E, N, K_w = weight_bank.shape
        assert K == K_w

        c = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

        if M_total == 0:
            return c

        offsets = torch.zeros(counts.shape[0] + 1, device=a.device, dtype=torch.int32)
        offsets[1:] = torch.cumsum(counts, dim=0)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        max_m = counts.max().item()
        grid = (
            triton.cdiv(max_m, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            unique_experts.shape[0],
        )

        _grouped_gemm_output_input_kernel[grid](
            a, weight_bank, c,
            a.stride(0), a.stride(1),
            weight_bank.stride(0), weight_bank.stride(1), weight_bank.stride(2),
            c.stride(0), c.stride(1),
            offsets, unique_experts,
            K, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return c

    @staticmethod
    def backward(ctx, grad_c):
        a, weight_bank, unique_experts, counts = ctx.saved_tensors

        grad_a = TritonGroupedGemmOutputInputFunction.apply(grad_c, weight_bank, unique_experts, counts)

        grad_weight_bank = torch.zeros_like(weight_bank, dtype=torch.float32)
        start = 0
        a_f32 = a.float()
        grad_c_f32 = grad_c.float()

        for expert, count in zip(unique_experts.tolist(), counts.tolist()):
            end = start + count
            if count > 0:
                grad_weight_bank[expert] = grad_c_f32[start:end].T @ a_f32[start:end]
            start = end

        grad_weight_bank = grad_weight_bank.to(weight_bank.dtype)
        return grad_a, grad_weight_bank, None, None


def triton_grouped_gemm_output_input(a, weight_bank, unique_experts, counts):
    return TritonGroupedGemmOutputInputFunction.apply(a, weight_bank, unique_experts, counts)
