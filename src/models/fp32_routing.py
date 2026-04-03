import torch

class Fp32IndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, dim, index):
        ctx.save_for_backward(index)
        ctx.dim = dim
        ctx.inputs_shape = inputs.shape
        ctx.inputs_dtype = inputs.dtype
        return inputs.index_select(dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_tensors
        dim = ctx.dim
        grad_inputs = torch.zeros(ctx.inputs_shape, device=grad_output.device, dtype=torch.float32)
        grad_inputs.index_add_(dim, index, grad_output.to(torch.float32))
        return grad_inputs.to(ctx.inputs_dtype), None, None

def fp32_index_select(inputs, dim, index):
    if inputs.requires_grad and inputs.dtype in (torch.float16, torch.bfloat16):
        return Fp32IndexSelect.apply(inputs, dim, index)
    return inputs.index_select(dim, index)

class Fp32IndexAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target, dim, index, source):
        ctx.save_for_backward(index)
        ctx.dim = dim
        # accumulate in fp32
        out = target.to(torch.float32)
        out.index_add_(dim, index, source.to(torch.float32))
        return out.to(target.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_tensors
        dim = ctx.dim
        grad_target = grad_output
        grad_source = fp32_index_select(grad_output, dim, index)
        return grad_target, None, None, grad_source

def fp32_index_add(target, dim, index, source):
    """
    Out-of-place index_add that accumulates in fp32 and has a stable fp32 backward.
    """
    if (target.requires_grad or source.requires_grad) and target.dtype in (torch.float16, torch.bfloat16):
        return Fp32IndexAdd.apply(target, dim, index, source)
    out = target.clone()
    out.index_add_(dim, index, source.to(target.dtype))
    return out

class Fp32IndexPut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target, index, source):
        ctx.save_for_backward(index)
        ctx.target_shape = target.shape
        ctx.target_dtype = target.dtype
        out = target.clone()
        out[index] = source.to(target.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_tensors
        grad_target = grad_output.clone()
        grad_target[index] = 0.0
        grad_source = fp32_index_select(grad_output, 0, index)
        return grad_target, None, grad_source

def fp32_index_put(target, index, source):
    """
    Out-of-place assignment out[index] = source with stable fp32 backward accumulation.
    """
    if (target.requires_grad or source.requires_grad) and target.dtype in (torch.float16, torch.bfloat16):
        return Fp32IndexPut.apply(target, index, source)
    out = target.clone()
    out[index] = source.to(target.dtype)
    return out
