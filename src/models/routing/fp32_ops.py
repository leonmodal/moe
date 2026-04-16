"""FP32 index operations for numerical stability in routing.

These ensure that index operations used in expert dispatch maintain
float32 precision to avoid numerical instability in mixed-precision training.
"""

import torch


def fp32_index_select(source: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """index_select in float32 for numerical stability."""
    return torch.index_select(source.float(), dim, index).to(source.dtype)


def fp32_index_put(
    target: torch.Tensor,
    indices: tuple[torch.Tensor, ...],
    values: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    """index_put in float32 for numerical stability."""
    target_f32 = target.float()
    target_f32.index_put_(indices, values.float(), accumulate=accumulate)
    return target_f32.to(target.dtype)


def fp32_index_add(
    target: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """index_add in float32 for numerical stability."""
    target_f32 = target.float()
    target_f32.index_add_(dim, index, source.float())
    return target_f32.to(target.dtype)
