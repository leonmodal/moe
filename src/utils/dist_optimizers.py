"""
Distributed Muon and Adam optimizers for the speedrun model.

Reference: /tmp/modded-nanogpt-prefa3/train_gpt.py (lines 388-527)

These optimizers handle their own gradient all-reduce internally via
reduce_scatter / all_gather, so the model should NOT be wrapped in DDP.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

from src.utils.triton_newton_schulz import newton_schulz_triton


class DistMuon(torch.optim.Optimizer):
    """
    Distributed Muon optimizer with Newton-Schulz orthogonalization.

    Groups parameters by shape and uses reduce_scatter / all_gather
    for efficient distributed gradient sync.

    Respects per-parameter `lr_mul` and `wd_mul` attributes.
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [(param.grad if param.grad is not None else torch.zeros_like(param)) for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p_grad = params[base_i + rank].grad
                    grad = p_grad if p_grad is not None else torch.zeros_like(params[base_i + rank])
                reduce_scatter_futures.append(
                    dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future()
                )

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad if p.grad is not None else torch.zeros_like(p)
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = newton_schulz_triton(grad)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_gather_futures.append(
                    dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future()
                )
        torch.futures.collect_all(all_gather_futures).wait()


class DistAdam(torch.optim.Optimizer):
    """
    Distributed Adam optimizer that shards state across ranks.

    Uses reduce_scatter_tensor / all_gather_into_tensor for
    efficient distributed gradient sync and parameter update.

    Respects per-parameter `lr_mul` and `wd_mul` attributes.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                if grad is None:
                    grad = torch.zeros_like(params[base_i])
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                )
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = group["params"]
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )
        torch.futures.collect_all(all_gather_futures).wait()
