"""
Muon optimizer: Newton-Schulz orthogonalization for projection matrices,
combined with AdamW for non-matrix parameters (embeddings, norms, biases).

Reference: KellerJordan/modded-nanogpt (the "Polar Express" variant).

Usage:
    muon_params, adam_decay, adam_no_decay = classify_muon_params(model)
    optimizer = Muon(
        muon_params, lr=0.02, momentum=0.95, weight_decay=1.2,
        adam_params=[
            {"params": adam_decay, "weight_decay": 0.1},
            {"params": adam_no_decay, "weight_decay": 0.0},
        ],
        adam_lr=0.008,
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


# Newton-Schulz iteration coefficients for polar decomposition.
# These approximate X @ (X^T X)^{-1/2} in 5 iterations.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate the polar factor of matrix G using Newton-Schulz iteration.

    The polar factor U of G (where G = U @ S for positive semidefinite S)
    is the closest orthogonal matrix to G in Frobenius norm. This is the
    optimal steepest descent direction for matrix-valued parameters.
    """
    assert G.ndim == 2
    a, b, c = _NS_COEFFS
    X = G.bfloat16()
    X /= X.norm() + 1e-7

    if X.size(0) > X.size(1):
        # Tall matrix: work with X^T @ X (smaller)
        for _ in range(steps):
            A = X.T @ X
            B = b * A + c * A @ A
            X = a * X + X @ B
    else:
        # Wide/square matrix: work with X @ X^T (smaller)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

    return X.to(G.dtype)


def classify_muon_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """
    Classify model parameters into Muon (projection matrices) and Adam groups.

    Returns:
        muon_params: 2D weight matrices (projections) — optimized with Muon.
        adam_decay_params: Other parameters that should get weight decay.
        adam_no_decay_params: Biases, norms, embeddings — no weight decay.
    """
    muon_params: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []

    no_decay_keywords = {"bias", "norm", "embedding", "wte", "wpe", "embed"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()

        # Embeddings, biases, norms, 1D params → Adam without weight decay
        if param.ndim < 2 or any(kw in name_lower for kw in no_decay_keywords):
            adam_no_decay.append(param)
        # 2D weight matrices → Muon
        elif param.ndim == 2:
            muon_params.append(param)
        # Higher-dim tensors (rare) → Adam with weight decay
        else:
            adam_decay.append(param)

    return muon_params, adam_decay, adam_no_decay


class Muon(Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization for 2D weight matrices,
    combined with AdamW for non-matrix parameters.

    The first parameter group uses Muon updates. Subsequent groups use AdamW.

    Args:
        muon_params: Iterable of parameters for Muon (2D projection matrices).
        lr: Learning rate for Muon (default: 0.02).
        momentum: Momentum coefficient for Muon (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Newton-Schulz iterations (default: 5).
        weight_decay: Weight decay for Muon parameters (default: 0.0).
        adam_params: List of dicts with 'params' and optional 'weight_decay'.
        adam_lr: Learning rate for Adam parameters (default: 3e-4).
        adam_betas: Beta coefficients for Adam (default: (0.9, 0.95)).
        adam_eps: Epsilon for Adam (default: 1e-8).
    """

    def __init__(
        self,
        muon_params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adam_params: list[dict] | None = None,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        muon_group = {
            "params": list(muon_params),
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "is_muon": True,
        }
        groups = [muon_group]

        if adam_params is not None:
            for group in adam_params:
                groups.append({
                    "params": list(group["params"]),
                    "lr": adam_lr,
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": group.get("weight_decay", 0.0),
                    "is_muon": False,
                })

        # defaults dict is used for state_dict serialization
        super().__init__(groups, defaults=dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
            is_muon=False, betas=adam_betas, eps=adam_eps,
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("is_muon", False):
                self._muon_step(group)
            else:
                self._adam_step(group)

        return loss

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        mom = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad.float()
            state = self.state[p]

            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.lerp_(g, 1 - mom)

            if nesterov:
                # Nesterov: look ahead using momentum
                update = g.lerp_(buf, mom)
            else:
                update = buf.clone()

            # Newton-Schulz orthogonalization → polar factor
            update = newton_schulz5(update, steps=ns_steps)

            # Decoupled weight decay
            if wd != 0:
                p.data.mul_(1 - lr * wd)

            # Scale update by sqrt(max(m,n)) to match gradient magnitude convention
            scale = max(p.size(0), p.size(1)) ** 0.5
            p.data.add_(update, alpha=-lr * scale)

    def _adam_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad.float()
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Bias correction
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step

            exp_avg.lerp_(g, 1 - beta1)
            exp_avg_sq.lerp_(g.square(), 1 - beta2)

            denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)
            step_size = lr / bc1

            # Decoupled weight decay
            if wd != 0:
                p.data.mul_(1 - lr * wd)

            p.data.addcdiv_(exp_avg, denom, value=-step_size)


def get_muon_momentum(
    step: int,
    warmup_steps: int = 300,
    momentum_min: float = 0.85,
    momentum_max: float = 0.95,
) -> float:
    """Compute momentum with linear warmup."""
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return momentum_min + frac * (momentum_max - momentum_min)
    return momentum_max
