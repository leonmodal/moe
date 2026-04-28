"""AC-8: attention aux loss must be gradient-bearing.

Before this fix, `AttentionExpertBank._store_router_info()` detached
`router_probs` before storing them in `self.last_router_info`. The downstream
`MoEverythingForCausalLM.forward()` then built `attention_aux_loss` from those
detached tensors and added `router_aux_loss_coef * attention_aux_loss` to the
total loss — but no gradient could flow back to the attention router weights.
The aux loss was numerically present and zero-impact in practice.

These tests pin the contract directly on `_store_router_info()` and on
`load_balancing_loss_func` consuming the stored payload. They run on CPU
without the full attention forward path (which has CUDA-only kernels and
several index-assignment shape constraints unrelated to AC-8).
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn

from src.models.moe_everything.attention_bank import AttentionExpertBank
from src.models.routing.load_balancing import load_balancing_loss_func


def _make_bank() -> AttentionExpertBank:
    """Build a minimal AttentionExpertBank without invoking the full forward.

    We only need the `_store_router_info` method, which doesn't depend on the
    bank's expert-weight buffers. We construct the module and patch what we
    need rather than relying on the real config-driven init path.
    """
    bank = AttentionExpertBank.__new__(AttentionExpertBank)
    nn.Module.__init__(bank)
    bank.last_router_info = {}
    return bank


def _gradient_sensitive_reduction(t: torch.Tensor) -> torch.Tensor:
    """Reduce a router-probs tensor to a scalar in a way that yields non-zero
    gradient back to its source.

    Plain `t.sum()` is zero-gradient when `t` came from a row-wise softmax —
    each row sums to 1, so the partial w.r.t. logits cancels. Squaring the
    elements first breaks that symmetry and gives a meaningful gradient.
    """
    return (t * t).sum()


def test_store_router_info_no_token_mask_keeps_gradient_bearing():
    """`_store_router_info` without a token_mask must keep `router_logits`
    differentiable so aux loss can propagate gradient back to the router weights.
    """
    bank = _make_bank()
    router = nn.Linear(8, 4, bias=False)
    x = torch.randn(6, 8)  # (N, hidden_size)
    router_probs = router(x).softmax(dim=-1)  # (N, num_experts), requires grad
    expert_idx = torch.randint(0, 4, (6, 1))
    bank._store_router_info("q", router_probs, expert_idx)

    info = bank.last_router_info["q"]
    assert info["router_logits"].requires_grad, (
        "stored router_logits must require grad — aux loss needs gradient flow "
        "back to attention router weights (AC-8)."
    )
    # The detached side-channel exists for telemetry.
    assert "router_logits_detached" in info
    assert not info["router_logits_detached"].requires_grad

    # End-to-end gradient sanity: a gradient-sensitive reduction on the stored
    # router_logits must produce a non-zero gradient on the router weight.
    _gradient_sensitive_reduction(info["router_logits"]).backward()
    assert router.weight.grad is not None
    assert router.weight.grad.norm().item() > 1e-12


def test_store_router_info_token_mask_path_keeps_gradient_bearing():
    """Same contract for the token-mask path (used when MLP-branch tokens are
    routed away from attention)."""
    bank = _make_bank()
    router = nn.Linear(8, 4, bias=False)
    active_x = torch.randn(3, 8)
    router_probs = router(active_x).softmax(dim=-1)  # (3, 4), requires grad
    expert_idx = torch.randint(0, 4, (3, 1))
    token_mask = torch.tensor([True, False, True, True, False, False])  # 3 active out of 6

    bank._store_router_info("q", router_probs, expert_idx, token_mask=token_mask)
    info = bank.last_router_info["q"]
    assert info["router_logits"].requires_grad, (
        "token-masked router_logits must keep gradient flow (AC-8)."
    )
    assert info["router_logits"].shape == (6, 4)  # densified to full T
    assert "router_logits_detached" in info
    assert info["token_mask"] is not None and info["token_mask"].dtype == torch.bool

    _gradient_sensitive_reduction(info["router_logits"]).backward()
    assert router.weight.grad is not None
    assert router.weight.grad.norm().item() > 1e-12


def test_aux_loss_consumer_receives_gradient_bearing_router_logits():
    """End-to-end: feed the stored `router_logits` into `load_balancing_loss_func`
    and assert the resulting aux loss carries gradient back to the router weight.

    This is the path that `MoEverythingForCausalLM.forward()` actually executes
    for attention aux loss. Fixes AC-8's regression at the integration boundary.
    """
    torch.manual_seed(0)
    bank = _make_bank()
    router = nn.Linear(8, 4, bias=False)
    active_x = torch.randn(5, 8)
    router_probs = router(active_x).softmax(dim=-1)
    expert_idx = router_probs.argmax(dim=-1, keepdim=True)
    token_mask = torch.tensor([True, True, False, True, True, False, False, True])  # 5 active

    bank._store_router_info("q", router_probs, expert_idx, token_mask=token_mask)
    info = bank.last_router_info["q"]
    aux = load_balancing_loss_func(
        gate_logits=(info["router_logits"],),
        num_experts=4,
        top_k=1,
        token_masks=(info["token_mask"],),
        selected_experts=(info["selected_experts"],),
    )
    assert isinstance(aux, torch.Tensor) and aux.requires_grad, (
        "aux loss must require grad — locks AC-8 attention aux gradient contract."
    )
    aux.backward()
    assert router.weight.grad is not None
    gnorm = router.weight.grad.norm().item()
    assert gnorm > 1e-12, (
        f"aux backward produced negligible gradient on the router weight: {gnorm}. "
        f"AC-8 regression — the stored router_logits is silently detached again."
    )


if __name__ == "__main__":
    test_store_router_info_no_token_mask_keeps_gradient_bearing()
    test_store_router_info_token_mask_path_keeps_gradient_bearing()
    test_aux_loss_consumer_receives_gradient_bearing_router_logits()
    print("ALL OK")
