"""
SpeedrunMoEGPT: Speedrun GPT with per-head routed attention + shared MLP experts.

Routing design:
- Each head slot has its own top-1 router (NOT one router picking top-K)
- per_head_fully_independent: 4H routers (Q/K/V independent, O on attn output)
- per_head_precompute_kv: H routers (QKVO bundled)
- Shared MLP expert bank with top-1 routing
- Switch-style aux loss for load balancing

Uses Triton grouped GEMM for efficient dispatch.
Supports batched (B, T) inputs with SDPA attention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.speedrun_gpt import (
    CastedLinear,
    Rotary,
    mm_op,  # noqa: F401 - registers the custom op
    norm,
    next_multiple_of_n,
)
from src.models.triton_grouped_gemm import triton_grouped_gemm
from src.utils.routing_loss import switch_load_balancing_loss


# ---------------------------------------------------------------------------
# Routing stats collector

@dataclass
class RoutingStats:
    """Accumulates routing statistics during a forward pass."""
    aux_losses: list[Tensor] = field(default_factory=list)
    # Per-router: (router_name, expert_ids, num_experts)
    router_records: list[tuple[str, Tensor, int]] = field(default_factory=list)

    def add(self, name: str, expert_ids: Tensor, probs: Tensor, num_experts: int):
        self.aux_losses.append(switch_load_balancing_loss(expert_ids, probs, num_experts))
        self.router_records.append((name, expert_ids.detach(), num_experts))

    def total_aux_loss(self) -> Tensor:
        if not self.aux_losses:
            return torch.tensor(0.0)
        return sum(self.aux_losses) / len(self.aux_losses)

    def summary(self) -> dict[str, float]:
        """Compute summary stats for wandb logging."""
        if not self.router_records:
            return {}
        stats = {}
        for name, eids, E in self.router_records:
            counts = torch.bincount(eids, minlength=E).float()
            total = counts.sum()
            if total == 0:
                continue
            fracs = counts / total
            # Load balance ratio: 1.0 = perfectly balanced, 0.0 = all on one expert
            # Defined as E * min(fracs) — equals 1.0 when uniform
            stats[f"routing/{name}_balance"] = (E * fracs.min()).item()
            # Utilization: fraction of experts that received at least 1 token
            stats[f"routing/{name}_utilization"] = (counts > 0).float().mean().item()
            # Entropy of routing distribution (normalized by log(E))
            entropy = -(fracs * (fracs + 1e-10).log()).sum()
            max_entropy = torch.tensor(E, dtype=torch.float32).log()
            stats[f"routing/{name}_entropy"] = (entropy / max_entropy).item()
        return stats


# ---------------------------------------------------------------------------
# Per-head-slot top-1 routing + grouped GEMM projection

def _route_top1(router: nn.Linear, x: Tensor, stats: RoutingStats | None = None,
                router_name: str = ""):
    """Top-1 routing with optional stats/aux-loss collection."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = router(x.float())
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        expert_ids = probs.argmax(dim=-1)  # (N,)
        weights = probs.gather(-1, expert_ids.unsqueeze(-1)).squeeze(-1)  # (N,)
    if stats is not None:
        stats.add(router_name, expert_ids, probs, logits.shape[-1])
    return expert_ids, weights


def _grouped_project(x: Tensor, weight_bank: Tensor, expert_ids: Tensor, weights: Tensor):
    """Sort by expert, Triton grouped GEMM, weight, unsort."""
    sort_order = expert_ids.argsort()
    sorted_expert = expert_ids[sort_order]
    sorted_inputs = x[sort_order].contiguous()
    unique_experts, counts = sorted_expert.unique_consecutive(return_counts=True)

    proj = triton_grouped_gemm(sorted_inputs, weight_bank.to(x.dtype), unique_experts, counts)
    proj = proj * weights[sort_order].unsqueeze(-1).to(proj.dtype)

    out = proj.new_zeros(x.shape[0], proj.shape[-1])
    out[sort_order] = proj
    return out


def _route_and_project_heads(routers: nn.ModuleList, x: Tensor, weight_bank: Tensor,
                             norm_weights: Tensor | None = None,
                             stats: RoutingStats | None = None,
                             name_prefix: str = "") -> Tensor:
    """Per-head-slot top-1 routing + projection."""
    H = len(routers)
    results = []
    for h in range(H):
        eid, ew = _route_top1(routers[h], x, stats, f"{name_prefix}_h{h}")
        proj = _grouped_project(x, weight_bank, eid, ew)
        if norm_weights is not None:
            proj_f = proj.float()
            var = proj_f.pow(2).mean(-1, keepdim=True)
            proj_normed = proj_f * torch.rsqrt(var + 1e-6)
            nw = norm_weights[eid].to(proj.dtype)
            proj = (nw * proj_normed.to(proj.dtype))
        results.append(proj)
    return torch.stack(results, dim=1)


def _route_and_project_reduce(routers: nn.ModuleList, x: Tensor, weight_bank: Tensor,
                              stats: RoutingStats | None = None,
                              name_prefix: str = "") -> Tensor:
    """Per-head-slot top-1 routing on per-head inputs, sum across heads."""
    N, H, _ = x.shape
    dim = weight_bank.shape[2]
    token_out = torch.zeros(N, dim, device=x.device, dtype=torch.float32)
    for h in range(H):
        x_h = x[:, h].contiguous()
        eid, ew = _route_top1(routers[h], x_h, stats, f"{name_prefix}_h{h}")
        proj = _grouped_project(x_h, weight_bank, eid, ew)
        token_out = token_out + proj.float()
    return token_out.to(x.dtype)


# ---------------------------------------------------------------------------
# Attention Expert Bank

class AttentionExpertBank(nn.Module):
    def __init__(self, num_experts: int, dim: int, num_heads: int, head_dim: int, mode: str):
        super().__init__()
        self.mode = mode
        self.num_experts = num_experts
        E, D, HD, H = num_experts, dim, head_dim, num_heads

        if mode == "per_head_fully_independent":
            self.q_routers = nn.ModuleList([nn.Linear(D, E, bias=False) for _ in range(H)])
            self.k_routers = nn.ModuleList([nn.Linear(D, E, bias=False) for _ in range(H)])
            self.v_routers = nn.ModuleList([nn.Linear(D, E, bias=False) for _ in range(H)])
            self.o_routers = nn.ModuleList([nn.Linear(HD, E, bias=False) for _ in range(H)])
            self.q_proj = nn.Parameter(torch.empty(E, D, HD))
            self.k_proj = nn.Parameter(torch.empty(E, D, HD))
            self.v_proj = nn.Parameter(torch.empty(E, D, HD))
            self.o_proj = nn.Parameter(torch.empty(E, HD, D))
            self.q_norm = nn.Parameter(torch.ones(E, HD))
            self.k_norm = nn.Parameter(torch.ones(E, HD))
        elif mode == "per_head_precompute_kv":
            self.routers = nn.ModuleList([nn.Linear(D, E, bias=False) for _ in range(H)])
            self.q_proj = nn.Parameter(torch.empty(E, D, HD))
            self.k_proj = nn.Parameter(torch.empty(E, D, HD))
            self.v_proj = nn.Parameter(torch.empty(E, D, HD))
            self.o_proj = nn.Parameter(torch.empty(E, HD, D))
            self.q_norm = nn.Parameter(torch.ones(E, HD))
            self.k_norm = nn.Parameter(torch.ones(E, HD))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self._init_weights(dim)

    def _init_weights(self, dim):
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.uniform_(proj, -bound, bound)
        nn.init.zeros_(self.o_proj)


# ---------------------------------------------------------------------------
# MLP Expert Bank

class MLPExpertBank(nn.Module):
    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.c_fc = nn.Parameter(torch.empty(num_experts, dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(num_experts, hdim, dim))
        self._init_weights(dim)

    def _init_weights(self, dim):
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        nn.init.uniform_(self.c_fc, -bound, bound)
        nn.init.zeros_(self.c_proj)

    def forward(self, x: Tensor, stats: RoutingStats | None = None, depth_idx: int = 0) -> Tensor:
        eid, ew = _route_top1(self.router, x, stats, f"mlp_d{depth_idx}")
        h = _grouped_project(x, self.c_fc, eid, torch.ones_like(ew))
        h = F.relu(h).square()
        out = _grouped_project(h, self.c_proj, eid, ew)
        return out


# ---------------------------------------------------------------------------
# Routed attention modules

class RoutedAttentionFullyIndependent(nn.Module):
    def __init__(self, bank: AttentionExpertBank, num_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.bank = bank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12
        self.attn_gate_dim = 12
        self.attn_gate = CastedLinear(self.attn_gate_dim, num_heads)
        self.attn_gate.weight.detach().zero_()

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor,
                stats: RoutingStats | None = None, depth_idx: int = 0):
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat = x.reshape(B * T, D)
        dp = f"d{depth_idx}"

        q = _route_and_project_heads(bank.q_routers, flat, bank.q_proj, bank.q_norm, stats, f"q_{dp}")
        k = _route_and_project_heads(bank.k_routers, flat, bank.k_proj, bank.k_norm, stats, f"k_{dp}")
        v = _route_and_project_heads(bank.v_routers, flat, bank.v_proj, None, stats, f"v_{dp}")

        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.reshape(B * T, H, HD).to(v.dtype)
        else:
            v = lambdas[0] * v

        q = q.view(B, T, H, HD).transpose(1, 2)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)
        q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
        k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        y = y.transpose(1, 2)

        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim]))
        y = y * gate.unsqueeze(-1)

        y_flat = y.reshape(B * T, H, HD)
        out = _route_and_project_reduce(bank.o_routers, y_flat, bank.o_proj, stats, f"o_{dp}")
        return out.view(B, T, D)


class RoutedAttentionPrecomputeKV(nn.Module):
    def __init__(self, bank: AttentionExpertBank, num_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.bank = bank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12
        self.attn_gate_dim = 12
        self.attn_gate = CastedLinear(self.attn_gate_dim, num_heads)
        self.attn_gate.weight.detach().zero_()

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor,
                stats: RoutingStats | None = None, depth_idx: int = 0):
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat = x.reshape(B * T, D)
        dp = f"d{depth_idx}"

        # Route per head slot (QKVO bundled)
        head_eids, head_ews = [], []
        for h in range(H):
            eid, ew = _route_top1(bank.routers[h], flat, stats, f"qkvo_{dp}_h{h}")
            head_eids.append(eid)
            head_ews.append(ew)

        # Project Q
        q_parts = []
        for h in range(H):
            proj = _grouped_project(flat, bank.q_proj, head_eids[h], head_ews[h])
            proj_f = proj.float()
            var = proj_f.pow(2).mean(-1, keepdim=True)
            proj_normed = proj_f * torch.rsqrt(var + 1e-6)
            nw = bank.q_norm[head_eids[h]].to(proj.dtype)
            q_parts.append((nw * proj_normed.to(proj.dtype)))
        q = torch.stack(q_parts, dim=1)

        q_4d = q.view(B, T, H, HD).transpose(1, 2)
        q_4d = self.rotary(q_4d.transpose(1, 2)).transpose(1, 2)

        output = torch.zeros(B, H, T, HD, device=flat.device, dtype=flat.dtype)

        for h in range(H):
            eid, ew = head_eids[h], head_ews[h]
            active_experts = eid.unique().tolist()
            q_h = q_4d[:, h:h + 1]
            attn_h = torch.zeros(B, 1, T, HD, device=flat.device, dtype=flat.dtype)

            for e in active_experts:
                k_e = flat @ bank.k_proj[e].to(flat.dtype)
                k_f = k_e.float()
                k_normed = (k_f * torch.rsqrt(k_f.pow(2).mean(-1, keepdim=True) + 1e-6)).to(k_e.dtype)
                k_normed = k_normed * bank.k_norm[e].to(k_e.dtype)
                v_e = flat @ bank.v_proj[e].to(flat.dtype)

                k_4d = self.rotary(k_normed.view(B, T, 1, HD)).view(B, 1, T, HD)
                v_4d = v_e.view(B, 1, T, HD)

                if ve is not None:
                    ve_h = ve.view(B, T, H, HD)[:, :, h:h + 1].transpose(1, 2).to(v_4d.dtype)
                    v_4d = lambdas[0] * v_4d + lambdas[1] * ve_h
                else:
                    v_4d = lambdas[0] * v_4d

                attn_e = F.scaled_dot_product_attention(q_h, k_4d, v_4d, is_causal=True, scale=self.attn_scale)
                mask_w = (ew * (eid == e).float()).view(B, T).unsqueeze(1).unsqueeze(-1)
                attn_h = attn_h + attn_e * mask_w

            output[:, h:h + 1] = attn_h

        y = output.transpose(1, 2)
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim]))
        y = y * gate.unsqueeze(-1)

        # O projection (same routing as QKV — bundled)
        y_flat = y.reshape(B * T, H, HD)
        o_parts = []
        for h in range(H):
            o_h = _grouped_project(y_flat[:, h].contiguous(), bank.o_proj, head_eids[h], head_ews[h])
            o_parts.append(o_h.float())
        out = sum(o_parts)
        return out.to(x.dtype).view(B, T, D)


# ---------------------------------------------------------------------------
# Block and full model

class MoEBlock(nn.Module):
    def __init__(self, attn_bank: AttentionExpertBank, mlp_bank: MLPExpertBank,
                 num_heads: int, head_dim: int, max_seq_len: int, layer_idx: int, mode: str,
                 skip_attn: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        if skip_attn:
            self.attn = None
        elif mode == "per_head_fully_independent":
            self.attn = RoutedAttentionFullyIndependent(attn_bank, num_heads, head_dim, max_seq_len)
        elif mode == "per_head_precompute_kv":
            self.attn = RoutedAttentionPrecomputeKV(attn_bank, num_heads, head_dim, max_seq_len)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.mlp_bank = mlp_bank

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor,
                sa_lambdas: Tensor, stats: RoutingStats | None = None):
        B, T, D = x.shape
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, stats, self.layer_idx)
        x = x + self.mlp_bank(norm(x).reshape(-1, D), stats, self.layer_idx).view(B, T, D)
        return x


class SpeedrunMoEGPT(nn.Module):
    """Speedrun GPT with shared attention + MLP expert banks.

    After forward(), access:
      self._aux_loss: scalar tensor (switch load balancing loss)
      self._routing_stats: dict of routing metrics for wandb logging
    """

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int,
                 head_dim: int, max_seq_len: int, mode: str,
                 num_attn_experts: int = 66, num_mlp_experts: int = 12):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.mode = mode
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

        self.attn_bank = AttentionExpertBank(num_attn_experts, model_dim, num_heads, head_dim, mode)
        self.mlp_bank = MLPExpertBank(num_mlp_experts, model_dim)

        self.blocks = nn.ModuleList([
            MoEBlock(self.attn_bank, self.mlp_bank, num_heads, head_dim, max_seq_len, i, mode)
            for i in range(num_layers)
        ])

        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8,
                                     x_s=(model_dim ** 0.5) / 448, w_s=2 ** -9, grad_s=1 / 448)
        self.lm_head.weight.detach().zero_()

        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % max(dist.get_world_size(), 1)
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
            torch.ones(max(pad, 0)),
        ]))

        for param in self.embed.parameters():
            param.lr_mul = 75.0
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

        # Populated after each forward
        self._aux_loss = torch.tensor(0.0)
        self._routing_stats = {}

    def forward(self, input_ids: Tensor, labels: Tensor):
        B, T = input_ids.shape
        stats = RoutingStats() if self.training else None

        ve = [vemb(input_ids) for vemb in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (self.num_layers - 6) + [ve[0], ve[1], ve[2]]

        x = x0 = norm(self.embed(input_ids))

        skip_connections = []
        skip_weights = self.scalars[:self.num_layers // 2]
        lambdas = self.scalars[1 * self.num_layers:3 * self.num_layers].view(-1, 2)
        sa_lambdas = self.scalars[3 * self.num_layers:5 * self.num_layers].view(-1, 2)
        n = self.num_layers // 2

        for i in range(self.num_layers):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], stats)
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               reduction="sum" if self.training else "mean")

        if stats is not None:
            self._aux_loss = stats.total_aux_loss()
            self._routing_stats = stats.summary()
        else:
            self._aux_loss = torch.tensor(0.0, device=loss.device)
            self._routing_stats = {}

        return loss
