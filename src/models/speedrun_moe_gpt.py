"""
SpeedrunMoEGPT: Speedrun GPT with per-head routed attention + shared MLP experts.

Routing design:
- Each head slot has its own top-1 DeepSeek-style router (sigmoid + expert bias)
- per_head_fully_independent: 4H routers (Q/K/V independent, O on attn output)
- per_head_precompute_kv: H routers (QKVO bundled)
- Shared MLP expert bank with top-1 routing
- Switch-style aux loss for load balancing
- Expert bias updates (DeepSeek V3 style)

Uses Triton grouped GEMM for efficient dispatch.
Supports batched (B, T) inputs with SDPA attention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

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
from src.models.router import DeepSeekRouter
from src.models.load_balancing import seq_load_balancing_loss_func
from src.models.fp32_routing import fp32_index_put, fp32_index_select
from src.utils.routing_loss import switch_load_balancing_loss


# ---------------------------------------------------------------------------
# Routing stats collector

@dataclass
class RoutingStats:
    """Accumulates routing statistics during a forward pass."""
    aux_losses: list[Tensor] = field(default_factory=list)
    router_records: list[tuple[str, Tensor, int]] = field(default_factory=list)
    branch_records: list[Tensor] = field(default_factory=list)
    seq_router_records: list[tuple[Tensor, Tensor, int]] = field(default_factory=list)

    def add(self, name: str, expert_ids: Tensor, probs: Tensor, num_experts: int):
        self.aux_losses.append(switch_load_balancing_loss(expert_ids, probs, num_experts))
        self.router_records.append((name, expert_ids.detach(), num_experts))
        self.seq_router_records.append((probs, expert_ids.unsqueeze(-1), num_experts))

    def total_aux_loss(self) -> Tensor:
        if not self.aux_losses:
            return torch.tensor(0.0)
        return sum(self.aux_losses) / len(self.aux_losses)

    def add_branch(self, probs: Tensor):
        self.branch_records.append(probs.detach())

    def total_seq_aux_loss(self, batch_size: int) -> Tensor:
        if not self.seq_router_records:
            return torch.tensor(0.0)
        seq_terms: list[Tensor] = []
        for probs, selected, num_experts in self.seq_router_records:
            seq_terms.append(
                seq_load_balancing_loss_func(
                    (probs,),
                    num_experts=num_experts,
                    top_k=1,
                    batch_size=batch_size,
                    selected_experts=(selected,),
                )
            )
        return sum(seq_terms) / len(seq_terms)

    def summary(self) -> dict[str, float]:
        if not self.router_records:
            return {}
        stats = {}
        for name, eids, E in self.router_records:
            counts = torch.bincount(eids, minlength=E).float()
            total = counts.sum()
            if total == 0:
                continue
            fracs = counts / total
            stats[f"routing/{name}_balance"] = (E * fracs.min()).item()
            stats[f"routing/{name}_utilization"] = (counts > 0).float().mean().item()
            entropy = -(fracs * (fracs + 1e-10).log()).sum()
            max_entropy = torch.tensor(E, dtype=torch.float32).log()
            stats[f"routing/{name}_entropy"] = (entropy / max_entropy).item()
        if self.branch_records:
            probs = torch.cat([p.reshape(-1, 2) for p in self.branch_records], dim=0)
            mean_probs = probs.mean(dim=0)
            stats["routing/branch_attn_frac"] = mean_probs[0].item()
            stats["routing/branch_mlp_frac"] = mean_probs[1].item()
        return stats


class BranchRouter(nn.Module):
    """Binary argmax router: choose attention or MLP per token."""

    def __init__(self, hidden_size: int, exploration_rate: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)
        self.exploration_rate = exploration_rate
        self.last_probs = None
        self.last_selected = None

    def forward(self, hidden_states: Tensor):
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = self.gate(hidden_states.float())
            probs = F.softmax(logits, dim=-1)
            choice_scores = probs.float()
            if self.training and self.exploration_rate > 0.0:
                explore = torch.rand(choice_scores.shape[:-1], device=choice_scores.device) < self.exploration_rate
                if explore.any():
                    choice_scores = choice_scores.clone()
                    choice_scores[explore] = torch.rand_like(choice_scores[explore])
            choice = choice_scores.argmax(dim=-1)
        attn_mask = (choice == 0).unsqueeze(-1)
        mlp_mask = ~attn_mask
        probs_out = probs.to(hidden_states.dtype)
        w_attn = probs_out[..., 0:1] * attn_mask
        w_mlp = probs_out[..., 1:2] * mlp_mask
        self.last_probs = probs_out
        self.last_selected = choice.unsqueeze(-1).detach()
        return w_attn, w_mlp, attn_mask, mlp_mask


# ---------------------------------------------------------------------------
# DeepSeek router factory

def _make_router(input_dim: int, num_experts: int, exploration_rate: float = 0.02) -> DeepSeekRouter:
    """Create a DeepSeek-style sigmoid + expert-bias router."""
    cfg = SimpleNamespace(
        hidden_size=input_dim,
        num_local_experts=num_experts,
        num_experts=num_experts,
        num_experts_per_tok=1,  # top-1
        norm_topk_prob=True,
        router_exploration_rate=exploration_rate,
        topk_scaling_factor=None,
        num_groups=None,
        group_topk=None,
    )
    return DeepSeekRouter(cfg)


# ---------------------------------------------------------------------------
# Per-head-slot top-1 routing + grouped GEMM projection

def _route_top1(router: DeepSeekRouter, x: Tensor, stats: RoutingStats | None = None,
                router_name: str = ""):
    """Top-1 DeepSeek routing with optional stats collection."""
    probs, weights, indices = router(x)  # probs: (N, E), weights: (N, 1), indices: (N, 1)
    expert_ids = indices.squeeze(-1)  # (N,)
    expert_weights = weights.squeeze(-1)  # (N,)
    if stats is not None:
        stats.add(router_name, expert_ids, probs, probs.shape[-1])
    return expert_ids, expert_weights


def _grouped_project(x: Tensor, weight_bank: Tensor, expert_ids: Tensor, weights: Tensor):
    """Sort by expert, Triton grouped GEMM, weight, unsort."""
    sort_order = expert_ids.argsort()
    sorted_expert = expert_ids[sort_order]
    sorted_inputs = fp32_index_select(x, 0, sort_order).contiguous()
    unique_experts, counts = sorted_expert.unique_consecutive(return_counts=True)

    proj = triton_grouped_gemm(sorted_inputs, weight_bank.to(x.dtype), unique_experts, counts)
    proj = proj * weights[sort_order].unsqueeze(-1).to(proj.dtype)

    out = proj.new_zeros(x.shape[0], proj.shape[-1])
    return fp32_index_put(out, sort_order, proj)


def _route_and_project_heads(routers: nn.ModuleList, x: Tensor, weight_bank: Tensor,
                             stats: RoutingStats | None = None,
                             name_prefix: str = "") -> Tensor:
    """Per-head-slot top-1 routing + projection."""
    H = len(routers)
    results = []
    for h in range(H):
        eid, ew = _route_top1(routers[h], x, stats, f"{name_prefix}_h{h}")
        proj = _grouped_project(x, weight_bank, eid, ew)
        proj = norm(proj)
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
    def __init__(self, num_experts: int, dim: int, num_heads: int, head_dim: int, mode: str,
                 exploration_rate: float = 0.02):
        super().__init__()
        self.mode = mode
        self.num_experts = num_experts
        E, D, HD, H = num_experts, dim, head_dim, num_heads

        if mode == "per_head_fully_independent":
            self.q_routers = nn.ModuleList([_make_router(D, E, exploration_rate) for _ in range(H)])
            self.k_routers = nn.ModuleList([_make_router(D, E, exploration_rate) for _ in range(H)])
            self.v_routers = nn.ModuleList([_make_router(D, E, exploration_rate) for _ in range(H)])
            self.o_routers = nn.ModuleList([_make_router(HD, E, exploration_rate) for _ in range(H)])
            self.q_proj = nn.Parameter(torch.empty(E, D, HD))
            self.k_proj = nn.Parameter(torch.empty(E, D, HD))
            self.v_proj = nn.Parameter(torch.empty(E, D, HD))
            self.o_proj = nn.Parameter(torch.empty(E, HD, D))
        elif mode == "per_head_precompute_kv":
            self.routers = nn.ModuleList([_make_router(D, E, exploration_rate) for _ in range(H)])
            self.q_proj = nn.Parameter(torch.empty(E, D, HD))
            self.k_proj = nn.Parameter(torch.empty(E, D, HD))
            self.v_proj = nn.Parameter(torch.empty(E, D, HD))
            self.o_proj = nn.Parameter(torch.empty(E, HD, D))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self._init_weights(dim)

    def _init_weights(self, dim):
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.uniform_(proj, -bound, bound)
        nn.init.zeros_(self.o_proj)

    def get_all_routers(self) -> list[DeepSeekRouter]:
        """Return all DeepSeek routers for bias updates."""
        routers = []
        for attr in ["q_routers", "k_routers", "v_routers", "o_routers", "routers"]:
            module_list = getattr(self, attr, None)
            if module_list is not None:
                routers.extend(module_list)
        return routers


# ---------------------------------------------------------------------------
# MLP Expert Bank

class MLPExpertBank(nn.Module):
    def __init__(self, num_experts: int, dim: int, exploration_rate: float = 0.02):
        super().__init__()
        hdim = 4 * dim
        self.num_experts = num_experts
        self.router = _make_router(dim, num_experts, exploration_rate)
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

    def get_all_routers(self) -> list[DeepSeekRouter]:
        return [self.router]


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

        q = _route_and_project_heads(bank.q_routers, flat, bank.q_proj, stats, f"q_{dp}")
        k = _route_and_project_heads(bank.k_routers, flat, bank.k_proj, stats, f"k_{dp}")
        v = _route_and_project_heads(bank.v_routers, flat, bank.v_proj, stats, f"v_{dp}")

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

        head_eids, head_ews = [], []
        for h in range(H):
            eid, ew = _route_top1(bank.routers[h], flat, stats, f"qkvo_{dp}_h{h}")
            head_eids.append(eid)
            head_ews.append(ew)

        q_parts = []
        for h in range(H):
            proj = _grouped_project(flat, bank.q_proj, head_eids[h], head_ews[h])
            q_parts.append(norm(proj))
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
                k_normed = norm(k_e)
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

        y_flat = y.reshape(B * T, H, HD)
        o_parts = []
        for h in range(H):
            o_h = _grouped_project(y_flat[:, h].contiguous(), bank.o_proj, head_eids[h], head_ews[h])
            o_parts.append(o_h.float())
        out = sum(o_parts)
        return out.to(x.dtype).view(B, T, D)


# ---------------------------------------------------------------------------
# Block and full model

class BranchRoutedDepthStep(nn.Module):
    def __init__(self, attn_bank: AttentionExpertBank, mlp_bank: MLPExpertBank,
                 num_heads: int, head_dim: int, max_seq_len: int, depth_idx: int, mode: str):
        super().__init__()
        self.depth_idx = depth_idx
        if mode == "per_head_fully_independent":
            self.attn = RoutedAttentionFullyIndependent(attn_bank, num_heads, head_dim, max_seq_len)
        elif mode == "per_head_precompute_kv":
            self.attn = RoutedAttentionPrecomputeKV(attn_bank, num_heads, head_dim, max_seq_len)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.mlp_bank = mlp_bank

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor,
                sa_lambdas: Tensor, branch_router: BranchRouter, stats: RoutingStats | None = None):
        B, T, D = x.shape
        x = lambdas[0] * x + lambdas[1] * x0
        x_norm = norm(x)
        w_attn, w_mlp, _, _ = branch_router(x)
        if stats is not None and branch_router.last_probs is not None:
            stats.add_branch(branch_router.last_probs)
        attn_out = self.attn(x_norm, ve, sa_lambdas, stats, self.depth_idx)
        mlp_out = self.mlp_bank(x_norm.reshape(-1, D), stats, self.depth_idx).view(B, T, D)
        x = x + w_attn * attn_out + w_mlp * mlp_out
        return x


class SpeedrunMoEGPT(nn.Module):
    """Speedrun GPT with shared attention + MLP expert banks.

    After forward(), access:
      self._aux_loss: scalar tensor (switch load balancing loss)
      self._routing_stats: dict of routing metrics for wandb logging
    """

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int,
                 head_dim: int, max_seq_len: int, mode: str,
                 num_attn_experts: int = 66, num_mlp_experts: int = 12,
                 exploration_rate: float = 0.02):
        super().__init__()
        self.num_blocks = num_layers
        self.num_depths = num_layers * 2
        self.model_dim = model_dim
        self.mode = mode
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.branch_router = BranchRouter(model_dim, exploration_rate=exploration_rate)

        self.attn_bank = AttentionExpertBank(num_attn_experts, model_dim, num_heads, head_dim, mode, exploration_rate)
        self.mlp_bank = MLPExpertBank(num_mlp_experts, model_dim, exploration_rate)

        self.blocks = nn.ModuleList([
            BranchRoutedDepthStep(self.attn_bank, self.mlp_bank, num_heads, head_dim, max_seq_len, i, mode)
            for i in range(self.num_depths)
        ])

        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8,
                                     x_s=(model_dim ** 0.5) / 448, w_s=2 ** -9, grad_s=1 / 448)
        self.lm_head.weight.detach().zero_()

        assert self.num_depths % 2 == 0
        pad = (-self.num_depths * 5) % max(dist.get_world_size(), 1)
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(self.num_depths),
            *[torch.tensor([1.0, 0.0]) for _ in range(self.num_depths)],
            *[torch.tensor([0.5, 0.5]) for _ in range(self.num_depths)],
            torch.ones(max(pad, 0)),
        ]))

        for param in self.embed.parameters():
            param.lr_mul = 75.0
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

        self._aux_loss = torch.tensor(0.0)
        self._seq_aux_loss = torch.tensor(0.0)
        self._routing_stats = {}

    def get_all_routers(self) -> list[DeepSeekRouter]:
        """Return all DeepSeek routers for bias updates."""
        return self.attn_bank.get_all_routers() + self.mlp_bank.get_all_routers()

    def forward(self, input_ids: Tensor, labels: Tensor):
        B, T = input_ids.shape
        stats = RoutingStats() if self.training else None

        ve = [vemb(input_ids) for vemb in self.value_embeds]
        old_schedule = [ve[0], ve[1], ve[2]] + [None] * (self.num_blocks - 6) + [ve[0], ve[1], ve[2]]
        ve = [v for item in old_schedule for v in (item, item)]

        x = x0 = norm(self.embed(input_ids))

        skip_connections = []
        skip_weights = self.scalars[:self.num_depths // 2]
        lambdas = self.scalars[1 * self.num_depths:3 * self.num_depths].view(-1, 2)
        sa_lambdas = self.scalars[3 * self.num_depths:5 * self.num_depths].view(-1, 2)
        n = self.num_depths // 2

        for i in range(self.num_depths):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], self.branch_router, stats)
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               reduction="sum" if self.training else "mean")

        if stats is not None:
            self._aux_loss = stats.total_aux_loss()
            self._seq_aux_loss = stats.total_seq_aux_loss(batch_size=B)
            self._routing_stats = stats.summary()
        else:
            self._aux_loss = torch.tensor(0.0, device=loss.device)
            self._seq_aux_loss = torch.tensor(0.0, device=loss.device)
            self._routing_stats = {}

        return loss
