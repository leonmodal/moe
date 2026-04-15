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

import math
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

AUTO_QUERY_SPARSE_THRESHOLDS = {
    "per_head_fully_independent": 0.75,
    "per_head_precompute_kv": 0.25,
}


# ---------------------------------------------------------------------------
# Routing stats collector

@dataclass
class RoutingStats:
    """Accumulates routing statistics during a forward pass."""
    aux_losses: list[Tensor] = field(default_factory=list)
    router_records: list[tuple[str, Tensor, int]] = field(default_factory=list)
    branch_records: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    seq_router_records: list[tuple[Tensor, Tensor, int, Tensor | None]] = field(default_factory=list)

    def add(
        self,
        name: str,
        expert_ids: Tensor,
        probs: Tensor,
        num_experts: int,
        token_mask: Tensor | None = None,
    ):
        self.aux_losses.append(switch_load_balancing_loss(expert_ids, probs, num_experts))
        self.router_records.append((name, expert_ids.detach(), num_experts))
        probs_for_seq = probs
        selected_for_seq = expert_ids.unsqueeze(-1)
        token_mask_for_seq = None
        if token_mask is not None:
            flat_mask = token_mask.reshape(-1).bool()
            if probs.shape[0] != flat_mask.numel():
                selected_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
                probs_for_seq = probs.new_zeros(flat_mask.numel(), num_experts)
                probs_for_seq = fp32_index_put(probs_for_seq, selected_idx, probs)
                selected_for_seq = expert_ids.new_zeros(flat_mask.numel(), 1)
                selected_for_seq = fp32_index_put(selected_for_seq, selected_idx, expert_ids.unsqueeze(-1))
                token_mask_for_seq = flat_mask
        self.seq_router_records.append((probs_for_seq, selected_for_seq, num_experts, token_mask_for_seq))

    def total_aux_loss(self) -> Tensor:
        if not self.aux_losses:
            return torch.tensor(0.0)
        return sum(self.aux_losses) / len(self.aux_losses)

    def add_branch(self, probs: Tensor, selected: Tensor):
        """Record branch routing.

        probs: soft scores (B, T, 2) or (B*T, 2)
        selected: hard decisions (B, T, 1) or (B*T, 1), 0=attn, 1=mlp
        """
        self.branch_records.append((probs.detach(), selected.detach()))

    def total_seq_aux_loss(self, batch_size: int) -> Tensor:
        if not self.seq_router_records:
            return torch.tensor(0.0)
        seq_terms: list[Tensor] = []
        for probs, selected, num_experts, token_mask in self.seq_router_records:
            seq_terms.append(
                seq_load_balancing_loss_func(
                    (probs,),
                    num_experts=num_experts,
                    top_k=1,
                    batch_size=batch_size,
                    selected_experts=(selected,),
                    token_masks=((token_mask,) if token_mask is not None else None),
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
            all_selected = torch.cat([s.reshape(-1) for _, s in self.branch_records], dim=0)
            n = all_selected.numel()
            stats["routing/branch_attn_frac"] = (all_selected == 0).sum().item() / max(n, 1)
            stats["routing/branch_mlp_frac"] = (all_selected == 1).sum().item() / max(n, 1)
        return stats

    def routing_snapshot(self) -> dict:
        """Build a snapshot dict compatible with plot_routing_snapshot()."""
        import re
        snapshot: dict = {}

        # Branch data — use hard routing decisions for fractions
        if self.branch_records:
            layers = {}
            total_attn = 0
            total_mlp = 0
            total_tokens = 0
            total_attn_weight_sum = 0.0
            total_mlp_weight_sum = 0.0
            total_attn_count = 0
            total_mlp_count = 0
            for i, (probs, selected) in enumerate(self.branch_records):
                sel_flat = selected.reshape(-1)  # 0=attn, 1=mlp
                p_flat = probs.reshape(-1, 2)
                n = sel_flat.numel()
                attn_mask = (sel_flat == 0)
                mlp_mask = (sel_flat == 1)
                n_attn = attn_mask.sum().item()
                n_mlp = mlp_mask.sum().item()
                attn_f = n_attn / max(n, 1)
                mlp_f = n_mlp / max(n, 1)
                # Mean sigmoid weight for tokens that chose each branch
                mean_attn_w = p_flat[attn_mask, 0].mean().item() if n_attn > 0 else 0.0
                mean_mlp_w = p_flat[mlp_mask, 1].mean().item() if n_mlp > 0 else 0.0
                layers[str(i)] = {
                    "attn_frac": attn_f,
                    "mlp_frac": mlp_f,
                    "attn_to_mlp_ratio": attn_f / max(mlp_f, 1e-10),
                    "mean_attn_weight": mean_attn_w,
                    "mean_mlp_weight": mean_mlp_w,
                }
                total_attn += n_attn
                total_mlp += n_mlp
                total_tokens += n
                total_attn_weight_sum += mean_attn_w * n_attn
                total_mlp_weight_sum += mean_mlp_w * n_mlp
                total_attn_count += n_attn
                total_mlp_count += n_mlp
            total_attn_f = total_attn / max(total_tokens, 1)
            total_mlp_f = total_mlp / max(total_tokens, 1)
            snapshot["branch"] = {
                "layers": layers,
                "total": {
                    "attn_frac": total_attn_f,
                    "mlp_frac": total_mlp_f,
                    "attn_to_mlp_ratio": total_attn_f / max(total_mlp_f, 1e-10),
                    "mean_attn_weight": total_attn_weight_sum / max(total_attn_count, 1),
                    "mean_mlp_weight": total_mlp_weight_sum / max(total_mlp_count, 1),
                },
            }

        # Expert histograms grouped by family
        families: dict[str, dict[int, dict]] = {}
        global_pools: dict[str, list] = {}
        for name, eids, E in self.router_records:
            m = re.match(r"^(\w+?)_d(\d+)(?:_h(\d+))?$", name)
            if not m:
                continue
            prefix, depth_str, head_str = m.group(1), m.group(2), m.group(3)
            depth = int(depth_str)
            family = f"{prefix}_h{head_str}" if head_str is not None else prefix
            counts = torch.bincount(eids, minlength=E).float()
            total = counts.sum()
            fracs = (counts / total).tolist() if total > 0 else [0.0] * E
            families.setdefault(family, {})[depth] = {"token_fracs": fracs}
            global_pools.setdefault(family, torch.zeros(E, device=counts.device))
            global_pools[family] += counts

        # MLP histograms
        if "mlp" in families:
            mlp_pool = global_pools["mlp"]
            mlp_total = mlp_pool.sum()
            snapshot["layers"] = families["mlp"]
            snapshot["global_pool"] = {
                "token_fracs": (mlp_pool / max(mlp_total.item(), 1e-10)).tolist()
            }

        # Attention histograms
        attn_families = {k: v for k, v in families.items() if k != "mlp"}
        if attn_families:
            snapshot["attention"] = {}
            for family, depth_map in attn_families.items():
                pool = global_pools[family]
                pool_total = pool.sum()
                snapshot["attention"][family] = {
                    "layers": depth_map,
                    "global_pool": {
                        "token_fracs": (pool / max(pool_total.item(), 1e-10)).tolist()
                    },
                }

        # Global per-projection-type pools (aggregate across all heads)
        proj_pools: dict[str, torch.Tensor] = {}
        for name, eids, E in self.router_records:
            m = re.match(r"^(\w+?)_d(\d+)(?:_h(\d+))?$", name)
            if not m:
                continue
            prefix = m.group(1)  # "q", "k", "v", "o", "mlp", "qkvo"
            if prefix not in proj_pools:
                proj_pools[prefix] = torch.zeros(E, device=eids.device)
            proj_pools[prefix] += torch.bincount(eids, minlength=E).float()
        snapshot["global_projection_pools"] = {}
        for prefix, counts in proj_pools.items():
            total = counts.sum()
            snapshot["global_projection_pools"][prefix] = {
                "token_fracs": (counts / max(total.item(), 1e-10)).tolist()
            }

        return snapshot

    def expert_heatmap_data(self) -> dict[str, list[list[float]]]:
        """Return per-router-family heatmap data: {family: [[fracs per expert] per layer]}.

        Router names like ``mlp_d0``, ``qkvo_d0_h0`` are grouped by prefix
        (``mlp``, ``qkvo_h0``, etc.) with one row per depth index, suitable for
        plotting as a 2-D heatmap (y=depth, x=expert).
        """
        import re
        if not self.router_records:
            return {}

        # Group records by family
        # mlp_d{depth} -> family="mlp"
        # qkvo_d{depth}_h{head} -> family="qkvo_h{head}"
        families: dict[str, dict[int, torch.Tensor]] = {}
        for name, eids, E in self.router_records:
            m = re.match(r"^(\w+?)_d(\d+)(?:_h(\d+))?$", name)
            if not m:
                continue
            prefix, depth_str, head_str = m.group(1), m.group(2), m.group(3)
            depth = int(depth_str)
            if head_str is not None:
                family = f"{prefix}_h{head_str}"
            else:
                family = prefix
            counts = torch.bincount(eids, minlength=E).float()
            total = counts.sum()
            fracs = (counts / total) if total > 0 else counts
            families.setdefault(family, {})[depth] = fracs

        result: dict[str, list[list[float]]] = {}
        for family, depth_map in families.items():
            depths = sorted(depth_map.keys())
            result[family] = [depth_map[d].cpu().tolist() for d in depths]
        return result


class BranchRouter(nn.Module):
    """Binary router: choose attention or MLP per token.

    Args:
        use_sampling: If True, sample from the softmax distribution during training.
        use_seq_level: If True, mean-pool tokens and make one decision per sequence.
        use_deepseek_style: If True, use sigmoid + bias (DeepSeek V3 style) instead of softmax.
    """

    def __init__(self, hidden_size: int, exploration_rate: float = 0.0,
                 use_sampling: bool = False, use_seq_level: bool = False,
                 use_deepseek_style: bool = False):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 2, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        self.exploration_rate = exploration_rate
        self.use_sampling = use_sampling
        self.use_seq_level = use_seq_level
        self.use_deepseek_style = use_deepseek_style
        self.last_probs = None
        self.last_selected = None
        if use_deepseek_style:
            self.register_buffer("branch_bias", torch.zeros(2))
            self.register_buffer("local_counts", torch.zeros(2), persistent=False)

    def forward(self, hidden_states: Tensor):
        # hidden_states: (B, T, D) or (B*T, D)
        device_type = hidden_states.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            if self.use_seq_level and hidden_states.ndim == 3:
                B, T, D = hidden_states.shape
                seq_repr = hidden_states.float().mean(dim=1)  # (B, D)
                logits = self.gate(seq_repr)  # (B, 2)
            else:
                B, T = None, None
                logits = self.gate(hidden_states.float())

            if self.use_deepseek_style:
                # Sigmoid + bias selection, unbiased weight
                scores = torch.sigmoid(logits)  # (*, 2)
                biased = scores + self.branch_bias
                if self.training and self.use_sampling:
                    flat = scores.view(-1, 2)
                    choice = torch.multinomial(flat, 1).view(scores.shape[:-1])
                else:
                    choice = biased.argmax(dim=-1)
                # Gather unbiased score for selected branch
                probs = scores
            else:
                # Softmax style
                probs = F.softmax(logits, dim=-1)
                if self.training and self.use_sampling:
                    flat = probs.view(-1, 2)
                    choice = torch.multinomial(flat, 1).view(probs.shape[:-1])
                else:
                    choice_scores = probs.float()
                    if self.training and self.exploration_rate > 0.0:
                        explore = torch.rand(choice_scores.shape[:-1], device=choice_scores.device) < self.exploration_rate
                        if explore.any():
                            choice_scores = choice_scores.clone()
                            choice_scores[explore] = torch.rand_like(choice_scores[explore])
                    choice = choice_scores.argmax(dim=-1)

            # Broadcast seq-level decision to all tokens
            if self.use_seq_level and B is not None:
                choice = choice.unsqueeze(1).expand(B, T)
                probs = probs.unsqueeze(1).expand(B, T, 2)

        # Track counts for bias update (DeepSeek style)
        if self.use_deepseek_style and self.training:
            with torch.no_grad():
                flat_choice = choice.reshape(-1)
                counts = torch.bincount(flat_choice, minlength=2).float()
                self.local_counts += counts

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
    router = DeepSeekRouter(cfg)
    # Initialize router weights with small random values so tokens
    # get diverse expert preferences from the start (instead of all-zero
    # which collapses to a single expert via tiebreaking).
    nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))
    return router


# ---------------------------------------------------------------------------
# Per-head-slot top-1 routing + grouped GEMM projection

def _route_top1(router: DeepSeekRouter, x: Tensor, stats: RoutingStats | None = None,
                router_name: str = "", token_mask: Tensor | None = None):
    """Top-1 DeepSeek routing with optional stats collection."""
    probs, weights, indices = router(x)  # probs: (N, E), weights: (N, 1), indices: (N, 1)
    expert_ids = indices.squeeze(-1)  # (N,)
    expert_weights = weights.squeeze(-1)  # (N,)
    if stats is not None:
        stats.add(router_name, expert_ids, probs, probs.shape[-1], token_mask=token_mask)
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
                             name_prefix: str = "",
                             token_mask: Tensor | None = None) -> Tensor:
    """Per-head-slot top-1 routing + projection."""
    H = len(routers)
    results = []
    for h in range(H):
        eid, ew = _route_top1(routers[h], x, stats, f"{name_prefix}_h{h}", token_mask=token_mask)
        proj = _grouped_project(x, weight_bank, eid, ew)
        proj = norm(proj)
        results.append(proj)
    return torch.stack(results, dim=1)


def _seq_route_and_project_heads(routers: nn.ModuleList, x: Tensor, weight_bank: Tensor,
                                  B: int, T: int,
                                  stats: RoutingStats | None = None,
                                  name_prefix: str = "",
                                  token_mask: Tensor | None = None) -> Tensor:
    """Seq-level routing: mean pool → one expert per head per sequence, apply to all tokens."""
    H = len(routers)
    # x: (B*T, D) — reshape to (B, T, D) for mean pooling
    x_3d = x.view(B, T, -1)
    seq_repr = x_3d.mean(dim=1)  # (B, D)
    results = []
    for h in range(H):
        # Route on seq representation
        eid, ew = _route_top1(routers[h], seq_repr, stats, f"{name_prefix}_h{h}", token_mask=token_mask)
        # Expand to all tokens: (B,) -> (B*T,)
        eid_expanded = eid.unsqueeze(1).expand(B, T).reshape(B * T)
        ew_expanded = ew.unsqueeze(1).expand(B, T).reshape(B * T)
        proj = _grouped_project(x, weight_bank, eid_expanded, ew_expanded)
        proj = norm(proj)
        results.append(proj)
    return torch.stack(results, dim=1)


def _seq_route_and_project_reduce(routers: nn.ModuleList, x: Tensor, weight_bank: Tensor,
                                   B: int, T: int,
                                   stats: RoutingStats | None = None,
                                   name_prefix: str = "",
                                   token_mask: Tensor | None = None) -> Tensor:
    """Seq-level routing for output projection, sum across heads."""
    N, H, _ = x.shape
    dim = weight_bank.shape[2]
    # Use the attention output mean for routing
    x_mean = x.view(B, T, H, -1).mean(dim=1)  # (B, H, HD)
    token_out = torch.zeros(N, dim, device=x.device, dtype=torch.float32)
    for h in range(H):
        x_h = x[:, h].contiguous()
        seq_repr_h = x_mean[:, h].contiguous()  # (B, HD)
        eid, ew = _route_top1(routers[h], seq_repr_h, stats, f"{name_prefix}_h{h}", token_mask=token_mask)
        eid_expanded = eid.unsqueeze(1).expand(B, T).reshape(B * T)
        ew_expanded = ew.unsqueeze(1).expand(B, T).reshape(B * T)
        token_out += _grouped_project(x_h, weight_bank, eid_expanded, ew_expanded).float()
    return token_out


def _route_and_project_reduce(routers: nn.ModuleList, x: Tensor, weight_bank: Tensor,
                              stats: RoutingStats | None = None,
                              name_prefix: str = "",
                              token_mask: Tensor | None = None) -> Tensor:
    """Per-head-slot top-1 routing on per-head inputs, sum across heads."""
    N, H, _ = x.shape
    dim = weight_bank.shape[2]
    token_out = torch.zeros(N, dim, device=x.device, dtype=torch.float32)
    for h in range(H):
        x_h = x[:, h].contiguous()
        eid, ew = _route_top1(routers[h], x_h, stats, f"{name_prefix}_h{h}", token_mask=token_mask)
        proj = _grouped_project(x_h, weight_bank, eid, ew)
        token_out = token_out + proj.float()
    return token_out.to(x.dtype)


def _apply_rotary_at_positions(rotary: Rotary, x_QHD: Tensor, positions: Tensor) -> Tensor:
    """Apply speedrun RoPE to query vectors at explicit sequence positions."""
    cos = fp32_index_select(rotary.cos, 0, positions).unsqueeze(1)
    sin = fp32_index_select(rotary.sin, 0, positions).unsqueeze(1)
    x1, x2 = x_QHD.to(dtype=torch.float32).chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1).type_as(x_QHD)


def _build_query_position_mask(query_positions: Tensor, key_length: int, *, device, dtype) -> Tensor:
    key_positions = torch.arange(key_length, device=device)
    future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros((query_positions.shape[0], key_length), device=device, dtype=dtype)
    mask.masked_fill_(future_mask, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _should_use_sparse_query_path(token_mask: Tensor, threshold: float) -> bool:
    flat_mask = token_mask.reshape(-1).bool()
    if not flat_mask.any():
        return True
    if flat_mask.all():
        return False
    return flat_mask.float().mean().item() < threshold


# ---------------------------------------------------------------------------
# Attention Expert Bank

class AttentionExpertBank(nn.Module):
    def __init__(self, num_experts: int, dim: int, num_heads: int, head_dim: int, mode: str,
                 exploration_rate: float = 0.02):
        super().__init__()
        self.mode = mode
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim
        self.exploration_rate = exploration_rate
        E, D, HD, H = num_experts, dim, head_dim, num_heads

        # Plain dict for per-depth routers (set by BranchRoutedDepthStep._install_routers)
        self._active_routers = {}

        # Shared expert projection weights (used by all depths)
        self.q_proj = nn.Parameter(torch.empty(E, D, HD))
        self.k_proj = nn.Parameter(torch.empty(E, D, HD))
        self.v_proj = nn.Parameter(torch.empty(E, D, HD))
        self.o_proj = nn.Parameter(torch.empty(E, HD, D))
        self._init_weights(dim)

    def _init_weights(self, dim):
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.uniform_(proj, -bound, bound)
        nn.init.zeros_(self.o_proj)

    def make_routers(self) -> nn.ModuleDict:
        """Create a fresh set of routers for one depth step."""
        E, D, HD, H = self.num_experts, self.dim, self.head_dim, self.num_heads
        er = self.exploration_rate
        if self.mode == "per_head_fully_independent":
            return nn.ModuleDict({
                "q_routers": nn.ModuleList([_make_router(D, E, er) for _ in range(H)]),
                "k_routers": nn.ModuleList([_make_router(D, E, er) for _ in range(H)]),
                "v_routers": nn.ModuleList([_make_router(D, E, er) for _ in range(H)]),
                "o_routers": nn.ModuleList([_make_router(HD, E, er) for _ in range(H)]),
            })
        elif self.mode == "per_head_precompute_kv":
            return nn.ModuleDict({
                "routers": nn.ModuleList([_make_router(D, E, er) for _ in range(H)]),
            })
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ---------------------------------------------------------------------------
# MLP Expert Bank

class MLPExpertBank(nn.Module):
    def __init__(self, num_experts: int, dim: int, exploration_rate: float = 0.02):
        super().__init__()
        hdim = 4 * dim
        self.num_experts = num_experts
        self.dim = dim
        self.exploration_rate = exploration_rate
        self.c_fc = nn.Parameter(torch.empty(num_experts, dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(num_experts, hdim, dim))
        self._init_weights(dim)

    def _init_weights(self, dim):
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        nn.init.uniform_(self.c_fc, -bound, bound)
        nn.init.zeros_(self.c_proj)

    def make_router(self) -> DeepSeekRouter:
        """Create a fresh router for one depth step."""
        return _make_router(self.dim, self.num_experts, self.exploration_rate)

    def forward(
        self,
        router: DeepSeekRouter,
        x: Tensor,
        stats: RoutingStats | None = None,
        depth_idx: int = 0,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        if token_mask is None:
            selected_idx = None
            x_selected = x
        else:
            flat_mask = token_mask.reshape(-1).bool()
            if not flat_mask.any():
                return x.new_zeros(x.shape[0], x.shape[1])
            if flat_mask.all():
                selected_idx = None
                x_selected = x
            else:
                selected_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
                x_selected = fp32_index_select(x, 0, selected_idx)

        seq_token_mask = None if selected_idx is None else token_mask
        eid, ew = _route_top1(router, x_selected, stats, f"mlp_d{depth_idx}", token_mask=seq_token_mask)
        h = _grouped_project(x_selected, self.c_fc, eid, torch.ones_like(ew))
        h = F.relu(h).square()
        out = _grouped_project(h, self.c_proj, eid, ew)
        if selected_idx is None:
            return out
        dense_out = x.new_zeros(x.shape[0], x.shape[1])
        return fp32_index_put(dense_out, selected_idx, out)


# ---------------------------------------------------------------------------
# Routed attention modules

class RoutedAttentionFullyIndependent(nn.Module):
    def __init__(self, bank: AttentionExpertBank, num_heads: int, head_dim: int, max_seq_len: int,
                 attn_routing_level: str = "token"):
        super().__init__()
        self.bank = bank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_routing_level = attn_routing_level  # "token", "seq", or "seq_qk"
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12
        self.attn_gate_dim = 12
        self.attn_gate = CastedLinear(self.attn_gate_dim, num_heads)
        self.attn_gate.weight.detach().zero_()
        self.query_sparse_fraction_threshold = AUTO_QUERY_SPARSE_THRESHOLDS["per_head_fully_independent"]

    def _forward_sparse_queries(
        self,
        x: Tensor,
        flat: Tensor,
        token_mask: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        stats: RoutingStats | None,
        dp: str,
    ) -> Tensor:
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat_mask = token_mask.reshape(-1).bool()
        selected_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        flat_selected = fp32_index_select(flat, 0, selected_idx)

        q_selected = _route_and_project_heads(
            bank._active_routers["q_routers"],
            flat_selected,
            bank.q_proj,
            stats,
            f"q_{dp}",
            token_mask=flat_mask,
        )
        k = _route_and_project_heads(bank._active_routers["k_routers"], flat, bank.k_proj, stats, f"k_{dp}")
        v = _route_and_project_heads(bank._active_routers["v_routers"], flat, bank.v_proj, stats, f"v_{dp}")

        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.reshape(B * T, H, HD).to(v.dtype)
        else:
            v = lambdas[0] * v

        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)
        k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

        attn_heads = x.new_zeros(B, H, T, HD)
        token_mask_2d = token_mask.squeeze(-1).bool()
        offset = 0
        for b in range(B):
            pos = token_mask_2d[b].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            q_b = q_selected[offset:offset + pos.numel()]
            q_b = _apply_rotary_at_positions(self.rotary, q_b, pos).transpose(0, 1).unsqueeze(0)
            attn_mask = _build_query_position_mask(pos, T, device=q_b.device, dtype=q_b.dtype)
            attn_b = F.scaled_dot_product_attention(
                q_b,
                k[b:b + 1],
                v[b:b + 1],
                attn_mask=attn_mask,
                scale=self.attn_scale,
            )
            attn_heads[b, :, pos, :] = attn_b.squeeze(0).to(attn_heads.dtype)
            offset += pos.numel()

        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim]))
        attn_selected = fp32_index_select(
            attn_heads.transpose(1, 2).reshape(B * T, H, HD),
            0,
            selected_idx,
        )
        gate_selected = fp32_index_select(gate.reshape(B * T, H), 0, selected_idx).unsqueeze(-1)
        y_selected = attn_selected * gate_selected.to(attn_selected.dtype)
        out_selected = _route_and_project_reduce(
            bank._active_routers["o_routers"],
            y_selected,
            bank.o_proj,
            stats,
            f"o_{dp}",
            token_mask=flat_mask,
        )
        out = x.new_zeros(B * T, D)
        out = fp32_index_put(out, selected_idx, out_selected)
        return out.view(B, T, D)

    def _forward_dense_queries(
        self,
        x: Tensor,
        flat: Tensor,
        token_mask: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        stats: RoutingStats | None,
        dp: str,
    ) -> Tensor:
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat_mask = token_mask.reshape(-1).bool()
        selected_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        flat_selected = fp32_index_select(flat, 0, selected_idx)

        q_selected = _route_and_project_heads(
            bank._active_routers["q_routers"],
            flat_selected,
            bank.q_proj,
            stats,
            f"q_{dp}",
            token_mask=flat_mask,
        )
        k = _route_and_project_heads(bank._active_routers["k_routers"], flat, bank.k_proj, stats, f"k_{dp}")
        v = _route_and_project_heads(bank._active_routers["v_routers"], flat, bank.v_proj, stats, f"v_{dp}")

        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.reshape(B * T, H, HD).to(v.dtype)
        else:
            v = lambdas[0] * v

        q_flat = flat.new_zeros(B * T, H, HD)
        q_flat = fp32_index_put(q_flat, selected_idx, q_selected)

        q = q_flat.view(B, T, H, HD).transpose(1, 2)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)
        q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
        k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim]))
        y = y.transpose(1, 2)
        y_selected = fp32_index_select((y * gate.unsqueeze(-1)).reshape(B * T, H, HD), 0, selected_idx)
        out_selected = _route_and_project_reduce(
            bank._active_routers["o_routers"],
            y_selected,
            bank.o_proj,
            stats,
            f"o_{dp}",
            token_mask=flat_mask,
        )
        out = x.new_zeros(B * T, D)
        out = fp32_index_put(out, selected_idx, out_selected)
        return out.view(B, T, D)

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        stats: RoutingStats | None = None,
        depth_idx: int = 0,
        token_mask: Tensor | None = None,
    ):
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat = x.reshape(B * T, D)
        dp = f"d{depth_idx}"

        if token_mask is not None and self.attn_routing_level == "token":
            # Sparse/dense query optimization only for token-level routing
            # Seq-level routing needs full B*T tensors for correct mean pooling
            flat_mask = token_mask.reshape(-1).bool()
            if not flat_mask.any():
                return x.new_zeros(B, T, D)
            if not flat_mask.all():
                if _should_use_sparse_query_path(token_mask, self.query_sparse_fraction_threshold):
                    return self._forward_sparse_queries(x, flat, token_mask, ve, lambdas, stats, dp)
                return self._forward_dense_queries(x, flat, token_mask, ve, lambdas, stats, dp)

        # Select routing function based on attn_routing_level
        level = self.attn_routing_level
        if level == "seq":
            # All Q/K/V/O use seq-level routing
            q = _seq_route_and_project_heads(bank._active_routers["q_routers"], flat, bank.q_proj, B, T, stats, f"q_{dp}")
            k = _seq_route_and_project_heads(bank._active_routers["k_routers"], flat, bank.k_proj, B, T, stats, f"k_{dp}")
            v = _seq_route_and_project_heads(bank._active_routers["v_routers"], flat, bank.v_proj, B, T, stats, f"v_{dp}")
        elif level == "seq_qk":
            # Q/K use seq-level, V uses token-level
            q = _seq_route_and_project_heads(bank._active_routers["q_routers"], flat, bank.q_proj, B, T, stats, f"q_{dp}")
            k = _seq_route_and_project_heads(bank._active_routers["k_routers"], flat, bank.k_proj, B, T, stats, f"k_{dp}")
            v = _route_and_project_heads(bank._active_routers["v_routers"], flat, bank.v_proj, stats, f"v_{dp}")
        else:
            # Token-level (default)
            q = _route_and_project_heads(bank._active_routers["q_routers"], flat, bank.q_proj, stats, f"q_{dp}")
            k = _route_and_project_heads(bank._active_routers["k_routers"], flat, bank.k_proj, stats, f"k_{dp}")
            v = _route_and_project_heads(bank._active_routers["v_routers"], flat, bank.v_proj, stats, f"v_{dp}")

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
        if level == "seq":
            out = _seq_route_and_project_reduce(bank._active_routers["o_routers"], y_flat, bank.o_proj, B, T, stats, f"o_{dp}")
        else:
            # Token-level O for both "token" and "seq_qk"
            out = _route_and_project_reduce(bank._active_routers["o_routers"], y_flat, bank.o_proj, stats, f"o_{dp}")
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
        self.query_sparse_fraction_threshold = AUTO_QUERY_SPARSE_THRESHOLDS["per_head_precompute_kv"]

    def _route_selected_queries(
        self,
        flat: Tensor,
        token_mask: Tensor,
        stats: RoutingStats | None,
        dp: str,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor]:
        B, T, _ = token_mask.shape
        H = self.num_heads
        bank = self.bank
        flat_mask = token_mask.reshape(-1).bool()
        selected_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        flat_selected = fp32_index_select(flat, 0, selected_idx)

        head_eids, head_ews = [], []
        for h in range(H):
            eid, ew = _route_top1(
                bank._active_routers["routers"][h],
                flat_selected,
                stats,
                f"qkvo_{dp}_h{h}",
                token_mask=flat_mask,
            )
            head_eids.append(eid)
            head_ews.append(ew)

        q_parts = []
        for h in range(H):
            proj = _grouped_project(flat_selected, bank.q_proj, head_eids[h], head_ews[h])
            q_parts.append(norm(proj))
        q_selected = torch.stack(q_parts, dim=1)
        return selected_idx, head_eids, head_ews, q_selected

    def _forward_sparse_queries(
        self,
        x: Tensor,
        flat: Tensor,
        token_mask: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        stats: RoutingStats | None,
        dp: str,
    ) -> Tensor:
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        token_mask_2d = token_mask.squeeze(-1).bool()
        ve_heads = None if ve is None else ve.view(B, T, H, HD).transpose(1, 2)
        selected_idx, head_eids, head_ews, q_selected = self._route_selected_queries(flat, token_mask, stats, dp)
        attn_selected = q_selected.new_zeros(q_selected.shape[0], H, HD)

        batch_slices: list[tuple[int, Tensor, int, int]] = []
        offset = 0
        for b in range(B):
            pos = token_mask_2d[b].nonzero(as_tuple=False).squeeze(-1)
            q_count = pos.numel()
            batch_slices.append((b, pos, offset, offset + q_count))
            offset += q_count

        for h in range(H):
            eid = head_eids[h]
            ew = head_ews[h]
            q_h = q_selected[:, h:h + 1, :]
            for e in eid.unique().tolist():
                k_e = flat @ bank.k_proj[e].to(flat.dtype)
                k_normed = norm(k_e)
                v_e = flat @ bank.v_proj[e].to(flat.dtype)

                k_4d = self.rotary(k_normed.view(B, T, 1, HD))
                v_4d = v_e.view(B, T, 1, HD).transpose(1, 2)
                if ve_heads is not None:
                    v_4d = lambdas[0] * v_4d + lambdas[1] * ve_heads[:, h:h + 1].to(v_4d.dtype)
                else:
                    v_4d = lambdas[0] * v_4d

                for b, pos, start, end in batch_slices:
                    if pos.numel() == 0:
                        continue
                    eid_b = eid[start:end]
                    match = eid_b == e
                    if not match.any():
                        continue
                    local_idx = torch.arange(start, end, device=flat.device)[match]
                    pos_e = pos[match]
                    q_b = q_h[local_idx]
                    q_b = _apply_rotary_at_positions(self.rotary, q_b, pos_e).transpose(0, 1).unsqueeze(0)
                    attn_mask = _build_query_position_mask(pos_e, T, device=q_b.device, dtype=q_b.dtype)
                    attn_e = F.scaled_dot_product_attention(
                        q_b,
                        k_4d[b:b + 1].transpose(1, 2),
                        v_4d[b:b + 1],
                        attn_mask=attn_mask,
                        scale=self.attn_scale,
                    )
                    weighted = attn_e.squeeze(0).squeeze(0) * ew[local_idx].unsqueeze(-1).to(attn_e.dtype)
                    attn_selected[local_idx, h] = weighted.to(attn_selected.dtype)

        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim]))
        gate_selected = fp32_index_select(gate.reshape(B * T, H), 0, selected_idx).unsqueeze(-1)
        y_selected = attn_selected * gate_selected.to(attn_selected.dtype)

        out_parts = []
        for h in range(H):
            o_h = _grouped_project(
                y_selected[:, h].contiguous(),
                bank.o_proj,
                head_eids[h],
                head_ews[h],
            )
            out_parts.append(o_h.float())
        out_selected = sum(out_parts).to(x.dtype)
        out = x.new_zeros(B * T, D)
        out = fp32_index_put(out, selected_idx, out_selected)
        return out.view(B, T, D)

    def _forward_dense_queries(
        self,
        x: Tensor,
        flat: Tensor,
        token_mask: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        stats: RoutingStats | None,
        dp: str,
    ) -> Tensor:
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat_mask = token_mask.reshape(-1).bool()
        selected_idx, head_eids, head_ews, q_selected = self._route_selected_queries(flat, token_mask, stats, dp)
        q_flat = flat.new_zeros(B * T, H, HD)
        q_flat = fp32_index_put(q_flat, selected_idx, q_selected)
        q_4d = q_flat.view(B, T, H, HD).transpose(1, 2)
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
                mask_w = flat.new_zeros(B * T, dtype=ew.dtype)
                selected_mask_w = ew * (eid == e).float()
                mask_w = fp32_index_put(mask_w, selected_idx, selected_mask_w)
                mask_w = mask_w.view(B, T).unsqueeze(1).unsqueeze(-1)
                attn_h = attn_h + attn_e * mask_w.to(attn_e.dtype)

            output[:, h:h + 1] = attn_h

        y = output.transpose(1, 2)
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_dim]))
        y = y * gate.unsqueeze(-1)
        y_selected = fp32_index_select(y.reshape(B * T, H, HD), 0, selected_idx)

        out_parts = []
        for h in range(H):
            o_h = _grouped_project(
                y_selected[:, h].contiguous(),
                bank.o_proj,
                head_eids[h],
                head_ews[h],
            )
            out_parts.append(o_h.float())
        out_selected = sum(out_parts).to(x.dtype)
        out = x.new_zeros(B * T, D)
        out = fp32_index_put(out, selected_idx, out_selected)
        return out.view(B, T, D)

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        stats: RoutingStats | None = None,
        depth_idx: int = 0,
        token_mask: Tensor | None = None,
    ):
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim
        bank = self.bank
        flat = x.reshape(B * T, D)
        dp = f"d{depth_idx}"

        if token_mask is not None:
            flat_mask = token_mask.reshape(-1).bool()
            if not flat_mask.any():
                return x.new_zeros(B, T, D)
            if not flat_mask.all():
                if _should_use_sparse_query_path(token_mask, self.query_sparse_fraction_threshold):
                    return self._forward_sparse_queries(x, flat, token_mask, ve, lambdas, stats, dp)
                return self._forward_dense_queries(x, flat, token_mask, ve, lambdas, stats, dp)

        head_eids, head_ews = [], []
        for h in range(H):
            eid, ew = _route_top1(bank._active_routers["routers"][h], flat, stats, f"qkvo_{dp}_h{h}")
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
                 num_heads: int, head_dim: int, max_seq_len: int, depth_idx: int, mode: str,
                 attn_routing_level: str = "token", branch_mode: str = "router"):
        super().__init__()
        self.depth_idx = depth_idx
        self.mode = mode
        self.branch_mode = branch_mode
        if mode == "per_head_fully_independent":
            self.attn = RoutedAttentionFullyIndependent(attn_bank, num_heads, head_dim, max_seq_len,
                                                         attn_routing_level=attn_routing_level)
        elif mode == "per_head_precompute_kv":
            self.attn = RoutedAttentionPrecomputeKV(attn_bank, num_heads, head_dim, max_seq_len)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.mlp_bank = mlp_bank

        # Per-depth routers (not shared across depths)
        self.attn_routers = attn_bank.make_routers()
        self.mlp_router = mlp_bank.make_router()

    def _install_routers(self):
        """Temporarily install this depth's routers onto the shared banks.

        Stores routers in a plain dict (_active_routers) on the bank, not as
        nn.Module attributes, to avoid polluting state_dict/torch.compile.
        The attention forward methods read from bank._active_routers.
        """
        bank = self.attn.bank
        bank._active_routers = dict(self.attn_routers)

    def get_all_routers(self) -> list:
        """Return all DeepSeek routers owned by this depth step."""
        routers = [self.mlp_router]
        for val in self.attn_routers.values():
            if isinstance(val, nn.ModuleList):
                routers.extend(val)
        return routers

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor,
                sa_lambdas: Tensor, branch_router: BranchRouter | None, stats: RoutingStats | None = None):
        B, T, D = x.shape
        already_recorded = (stats is not None and
                           len(stats.branch_records) > self.depth_idx)
        effective_stats = None if already_recorded else stats
        x = lambdas[0] * x + lambdas[1] * x0
        x_norm = norm(x)

        if self.branch_mode == "alternating":
            # Hardcoded: even depths = attention, odd depths = MLP
            self._install_routers()
            if self.depth_idx % 2 == 0:
                # Attention depth
                attn_out = self.attn(x_norm, ve, sa_lambdas, effective_stats, self.depth_idx)
                x = x + attn_out
            else:
                # MLP depth
                mlp_out = self.mlp_bank(
                    self.mlp_router, x_norm.reshape(-1, D), effective_stats, self.depth_idx,
                ).view(B, T, D)
                x = x + mlp_out
        else:
            # Router-based branch selection
            w_attn, w_mlp, _, _ = branch_router(x)
            if effective_stats is not None and branch_router.last_probs is not None:
                effective_stats.add_branch(branch_router.last_probs, branch_router.last_selected)
            self._install_routers()
            attn_out = self.attn(
                x_norm, ve, sa_lambdas, effective_stats, self.depth_idx,
                token_mask=w_attn.bool(),
            )
            mlp_out = self.mlp_bank(
                self.mlp_router, x_norm.reshape(-1, D), effective_stats, self.depth_idx,
                token_mask=w_mlp.bool(),
            ).view(B, T, D)
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
                 exploration_rate: float = 0.02,
                 branch_sampling: bool = False,
                 branch_level: str = "token",       # "token" or "seq"
                 branch_mode: str = "router",       # "router" or "alternating"
                 branch_deepseek: bool = False,     # use DeepSeek-style sigmoid+bias for branch router
                 attn_routing_level: str = "token",  # "token", "seq", or "seq_qk" (Q/K seq, V/O token)
                 global_load_balancing: bool = False):
        super().__init__()
        self.num_blocks = num_layers
        self.num_depths = num_layers * 2
        self.model_dim = model_dim
        self.mode = mode
        self.branch_level = branch_level
        self.branch_mode = branch_mode
        self.attn_routing_level = attn_routing_level
        self.global_load_balancing = global_load_balancing
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

        self.attn_bank = AttentionExpertBank(num_attn_experts, model_dim, num_heads, head_dim, mode, exploration_rate)
        self.mlp_bank = MLPExpertBank(num_mlp_experts, model_dim, exploration_rate)

        self.blocks = nn.ModuleList([
            BranchRoutedDepthStep(self.attn_bank, self.mlp_bank, num_heads, head_dim, max_seq_len, i, mode,
                                  attn_routing_level=attn_routing_level, branch_mode=branch_mode)
            for i in range(self.num_depths)
        ])
        # Per-depth branch routers
        self.branch_routers = nn.ModuleList([
            BranchRouter(model_dim, exploration_rate=exploration_rate,
                         use_sampling=branch_sampling, use_seq_level=(branch_level == "seq"),
                         use_deepseek_style=branch_deepseek)
            for _ in range(self.num_depths)
        ])

        # Global load balancing: per-projection-type bias buffers
        if global_load_balancing:
            self.register_buffer("_global_q_bias", torch.zeros(num_attn_experts))
            self.register_buffer("_global_k_bias", torch.zeros(num_attn_experts))
            self.register_buffer("_global_v_bias", torch.zeros(num_attn_experts))
            self.register_buffer("_global_o_bias", torch.zeros(num_attn_experts))
            self.register_buffer("_global_mlp_bias", torch.zeros(num_mlp_experts))
            self._install_global_bias()

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
        self._routing_stats_obj = None

    def _install_global_bias(self):
        """Sync per-projection-type global bias to all routers."""
        from src.models.router import DeepSeekRouter
        bias_map = {
            "q_routers": self._global_q_bias,
            "k_routers": self._global_k_bias,
            "v_routers": self._global_v_bias,
            "o_routers": self._global_o_bias,
            "routers": self._global_q_bias,  # precompute_kv bundled QKVO uses Q bias
        }
        for block in self.blocks:
            for key, val in block.attn_routers.items():
                bias = bias_map.get(key)
                if bias is not None and isinstance(val, nn.ModuleList):
                    for router in val:
                        if isinstance(router, DeepSeekRouter):
                            router.expert_bias.copy_(bias)
            if isinstance(block.mlp_router, DeepSeekRouter):
                block.mlp_router.expert_bias.copy_(self._global_mlp_bias)

    def get_bias_rate(self, step: int, base_rate: float,
                      warmup_start: float = 0.01, warmup_steps: int = 100) -> float:
        """Linear decay from warmup_start to base_rate over warmup_steps, then constant."""
        if step < warmup_steps:
            frac = step / warmup_steps
            return warmup_start + (base_rate - warmup_start) * frac
        return base_rate

    def update_global_bias(self, bias_rate: float, bias_rates: dict | None = None):
        """Global load balancing: pool counts per projection type, update bias, broadcast."""
        from src.models.router import DeepSeekRouter
        # Accumulate counts per projection type
        counts_map = {
            "q_routers": torch.zeros_like(self._global_q_bias),
            "k_routers": torch.zeros_like(self._global_k_bias),
            "v_routers": torch.zeros_like(self._global_v_bias),
            "o_routers": torch.zeros_like(self._global_o_bias),
            "routers": torch.zeros_like(self._global_q_bias),  # precompute_kv bundled
        }
        mlp_counts = torch.zeros_like(self._global_mlp_bias)
        for block in self.blocks:
            for key, val in block.attn_routers.items():
                if key in counts_map and isinstance(val, nn.ModuleList):
                    for router in val:
                        if isinstance(router, DeepSeekRouter):
                            counts_map[key] += router.local_tokens_per_expert
                            router.local_tokens_per_expert.zero_()
            if isinstance(block.mlp_router, DeepSeekRouter):
                mlp_counts += block.mlp_router.local_tokens_per_expert
                block.mlp_router.local_tokens_per_expert.zero_()
        # All-reduce counts across ranks so all ranks see global token distribution
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            for counts in counts_map.values():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(mlp_counts, op=dist.ReduceOp.SUM)

        bias_map = {
            "q_routers": self._global_q_bias,
            "k_routers": self._global_k_bias,
            "v_routers": self._global_v_bias,
            "o_routers": self._global_o_bias,
            "routers": self._global_q_bias,
        }
        # Per-projection bias rates (fall back to default)
        rate_map = {
            "q_routers": bias_rates.get("q", bias_rate) if bias_rates else bias_rate,
            "k_routers": bias_rates.get("k", bias_rate) if bias_rates else bias_rate,
            "v_routers": bias_rates.get("v", bias_rate) if bias_rates else bias_rate,
            "o_routers": bias_rates.get("o", bias_rate) if bias_rates else bias_rate,
            "routers": bias_rates.get("q", bias_rate) if bias_rates else bias_rate,
        }
        mlp_rate = bias_rates.get("mlp", bias_rate) if bias_rates else bias_rate
        branch_rate = bias_rates.get("branch", bias_rate) if bias_rates else bias_rate
        with torch.no_grad():
            for key, counts in counts_map.items():
                bias = bias_map.get(key)
                r = rate_map.get(key, bias_rate)
                if bias is not None and counts.sum() > 0:
                    total = counts.sum()
                    loads = counts / total
                    expected = 1.0 / counts.shape[0]
                    s = torch.sign(loads - expected)
                    bias -= (s - s.mean()) * r
                    bias.clamp_(-16.0, 16.0)
            if mlp_counts.sum() > 0:
                total = mlp_counts.sum()
                loads = mlp_counts / total
                expected = 1.0 / mlp_counts.shape[0]
                s = torch.sign(loads - expected)
                self._global_mlp_bias -= (s - s.mean()) * mlp_rate
                self._global_mlp_bias.clamp_(-16.0, 16.0)
            # Update branch router biases (per-depth, all-reduce then update)
            for br in self.branch_routers:
                if hasattr(br, 'branch_bias') and hasattr(br, 'local_counts'):
                    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                        dist.all_reduce(br.local_counts, op=dist.ReduceOp.SUM)
                    if br.local_counts.sum() > 0:
                        total = br.local_counts.sum()
                        loads = br.local_counts / total
                        expected = 1.0 / br.local_counts.shape[0]
                        s = torch.sign(loads - expected)
                        br.branch_bias -= (s - s.mean()) * branch_rate
                        br.branch_bias.clamp_(-16.0, 16.0)
                    br.local_counts.zero_()
            # Broadcast updated global bias to all routers
            self._install_global_bias()

    def get_all_routers(self) -> list[DeepSeekRouter]:
        """Return all DeepSeek routers for bias updates (per-depth)."""
        routers = []
        for block in self.blocks:
            routers.extend(block.get_all_routers())
        return routers

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
            br = self.branch_routers[i] if self.branch_mode == "router" else None
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], br, stats)
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
            self._routing_stats_obj = stats
        else:
            self._aux_loss = torch.tensor(0.0, device=loss.device)
            self._seq_aux_loss = torch.tensor(0.0, device=loss.device)
            self._routing_stats = {}
            self._routing_stats_obj = None

        return loss
