"""RoutingStats dataclass for per-forward-pass accumulation.

Extracted from speedrun_moe_gpt.py. This is separate from src/utils/routing_stats.py
which handles count-based statistics over time. This module handles per-forward-pass
accumulation of aux losses, router records, and branch records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RoutingStats:
    """Accumulates routing statistics during a single forward pass.

    Used by MoE models to collect per-layer routing decisions for
    logging, visualization, and bias updates.
    """
    # Accumulated losses
    aux_loss: float = 0.0
    seq_aux_loss: float = 0.0
    branch_aux_loss: float = 0.0
    attention_aux_loss: float = 0.0

    # Per-layer routing records: list of dicts with keys like
    # 'selected_experts', 'routing_weights', 'exploration_mask'
    router_records: list[dict[str, Any]] = field(default_factory=list)
    branch_records: list[dict[str, Any]] = field(default_factory=list)
    attn_records: list[dict[str, Any]] = field(default_factory=list)

    def add_router_record(self, record: dict[str, Any]) -> None:
        self.router_records.append(record)

    def add_branch_record(self, record: dict[str, Any]) -> None:
        self.branch_records.append(record)

    def add_attn_record(self, record: dict[str, Any]) -> None:
        self.attn_records.append(record)

    def expert_heatmap_data(self) -> dict[str, Any]:
        """Extract data for expert heatmap visualization."""
        heatmap = {}
        for i, rec in enumerate(self.router_records):
            if "selected_experts" in rec:
                sel = rec["selected_experts"]
                if isinstance(sel, torch.Tensor):
                    sel = sel.detach().cpu()
                heatmap[f"layer_{i}"] = sel
        return heatmap

    def routing_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of routing state for visualization."""
        return {
            "router_records": self.router_records,
            "branch_records": self.branch_records,
            "attn_records": self.attn_records,
        }

    def reset(self) -> None:
        """Reset all accumulated stats."""
        self.aux_loss = 0.0
        self.seq_aux_loss = 0.0
        self.branch_aux_loss = 0.0
        self.attention_aux_loss = 0.0
        self.router_records.clear()
        self.branch_records.clear()
        self.attn_records.clear()
