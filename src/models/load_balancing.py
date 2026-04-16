"""Backward compatibility: re-export from src.models.routing.load_balancing.

The source of truth for load-balancing loss functions is now
src/models/routing/load_balancing.py.
"""

from .routing.load_balancing import (
    load_balancing_loss_func,
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
    normalize_router_scores,
)

__all__ = [
    "load_balancing_loss_func",
    "normalized_load_balancing_loss_func",
    "seq_load_balancing_loss_func",
    "normalize_router_scores",
]
