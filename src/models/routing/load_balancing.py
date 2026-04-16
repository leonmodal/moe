"""Unified load balancing loss functions.

Re-exports from src.models.load_balancing for the routing module.
All MoE models should use these functions for consistency.
"""

from src.models.load_balancing import (
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
    switch_load_balancing_loss_func,
    batch_load_balancing_loss_func,
)

__all__ = [
    "normalized_load_balancing_loss_func",
    "seq_load_balancing_loss_func",
    "switch_load_balancing_loss_func",
    "batch_load_balancing_loss_func",
]
