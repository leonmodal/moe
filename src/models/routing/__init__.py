"""Routing module: routers, load balancing, bias management, stats.

This package is the source of truth for:
- BranchRouter and BranchRouterRecorder (defined in routers.py)
- Load balancing loss functions (defined in load_balancing.py)
- DeepSeekRouter, ExplorationTopKRouter (defined in src.models.router, re-exported here)
"""

from .routers import (
    DeepSeekRouter,
    ExplorationTopKRouter,
    BranchRouter,
    BranchRouterRecorder,
)
from .load_balancing import (
    load_balancing_loss_func,
    normalized_load_balancing_loss_func,
    seq_load_balancing_loss_func,
)

__all__ = [
    "DeepSeekRouter",
    "ExplorationTopKRouter",
    "BranchRouter",
    "BranchRouterRecorder",
    "load_balancing_loss_func",
    "normalized_load_balancing_loss_func",
    "seq_load_balancing_loss_func",
]
