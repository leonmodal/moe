"""MoE-Everything model package.

Imports from the monolithic mixture_of_everything.py which still holds
the model classes. BranchRouter and BranchRouterRecorder have been
extracted to src/models/routing/routers.py (the source of truth).

Module organization by API seam:
- Config: MoEverythingConfig
- Routing: BranchRouter (from src/models/routing/routers.py)
- Attention bank: AttentionExpertBank, NormExpertBank
- MLP bank: MlpExpertBank
- Model assembly: MoEverythingModel, MoEverythingForCausalLM
"""

from src.models.mixture_of_everything import (
    MoEverythingConfig,
    MoEverythingForCausalLM,
    MoEverythingModel,
    AttentionExpertBank,
    MlpExpertBank,
    NormExpertBank,
)
from src.models.routing.routers import BranchRouter, BranchRouterRecorder

__all__ = [
    "MoEverythingConfig",
    "MoEverythingForCausalLM",
    "MoEverythingModel",
    "AttentionExpertBank",
    "MlpExpertBank",
    "NormExpertBank",
    "BranchRouter",
    "BranchRouterRecorder",
]
