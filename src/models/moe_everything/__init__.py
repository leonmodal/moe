"""MoE-Everything model: split by API seam from mixture_of_everything.py.

This package provides the reorganized MoE-Everything model with coherent
module boundaries. It re-exports from the original mixture_of_everything.py
for compatibility during the transition.

Module structure:
- config.py: MoEverythingConfig
- attention_bank.py: AttentionExpertBank (per-head routed attention)
- mlp_bank.py: MlpExpertBank (routed MLP experts)
- model.py: MoEverythingModel, MoEverythingForCausalLM (assembly)
"""

from src.models.mixture_of_everything import (
    MoEverythingConfig,
    MoEverythingForCausalLM,
    MoEverythingModel,
    AttentionExpertBank,
    MlpExpertBank,
    NormExpertBank,
    BranchRouter,
    BranchRouterRecorder,
)

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
