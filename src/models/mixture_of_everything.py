"""Backward compatibility wrapper.

The MoE-Everything implementation has been split into modules under
src/models/moe_everything/:
- config.py: MoEverythingConfig
- attention_bank.py: AttentionExpertBank, NormExpertBank
- mlp_bank.py: MlpExpertBank
- model.py: MoEverythingModel, MoEverythingForCausalLM

This file re-exports for backward compatibility with existing imports.
"""

from .moe_everything.config import MoEverythingConfig
from .moe_everything.attention_bank import AttentionExpertBank, NormExpertBank
from .moe_everything.mlp_bank import MlpExpertBank
from .moe_everything.model import MoEverythingModel, MoEverythingForCausalLM
from .routing.routers import BranchRouter, BranchRouterRecorder

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
