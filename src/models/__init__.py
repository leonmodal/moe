"""Model exports.

All model configs and classes are imported from local files (Bagel-style),
not from the transformers package. Local files use transformers utilities
and PreTrainedModel for checkpoint compatibility, but model forward-pass
logic is defined locally with our bug fixes and customizations.
"""

from importlib import import_module

# Import configs and dense model from local files
from .configuration_qwen3 import Qwen3Config
from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3 import Qwen3ForCausalLM

__all__ = [
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "StandardMoEConfig",
    "StandardMoEModel",
    "DeepSeekStandardMoEModel",
    "DeepSeekRouter",
    "GlobalMoEConfig",
    "GlobalMoEModel",
    "GlobalMoEForCausalLM",
    "DeepSeekGlobalMoEForCausalLM",
    "MoEverythingConfig",
    "MoEverythingForCausalLM",
    "MoEverythingModel",
    "RecurrentMoEConfig",
    "RecurrentMoEForCausalLM",
    "RecurrentMoEModel",
]

_LAZY_IMPORTS = {
    # MoE model from local file (with double-softmax fix and Triton GEMM)
    "Qwen3MoeForCausalLM": ("src.models.modeling_qwen3_moe", "Qwen3MoeForCausalLM"),
    "DeepSeekRouter": ("src.models.router", "DeepSeekRouter"),
    "StandardMoEConfig": ("src.models.standard_moe", "StandardMoEConfig"),
    "StandardMoEModel": ("src.models.standard_moe", "StandardMoEModel"),
    "DeepSeekStandardMoEModel": ("src.models.standard_moe", "DeepSeekStandardMoEModel"),
    "GlobalMoEConfig": ("src.models.global_moe", "GlobalMoEConfig"),
    "GlobalMoEForCausalLM": ("src.models.global_moe", "GlobalMoEForCausalLM"),
    "GlobalMoEModel": ("src.models.global_moe", "GlobalMoEModel"),
    "DeepSeekGlobalMoEForCausalLM": (
        "src.models.global_moe",
        "DeepSeekGlobalMoEForCausalLM",
    ),
    "MoEverythingConfig": ("src.models.moe_everything", "MoEverythingConfig"),
    "MoEverythingForCausalLM": ("src.models.moe_everything", "MoEverythingForCausalLM"),
    "MoEverythingModel": ("src.models.moe_everything", "MoEverythingModel"),
    "RecurrentMoEConfig": ("src.models.recurrent_moe", "RecurrentMoEConfig"),
    "RecurrentMoEForCausalLM": ("src.models.recurrent_moe", "RecurrentMoEForCausalLM"),
    "RecurrentMoEModel": ("src.models.recurrent_moe", "RecurrentMoEModel"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
