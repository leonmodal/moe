from importlib import import_module

from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig

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
]

_LAZY_IMPORTS = {
    "Qwen3MoeForCausalLM": (
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "Qwen3MoeForCausalLM",
    ),
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
    "MoEverythingConfig": ("src.models.mixture_of_everything", "MoEverythingConfig"),
    "MoEverythingForCausalLM": ("src.models.mixture_of_everything", "MoEverythingForCausalLM"),
    "MoEverythingModel": ("src.models.mixture_of_everything", "MoEverythingModel"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
