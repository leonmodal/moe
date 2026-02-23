"""
Standard MoE — this is just Qwen3MoE as-is.

16 layers × 128 experts/layer = 2048 total independent experts.
Nothing custom here; use Qwen3MoeConfig + Qwen3MoeForCausalLM directly.
"""
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

StandardMoEConfig = Qwen3MoeConfig
StandardMoEModel = Qwen3MoeForCausalLM
