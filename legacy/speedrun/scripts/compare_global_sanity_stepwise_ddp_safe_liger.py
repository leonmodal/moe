"""
Run the DDP parity harness with the validated-safe Liger subset enabled for
both compared models.

This keeps the comparison symmetric:
  - RoPE and RMSNorm patched on both models
  - SWIGLU and fused CE disabled, because those are not safe for moe_everything
"""

from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

apply_liger_kernel_to_qwen3_moe(
    rope=True,
    rms_norm=True,
    swiglu=False,
    fused_linear_cross_entropy=False,
    cross_entropy=False,
)

from scripts.compare_global_sanity_stepwise_ddp import main


if __name__ == "__main__":
    main()
