#!/bin/bash
# Launch all 4 MoE-Everything per-head + per-layer router experiments sequentially
# batch_size=32 × grad_accum=2 = 64 per rank, gradient checkpointing ON
set -e

ACCEL_CONFIG="accelerate_configs/ddp_8gpu.yaml"

CONFIGS=(
  "configs/moe_everything_per_head_independent_prenorm.yaml"
  "configs/moe_everything_per_head_independent_bothnorm.yaml"
  "configs/moe_everything_per_head_precompute_kv_prenorm.yaml"
  "configs/moe_everything_per_head_precompute_kv_bothnorm.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "=========================================="
  echo "Launching: $name"
  echo "Config:    $cfg"
  echo "=========================================="
  uv run accelerate launch --config_file "$ACCEL_CONFIG" \
    train.py --config "$cfg" \
    --output_dir "./outputs/${name}"
  echo ""
  echo "$name finished."
  echo ""
done

echo "All 4 runs complete."
