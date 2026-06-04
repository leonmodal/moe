#!/bin/bash
# Launch the active qkvo recompute-KV branch-router ablation set.
set -e

COMMON_ARGS=(
  --background
  --batch-size 16
  --gradient-accumulation 1
  --save-every 100
  --output-suffix h200_8gpu_bs16_ddp
  --dist-strategy ddp
  --disable-grouped-mm
)

export MOE_MODAL_GPU_TYPE=H200
export MOE_MODAL_GPUS_PER_NODE=8
export MOE_MODAL_N_NODES=1
export MOE_MODAL_DIST_STRATEGY=ddp

echo "Launching active qkvo recompute-KV branch-router runs..."

modal run --detach modal_train.py --config configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_sampling_entropy.yaml "${COMMON_ARGS[@]}" &
modal run --detach modal_train.py --config configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_top1_explore_decay.yaml "${COMMON_ARGS[@]}" &
modal run --detach modal_train.py --config configs/16_layers/moe_everything_per_head_recompute_kv_qkvo_branch_fixed_alternating.yaml "${COMMON_ARGS[@]}" &

wait
echo "All runs launched."
