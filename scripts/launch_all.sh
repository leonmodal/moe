#!/bin/bash
# Launch all DeepSeek MoE experiments in parallel
set -e

echo "Launching all runs in parallel..."

# modal run --detach modal_train.py --config configs/global_moe.yaml &
# modal run --detach modal_train.py --config configs/global_moe_nointerp.yaml &
# modal run --detach modal_train.py --config configs/standard_moe.yaml &
# modal run --detach modal_train.py --config configs/scaling/xs_dense_baseline.yaml &

modal run --detach modal_train.py --config configs/moe_everything_bundled.yaml &
modal run --detach modal_train.py --config configs/moe_everything_kv_paired.yaml &
modal run --detach modal_train.py --config configs/moe_everything_qk_paired.yaml &
modal run --detach modal_train.py --config configs/moe_everything_fully_independent.yaml &
modal run --detach modal_train.py --config configs/moe_everything_precompute_kv.yaml &

wait
echo "All runs launched."
