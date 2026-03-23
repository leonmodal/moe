#!/bin/bash
# Launch all DeepSeek MoE experiments in parallel
set -e

echo "Launching all runs in parallel..."

modal run --detach modal_train.py --config configs/global_moe.yaml &
# modal run --detach modal_train.py --config configs/global_moe_nointerp.yaml &
# modal run --detach modal_train.py --config configs/standard_moe.yaml &
# modal run --detach modal_train.py --config configs/scaling/xs_dense_baseline.yaml &

wait
echo "All runs launched."
