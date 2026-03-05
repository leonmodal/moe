#!/bin/bash
# Launch all DeepSeek MoE experiments in parallel
set -e

echo "Launching all runs in parallel..."

modal run --detach modal_train.py --config configs/scaling/xs_deepseek_global.yaml &
# modal run --detach modal_train.py --config configs/scaling/xs_deepseek_global_nointerp.yaml &
# modal run --detach modal_train.py --config configs/scaling/xs_deepseek_standard.yaml &
# modal run --detach modal_train.py --config configs/scaling/xs_dense_baseline.yaml &

wait
echo "All runs launched."
