#!/bin/bash
# Launch all 5 MoE-Everything per-head experiments, including sanity routing
set -e

echo "Launching all runs..."

modal run --detach modal_train.py --config configs/moe_everything_per_head_independent_prenorm.yaml &
modal run --detach modal_train.py --config configs/moe_everything_per_head_independent_bothnorm.yaml &
modal run --detach modal_train.py --config configs/moe_everything_per_head_precompute_kv_prenorm.yaml &
modal run --detach modal_train.py --config configs/moe_everything_per_head_precompute_kv_bothnorm.yaml &
modal run --detach modal_train.py --config configs/moe_everything_per_head_precompute_kv_sanity.yaml &

wait
echo "All runs launched."
