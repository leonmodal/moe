#!/bin/bash
# Launch the 3 prenorm MoE-Everything per-head experiments with per-layer
# shared Q/K norm, including sanity routing
set -e

echo "Launching all runs..."

modal run --detach modal_train.py --config configs/moe_everything_per_head_independent_perlayer_prenorm_per_layer_qk_norm.yaml &
modal run --detach modal_train.py --config configs/moe_everything_per_head_precompute_kv_perlayer_prenorm_per_layer_qk_norm.yaml &
modal run --detach modal_train.py --config configs/moe_everything_per_head_precompute_kv_sanity.yaml &

wait
echo "All runs launched."
