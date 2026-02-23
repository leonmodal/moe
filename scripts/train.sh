#!/usr/bin/env bash
# Launch pretraining for a given scale and model type.
#
# Usage:
#   ./scripts/train.sh <scale> <type> [--smoke_test] [--resume <ckpt_dir>]
#
#   scale : xs | s | m
#   type  : standard | global
#
# Examples:
#   ./scripts/train.sh xs standard
#   ./scripts/train.sh xs global
#   ./scripts/train.sh m  standard --smoke_test
#   ./scripts/train.sh s  global   --resume outputs/s_global_moe/checkpoint-5000

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

# ── Load API keys ────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
  set -a; source .env; set +a
else
  echo "WARNING: .env not found — wandb/HF may not authenticate"
fi

# ── Parse positional args ────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <scale> <type> [--smoke_test] [--resume <dir>]"
  echo "  scale : xs | s | m"
  echo "  type  : standard | global"
  exit 1
fi

SCALE="$1"; shift
TYPE="$1";  shift

CONFIG="configs/scaling/${SCALE}_${TYPE}.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi

EXTRA_ARGS=("$@")   # pass remaining args (--smoke_test, --resume, etc.) through

# ── Launch ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Scale  : $SCALE"
echo "  Type   : $TYPE"
echo "  Config : $CONFIG"
echo "  Extra  : ${EXTRA_ARGS[*]:-none}"
echo "========================================"

uv run accelerate launch \
  --config_file accelerate_configs/ddp_8gpu.yaml \
  train.py \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}"
