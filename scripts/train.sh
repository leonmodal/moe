#!/usr/bin/env bash
# Launch pretraining with a config file.
#
# Usage:
#   ./scripts/train.sh <config> [--resume <ckpt_dir>]
#
# Examples:
#   ./scripts/train.sh configs/scaling/xs_standard.yaml
#   ./scripts/train.sh configs/scaling/xs_global.yaml
#   ./scripts/train.sh configs/scaling/l_standard.yaml --resume outputs/l_standard_moe/checkpoint-5000

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

# ── Load API keys ────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
  set -a; source .env; set +a
else
  echo "WARNING: .env not found — wandb/HF may not authenticate"
fi

# ── Parse args ─────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config> [--resume <dir>]"
  echo ""
  echo "Available configs:"
  find configs -name '*.yaml' | sort
  exit 1
fi

CONFIG="$1"; shift

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi

EXTRA_ARGS=("$@")   # pass remaining args (--resume, etc.) through

# ── Launch ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Config : $CONFIG"
echo "  Extra  : ${EXTRA_ARGS[*]:-none}"
echo "========================================"

TRAIN_ENTRYPOINT="scripts/train.py"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  uv run torchrun \
    --standalone \
    --nproc_per_node "${NPROC_PER_NODE}" \
    "$TRAIN_ENTRYPOINT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"
else
  uv run python "$TRAIN_ENTRYPOINT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"
fi
