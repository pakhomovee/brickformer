#!/usr/bin/env bash
# Command 2/2: train the native LEGO transformer on the prepared token stream.
#
#   bash scripts/train.sh                 # 25M ablation twin, sensible defaults
#   bash scripts/train.sh --size 250M --batch 16 --grad-accum 4 --max-iters 60000
#
# Auto-uses CUDA if available. Any extra flags pass straight to lego_tf.bnet.train.
# Override the preset defaults via env: SIZE, CTX, BATCH, MAX_ITERS, OUT.
# USE_POSE=1 trains the v1 resolved-pose model (needs a pose stream from POSES=1 prepare).
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-25M}"
CTX="${CTX:-1024}"
BATCH="${BATCH:-32}"
MAX_ITERS="${MAX_ITERS:-20000}"
USE_POSE="${USE_POSE:-}"
DEFAULT_OUT="runs/pretrain-$SIZE"; [ -n "$USE_POSE" ] && DEFAULT_OUT="$DEFAULT_OUT-v1"
OUT="${OUT:-$DEFAULT_OUT}"

POSE_ARG=""; [ -n "$USE_POSE" ] && POSE_ARG="--use-pose"
python -m lego_tf.bnet.train \
  --train data/pretrain.bin --val data/val.bin \
  --size "$SIZE" --ctx "$CTX" --batch "$BATCH" --max-iters "$MAX_ITERS" \
  --out "$OUT" $POSE_ARG "$@"
