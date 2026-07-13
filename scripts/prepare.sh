#!/usr/bin/env bash
# Command 1/2: install deps, download the BrickNet graph splits, tokenize to .bin.
#
#   bash scripts/prepare.sh
#
# Needs the AWS CLI configured with a profile for Yandex Object Storage (once):
#   aws configure --profile brickformer          # enter your key id + secret
#   aws configure set profile.brickformer.region ru-central1
#   aws configure set profile.brickformer.endpoint_url https://storage.yandexcloud.net
#
# Override defaults via env: S3_PROFILE, S3_BUCKET, SPLITS.
set -euo pipefail
cd "$(dirname "$0")/.."

S3_PROFILE="${S3_PROFILE:-brickformer}"
S3_BUCKET="${S3_BUCKET:-brickformer}"
S3_PREFIX="${S3_PREFIX:-bricknet/}"    # objects live under this key prefix in the bucket
SPLITS="${SPLITS:-pt.npz val.npz}"     # graphs only — the big paths_*.jsonl are NOT needed
WORKERS="${WORKERS:-}"                  # tokenizer processes (default: all CPU cores)

mkdir -p data
echo ">> installing deps (torch: CUDA wheel on a GPU box, CPU wheel otherwise)"
pip install -q -r requirements.txt

for npz in $SPLITS; do
  if [ ! -f "data/$npz" ]; then
    echo ">> downloading $npz from s3://$S3_BUCKET/$S3_PREFIX"
    aws --profile "$S3_PROFILE" s3 cp "s3://$S3_BUCKET/$S3_PREFIX$npz.xz" "data/$npz.xz"
    unxz -kf "data/$npz.xz"
  else
    echo ">> data/$npz already present, skipping download"
  fi
done

WORKER_ARG=""; [ -n "$WORKERS" ] && WORKER_ARG="--workers $WORKERS"
echo ">> tokenizing pretrain split -> data/pretrain.bin (mmap streaming, all cores)"
python -m lego_tf.bnet.prepare_data --split data/pt.npz  --out data/pretrain.bin $WORKER_ARG
echo ">> tokenizing val split -> data/val.bin"
python -m lego_tf.bnet.prepare_data --split data/val.npz --out data/val.bin $WORKER_ARG

echo ">> prepare done. Next:  bash scripts/train.sh"
