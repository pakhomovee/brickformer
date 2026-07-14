#!/usr/bin/env bash
# Caption-conditioned SFT: fine-tune a pretrained checkpoint on (caption -> build) pairs.
#
#   INIT=weights/brickformer_25M_v0_fix.pt bash scripts/sft.sh
#
# Needs the sft graphs + captions (downloaded from the dataset bucket) and a text encoder.
# Override via env: INIT, SPLIT, CAPTIONS, OUT, MAXITERS, BATCH, LR, CFG_DROP, CAPS_MODEL.
set -euo pipefail
cd "$(dirname "$0")/.."

INIT="${INIT:?set INIT=<pretrained .pt to fine-tune from>}"
SPLIT="${SPLIT:-data/sft.npz}"
CAPTIONS="${CAPTIONS:-data/captions_sft.jsonl}"
OUT="${OUT:-runs/sft-25M}"
CAPS="${CAPS:-data/sft}"                 # prefix for <CAPS>.capemb.f16 + .capmap.json
CAPS_MODEL="${CAPS_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
MAXITERS="${MAXITERS:-2000}"; BATCH="${BATCH:-64}"; LR="${LR:-1e-4}"; CFG_DROP="${CFG_DROP:-0.1}"
S3_PROFILE="${S3_PROFILE:-brickformer}"; S3_BUCKET="${S3_BUCKET:-brickformer}"; S3_PREFIX="${S3_PREFIX:-bricknet/}"

echo ">> installing SFT deps (text encoder)"
pip install -q -r requirements-sft.txt

for f in "$(basename "$SPLIT")" "$(basename "$CAPTIONS")"; do
  if [ ! -f "data/$f" ]; then
    echo ">> downloading $f"
    aws --profile "$S3_PROFILE" s3 cp "s3://$S3_BUCKET/$S3_PREFIX$f.xz" "data/$f.xz" && unxz -kf "data/$f.xz"
  fi
done

if [ ! -f "$CAPS.capemb.f16" ]; then
  echo ">> precomputing caption embeddings ($CAPS_MODEL) -> $CAPS.capemb.f16"
  python -m lego_tf.bnet.captions --split "$SPLIT" --captions "$CAPTIONS" --out "$CAPS" --model "$CAPS_MODEL"
fi

echo ">> SFT: fine-tuning $INIT -> $OUT"
python -m lego_tf.bnet.train_sft --split "$SPLIT" --caps "$CAPS" --init "$INIT" --out "$OUT" \
  --max-iters "$MAXITERS" --batch "$BATCH" --lr "$LR" --cond-drop "$CFG_DROP"

echo ">> done. Generate with a prompt:"
echo "   python -m lego_tf.bnet.evaluate --ckpt $OUT/best.pt --n 16 --collision-free \\"
echo "       --prompt 'a red race car' --cfg-weight 4 --export runs/eval-sft"
