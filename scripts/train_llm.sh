#!/usr/bin/env bash
# Track B: LoRA fine-tune a pretrained backbone (Qwen2.5-0.5B) on (caption -> native LEGO tokens).
# Native tokens + validity-by-construction decoding, now on top of pretrained priors.
#
#   bash scripts/train_llm.sh                       # vehicle-only by default
#   VEHICLES_ONLY= bash scripts/train_llm.sh        # all captions
#
# Override via env: BACKBONE, SPLIT, CAPTIONS, OUT, MAXITERS, BATCH, LR, LORA_R, CFG_DROP, PROMPT.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE="${BACKBONE:-Qwen/Qwen2.5-0.5B}"
SPLIT="${SPLIT:-data/sft.npz}"
CAPTIONS="${CAPTIONS:-data/captions_sft.jsonl}"
OUT="${OUT:-runs/llm-qwen0.5b}"
MAXITERS="${MAXITERS:-2000}"; BATCH="${BATCH:-8}"; LR="${LR:-2e-4}"; LORA_R="${LORA_R:-16}"
CFG_DROP="${CFG_DROP:-0.1}"
VEHICLES_ONLY="${VEHICLES_ONLY:-1}"                  # set empty for all captions
PROMPT="${PROMPT:-a small red race car}"
S3_PROFILE="${S3_PROFILE:-brickformer}"; S3_BUCKET="${S3_BUCKET:-brickformer}"; S3_PREFIX="${S3_PREFIX:-bricknet/}"

echo ">> installing Track-B deps (transformers + peft + accelerate)"
pip install -q -r requirements-llm.txt

for f in "$(basename "$SPLIT")" "$(basename "$CAPTIONS")"; do
  if [ ! -f "data/$f" ]; then
    echo ">> downloading $f"
    aws --profile "$S3_PROFILE" s3 cp "s3://$S3_BUCKET/$S3_PREFIX$f.xz" "data/$f.xz" && unxz -kf "data/$f.xz"
  fi
done

FILTER_ARG=""; [ -n "$VEHICLES_ONLY" ] && FILTER_ARG="--vehicles-only"
echo ">> LoRA SFT: $BACKBONE -> $OUT ${VEHICLES_ONLY:+(vehicles only)}"
python -m lego_tf.bnet.train_llm --split "$SPLIT" --captions "$CAPTIONS" --out "$OUT" \
  --backbone "$BACKBONE" --max-iters "$MAXITERS" --batch "$BATCH" --lr "$LR" \
  --lora-r "$LORA_R" --cond-drop "$CFG_DROP" $FILTER_ARG

echo ">> done. Generate with a prompt (collision-free, same harness as Track A):"
echo "   python -m lego_tf.bnet.evaluate --ckpt $OUT/best --n 16 --collision-free \\"
echo "       --prompt '$PROMPT' --cfg-weight 4 --min-bricks 8 --export runs/eval-llm"
