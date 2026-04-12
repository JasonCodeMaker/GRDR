#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"

DATASET="${DATASET:-msrvtt}"
MODE="${MODE:-none}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-output/baseline}"
VERSION="${VERSION:-2.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$DATASET" in
  actnet|didemo)
    DEFAULT_EPOCHS=50
    ;;
  lsmdc)
    DEFAULT_EPOCHS=30
    ;;
  *)
    DEFAULT_EPOCHS=20
    ;;
esac
TRAIN_EPOCH="${TRAIN_EPOCH:-$DEFAULT_EPOCHS}"

PRECISION_ARGS=()
if [[ "${BF16:-1}" == "1" ]]; then
  PRECISION_ARGS+=(--bf16)
elif [[ "${FP16:-0}" == "1" ]]; then
  PRECISION_ARGS+=(--float16)
fi

extra_args=("$@")

run_cmd "$PYTHON_BIN" -m baselines.mm_semantictvr.retriever.avg_train_retriever_t5 \
  --dataset "$DATASET" \
  --mode "$MODE" \
  --index_type text_guided \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --train_epoch "$TRAIN_EPOCH" \
  --learning_rate "$LEARNING_RATE" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gpu_id "$GPU_ID" \
  --version "$VERSION" \
  "${PRECISION_ARGS[@]}" \
  "${extra_args[@]}"
