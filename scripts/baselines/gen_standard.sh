#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"

DATASET="${DATASET:-msrvtt}"
MODE="${MODE:-none}"
CODE_NUM="${CODE_NUM:-256}"
CODEBOOK_LAYERS="${CODEBOOK_LAYERS:-4}"
DEVICE="${DEVICE:-cuda}"
FEATURES_ROOT="${FEATURES_ROOT:-$SEMANTIC_FEATURES_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/${DATASET}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INDEX_DATASET="$DATASET"
if [[ "$INDEX_DATASET" == "actnet" ]]; then
  INDEX_DATASET="activitynet"
fi

DEFAULT_CKPT="./index/log/${INDEX_DATASET}/standard/code_num_${CODE_NUM}_codebook_layers_${CODEBOOK_LAYERS}/best_collision_model.pth"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT}"
FEATURES_DATASET_DIR=$(feature_dataset_dir "$DATASET")

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi
require_dir "$FEATURES_ROOT/InternVideo2/$FEATURES_DATASET_DIR"

extra_args=("$@")

run_cmd "$PYTHON_BIN" -m baselines.mm_semantictvr.index.gen_sid_rqvae \
  --dataset "$INDEX_DATASET" \
  --features_root "$FEATURES_ROOT" \
  --mode "$MODE" \
  --type standard \
  --ckpt_path "$CKPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --code_num "$CODE_NUM" \
  --codebook_layers "$CODEBOOK_LAYERS" \
  "${extra_args[@]}" \
  --split train

run_cmd "$PYTHON_BIN" -m baselines.mm_semantictvr.index.gen_sid_rqvae \
  --dataset "$INDEX_DATASET" \
  --features_root "$FEATURES_ROOT" \
  --mode "$MODE" \
  --type standard \
  --ckpt_path "$CKPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --code_num "$CODE_NUM" \
  --codebook_layers "$CODEBOOK_LAYERS" \
  "${extra_args[@]}" \
  --split test
