#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$ROOT_DIR"

DATASET="${DATASET:-msrvtt}"
VERSION="${VERSION:-2.0}"
DEVICE="${DEVICE:-cuda}"
FEATURES_ROOT="${FEATURES_ROOT:-./data_process/datasets/features}"
OUTPUT_DIR="${OUTPUT_DIR:-./data}"

INDEX_DATASET="$DATASET"
if [[ "$INDEX_DATASET" == "actnet" ]]; then
  INDEX_DATASET="activitynet"
fi

if [[ -n "${CKPT_PATH:-}" ]]; then
  RESOLVED_CKPT="$CKPT_PATH"
else
  SEARCH_ROOT="./index/log/${INDEX_DATASET}/videorqvae_v${VERSION}"
  if [[ ! -d "$SEARCH_ROOT" ]]; then
    echo "Checkpoint root not found: $SEARCH_ROOT" >&2
    echo "Set CKPT_PATH explicitly for videorqvae generation." >&2
    exit 1
  fi
  RESOLVED_CKPT=$(
    find "$SEARCH_ROOT" -type f -name "best_recall_at_1_model.pth" -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )
fi

if [[ -z "${RESOLVED_CKPT:-}" || ! -f "$RESOLVED_CKPT" ]]; then
  echo "Unable to resolve videorqvae checkpoint. Set CKPT_PATH explicitly." >&2
  exit 1
fi

python -m baselines.mm_semantictvr.index.gen_sid_videorqvae_v2 \
  --dataset "$INDEX_DATASET" \
  --features_root "$FEATURES_ROOT" \
  --version "$VERSION" \
  --ckpt_path "$RESOLVED_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@" \
  --split train

python -m baselines.mm_semantictvr.index.gen_sid_videorqvae_v2 \
  --dataset "$INDEX_DATASET" \
  --features_root "$FEATURES_ROOT" \
  --version "$VERSION" \
  --ckpt_path "$RESOLVED_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@" \
  --split test
