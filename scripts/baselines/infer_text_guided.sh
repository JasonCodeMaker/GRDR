#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$ROOT_DIR"

DATASET="${DATASET:-msrvtt}"
MODE="${MODE:-none}"
VERSION="${VERSION:-2.0}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
CODE_BOOK_SIZE="${CODE_BOOK_SIZE:-256}"
CODE_BOOK_NUM="${CODE_BOOK_NUM:-4}"
NUM_CANDIDATES="${NUM_CANDIDATES:-100}"
GPU_ID="${GPU_ID:-0}"
SETTING="${SETTING:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-output/baseline}"
CANDIDATE_OUTPUT_DIR="${CANDIDATE_OUTPUT_DIR:-candidates/baseline}"

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  RESOLVED_CHECKPOINT="$CHECKPOINT_PATH"
else
  SEARCH_ROOT="${OUTPUT_DIR}/${DATASET}/text_guided"
  if [[ ! -d "$SEARCH_ROOT" ]]; then
    echo "Checkpoint root not found: $SEARCH_ROOT" >&2
    echo "Set CHECKPOINT_PATH explicitly." >&2
    exit 1
  fi
  RESOLVED_CHECKPOINT=$(
    find "$SEARCH_ROOT" -type d -name best_model -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )
fi

if [[ -z "${RESOLVED_CHECKPOINT:-}" || ! -d "$RESOLVED_CHECKPOINT" ]]; then
  echo "Unable to resolve text_guided baseline checkpoint." >&2
  exit 1
fi

if [[ "$(basename "$RESOLVED_CHECKPOINT")" != "best_model" ]]; then
  echo "Baseline inference only supports checkpoint directories named best_model." >&2
  echo "Got: $RESOLVED_CHECKPOINT" >&2
  exit 1
fi

python -m baselines.mm_semantictvr.retriever.avg_train_retriever_t5 \
  --dataset "$DATASET" \
  --mode "$MODE" \
  --index_type text_guided \
  --model_name "$MODEL_NAME" \
  --code_book_size "$CODE_BOOK_SIZE" \
  --code_book_num "$CODE_BOOK_NUM" \
  --version "$VERSION" \
  --gpu_id "$GPU_ID" \
  --num_candidates "$NUM_CANDIDATES" \
  --setting "$SETTING" \
  --output_dir "$OUTPUT_DIR" \
  --candidate_output_dir "$CANDIDATE_OUTPUT_DIR" \
  --eval_checkpoint "$RESOLVED_CHECKPOINT" \
  --eval \
  "$@"
