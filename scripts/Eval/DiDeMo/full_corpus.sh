#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"

SEMANTICTVR_ENV=${SEMANTICTVR_ENV:-semantictvr}
XPOOL_ENV=${XPOOL_ENV:-xpool}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RUN_DEVICE=${RUN_DEVICE:-0}
XPOOL_PYTHONPATH="$REPO_ROOT/reranker/xpool${PYTHONPATH:+:$PYTHONPATH}"

DATASET=didemo
DATASET_NAME=DIDEMO
SETTING=2
SETTING_NAME=full_corpus
NUM_CANDIDATES=100
CANDIDATE_HANDOFF_CAP=300
EXPORT_BATCH_SIZE=${EXPORT_BATCH_SIZE:-40}
RERANK_BATCH_SIZE=${RERANK_BATCH_SIZE:-32}
POOL_BATCH_SIZE=${POOL_BATCH_SIZE:-64}

GRDR_CKPT=${GRDR_CKPT:-output/checkpoints/GRDR/didemo/best_model/best_model.pt}
XPOOL_CKPT=${XPOOL_CKPT:-reranker/xpool/ckpt/didemo_model_best.pth}
VIDEOS_DIR=${VIDEOS_DIR:-dataset/DiDeMo}
XPOOL_CACHE_ROOT=${XPOOL_CACHE_ROOT:-reranker/xpool/video_features_cache/Xpool}
CANDIDATE_FILE=${CANDIDATE_FILE:-candidates/GRDR/didemo/didemo_t2_100_candidates.json}
RESULT_FILE=${RESULT_FILE:-reproduction/GRDR/didemo/full_corpus/result.csv}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

require_file() {
    if [[ "${DRY_RUN:-0}" != "1" && ! -f "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ "${DRY_RUN:-0}" != "1" && ! -d "$1" ]]; then
        echo "Missing required directory: $1" >&2
        exit 1
    fi
}

require_file "$GRDR_CKPT"
require_file "$GRDR_CKPT.code"
require_file "$XPOOL_CKPT"
require_dir "$XPOOL_CACHE_ROOT"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$(dirname "$CANDIDATE_FILE")"
fi

run env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    conda run -n "$SEMANTICTVR_ENV" python run.py \
    --candidate_export \
    --eval_checkpoint "$GRDR_CKPT" \
    --model_name t5-small \
    --dataset "$DATASET" \
    --setting "$SETTING" \
    --code_num 128 \
    --max_length 3 \
    --batch_size "$EXPORT_BATCH_SIZE" \
    --eval_batch_size "$EXPORT_BATCH_SIZE" \
    --num_latent_tokens 4 \
    --num_candidates "$NUM_CANDIDATES" \
    --candidate_handoff_cap "$CANDIDATE_HANDOFF_CAP" \
    --inference_reorder_by_access_score \
    --access_score_bucket_gamma 0.50 \
    --output_json "$CANDIDATE_FILE" \
    --seed 42 \
    --device "$RUN_DEVICE" \
    --use_pseudo_queries

run env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$XPOOL_PYTHONPATH" \
    conda run -n "$XPOOL_ENV" python reranker/xpool/test.py \
    --dataset_name "$DATASET_NAME" \
    --videos_dir "$VIDEOS_DIR" \
    --eval_checkpoint "$XPOOL_CKPT" \
    --candidate_file "$CANDIDATE_FILE" \
    --rerank_mode \
    --expanded_pool \
    --use_cached_video_features \
    --video_cache_dir "$XPOOL_CACHE_ROOT" \
    --batch_size "$RERANK_BATCH_SIZE" \
    --pool_batch_size "$POOL_BATCH_SIZE" \
    --result_file "$RESULT_FILE" \
    --huggingface \
    --seed 42 \
    --no_tensorboard
