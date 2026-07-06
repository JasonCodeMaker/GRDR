#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"

SEMANTICTVR_ENV=${SEMANTICTVR_ENV:-semantictvr}
XPOOL_ENV=${XPOOL_ENV:-xpool}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RUN_DEVICE=${RUN_DEVICE:-0}
XPOOL_PYTHONPATH="$REPO_ROOT/reranker/xpool${PYTHONPATH:+:$PYTHONPATH}"

DATASET=panda
SETTING=2
SETTING_NAME=full_corpus
NUM_CANDIDATES=200
CANDIDATE_HANDOFF_CAP=600
EXPORT_BATCH_SIZE=${EXPORT_BATCH_SIZE:-20}
RERANK_BATCH_SIZE=${RERANK_BATCH_SIZE:-32}

GRDR_CKPT=${GRDR_CKPT:-output/checkpoints/GRDR/panda/best_model/best_model.pt}
XPOOL_CKPT=${XPOOL_CKPT:-reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}
XPOOL_CACHE_ROOT=${XPOOL_CACHE_ROOT:-reranker/xpool/video_features_cache/Xpool-Panda}
CANDIDATE_FILE=${CANDIDATE_FILE:-candidates/GRDR/panda/panda_t2_200_candidates.json}
RESULT_FILE=${RESULT_FILE:-reproduction/GRDR/panda/full_corpus/result.csv}
if [[ "$RESULT_FILE" = /* ]]; then
    RERANK_JSON=${RERANK_JSON:-${RESULT_FILE%.csv}.json}
else
    RERANK_JSON=${RERANK_JSON:-output/evaluation_results/rerank/${RESULT_FILE%.csv}.json}
fi
XPOOL_DEVICE=${XPOOL_DEVICE:-0}
LOAD_WORKERS=${LOAD_WORKERS:-16}

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
    --code_num 4096 \
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
    conda run -n "$XPOOL_ENV" python reranker/xpool/candidate_rerank.py \
    --eval_checkpoint "$XPOOL_CKPT" \
    --candidate_file "$CANDIDATE_FILE" \
    --video_cache_dir "$XPOOL_CACHE_ROOT" \
    --num_frames 4 \
    --device "$XPOOL_DEVICE" \
    --batch_size "$RERANK_BATCH_SIZE" \
    --result_file "$RESULT_FILE" \
    --out_json "$RERANK_JSON" \
    --seed 42 \
    --load_workers "$LOAD_WORKERS"
