#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/_common.sh
source_conda_sh

DEVICE="${DEVICE:-0}"
NUM_WARMUP="${NUM_WARMUP:-10}"
INDEX_TYPES_STR="${INDEX_TYPES:-flat hnsw ivf}"
STAGE2_INDEX_TYPE="${STAGE2_INDEX_TYPE:-flat}"
STAGE2_NUM_CANDIDATES="${STAGE2_NUM_CANDIDATES:-100}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/ann_latency}"
STAGE1_ROOT="${STAGE1_ROOT:-$OUTPUT_ROOT/stage1}"
STAGE2_ROOT="${STAGE2_ROOT:-$OUTPUT_ROOT/stage2}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$OUTPUT_ROOT/candidates}"
SUMMARY_CSV="${SUMMARY_CSV:-$OUTPUT_ROOT/ann_per_query_latency_summary.csv}"
STAGE2_CSV="${STAGE2_CSV:-$OUTPUT_ROOT/ann_stage2_latency_once_per_dataset.csv}"
COMPACT_CSV="${COMPACT_CSV:-$OUTPUT_ROOT/ann_per_query_latency_compact.csv}"

read -r -a INDEX_TYPES <<<"$INDEX_TYPES_STR"
DATASETS=(msrvtt actnet didemo lsmdc)

ckpt_for() {
  case "$1" in
    msrvtt) printf '%s\n' "reranker/xpool/ckpt/msrvtt9k_model_best.pth" ;;
    actnet) printf '%s\n' "reranker/xpool/ckpt/actnet_model_best.pth" ;;
    didemo) printf '%s\n' "reranker/xpool/ckpt/didemo_model_best.pth" ;;
    lsmdc) printf '%s\n' "reranker/xpool/ckpt/lsmdc_model_best.pth" ;;
  esac
}

xpool_latency_args() {
  case "$1" in
    msrvtt)
      printf '%s\n' "MSRVTT;dataset/msrvtt_data/MSRVTT_Videos;reranker/xpool/video_features_cache/Xpool/MSRVTT;--msrvtt_train_file 9k --num_frames 12"
      ;;
    actnet)
      printf '%s\n' "ACTNET;dataset/ActivityNet/Activity_Videos;reranker/xpool/video_features_cache/Xpool/ACTNET;--num_frames 12"
      ;;
    didemo)
      printf '%s\n' "DIDEMO;dataset/DiDeMo;reranker/xpool/video_features_cache/Xpool/DIDEMO;--num_frames 16"
      ;;
    lsmdc)
      printf '%s\n' "LSMDC;dataset/LSMDC/LSMDC_Videos;reranker/xpool/video_features_cache/Xpool/LSMDC;--num_frames 12"
      ;;
  esac
}

resolve_stage2_videos_dir() {
  local requested_dir="$1"
  if [[ -d "$requested_dir" ]]; then
    printf '%s\n' "$requested_dir"
  else
    printf '%s\n' "dataset"
  fi
}

run_stage2_latency_once() {
  local dataset="$1"
  local candidate_file="$CANDIDATE_DIR/${dataset}_ann_${STAGE2_INDEX_TYPE}_${STAGE2_NUM_CANDIDATES}_candidates_t1.json"
  local triple ds_name videos_dir cache_dir extra_str
  local ckpt
  triple=$(xpool_latency_args "$dataset")
  IFS=';' read -r ds_name videos_dir cache_dir extra_str <<<"$triple"
  videos_dir=$(resolve_stage2_videos_dir "$videos_dir")
  ckpt=$(ckpt_for "$dataset")

  require_file "$candidate_file"
  require_file "$ckpt"
  require_dir "$videos_dir"
  require_dir "$cache_dir"

  activate_conda_env "$XPOOL_ENV"
  # shellcheck disable=SC2206
  local extra=( $extra_str )
  run_cmd env CUDA_VISIBLE_DEVICES="$DEVICE" python reranker/xpool/test_perquery.py \
    --dataset_name "$ds_name" \
    --videos_dir "$videos_dir" \
    --huggingface \
    --checkpoint "$ckpt" \
    --cache_dir "$cache_dir" \
    --candidate_file "$candidate_file" \
    --report_dir "$STAGE2_ROOT/$dataset" \
    --seed 42 \
    "${extra[@]}"
  deactivate_conda_env
}

if [[ "$RUN_STAGE1" == "1" ]]; then
  for index_type in "${INDEX_TYPES[@]}"; do
    echo ""
    echo "########################################################################"
    echo "Stage 1 per-query latency | INDEX_TYPE=${index_type}"
    echo "########################################################################"
    INDEX_TYPE="$index_type" \
    OUTPUT_DIR="$STAGE1_ROOT/$index_type" \
    CANDIDATE_DIR="$CANDIDATE_DIR" \
    RUN_STAGE1=1 \
    RUN_STAGE2=0 \
    PER_QUERY_TIMING=1 \
    NUM_WARMUP="$NUM_WARMUP" \
    DEVICE="$DEVICE" \
    bash scripts/baselines/eval_ann.sh
  done
fi

if [[ "$RUN_STAGE2" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "########################################################################"
    echo "Stage 2 per-query latency | dataset=${dataset} | candidates=${STAGE2_NUM_CANDIDATES}"
    echo "########################################################################"
    run_stage2_latency_once "$dataset"
  done
fi

activate_conda_env "$ANN_BASELINE_ENV"
run_cmd python scripts/baselines/summarize_ann_latency_results.py \
  --stage1_root "$STAGE1_ROOT" \
  --stage2_root "$STAGE2_ROOT" \
  --output "$SUMMARY_CSV" \
  --stage2-output "$STAGE2_CSV" \
  --compact-output "$COMPACT_CSV"
deactivate_conda_env

echo ""
echo "Latency summary CSV written to: $SUMMARY_CSV"
echo "Stage 2 latency CSV written to: $STAGE2_CSV"
echo "Compact latency CSV written to: $COMPACT_CSV"
