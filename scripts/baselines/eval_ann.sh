#!/usr/bin/env bash
# ANN dense-retrieval baseline: Stage 1 candidate selection + Stage 2 X-Pool reranking.
# Stage 2 invocation matches scripts/eval_t1.sh / eval_t2.sh exactly.
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/_common.sh
source_conda_sh

DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
OUTPUT_DIR="${OUTPUT_DIR:-output/evaluation_results/ann_baseline}"
CANDIDATE_DIR="${CANDIDATE_DIR:-candidates}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
PER_QUERY_TIMING="${PER_QUERY_TIMING:-1}"
NUM_WARMUP="${NUM_WARMUP:-0}"

# ANN baseline uses 100 candidates in both settings for all datasets.
DEFAULT_PAIRS=(
  "msrvtt:1:100"  "msrvtt:2:100"
  "actnet:1:100"  "actnet:2:100"
  "didemo:1:100"  "didemo:2:100"
  "lsmdc:1:100"   "lsmdc:2:100"
)

if [[ -n "${PAIRS_OVERRIDE:-}" ]]; then
  read -r -a PAIRS <<<"$PAIRS_OVERRIDE"
else
  PAIRS=("${DEFAULT_PAIRS[@]}")
fi

ckpt_for() {
  case "$1" in
    msrvtt) printf '%s\n' "reranker/xpool/ckpt/msrvtt9k_model_best.pth" ;;
    actnet) printf '%s\n' "reranker/xpool/ckpt/actnet_model_best.pth" ;;
    didemo) printf '%s\n' "reranker/xpool/ckpt/didemo_model_best.pth" ;;
    lsmdc)  printf '%s\n' "reranker/xpool/ckpt/lsmdc_model_best.pth" ;;
  esac
}

xpool_dataset_args() {
  local dataset="$1" setting="$2"
  case "$dataset" in
    msrvtt) printf '%s\n' "MSRVTT;dataset/msrvtt_data/MSRVTT_Frames;--msrvtt_train_file 9k" ;;
    actnet) printf '%s\n' "ACTNET;dataset/ActivityNet/Activity_Frames;" ;;
    didemo) printf '%s\n' "DIDEMO;dataset/DiDeMo;" ;;
    lsmdc)  printf '%s\n' "LSMDC;dataset/LSMDC/LSMDC_Frames_256;" ;;
  esac
}

xpool_num_frames_for() {
  local dataset="$1"
  case "$dataset" in
    didemo) printf '%s\n' "${DIDEMO_XPOOL_NUM_FRAMES:-16}" ;;
    *) printf '%s\n' "${XPOOL_NUM_FRAMES:-12}" ;;
  esac
}

resolve_xpool_media_dir() {
  local dataset="$1" requested_dir="$2"
  local -a candidates=()
  case "$dataset" in
    msrvtt)
      candidates=(
        "$requested_dir"
        "${requested_dir/MSRVTT_Frames/MSRVTT_Videos}"
      )
      ;;
    actnet)
      candidates=(
        "$requested_dir"
        "${requested_dir/Activity_Frames/Activity_Videos}"
      )
      ;;
    didemo)
      candidates=("$requested_dir")
      ;;
    lsmdc)
      candidates=(
        "$requested_dir"
        "${requested_dir/LSMDC_Frames_256/LSMDC_Videos}"
      )
      ;;
  esac

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  printf '%s\n' "$requested_dir"
}

run_ann_recall() {
  local dataset="$1" setting="$2" k="$3" ckpt
  ckpt=$(ckpt_for "$dataset")
  require_file "$ckpt"
  local extra=()
  if [[ "$PER_QUERY_TIMING" == "1" ]]; then
    extra+=(--per_query_timing --num_warmup "$NUM_WARMUP")
  fi
  # Stage 1 uses the repo's FAISS-enabled retrieval stack, not the X-Pool env.
  activate_conda_env "$ANN_BASELINE_ENV"
  run_cmd python -m baselines.ann_dense_retrieval.eval_ann \
    --dataset "$dataset" \
    --setting "$setting" \
    --index_type "$INDEX_TYPE" \
    --checkpoint "$ckpt" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_candidates "$k" \
    --candidate_dir "$CANDIDATE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "${extra[@]}"
  deactivate_conda_env
}

run_xpool_rerank() {
  local dataset="$1" setting="$2" candidate_file="$3" k="$4"
  local triple ds_name videos_dir extra_str num_frames
  triple=$(xpool_dataset_args "$dataset" "$setting")
  IFS=';' read -r ds_name videos_dir extra_str <<<"$triple"
  videos_dir=$(resolve_xpool_media_dir "$dataset" "$videos_dir")
  num_frames=$(xpool_num_frames_for "$dataset")
  local -a xpool_extra=(--videos_dir "$videos_dir" --num_frames "$num_frames")
  if [[ -n "$extra_str" ]]; then
    # shellcheck disable=SC2206
    xpool_extra+=( $extra_str )
  fi
  if [[ "$setting" == "2" ]]; then
    xpool_extra+=(--expanded_pool)
  fi
  local ckpt
  ckpt=$(ckpt_for "$dataset")
  local result_file="${dataset}/ann_${INDEX_TYPE}_${k}_candidates_t${setting}.csv"
  require_file "$candidate_file"
  require_file "$ckpt"
  require_dir "$videos_dir"
  activate_conda_env "$XPOOL_ENV"
  run_cmd env CUDA_VISIBLE_DEVICES="$DEVICE" python reranker/xpool/test.py \
    --exp_name test \
    --batch_size 32 \
    --huggingface \
    --dataset_name "$ds_name" \
    "${xpool_extra[@]}" \
    --eval_checkpoint "$ckpt" \
    --rerank_mode \
    --candidate_file "$candidate_file" \
    --result_file "$result_file"
  deactivate_conda_env
}

for spec in "${PAIRS[@]}"; do
  IFS=':' read -r dataset setting k <<<"$spec"
  candidate_file="${CANDIDATE_DIR}/${dataset}_ann_${INDEX_TYPE}_${k}_candidates_t${setting}.json"
  echo ""
  echo "========================================================================"
  echo "  ${dataset} | setting=${setting} | index=${INDEX_TYPE} | K=${k} | PQT=${PER_QUERY_TIMING}"
  echo "========================================================================"
  if [[ "$RUN_STAGE1" == "1" ]]; then
    echo "--- Stage 1: ANN Recall ---"
    run_ann_recall "$dataset" "$setting" "$k"
  fi
  if [[ "$RUN_STAGE2" == "1" ]]; then
    echo "--- Stage 2: X-Pool Reranking ---"
    run_xpool_rerank "$dataset" "$setting" "$candidate_file" "$k"
  fi
done
