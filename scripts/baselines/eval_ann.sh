#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
source_conda_sh

DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_CANDIDATES="${NUM_CANDIDATES:-50}"
INDEX_TYPE="${INDEX_TYPE:-flat}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ann_baseline}"
CANDIDATE_DIR="${CANDIDATE_DIR:-candidates}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
DATASETS="${DATASETS:-msrvtt actnet didemo lsmdc}"
SETTINGS="${SETTINGS:-1 2}"

# ── Stage 1: ANN Recall ─────────────────────────────────────────────────────

run_ann_recall() {
  local dataset="$1"
  local setting="$2"
  local checkpoint

  case "$dataset" in
    msrvtt)  checkpoint="reranker/xpool/ckpt/msrvtt9k_model_best.pth" ;;
    actnet)  checkpoint="reranker/xpool/ckpt/actnet_model_best.pth" ;;
    didemo)  checkpoint="reranker/xpool/ckpt/didemo_model_best.pth" ;;
    lsmdc)   checkpoint="reranker/xpool/ckpt/lsmdc_model_best.pth" ;;
    *) echo "Unsupported dataset: $dataset" >&2; return 1 ;;
  esac

  require_file "$checkpoint"

  activate_conda_env "$XPOOL_ENV"
  run_cmd python -m baselines.ann_dense_retrieval.eval_ann \
    --dataset "$dataset" \
    --setting "$setting" \
    --index_type "$INDEX_TYPE" \
    --checkpoint "$checkpoint" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_candidates "$NUM_CANDIDATES" \
    --candidate_dir "$CANDIDATE_DIR" \
    --output_dir "$OUTPUT_DIR"
  deactivate_conda_env
}

# ── Stage 2: X-Pool Dense Reranking ─────────────────────────────────────────

run_xpool_rerank() {
  local dataset="$1"
  local setting="$2"
  local candidate_file="$3"
  local dataset_name checkpoint videos_dir
  local -a xpool_extra=()

  case "$dataset" in
    msrvtt)
      dataset_name="MSRVTT"
      checkpoint="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
      videos_dir="dataset/msrvtt_data/MSRVTT_Videos"
      if [[ "$setting" == "1" ]]; then
        xpool_extra=(--msrvtt_train_file 9k --videos_dir "$videos_dir")
      else
        xpool_extra=(--expanded_pool --msrvtt_train_file 9k --videos_dir "$videos_dir")
      fi
      ;;
    actnet)
      dataset_name="ACTNET"
      checkpoint="reranker/xpool/ckpt/actnet_model_best.pth"
      videos_dir="dataset/ActivityNet/Activity_Videos"
      if [[ "$setting" == "1" ]]; then
        xpool_extra=(--videos_dir "$videos_dir")
      else
        xpool_extra=(--expanded_pool --videos_dir "$videos_dir")
      fi
      ;;
    didemo)
      dataset_name="DIDEMO"
      checkpoint="reranker/xpool/ckpt/didemo_model_best.pth"
      videos_dir="dataset/DiDeMo"
      if [[ "$setting" == "1" ]]; then
        xpool_extra=(--videos_dir "$videos_dir")
      else
        xpool_extra=(--expanded_pool --videos_dir "$videos_dir")
      fi
      ;;
    lsmdc)
      dataset_name="LSMDC"
      checkpoint="reranker/xpool/ckpt/lsmdc_model_best.pth"
      videos_dir="dataset/LSMDC/LSMDC_Videos"
      if [[ "$setting" == "1" ]]; then
        xpool_extra=(--videos_dir "$videos_dir")
      else
        xpool_extra=(--expanded_pool --videos_dir "$videos_dir")
      fi
      ;;
    *) echo "Unsupported dataset: $dataset" >&2; return 1 ;;
  esac

  local result_file="${dataset}/ann_${INDEX_TYPE}_${NUM_CANDIDATES}_candidates_t${setting}.csv"

  require_file "$candidate_file"
  require_file "$checkpoint"
  require_dir "$videos_dir"

  activate_conda_env "$XPOOL_ENV"
  run_cmd env CUDA_VISIBLE_DEVICES="$DEVICE" python reranker/xpool/test.py \
    --exp_name test \
    --batch_size 32 \
    --huggingface \
    --dataset_name "$dataset_name" \
    "${xpool_extra[@]}" \
    --eval_checkpoint "$checkpoint" \
    --rerank_mode \
    --candidate_file "$candidate_file" \
    --result_file "$result_file"
  deactivate_conda_env
}

# ── Main loop ────────────────────────────────────────────────────────────────

for dataset in $DATASETS; do
  for setting in $SETTINGS; do
    echo ""
    echo "========================================================================"
    echo "  Dataset: $dataset | Setting: $setting | Index: $INDEX_TYPE | K=$NUM_CANDIDATES"
    echo "========================================================================"

    candidate_file="${CANDIDATE_DIR}/${dataset}_ann_${INDEX_TYPE}_${NUM_CANDIDATES}_candidates_t${setting}.json"

    if [[ "$RUN_STAGE1" == "1" ]]; then
      echo "--- Stage 1: ANN Recall ---"
      run_ann_recall "$dataset" "$setting"
    fi

    if [[ "$RUN_STAGE2" == "1" ]]; then
      echo "--- Stage 2: X-Pool Reranking ---"
      run_xpool_rerank "$dataset" "$setting" "$candidate_file"
    fi
  done
done
