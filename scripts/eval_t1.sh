#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "$ROOT_DIR"

source "$SCRIPT_DIR/_common.sh"
source_conda_sh

DEVICE="${DEVICE:-0}"
RUN_GRDR_STAGE1="${RUN_GRDR_STAGE1:-1}"
RUN_XPOOL_STAGE2="${RUN_XPOOL_STAGE2:-1}"
DATASETS="${DATASETS:-lsmdc}"

run_grdr_eval() {
  local dataset="$1"
  local num_latent_tokens="$2"
  local code_num="$3"
  local code_length="$4"
  local batch_size="$5"
  local num_candidates="$6"
  local eval_checkpoint="$7"
  local features_dataset_dir
  features_dataset_dir=$(feature_dataset_dir "$dataset")

  require_file "$eval_checkpoint"
  require_dir "$SEMANTIC_FEATURES_ROOT/InternVideo2/$features_dataset_dir"

  activate_conda_env "$SEMANTICTVR_ENV"
  run_cmd python run.py \
    --device "$DEVICE" \
    --dataset "$dataset" \
    --features_root "$SEMANTIC_FEATURES_ROOT" \
    --num_latent_tokens "$num_latent_tokens" \
    --code_num "$code_num" \
    --max_length "$code_length" \
    --eval_checkpoint "$eval_checkpoint" \
    --batch_size "$batch_size" \
    --num_candidates "$num_candidates" \
    --setting 1 \
    --eval
  deactivate_conda_env
}

run_xpool_eval() {
  local dataset_name="$1"
  local candidate_file="$2"
  local result_file="$3"
  local checkpoint="$4"
  local videos_dir="$5"
  shift 5
  local extra_args=("$@")

  require_file "$candidate_file"
  require_file "$checkpoint"
  require_dir "$videos_dir"

  activate_conda_env "$XPOOL_ENV"
  run_cmd env CUDA_VISIBLE_DEVICES="$DEVICE" python reranker/xpool/test.py \
    --exp_name test \
    --batch_size 32 \
    --huggingface \
    --dataset_name "$dataset_name" \
    "${extra_args[@]}" \
    --eval_checkpoint "$checkpoint" \
    --rerank_mode \
    --candidate_file "$candidate_file" \
    --result_file "$result_file"
  deactivate_conda_env
}

for dataset in $DATASETS; do
  case "$dataset" in
    msrvtt)
      num_latent_tokens=4
      code_num=128
      code_length=3
      batch_size=1
      num_candidates=100
      eval_checkpoint="output/GRDR/msrvtt/best_model/best_model.pt"
      candidate_file="candidates/msrvtt_c${code_num}l${code_length}_${num_candidates}_candidates_t1.json"
      result_file="msrvtt/c${code_num}l${code_length}_${num_candidates}_candidates.csv"
      xpool_checkpoint="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
      videos_dir="dataset/msrvtt_data/MSRVTT_Videos"
      xpool_extra=(--msrvtt_train_file 9k --videos_dir "$videos_dir")
      dataset_name="MSRVTT"
      ;;
    actnet)
      num_latent_tokens=4
      code_num=128
      code_length=3
      batch_size=32
      num_candidates=100
      eval_checkpoint="output/GRDR/actnet/best_model/best_model.pt"
      candidate_file="candidates/actnet_c${code_num}l${code_length}_${num_candidates}_candidates_t1.json"
      result_file="actnet/c${code_num}l${code_length}_${num_candidates}_candidates.csv"
      xpool_checkpoint="reranker/xpool/ckpt/actnet_model_best.pth"
      videos_dir="dataset/ActivityNet/Activity_Videos"
      xpool_extra=(--videos_dir "$videos_dir")
      dataset_name="ACTNET"
      ;;
    didemo)
      num_latent_tokens=4
      code_num=96
      code_length=3
      batch_size=32
      num_candidates=100
      eval_checkpoint="output/GRDR/didemo/best_model/best_model.pt"
      candidate_file="candidates/didemo_c${code_num}l${code_length}_${num_candidates}_candidates_t1.json"
      result_file="didemo/c${code_num}l${code_length}_${num_candidates}_candidates.csv"
      xpool_checkpoint="reranker/xpool/ckpt/didemo_model_best.pth"
      videos_dir="dataset/DiDeMo"
      xpool_extra=(--videos_dir "$videos_dir")
      dataset_name="DIDEMO"
      ;;
    lsmdc)
      num_latent_tokens=1
      code_num=200
      code_length=3
      batch_size=32
      num_candidates=100
      eval_checkpoint="output/GRDR/lsmdc/best_model/best_model.pt"
      candidate_file="candidates/lsmdc_c${code_num}l${code_length}_${num_candidates}_candidates_t1.json"
      result_file="lsmdc/c${code_num}l${code_length}_${num_candidates}_candidates.csv"
      xpool_checkpoint="reranker/xpool/ckpt/lsmdc_model_best.pth"
      videos_dir="dataset/LSMDC/LSMDC_Videos"
      xpool_extra=(--videos_dir "$videos_dir")
      dataset_name="LSMDC"
      ;;
    *)
      echo "Unsupported dataset in DATASETS: $dataset" >&2
      exit 1
      ;;
  esac

  if [[ "$RUN_GRDR_STAGE1" == "1" ]]; then
    run_grdr_eval "$dataset" "$num_latent_tokens" "$code_num" "$code_length" "$batch_size" "$num_candidates" "$eval_checkpoint"
  fi

  if [[ "$RUN_XPOOL_STAGE2" == "1" ]]; then
    run_xpool_eval "$dataset_name" "$candidate_file" "$result_file" "$xpool_checkpoint" "$videos_dir" "${xpool_extra[@]}"
  fi
done
