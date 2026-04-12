#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "$ROOT_DIR"

source "$SCRIPT_DIR/_common.sh"
source_conda_sh

DATASET="${DATASET:-lsmdc}"
DEVICE="${DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
MAX_LENGTH="${MAX_LENGTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
MAIN_LR="${MAIN_LR:-1e-4}"
FIT_LR="${FIT_LR:-1e-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
MAIN_EPOCHS="${MAIN_EPOCHS:-2}"
FIT_EPOCHS="${FIT_EPOCHS:-2}"
SAVE_PATH="${SAVE_PATH:-output/GRDR}"
EXP_NAME="${EXP_NAME:-$DATASET}"
SEED="${SEED:-42}"

case "$DATASET" in
  msrvtt)
    CODE_NUM="${CODE_NUM:-128}"
    NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-4}"
    DEFAULT_USE_PSEUDO_QUERIES=1
    ;;
  actnet)
    CODE_NUM="${CODE_NUM:-128}"
    NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-4}"
    DEFAULT_USE_PSEUDO_QUERIES=1
    ;;
  didemo)
    CODE_NUM="${CODE_NUM:-96}"
    NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-4}"
    DEFAULT_USE_PSEUDO_QUERIES=1
    ;;
  lsmdc)
    CODE_NUM="${CODE_NUM:-200}"
    NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-1}"
    DEFAULT_USE_PSEUDO_QUERIES=0
    ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    exit 1
    ;;
esac

FEATURES_DATASET_DIR=$(feature_dataset_dir "$DATASET")
require_dir "$SEMANTIC_FEATURES_ROOT/InternVideo2/$FEATURES_DATASET_DIR"
mkdir -p "$SAVE_PATH"

USE_PSEUDO_QUERIES="${USE_PSEUDO_QUERIES:-$DEFAULT_USE_PSEUDO_QUERIES}"
PSEUDO_ARGS=()
if [[ "$USE_PSEUDO_QUERIES" == "1" ]]; then
  PSEUDO_ARGS+=(--use_pseudo_queries)
fi

activate_conda_env "$SEMANTICTVR_ENV"
run_cmd python run.py \
  --device "$DEVICE" \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --features_root "$SEMANTIC_FEATURES_ROOT" \
  --code_num "$CODE_NUM" \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --num_latent_tokens "$NUM_LATENT_TOKENS" \
  --pretrain_lr "$PRETRAIN_LR" \
  --main_lr "$MAIN_LR" \
  --fit_lr "$FIT_LR" \
  --pretrain_epochs "$PRETRAIN_EPOCHS" \
  --main_epochs "$MAIN_EPOCHS" \
  --fit_epochs "$FIT_EPOCHS" \
  --save_path "$SAVE_PATH" \
  --exp_name "$EXP_NAME" \
  --seed "$SEED" \
  "${PSEUDO_ARGS[@]}"
deactivate_conda_env
