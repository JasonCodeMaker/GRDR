#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
DATASET="${DATASET:-lsmdc}"
CODE_NUM="${CODE_NUM:-200}"
MAX_LENGTH="${MAX_LENGTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-41}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
MAIN_LR="${MAIN_LR:-1e-4}"
FIT_LR="${FIT_LR:-1e-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
MAIN_EPOCHS="${MAIN_EPOCHS:-2}"
FIT_EPOCHS="${FIT_EPOCHS:-2}"
SAVE_PATH="${SAVE_PATH:-output/GRDR}"
EXP_NAME="${EXP_NAME:-$DATASET}"
SEED="${SEED:-42}"
ENABLE_FIT="${ENABLE_FIT:-1}"
USE_PSEUDO_QUERIES="${USE_PSEUDO_QUERIES:-0}"

fit_flag=()
if [[ "$ENABLE_FIT" == "0" ]]; then
    fit_flag+=(--no-enable_fit)
else
    fit_flag+=(--enable_fit)
fi

pseudo_query_flag=()
if [[ "$USE_PSEUDO_QUERIES" == "1" ]]; then
    pseudo_query_flag+=(--use_pseudo_queries)
fi

python run.py \
    --device "$DEVICE" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
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
    "${fit_flag[@]}" \
    "${pseudo_query_flag[@]}"
