#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
DATASET="${DATASET:-msrvtt}"
CODE_NUM="${CODE_NUM:-128}"
MAX_LENGTH="${MAX_LENGTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_CANDIDATES="${NUM_CANDIDATES:-20}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-4}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
MAIN_LR="${MAIN_LR:-1e-4}"
FIT_LR="${FIT_LR:-1e-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
MAIN_EPOCHS="${MAIN_EPOCHS:-2}"
FIT_EPOCHS="${FIT_EPOCHS:-2}"
SAVE_PATH="${SAVE_PATH:-output/checkpoints/GRDR/msrvtt/bucket_candidate_k20}"
EXP_NAME="${EXP_NAME:-fit_bucket_l010_g10_k20_s42}"
SEED="${SEED:-42}"
ENABLE_FIT="${ENABLE_FIT:-1}"
USE_PSEUDO_QUERIES="${USE_PSEUDO_QUERIES:-0}"
W2_ROUTE_AGREE_LOSS="${W2_ROUTE_AGREE_LOSS:-0}"
W2_BUCKET_ROUTE_LOSS="${W2_BUCKET_ROUTE_LOSS:-0}"
W2_VIDEO_RANK_LOSS="${W2_VIDEO_RANK_LOSS:-0}"
W2_EXPANDED_SIZE_LOSS="${W2_EXPANDED_SIZE_LOSS:-0}"
W3_ROUTE_AGREE_LOSS="${W3_ROUTE_AGREE_LOSS:-0}"
W3_BUCKET_ROUTE_LOSS="${W3_BUCKET_ROUTE_LOSS:-0.10}"
W3_VIDEO_RANK_LOSS="${W3_VIDEO_RANK_LOSS:-0}"
W3_EXPANDED_SIZE_LOSS="${W3_EXPANDED_SIZE_LOSS:-0}"
ROUTE_BUCKET_GAMMA="${ROUTE_BUCKET_GAMMA:-1.0}"
VIDEO_RANK_BETA="${VIDEO_RANK_BETA:-0.5}"
ROUTE_AGREE_STOPGRAD_VIDEO="${ROUTE_AGREE_STOPGRAD_VIDEO:-1}"

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

stopgrad_flag=()
if [[ "$ROUTE_AGREE_STOPGRAD_VIDEO" == "0" ]]; then
    stopgrad_flag+=(--no-route_agree_stopgrad_video)
else
    stopgrad_flag+=(--route_agree_stopgrad_video)
fi

python run.py \
    --device "$DEVICE" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --code_num "$CODE_NUM" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_candidates "$NUM_CANDIDATES" \
    --num_latent_tokens "$NUM_LATENT_TOKENS" \
    --pretrain_lr "$PRETRAIN_LR" \
    --main_lr "$MAIN_LR" \
    --fit_lr "$FIT_LR" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --main_epochs "$MAIN_EPOCHS" \
    --fit_epochs "$FIT_EPOCHS" \
    --w2_route_agree_loss "$W2_ROUTE_AGREE_LOSS" \
    --w2_bucket_route_loss "$W2_BUCKET_ROUTE_LOSS" \
    --w2_video_rank_loss "$W2_VIDEO_RANK_LOSS" \
    --w2_expanded_size_loss "$W2_EXPANDED_SIZE_LOSS" \
    --w3_route_agree_loss "$W3_ROUTE_AGREE_LOSS" \
    --w3_bucket_route_loss "$W3_BUCKET_ROUTE_LOSS" \
    --w3_video_rank_loss "$W3_VIDEO_RANK_LOSS" \
    --w3_expanded_size_loss "$W3_EXPANDED_SIZE_LOSS" \
    --route_bucket_gamma "$ROUTE_BUCKET_GAMMA" \
    --video_rank_beta "$VIDEO_RANK_BETA" \
    --save_path "$SAVE_PATH" \
    --exp_name "$EXP_NAME" \
    --seed "$SEED" \
    "${fit_flag[@]}" \
    "${stopgrad_flag[@]}" \
    "${pseudo_query_flag[@]}"
