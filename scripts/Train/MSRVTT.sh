#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

SEMANTICTVR_ENV=${SEMANTICTVR_ENV:-semantictvr}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RUN_DEVICE=${RUN_DEVICE:-0}
export WANDB_MODE=${WANDB_MODE:-offline}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

run env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" WANDB_MODE="$WANDB_MODE" \
    conda run -n "$SEMANTICTVR_ENV" python run.py \
    --model_name t5-small \
    --dataset msrvtt \
    --code_num 128 \
    --max_length 3 \
    --batch_size 512 \
    --eval_batch_size 32 \
    --num_candidates 20 \
    --num_latent_tokens 4 \
    --pretrain_lr 0.0001 \
    --main_lr 0.0001 \
    --fit_lr 0.0001 \
    --pretrain_epochs 1 \
    --main_epochs 2 \
    --fit_epochs 2 \
    --w2_route_agree_loss 0 \
    --w2_bucket_route_loss 0 \
    --w2_video_rank_loss 0 \
    --w2_expanded_size_loss 0 \
    --w3_route_agree_loss 0 \
    --w3_bucket_route_loss 0.10 \
    --w3_video_rank_loss 0 \
    --w3_expanded_size_loss 0 \
    --route_bucket_gamma 1.0 \
    --video_rank_beta 0.5 \
    --save_path output/checkpoints/GRDR/msrvtt/bucket_candidate_k20 \
    --exp_name GRDR_MSRVTT \
    --seed 42 \
    --device "$RUN_DEVICE" \
    --use_pseudo_queries \
    --enable_fit \
    --route_agree_stopgrad_video \
    --wandb_project GRDR-reproduction \
    --wandb_run_name GRDR_MSRVTT
