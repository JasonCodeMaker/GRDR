#!/usr/bin/env bash
set -euo pipefail

python run.py \
    --exp_name panda_train \
    --device 1 \
    --model_name t5-small \
    --dataset panda \
    --features_root dataset/features \
    --code_num 256 \
    --max_length 3 \
    --batch_size 512 \
    --num_latent_tokens 4 \
    --pretrain_lr 1e-4 \
    --main_lr 1e-4 \
    --fit_lr 5e-5 \
    --pretrain_epochs 1 \
    --main_epochs 1 \
    --fit_epochs 1 \
    --save_path output/GRDR \
    --seed 42 \
    --w2_rq_loss 0.5 \
    --use_pseudo_queries \
    --enable_fit
