#!/usr/bin/env bash

# # MSRVTT
# python run.py \
#     --device 0 \
#     --model_name t5-small \
#     --dataset msrvtt \
#     --code_num 128 \
#     --max_length 3 \
#     --batch_size 512 \
#     --num_latent_tokens 4 \
#     --pretrain_lr 1e-4 \
#     --main_lr 1e-4 \
#     --fit_lr 1e-4 \
#     --pretrain_epochs 1 \
#     --main_epochs 2 \
#     --fit_epochs 2 \
#     --save_path output/GRDR \
#     --exp_name msrvtt \
#     --use_pseudo_queries \
#     --seed 42

# # ActivityNet
# python run.py \
#     --device 0 \
#     --model_name t5-small \
#     --dataset actnet \
#     --code_num 128 \
#     --max_length 3 \
#     --batch_size 512 \
#     --num_latent_tokens 4 \
#     --pretrain_lr 1e-4 \
#     --main_lr 1e-4 \
#     --fit_lr 1e-4 \
#     --pretrain_epochs 1 \
#     --main_epochs 2 \
#     --fit_epochs 2 \
#     --save_path output/GRDR \
#     --exp_name actnet \
#     --use_pseudo_queries \
#     --seed 42

# # DiDeMo
# python run.py \
#     --device 0 \
#     --model_name t5-small \
#     --dataset didemo \
#     --code_num 96 \
#     --max_length 3 \
#     --batch_size 512 \
#     --num_latent_tokens 4 \
#     --pretrain_lr 1e-4 \
#     --main_lr 1e-4 \
#     --fit_lr 1e-4 \
#     --pretrain_epochs 1 \
#     --main_epochs 2 \
#     --fit_epochs 2 \
#     --save_path output/GRDR \
#     --exp_name didemo \
#     --use_pseudo_queries \
#     --seed 42

# LSMDC
python run.py \
    --device 0 \
    --model_name t5-small \
    --dataset lsmdc \
    --code_num 200 \
    --max_length 3 \
    --batch_size 512 \
    --num_latent_tokens 41\
    --pretrain_lr 1e-4 \
    --main_lr 1e-4 \
    --fit_lr 1e-4 \
    --pretrain_epochs 1 \
    --main_epochs 2 \
    --fit_epochs 2 \
    --save_path output/GRDR \
    --exp_name lsmdc \
    --seed 42
