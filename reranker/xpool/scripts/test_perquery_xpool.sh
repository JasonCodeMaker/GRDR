#!/usr/bin/env bash

DEVICE=0

## Setting 1: Inductive setting
# MSRVTT
VIDEOS_DIR="dataset/msrvtt_data/MSRVTT_Videos"
CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name MSRVTT \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/MSRVTT \
    --seed 42   

# DIDEMO
VIDEOS_DIR="dataset/DiDeMo"
CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name DIDEMO \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/DIDEMO \
    --seed 42

# ACTNET
VIDEOS_DIR="dataset/ActivityNet/Activity_Videos"
CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name ACTNET \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/ACTNET \
    --seed 42

# LSMDC
VIDEOS_DIR="dataset/LSMDC/LSMDC_Videos"
CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name LSMDC \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/LSMDC \
    --seed 42

## Setting 2: Full-corpus setting
# MSRVTT
VIDEOS_DIR="dataset/MSRVTT/MSRVTT_Videos"
CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name MSRVTT \
    --videos_dir ${VIDEOS_DIR} \
    --expanded_pool \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/MSRVTT \
    --seed 42

# ACTNET
VIDEOS_DIR="dataset/ActivityNet/Activity_Videos"
CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name ACTNET \
    --videos_dir ${VIDEOS_DIR} \
    --expanded_pool \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/ACTNET \
    --seed 42

# DIDEMO
VIDEOS_DIR="dataset/DiDeMo"
CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name DIDEMO \
    --videos_dir ${VIDEOS_DIR} \
    --expanded_pool \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/DIDEMO \
    --seed 42

# LSMDC
VIDEOS_DIR="dataset/LSMDC/LSMDC_Videos"
CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name LSMDC \
    --videos_dir ${VIDEOS_DIR} \
    --expanded_pool \
    --max_queries 5 \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/LSMDC \
    --seed 42