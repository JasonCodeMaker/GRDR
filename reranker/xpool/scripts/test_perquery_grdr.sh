#!/usr/bin/env bash

DEVICE=0

## Setting 1: Inductive setting
# MSRVTT
VIDEOS_DIR="dataset/msrvtt_data/MSRVTT_Videos"
CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
CANDIDATE_FILE="candidates/msrvtt_c128l3_100_candidates_t1.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name MSRVTT \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/MSRVTT \
    --seed 42

# ACTNET
VIDEOS_DIR="dataset/ActivityNet/Activity_Videos"
CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
CANDIDATE_FILE="candidates/actnet_c128l3_100_candidates_t1.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name ACTNET \
    --expanded_pool \
    --max_queries 1000 \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/ACTNET \
    --seed 42

# DIDEMO
VIDEOS_DIR="dataset/DiDeMo"
CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
CANDIDATE_FILE="candidates/didemo_c96l3_100_candidates_t1.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name DIDEMO \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/DIDEMO \
    --seed 42

# LSMDC
VIDEOS_DIR="dataset/LSMDC/LSMDC_Videos"
CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
CANDIDATE_FILE="candidates/lsmdc_c200l3_100_candidates_t1.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name LSMDC \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/LSMDC \
    --seed 42

## Setting 2: Full-corpus setting
# MSRVTT
VIDEOS_DIR="dataset/MSRVTT/MSRVTT_Videos"
CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
CANDIDATE_FILE="candidates/msrvtt_c128l3_20_candidates_t2.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name MSRVTT \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/MSRVTT \
    --seed 42

# ACTNET
VIDEOS_DIR="dataset/ActivityNet/Activity_Videos"
CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
CANDIDATE_FILE="candidates/actnet_c128l3_50_candidates_t2.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name ACTNET \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/ACTNET \
    --seed 42

# DIDEMO
VIDEOS_DIR="dataset/DiDeMo"
CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
CANDIDATE_FILE="candidates/didemo_c96l3_50_candidates_t2.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name DIDEMO \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/DIDEMO \
    --seed 42

# LSMDC
VIDEOS_DIR="dataset/LSMDC/LSMDC_Videos"
CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
CANDIDATE_FILE="candidates/lsmdc_c200l3_20_candidates_t2.json"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name LSMDC \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/LSMDC \
    --seed 42