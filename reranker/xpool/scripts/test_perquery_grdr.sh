#!/usr/bin/env bash

DEVICE=1

VIDEOS_DIR="dataset/msrvtt_data/MSRVTT_Videos"
CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"
CANDIDATE_FILE=${1:-"candidates/unimvp/msrvtt_videorqvae__c128l3_20_candidates_t2.json"}
echo "Running candidate reranking mode with: $CANDIDATE_FILE"
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
VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos"
CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
CANDIDATE_FILE=${1:-"candidates/actnet_videorqvae__c128l3_50_candidates_t2.json"}
echo "Running candidate reranking mode with: $CANDIDATE_FILE"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name ACTNET \
    --expanded_pool \
    --max_queries 1000 \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/ACTNET \
    --seed 42

# DIDEMO
VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/DiDeMo"
CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
CANDIDATE_FILE=${1:-"candidates/didemo_videorqvae__c96l3_50_candidates_t2.json"}
echo "Running candidate reranking mode with: $CANDIDATE_FILE"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name DIDEMO \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/DIDEMO \
    --seed 42

# LSMDC
VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos"
CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
CANDIDATE_FILE=${1:-"candidates/lsmdc_videorqvae__c200l3_50_candidates_t2.json"}
echo "Running candidate reranking mode with: $CANDIDATE_FILE"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name LSMDC \
    --expanded_pool \
    --videos_dir ${VIDEOS_DIR} \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --candidate_file ${CANDIDATE_FILE} \
    --cache_dir reranker/xpool/video_features_cache/LSMDC \
    --seed 42