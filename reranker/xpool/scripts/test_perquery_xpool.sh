#!/usr/bin/env bash

DEVICE=1
VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/MSRVTT/MSRVTT_Videos"
CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"

# # MSRVTT
# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name MSRVTT \
#     --videos_dir ${VIDEOS_DIR} \
#     --max_queries 20 \
#     --expanded_pool \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/MSRVTT \
#     --seed 42   

# # DIDEMO
# VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/DiDeMo"
# CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name DIDEMO \
#     --videos_dir ${VIDEOS_DIR} \
#     --max_queries 20 \
#     --expanded_pool \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/DIDEMO \
#     --seed 42

# # ACTNET
# VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos"
# CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name ACTNET \
#     --videos_dir ${VIDEOS_DIR} \
#     --max_queries 20 \
#     --expanded_pool \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/ACTNET \
#     --seed 42

# LSMDC
VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos"
CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
    --dataset_name LSMDC \
    --videos_dir ${VIDEOS_DIR} \
    --max_queries 5 \
    --expanded_pool \
    --huggingface \
    --checkpoint ${CHECKPOINT} \
    --cache_dir reranker/xpool/video_features_cache/Xpool/LSMDC \
    --seed 42



# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name MSRVTT \
#     --expanded_pool \
#     --max_queries 20 \
#     --videos_dir ${VIDEOS_DIR} \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/MSRVTT \
#     --seed 42   

# # ACTNET
# VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos"
# CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"
# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name ACTNET \
#     --expanded_pool \
#     --max_queries 20 \
#     --videos_dir ${VIDEOS_DIR} \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/ACTNET \
#     --seed 42

# # DIDEMO
# VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/DiDeMo"
# CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"
# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name DIDEMO \
#     --expanded_pool \
#     --max_queries 20 \
#     --videos_dir ${VIDEOS_DIR} \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/DIDEMO \
#     --seed 42

# # LSMDC
# VIDEOS_DIR="/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos"
# CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"
# CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test_perquery.py \
#     --dataset_name LSMDC \
#     --expanded_pool \
#     --max_queries 5 \
#     --videos_dir ${VIDEOS_DIR} \
#     --huggingface \
#     --checkpoint ${CHECKPOINT} \
#     --cache_dir reranker/xpool/video_features_cache/Xpool/LSMDC \
#     --seed 42