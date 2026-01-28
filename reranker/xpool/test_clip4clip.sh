#!/usr/bin/env bash

# MSRVTT
# python reranker/xpool/test.py \
#     --exp_name test \
#     --arch clip_baseline \
#     --pooling_type avg \
#     --batch_size 32 \
#     --huggingface \
#     --expanded_pool \
#     --dataset_name MSRVTT \
#     --msrvtt_train_file 9k \
#     --eval_checkpoint "reranker/xpool/clip4clip/msrvtt9k_model_best.pth" 

# # ACTNET
# python reranker/xpool/test.py \
#     --exp_name test \
#     --arch clip_baseline \
#     --pooling_type avg \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name ACTNET \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos' \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/clip4clip/actnet_model_best.pth" \

# DIDEMO
python reranker/xpool/test.py \
    --exp_name test \
    --arch clip_baseline \
    --pooling_type avg \
    --batch_size 32 \
    --huggingface \
    --dataset_name DIDEMO \
    --videos_dir '/data2/uqzzha35/VideoRetrieval/DiDeMo' \
    --expanded_pool \
    --eval_checkpoint "reranker/xpool/clip4clip/didemo_model_best.pth" 

# LSMDC
python reranker/xpool/test.py \
    --exp_name test \
    --arch clip_baseline \
    --pooling_type avg \
    --batch_size 32 \
    --huggingface \
    --dataset_name LSMDC \
    --videos_dir '/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos' \
    --expanded_pool \
    --eval_checkpoint "reranker/xpool/clip4clip/lsmdc_model_best.pth" 