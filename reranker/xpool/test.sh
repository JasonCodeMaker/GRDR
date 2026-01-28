#!/usr/bin/env bash
# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name MSRVTT \
#     --msrvtt_train_file 9k \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/ckpt/msrvtt9k_model_best.pth" \
#     --rerank_mode \
#     --candidate_file 'candidates/t2vindexer/msrvtt_t2vindexer_100_candidates_t2.json' 

# # ACTNET
# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name ACTNET \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos' \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/ckpt/actnet_model_best.pth" \
#     --rerank_mode \
#     --candidate_file 'candidates/t2vindexer/actnet_t2vindexer_100_candidates_t2.json' 

# # DIDEMO
# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name DIDEMO \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/DiDeMo' \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/ckpt/didemo_model_best.pth" \
#     --rerank_mode \
#     --candidate_file 'candidates/t2vindexer/didemo_t2vindexer_100_candidates_t2.json' 

# LSMDC
python reranker/xpool/test.py \
    --exp_name test \
    --batch_size 32 \
    --huggingface \
    --dataset_name LSMDC \
    --videos_dir '/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos' \
    --expanded_pool \
    --eval_checkpoint "reranker/xpool/ckpt/lsmdc_model_best.pth" \
    --rerank_mode \
    --candidate_file 'candidates/t2vindexer/lsmdc_t2vindexer_100_candidates_t2.json' 

# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name ACTNET \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos' \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/ckpt/actnet_model_best.pth" 

# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name DIDEMO \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/DiDeMo' \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/ckpt/didemo_model_best.pth" 

# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name LSMDC \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos' \
#     --expanded_pool \
#     --eval_checkpoint "reranker/xpool/ckpt/lsmdc_model_best.pth" 


# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name LSMDC \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Videos' \
#     --eval_checkpoint 'reranker/xpool/ckpt/lsmdc_model_best.pth' \

# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name DIDEMO \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/DiDeMo' \
#     --eval_checkpoint 'outputs/didemo_xpool_clip_lr8e-7/model_best.pth' \


# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name ACTNET \
#     --videos_dir '/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Videos' \
#     --eval_checkpoint 'outputs/actnet_xpool_clip_lr8e-7/model_best.pth' \


# python test.py --exp_name=test --num_frames=24 --batch_size=32 --huggingface --dataset_name=ACTNET
# python reranker/xpool/test.py \
#     --exp_name test \
#     --batch_size 32 \
#     --huggingface \
#     --dataset_name MSRVTT \
#     --msrvtt_train_file 9k \
#     --eval_checkpoint 'reranker/xpool/ckpt/msrvtt9k_model_best.pth' \
#     --candidate_file 'candidates/msrvtt_videorqvae_c256l4_20251002_134651_candidates.json' 

# python test.py --exp_name=test --batch_size=32 --huggingface --dataset_name=MSVD
# python test.py --exp_name=test --batch_size=32 --huggingface --dataset_name=LSMDC

# python reranker/xpool/test_candidates.py \
#     --candidate_file candidates/msrvtt_videorqvae_c256l4_20251002_134651_candidates.json \
#     --rerank_mode \
#     --dataset_name MSRVTT \
#     --msrvtt_train_file=9k