#!/usr/bin/env bash

source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh

DEVICE=0

# MSRVTT
EVAL_CHECKPOINT="output/GRDR/msrvtt/best_model/best_model.pt"
CODE_BOOK_SIZE=128   
CODE_LENGTH=3        
BATCH_SIZE=1      
NUM_CANDIDATES=(20)
for num_candidates in "${NUM_CANDIDATES[@]}"; do
    conda activate semantictvr
    python run.py \
        --device "${DEVICE}" \
        --dataset msrvtt \
        --num_latent_tokens 4 \
        --code_num "${CODE_BOOK_SIZE}" \
        --max_length "${CODE_LENGTH}" \
        --eval_checkpoint "${EVAL_CHECKPOINT}" \
        --batch_size "${BATCH_SIZE}" \
        --num_candidates "${num_candidates}" \
        --setting 2 \
        --eval
    conda deactivate

    CANDIDATE_FILE=candidates/msrvtt_c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates_t2.json
    RESULT_FILE=msrvtt/c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates.csv
    XPOOL_CHECKPOINT="reranker/xpool/ckpt/msrvtt9k_model_best.pth"

    conda activate xpool
    CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test.py \
        --exp_name test \
        --batch_size 32 \
        --huggingface \
        --expanded_pool \
        --dataset_name MSRVTT \
        --videos_dir dataset/msrvtt_data/MSRVTT_Videos \
        --msrvtt_train_file 9k \
        --eval_checkpoint "${XPOOL_CHECKPOINT}" \
        --rerank_mode \
        --candidate_file $CANDIDATE_FILE \
        --result_file $RESULT_FILE
    conda deactivate
done

# ACTNET
EVAL_CHECKPOINT="output/GRDR/actnet/best_model/best_model.pt"
CODE_BOOK_SIZE=128   
CODE_LENGTH=3        
BATCH_SIZE=32      
NUM_CANDIDATES=(50)
for num_candidates in "${NUM_CANDIDATES[@]}"; do
    conda activate semantictvr
    python run.py \
        --device "${DEVICE}" \
        --dataset actnet \
        --num_latent_tokens 4 \
        --code_num "${CODE_BOOK_SIZE}" \
        --max_length "${CODE_LENGTH}" \
        --eval_checkpoint "${EVAL_CHECKPOINT}" \
        --batch_size "${BATCH_SIZE}" \
        --num_candidates "${num_candidates}" \
        --setting 2 \
        --eval
    conda deactivate

    CANDIDATE_FILE=candidates/actnet_c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates_t2.json
    RESULT_FILE=actnet/c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates.csv
    XPOOL_CHECKPOINT="reranker/xpool/ckpt/actnet_model_best.pth"

    conda activate xpool
    CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test.py \
        --exp_name test \
        --batch_size 32 \
        --huggingface \
        --dataset_name ACTNET \
        --expanded_pool \
        --videos_dir dataset/ActivityNet/Activity_Videos \
        --eval_checkpoint "${XPOOL_CHECKPOINT}" \
        --rerank_mode \
        --candidate_file $CANDIDATE_FILE \
        --result_file $RESULT_FILE
    conda deactivate

done


# DIDEMO
EVAL_CHECKPOINT="output/GRDR/didemo/best_model/best_model.pt"
CODE_BOOK_SIZE=96   
CODE_LENGTH=3        
BATCH_SIZE=32      
NUM_CANDIDATES=(50)
for num_candidates in "${NUM_CANDIDATES[@]}"; do
    conda activate semantictvr
    python run.py \
        --device "${DEVICE}" \
        --dataset didemo \
        --num_latent_tokens 4 \
        --code_num "${CODE_BOOK_SIZE}" \
        --max_length "${CODE_LENGTH}" \
        --eval_checkpoint "${EVAL_CHECKPOINT}" \
        --batch_size "${BATCH_SIZE}" \
        --num_candidates "${num_candidates}" \
        --setting 2 \
        --eval
    conda deactivate

    CANDIDATE_FILE=candidates/didemo_c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates_t2.json
    RESULT_FILE=didemo/c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates.csv
    XPOOL_CHECKPOINT="reranker/xpool/ckpt/didemo_model_best.pth"

    conda activate xpool
    CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test.py \
        --exp_name test \
        --batch_size 32 \
        --huggingface \
        --dataset_name DIDEMO \
        --expanded_pool \
        --videos_dir dataset/DiDeMo \
        --eval_checkpoint "${XPOOL_CHECKPOINT}" \
        --rerank_mode \
        --candidate_file $CANDIDATE_FILE \
        --result_file $RESULT_FILE
    conda deactivate

done

# LSMDC
EVAL_CHECKPOINT="output/GRDR/lsmdc/best_model/best_model.pt"
CODE_BOOK_SIZE=200   
CODE_LENGTH=3        
BATCH_SIZE=32      
NUM_CANDIDATES=(20)
for num_candidates in "${NUM_CANDIDATES[@]}"; do
    conda activate semantictvr
    python run.py \
        --device "${DEVICE}" \
        --dataset lsmdc \
        --num_latent_tokens 1 \
        --code_num "${CODE_BOOK_SIZE}" \
        --max_length "${CODE_LENGTH}" \
        --eval_checkpoint "${EVAL_CHECKPOINT}" \
        --batch_size "${BATCH_SIZE}" \
        --num_candidates "${num_candidates}" \
        --setting 2 \
        --eval
    conda deactivate

    CANDIDATE_FILE=candidates/lsmdc_c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates_t2.json
    RESULT_FILE=lsmdc/c${CODE_BOOK_SIZE}l${CODE_LENGTH}_${num_candidates}_candidates.csv
    XPOOL_CHECKPOINT="reranker/xpool/ckpt/lsmdc_model_best.pth"

    conda activate xpool
    CUDA_VISIBLE_DEVICES="${DEVICE}" python reranker/xpool/test.py \
        --exp_name test \
        --batch_size 32 \
        --huggingface \
        --dataset_name LSMDC \
        --expanded_pool \
        --videos_dir dataset/LSMDC/LSMDC_Videos \
        --eval_checkpoint "${XPOOL_CHECKPOINT}" \
        --rerank_mode \
        --candidate_file $CANDIDATE_FILE \
        --result_file $RESULT_FILE
    conda deactivate

done


