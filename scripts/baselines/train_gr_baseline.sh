#!/usr/bin/env bash
# End-to-end GR-baseline training (TIGER / AVG) at C=128, L=3, in-distribution.
# Three stages, per the Effectiveness-Efficiency Figure spec:
#   1. tokenizer  : train_rqvae.py        -> RQ-VAE quantizer  (index/log/<ds>/<type>/...)
#   2. sID dataset: gen_sid_rqvae.py      -> per-video sIDs     (data/<ds>/none/<type>_c128_l3/...)
#   3. GR model   : avg_train_retriever_t5.py -> T5 retriever    (output/checkpoints/Baseline_c128l3/<baseline>/...)
# Each stage skips if its artifact already exists. Driven by env vars:
#   DATASET (msrvtt|actnet|didemo|lsmdc)  INDEX_TYPE (standard|text_guided)  DEVICE  SEED
# Called by the thin train_tiger.sh / train_avg.sh wrappers.
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MM_TVR_DIR=${MM_TVR_DIR:-/home/uqzzha35/Project/SemanticID/MM-SemanticTVR}
CONDA_SH=${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-semantictvr}

DATASET=${DATASET:?DATASET required (msrvtt|actnet|didemo|lsmdc)}
INDEX_TYPE=${INDEX_TYPE:?INDEX_TYPE required (standard|text_guided)}
DEVICE=${DEVICE:-0}
SEED=${SEED:-42}
CODE_NUM=${CODE_NUM:-128}
LAYERS=${LAYERS:-3}
baseline=$([ "${INDEX_TYPE}" = "standard" ] && echo tiger || echo avg)

# Dataset-name quirks: actnet trains/encodes under the "activitynet" feature+ckpt name
# (index prefix is activitynet) but the data + retriever dir use "actnet".
case "${DATASET}" in
    actnet) GEN_DS=activitynet; DATA_DS=actnet;  RET_EPOCH=50 ;;
    msrvtt) GEN_DS=msrvtt;      DATA_DS=msrvtt;  RET_EPOCH=20 ;;
    didemo) GEN_DS=didemo;      DATA_DS=didemo;  RET_EPOCH=50 ;;
    lsmdc)  GEN_DS=lsmdc;       DATA_DS=lsmdc;   RET_EPOCH=30 ;;
    *) echo "Unknown DATASET: ${DATASET}" >&2; exit 2 ;;
esac

# Tokenizer hyperparameters match index/scripts/tune_<ds>_<type>.sh, codebook -> 128/3.
if [ "${INDEX_TYPE}" = "text_guided" ]; then TOK_LR=5e-4; TG_FLAG=(--text_guided); TOK_DROPOUT=0.05
else TOK_LR=1e-4; TG_FLAG=(); TOK_DROPOUT=0.0; fi

TOK_CKPT="${MM_TVR_DIR}/index/log/${GEN_DS}/${INDEX_TYPE}/code_num_${CODE_NUM}_codebook_layers_${LAYERS}/best_collision_model.pth"
IDX_PREFIX=$([ "${DATASET}" = "actnet" ] && echo activitynet || echo "${DATASET}")
TRAIN_IDX="${MM_TVR_DIR}/data/${DATA_DS}/none/${INDEX_TYPE}_c${CODE_NUM}_l${LAYERS}/${IDX_PREFIX}_index_internvideo2_emb_train.json"
RET_OUT="${REPO_ROOT}/output/checkpoints/Baseline_c128l3/${baseline}"

echo "===== $(date -u +%FT%TZ) train ${baseline} (${INDEX_TYPE}) ${DATASET} c${CODE_NUM}l${LAYERS} dev=${DEVICE} ====="
# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
cd "${MM_TVR_DIR}"

# --- Stage 1: tokenizer (RQ-VAE quantizer) ---
if [ -f "${TOK_CKPT}" ]; then
    echo "  [1/3] skip tokenizer (exists): ${TOK_CKPT}"
else
    echo "  [1/3] train tokenizer -> ${TOK_CKPT}"
    python ./index/train_rqvae.py \
        --dataset "${GEN_DS}" --code_num "${CODE_NUM}" --codebook_layers "${LAYERS}" \
        --lr "${TOK_LR}" --epochs 800 --batch_size 4096 --dropout_prob "${TOK_DROPOUT}" \
        --eval_step 5 --device "${DEVICE}" --seed "${SEED}" "${TG_FLAG[@]}"
fi

# --- Stage 2: sID-video dataset (train + test splits) ---
if [ -f "${TRAIN_IDX}" ]; then
    echo "  [2/3] skip sID dataset (exists): ${TRAIN_IDX}"
else
    echo "  [2/3] generate sID dataset -> data/${DATA_DS}/none/${INDEX_TYPE}_c${CODE_NUM}_l${LAYERS}/"
    for split in train test; do
        python ./index/gen_sid_rqvae.py \
            --ckpt_path "${TOK_CKPT}" --dataset "${GEN_DS}" \
            --code_num "${CODE_NUM}" --codebook_layers "${LAYERS}" \
            --type "${INDEX_TYPE}" --split "${split}" --mode none \
            --output_dir "./data/${DATA_DS}" --device "cuda:${DEVICE}"
    done
fi

# --- Stage 3: GR retriever (T5) ---
echo "  [3/3] train GR retriever -> ${RET_OUT}/${DATASET}/${INDEX_TYPE}/.../best_model"
python avg_train_retriever_t5.py \
    --dataset "${DATASET}" --index_type "${INDEX_TYPE}" --mode none \
    --code_book_size "${CODE_NUM}" --code_book_num "${LAYERS}" \
    --model_name t5-small --train_epoch "${RET_EPOCH}" --learning_rate 1e-3 \
    --train_batch_size 256 --gpu_id "${DEVICE}" --bf16 --seed "${SEED}" \
    --callback_early_stop_patience "${EARLY_STOP_PATIENCE:-30}" \
    --no_wandb --output_dir "${RET_OUT}"

echo "===== $(date -u +%FT%TZ) done ${baseline}/${DATASET}: resolve ckpt under ${RET_OUT}/${DATASET}/${INDEX_TYPE} ====="
