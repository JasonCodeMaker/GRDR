#!/usr/bin/env bash
# End-to-end T2VIndexer training at C=128, L=3, in-distribution.
# Three stages, per the Effectiveness-Efficiency Figure spec:
#   1. sID gen : DatasetProcess/kmeans/panda_kmeans.py -> fixed depth-3 hierarchical k-means
#                (kary=128) on InternVideo2 features -> dataset/sID/<ds>_k128_c128_seed_<S>.pkl
#   2. tsv     : dataset/convert_to_tsv.py            -> dataset/<ds>_k128_l3/{train,test}.tsv
#   3. GR model: Model/main.py --mode train           -> Model/logs/<ds>/exp_<ts>/best_model.pt
# T2VIndexer's tokenizer IS the hierarchical k-means (no separate RQ-VAE).
# Each stage skips if its artifact already exists. Driven by env vars:
#   DATASET (msrvtt|actnet|didemo|lsmdc)  DEVICE  SEED  EPOCHS  SMOKE
set -uo pipefail

T2V_DIR=${T2V_DIR:-/home/uqzzha35/Project/SemanticID/T2VIndexer-generativeSearch}
FEATURES_ROOT=${FEATURES_ROOT:-/data2/uqzzha35/VideoRetrieval/features/InternVideo2}
CONDA_SH=${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-semantictvr}   # proven T2VIndexer env (panda-baselines P3c)

DATASET=${DATASET:?DATASET required (msrvtt|actnet|didemo|lsmdc)}
DEVICE=${DEVICE:-0}
SEED=${SEED:-42}
KARY=${KARY:-128}
DEPTH=${DEPTH:-3}
EPOCHS=${EPOCHS:-50}
TRAIN_BATCH=${TRAIN_BATCH:-128}
case "${DATASET}" in msrvtt|actnet|didemo|lsmdc) ;; *) echo "Unknown DATASET: ${DATASET}" >&2; exit 2 ;; esac

SID_PKL="${T2V_DIR}/dataset/sID/${DATASET}_k${KARY}_c${KARY}_seed_${SEED}.pkl"
TSV_DIR="${T2V_DIR}/dataset/${DATASET}_k${KARY}_l${DEPTH}"

echo "===== $(date -u +%FT%TZ) train t2vindexer ${DATASET} c${KARY}l${DEPTH} dev=${DEVICE} ====="
# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
cd "${T2V_DIR}"

# --- Stage 1: sID generation (fixed depth-3 hierarchical k-means) ---
if [ -f "${SID_PKL}" ]; then
    echo "  [1/3] skip sID (exists): ${SID_PKL}"
else
    echo "  [1/3] generate sID -> ${SID_PKL}"
    python DatasetProcess/kmeans/panda_kmeans.py \
        --name "${DATASET}" --kary "${KARY}" --depth "${DEPTH}" --seed "${SEED}" \
        --features-dir "${FEATURES_ROOT}/${DATASET}"
fi

# --- Stage 2: TSV build (caption x sID dataset) ---
if [ -f "${TSV_DIR}/train.tsv" ]; then
    echo "  [2/3] skip tsv (exists): ${TSV_DIR}/train.tsv"
else
    echo "  [2/3] build tsv -> ${TSV_DIR}/"
    python dataset/convert_to_tsv.py \
        --datasets "${DATASET}" --id_class "k${KARY}_l${DEPTH}" --sid_pkl "${SID_PKL}"
fi

# --- Stage 3: GR model (T5 with tree-constrained decoding); run from Model/ (proven P3c cwd) ---
echo "  [3/3] train GR model -> Model/logs/${DATASET}/exp_<ts>/best_model.pt"
smoke_epochs=$([ "${SMOKE:-0}" = "1" ] && echo 1 || echo "${EPOCHS}")
cd "${T2V_DIR}/Model"
CUDA_VISIBLE_DEVICES="${DEVICE}" SMOKE="${SMOKE:-0}" python main.py \
    --mode train --dataset "${DATASET}" \
    --id_class "k${KARY}_l${DEPTH}" --kary "${KARY}" --output_vocab_size "${KARY}" \
    --model_info small --n_gpu 1 --fp_16 0 \
    --num_train_epochs "${smoke_epochs}" --train_batch_size "${TRAIN_BATCH}" --eval_batch_size 16 \
    --learning_rate 2e-4 --decoder_learning_rate 1e-4 --seed "${SEED}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE:-3}"

echo "===== $(date -u +%FT%TZ) done t2vindexer/${DATASET}: ckpt under Model/logs/${DATASET}/ ====="
