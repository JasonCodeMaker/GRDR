#!/usr/bin/env bash
# T2VIndexer Stage-1 candidate export (in-distribution, C=128/L=3).
# main.py --mode eval runs T5FineTuner.test(), which writes the figure candidate-JSON
# contract ({metadata, metrics:{total_queries,avg_candidates_per_query,Recall@K}, results:
# [{ground_truth_video_id, candidates:[video_ids]}]}) to the T2VIndexer candidates/ dir.
# We rescue-copy it to CAND_OUT. Standard mode (no --detailed_generation) => candidates are
# plain video ids, which the figure aggregator's recompute_canhit consumes directly.
# Required env: DS_LOWER SETTING OP INFER_CKPT CAND_OUT.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

DS_LOWER=${DS_LOWER:?DS_LOWER required}
SETTING=${SETTING:?SETTING required}
OP=${OP:?OP required}
INFER_CKPT=${INFER_CKPT:?INFER_CKPT required}
CAND_OUT=${CAND_OUT:?CAND_OUT required}
DEVICE=${DEVICE:-0}
SEED=${SEED:-42}
T2V_DIR=${T2V_DIR:-/home/uqzzha35/Project/SemanticID/T2VIndexer-generativeSearch}
CONDA_ENV=${CONDA_ENV_T2V:-semantictvr}
LOG_DIR=${RECALL_STAGE_ROOT}/_logs/t2v
mkdir -p "${LOG_DIR}" "$(dirname "${CAND_OUT}")"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
cd "${T2V_DIR}/Model"

CUDA_VISIBLE_DEVICES="${DEVICE}" python main.py \
    --mode eval --dataset "${DS_LOWER}" --setting "${SETTING}" \
    --id_class k128_l3 --kary 128 --output_vocab_size 128 --model_info small --n_gpu 1 \
    --num_return_sequences "${OP}" --infer_ckpt "${INFER_CKPT}" --seed "${SEED}" \
    --eval_batch_size "${T2V_EVAL_BATCH_SIZE:-32}" \
    2>&1 | tee "${LOG_DIR}/t2v_${DS_LOWER}_t${SETTING}_op${OP}.log"

# Rescue the candidate JSON from T2VIndexer's hardcoded candidates/ path (test() naming).
src="${T2V_DIR}/candidates/${DS_LOWER}_t2vindexer_${OP}_candidates_t${SETTING}.json"
if [ -f "${src}" ]; then
    cp -f "${src}" "${CAND_OUT}"
    echo "  rescued ${src} -> ${CAND_OUT}"
else
    echo "ERROR: t2vindexer candidate JSON not produced at ${src}" >&2
    exit 2
fi
