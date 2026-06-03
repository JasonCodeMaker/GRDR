#!/usr/bin/env bash
# P4 Stage-1 full candidate export for Panda-trained TIGER/AVG plain-T5.
#
# This is Pass A effectiveness export, not the Pass B latency subset.
# It generates one top-K candidate JSON per (baseline, target dataset, setting)
# using target videos encoded under the Panda-trained RQ-VAE codebook.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/uqzzha35/Project/SemanticID/GRDR}
RUNTIME_ROOT=${RUNTIME_ROOT:-${REPO_ROOT}/output/evaluation_results/figures}
CAND_ROOT=${CAND_ROOT:-${RUNTIME_ROOT}/candidates}
LOG_ROOT=${LOG_ROOT:-${RUNTIME_ROOT}/logs/p4_stage1}

MM_TVR_DIR=${MM_TVR_DIR:-/home/uqzzha35/Project/SemanticID/MM-SemanticTVR}
PYTHON=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-0}
SEED=${SEED:-42}
TOPK=${TOPK:-100}
CODE_NUM=${CODE_NUM:-4096}
LAYERS=${LAYERS:-3}
BASELINES=${BASELINES:-"tiger avg"}
DATASETS=${DATASETS:-"MSRVTT ACTNET DIDEMO LSMDC"}
SETTINGS=${SETTINGS:-"1 2"}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-0}

TIGER_CKPT=${TIGER_CKPT:-}
AVG_CKPT=${AVG_CKPT:-}

mkdir -p "${CAND_ROOT}" "${LOG_ROOT}"

resolve_plain_t5_ckpt () {
    local explicit=$1
    local search_root=$2
    if [ -n "${explicit}" ] && [ -e "${explicit}" ]; then
        printf '%s\n' "${explicit}"
        return 0
    fi
    local found
    found=$(find "${search_root}" -path '*/best_model' -type d 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "${found}" ]; then
        printf '%s\n' "${found}"
        return 0
    fi
    echo "MISSING plain-T5 checkpoint under ${search_root}; set TIGER_CKPT/AVG_CKPT explicitly" >&2
    return 2
}

run_cell () {
    local baseline=$1 ds=$2 setting=$3
    local idx_type="standard"
    local ck="${TIGER_CKPT}"
    if [ "${baseline}" = "avg" ]; then
        idx_type="text_guided"
        ck="${AVG_CKPT}"
    fi

    ck=$(resolve_plain_t5_ckpt "${ck}" "${CKPT_ROOT:-${REPO_ROOT}/output/checkpoints/Baseline}/${baseline}/${ds,,}") || return $?

    local ds_lower="${ds,,}"
    local out_dir="${CAND_ROOT}/${baseline}/${ds_lower}"
    mkdir -p "${out_dir}"
    local out_json="${out_dir}/${ds_lower}_${baseline}_${TOPK}_candidates_t${setting}.json"
    local log="${LOG_ROOT}/${baseline}_${ds}_t${setting}.console.log"

    echo "===== $(date -u +%FT%TZ) P4 Stage-1 ${baseline}/${ds}/t${setting} ckpt=${ck} =====" | tee "${log}"
    (
        cd "${MM_TVR_DIR}"
        CUDA_VISIBLE_DEVICES="${DEVICE}" "${PYTHON}" avg_train_retriever_t5.py \
            --eval \
            --dataset "${ds_lower}" \
            --index_type "${idx_type}" \
            --mode none \
            --eval_checkpoint "${ck}" \
            --code_book_size "${CODE_NUM}" \
            --code_book_num "${LAYERS}" \
            --gpu_id "${DEVICE}" \
            --num_candidates "${TOPK}" \
            --setting "${setting}" \
            --eval_num_workers "${EVAL_NUM_WORKERS}" \
            --output_json "${out_json}" \
            --seed "${SEED}"
    ) 2>&1 | tee -a "${log}"
    echo "===== $(date -u +%FT%TZ) done ${baseline}/${ds}/t${setting}: ${out_json} =====" | tee -a "${log}"
}

for baseline in ${BASELINES}; do
    case "${baseline}" in
        tiger|avg) ;;
        *) echo "Unsupported baseline: ${baseline}" >&2; exit 2 ;;
    esac
    for ds in ${DATASETS}; do
        for setting in ${SETTINGS}; do
            run_cell "${baseline}" "${ds}" "${setting}"
        done
    done
done
