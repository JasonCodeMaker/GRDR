#!/usr/bin/env bash
# Recall-Latency cell for the baseline methods (tiger/avg/t2vindexer/eercf/hnsw/ivf).
# One (BASELINE,DATASET,SETTING,OP_VALUE) per call. grdr_ref is handled by
# GRDR/grdr_recall_latency.sh (volatility split); this script never sees it.
# Faithful port of the prior P4B stage-1 latency cell with native paths.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

BASELINE=${BASELINE:?BASELINE required}
DATASET=${DATASET:?DATASET required}
SETTING=${SETTING:?SETTING required}
OP_VALUE=${OP_VALUE:?OP_VALUE required}

# Output root: Recall-Latency tree (overridable; default native).
CAND_B_ROOT=${CAND_B_ROOT:-${RECALL_LATENCY_ROOT}}
LOG_ROOT=${LOG_ROOT:-${RECALL_LATENCY_ROOT}/_logs}
WARMUP_N_USED_DEFAULT=${WARMUP_N_USED_DEFAULT:-10}

# Per-baseline operating-point knob (defaults to ${OP_VALUE}).
TIGER_AVG_NUM_CANDIDATES=${TIGER_AVG_NUM_CANDIDATES:-${OP_VALUE}}
T2V_NUM_RETURN_SEQUENCES=${T2V_NUM_RETURN_SEQUENCES:-${OP_VALUE}}
EERCF_RERANTOPK=${EERCF_RERANTOPK:-${OP_VALUE}}
# ANN op = candidate budget K; ef_search/nprobe are paired to K by the caller
# (_cells.sh passes HNSW_EF_SEARCH/IVF_NPROBE per op from ANN_*_BY_K in _env.sh).
# These fallbacks only apply on direct invocation; they default to the max op (K=200).
ANN_NUM_CANDIDATES=${ANN_NUM_CANDIDATES:-${OP_VALUE}}
HNSW_EF_SEARCH=${HNSW_EF_SEARCH:-128}
IVF_NPROBE=${IVF_NPROBE:-32}

mkdir -p "${LOG_ROOT}" "${CAND_B_ROOT}"

# Stage-1 latency launchers (MM-SemanticTVR / T2VIndexer / eval_ann) need semantictvr.
if [ "${CONDA_DEFAULT_ENV:-}" != "${SEMANTICTVR_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
fi

# Per-baseline checkpoints + entry pointers — all native.
TIGER_CKPT=${TIGER_CKPT:-}
AVG_CKPT=${AVG_CKPT:-}
EERCF_CKPT=${EERCF_CKPT:-${EERCF_INIT_MODEL}}
EERCF_PERQUERY_PY=${EERCF_PERQUERY_PY:-${FUNC_DIR}/baselines/eercf/eercf_perquery_latency.py}
EERCF_POOL_CACHE=${EERCF_POOL_CACHE:-${EERCF_CACHE_ROOT}}

resolve_plain_t5_ckpt () {
    local explicit=$1 search_root=$2
    if [ -n "${explicit}" ] && [ -e "${explicit}" ]; then printf '%s\n' "${explicit}"; return 0; fi
    local found
    found=$(find "${search_root}" -path '*/best_model' -type d 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "${found}" ]; then printf '%s\n' "${found}"; return 0; fi
    echo "MISSING plain-T5 checkpoint under ${search_root}; set TIGER_CKPT/AVG_CKPT explicitly" >&2
    return 2
}

run_cell () {
    local baseline=$1 ds=$2 setting=$3 op=$4
    local manifest="${MANIFEST_DIR}/latency_subset_${ds}_t${setting}.json"
    if [ ! -f "${manifest}" ]; then echo "MISSING manifest: ${manifest}" >&2; return 2; fi
    local cand_dir="${CAND_B_ROOT}/${baseline}/${ds,,}"; mkdir -p "${cand_dir}"
    local cand_out="${cand_dir}/${ds,,}_t${setting}_${op}_latency.json"
    local log="${LOG_ROOT}/${baseline}_${ds}_t${setting}_op${op}_stage1.console.log"
    local warmup_n=${WARMUP_N_USED_DEFAULT}
    [ "${baseline}" = "eercf" ] && warmup_n=1
    echo "===== $(date -u +%FT%TZ) recall-latency cell: ${baseline}/${ds}/t${setting} op=${op} ====="

    case "${baseline}" in
        tiger|avg)
            local idx_type="standard"; [ "${baseline}" = "avg" ] && idx_type="text_guided"
            local ck="${TIGER_CKPT}"; [ "${baseline}" = "avg" ] && ck="${AVG_CKPT}"
            # indist uses the c128l3 baseline ckpts (code_book_size=128) under Baseline_c128l3;
            # zeroshot uses the 4096-code panda ckpts. code_book_size MUST match the resolved
            # ckpt or constrained generation trips a CUDA device-side assert.
            local cbsize=4096 ck_root="${TIGER_AVG_CKPT_ROOT}/${baseline}/${ds,,}"
            if [ "${EVAL_MODE:-zeroshot}" = "indist" ]; then cbsize=128; ck_root="${BASELINE_C128L3_ROOT}/${baseline}/${ds,,}"; fi
            ck=$(resolve_plain_t5_ckpt "${ck}" "${ck_root}") || return $?
            (cd "${MM_TVR_DIR}" && CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" avg_train_retriever_t5.py \
                --eval --dataset "${ds,,}" --index_type "${idx_type}" --mode none \
                --eval_checkpoint "${ck}" --code_book_size "${cbsize}" --code_book_num 3 \
                --gpu_id "${DEVICE}" --num_candidates "${TIGER_AVG_NUM_CANDIDATES}" --train_batch_size 1 \
                --setting "${setting}" --subset_manifest "${manifest}" \
                --latency_helpers_dir "${LATENCY_HELPERS_DIR}" \
                --warmup_n_used "${warmup_n}" --wall_time_cap_s "${WALL_CAP_S}" \
                --eval_num_workers 0 --output_json "${cand_out}" --seed "${SEED}") 2>&1 | tee "${log}"
            ;;
        t2vindexer)
            # indist uses the per-dataset t2v ckpt; zeroshot uses the panda T2V_CKPT.
            local t2v_ck="${T2V_CKPT}"
            [ "${EVAL_MODE:-zeroshot}" = "indist" ] && t2v_ck="${BASE_CKPT_ROOT}/t2vindexer/${ds,,}/best_model.pt"
            if [ ! -f "${t2v_ck}" ]; then
                echo "MISSING T2V ckpt: ${t2v_ck}" >&2; return 2
            fi
            (cd "${T2V_DIR}" && CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" Model/main.py \
                --mode eval --infer_ckpt "${t2v_ck}" --dataset "${ds,,}" --setting "${setting}" \
                --num_return_sequences "${T2V_NUM_RETURN_SEQUENCES}" --eval_batch_size 1 \
                --subset_manifest "${manifest}" --warmup_n_used "${warmup_n}" \
                --wall_time_cap_s "${WALL_CAP_S}") 2>&1 | tee "${log}"
            local t2v_out="${T2V_DIR}/candidates/${ds,,}_t2vindexer_${T2V_NUM_RETURN_SEQUENCES}_candidates_t${setting}.json"
            [ -f "${t2v_out}" ] && cp -f "${t2v_out}" "${cand_out}"
            ;;
        eercf)
            if [ ! -f "${EERCF_CKPT}" ]; then echo "MISSING EERCF ckpt: ${EERCF_CKPT}" >&2; return 2; fi
            local eercf_dt eercf_features eercf_test_csv="" eercf_test_json="" eercf_train_csv="" eercf_train_json=""
            case "${ds}" in
                MSRVTT) eercf_dt=msrvtt; eercf_features=/data2/uqzzha35/VideoRetrieval/msrvtt_data/MSRVTT_Frames
                        eercf_test_csv=${EERCF_DIR}/data/MSRVTT/raw/MSRVTT_JSFUSION_test.csv
                        eercf_train_csv=${EERCF_DIR}/data/MSRVTT/raw/MSRVTT_train.9k.csv ;;
                ACTNET) eercf_dt=activity; eercf_features=/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Frames_224x224
                        eercf_test_json=${EERCF_DIR}/data/ACTNET/video_retreival_caption/anet_ret_val_1.json
                        eercf_train_json=${EERCF_DIR}/data/ACTNET/video_retreival_caption/anet_ret_train.json ;;
                DIDEMO) eercf_dt=didemo; eercf_features=${REPO_ROOT}/dataset/DiDeMo
                        eercf_test_json=${EERCF_DIR}/data/DIDEMO/video_retreival_caption/didemo_ret_test.json
                        eercf_train_json=${EERCF_DIR}/data/DIDEMO/video_retreival_caption/didemo_ret_train.json ;;
                LSMDC)  eercf_dt=lsmdc; eercf_features=${REPO_ROOT}/dataset/LSMDC
                        eercf_test_json=${EERCF_DIR}/data/LSMDC/video_retreival_caption/lsmdc_ret_test_1000.json
                        eercf_train_json=${EERCF_DIR}/data/LSMDC/video_retreival_caption/lsmdc_ret_train.json ;;
                *) echo "Unknown EERCF dataset: ${ds}" >&2; return 2 ;;
            esac
            CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" "${EERCF_PERQUERY_PY}" \
                --eercf_dir "${EERCF_DIR}" --init_model "${EERCF_CKPT}" \
                --datatype "${eercf_dt}" --setting "${setting}" --features_path "${eercf_features}" \
                ${eercf_test_csv:+--test_csv "${eercf_test_csv}"} \
                ${eercf_test_json:+--test_json "${eercf_test_json}"} \
                ${eercf_train_csv:+--train_csv "${eercf_train_csv}"} \
                ${eercf_train_json:+--train_json "${eercf_train_json}"} \
                --pool_cache_dir "${EERCF_POOL_CACHE}/${ds}" \
                --frame_cache_dir "${EERCF_FRAME_CACHE}/${ds}" \
                --subset_manifest "${manifest}" --warmup_n_used "${warmup_n}" \
                --wall_time_cap_s "${WALL_CAP_S}" --rerantopk "${EERCF_RERANTOPK}" \
                --output_json "${cand_out}" --gpu 0 --seed "${SEED}" 2>&1 | tee "${log}"
            ;;
        hnsw)
            CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" "${REPO_ROOT}/baselines/ann_dense_retrieval/eval_ann.py" \
                --index_type "${baseline}" --dataset "${ds,,}" --setting "${setting}" \
                --hnsw_ef_search "${HNSW_EF_SEARCH}" --num_candidates "${ANN_NUM_CANDIDATES}" \
                --batch_size 1 --per_query_timing \
                --cache_dir "${CACHE_PARENT}/${ds}" --subset_manifest "${manifest}" \
                --warmup_n_used "${warmup_n}" --wall_time_cap_s "${WALL_CAP_S}" \
                --device "${DEVICE}" --output_json "${cand_out}" 2>&1 | tee "${log}"
            ;;
        ivf)
            CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" "${REPO_ROOT}/baselines/ann_dense_retrieval/eval_ann.py" \
                --index_type "${baseline}" --dataset "${ds,,}" --setting "${setting}" \
                --ivf_nprobe "${IVF_NPROBE}" --num_candidates "${ANN_NUM_CANDIDATES}" \
                --batch_size 1 --per_query_timing \
                --cache_dir "${CACHE_PARENT}/${ds}" --subset_manifest "${manifest}" \
                --warmup_n_used "${warmup_n}" --wall_time_cap_s "${WALL_CAP_S}" \
                --device "${DEVICE}" --output_json "${cand_out}" 2>&1 | tee "${log}"
            ;;
        *) echo "Unknown baseline: ${baseline}" >&2; return 2 ;;
    esac
    local rc=$?
    [ "${rc}" -eq 0 ] && echo "----- cell ${baseline}/${ds}/t${setting} op=${op} done -----" \
        || echo "----- cell ${baseline}/${ds}/t${setting} op=${op} failed rc=${rc} -----" >&2
    return "${rc}"
}

run_cell "${BASELINE}" "${DATASET}" "${SETTING}" "${OP_VALUE}"
