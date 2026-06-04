#!/usr/bin/env bash
# P4-B Stage-2 (Rerank) latency pass.
#
# Re-runs X-Pool rerank with the shared Panda-X-Pool ckpt on the Pass-B
# candidate JSONs produced by P4B_stage1_latency.sh. Records the
# rerank_latency_ms block via reranker/xpool/test_perquery.py's existing
# per-query timing.total field (CUDA-synced).
#
# Reference: docs/eval-efficiency.html.

set -u

REPO_ROOT=${REPO_ROOT:-/home/uqzzha35/Project/SemanticID/GRDR}
PKG_ROOT="${REPO_ROOT}/output/evaluation_results/figures"
RUNTIME_ROOT=${RUNTIME_ROOT:-${REPO_ROOT}/output/evaluation_results/figures}
MANIFEST_DIR=${MANIFEST_DIR:-${RUNTIME_ROOT}/manifests/latency}
CAND_B_ROOT=${CAND_B_ROOT:-${RUNTIME_ROOT}/candidates_b}
RESULTS_B_ROOT=${RESULTS_B_ROOT:-${RUNTIME_ROOT}/results_b}
LOG_ROOT=${LOG_ROOT:-${RUNTIME_ROOT}/logs/pass_b}

DEVICE=${DEVICE:-0}
SEED=${SEED:-42}
BASELINES=${BASELINES:-"grdr_ref tiger avg t2vindexer eercf hnsw ivf"}
DATASETS=${DATASETS:-"MSRVTT ACTNET DIDEMO LSMDC"}
SETTINGS=${SETTINGS:-"1 2"}
WARMUP_N_USED_DEFAULT=${WARMUP_N_USED_DEFAULT:-10}
WALL_CAP_S=${WALL_CAP_S:-300}
# Underscore form matches the on-disk file produced by P1; the
# .../panda_2150k_s42/model_best.pth directory form does not exist.
PANDA_CKPT=${PANDA_CKPT:-${REPO_ROOT}/reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}
CACHE_PARENT=${CACHE_PARENT:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool-Panda}
# EERCF native-rerank path: uses EERCF's get_similarity_logits + fusion (not
# X-Pool's pool_frames), cold-loading patch+frame features from the P3.6 cache.
EERCF_NATIVE_PY=${EERCF_NATIVE_PY:-${REPO_ROOT}/scripts/latency_recall_figure/baselines/eercf/eercf_native_rerank_latency.py}
EERCF_CKPT=${EERCF_CKPT:-${REPO_ROOT}/output/checkpoints/Baseline/eercf/panda/pytorch_model.bin.best.0}
EERCF_FRAME_CACHE=${EERCF_FRAME_CACHE:-/data2/uqzzha35/VideoRetrieval/eercf_cache/cached_frame_features_p3d}
EERCF_DIR=${EERCF_DIR:-/home/uqzzha35/Project/SemanticID/EERCF}
LATENCY_HELPERS_DIR=${LATENCY_HELPERS_DIR:-${REPO_ROOT}/scripts/latency_recall_figure/utils}

# Canonical Pass-A mapping (from P4_grdr_mv4.sh REREANK_VIDEOS_DIR). With a
# full Xpool-Panda cache hit X-Pool never reads from videos_dir, but the path
# must be a real directory; these are the dirs Pass-A validated against.
declare -A VIDEOS_DIR=(
    [MSRVTT]=${REPO_ROOT}/dataset/msrvtt_data/MSRVTT_Frames
    [ACTNET]=${REPO_ROOT}/dataset/ActivityNet/Activity_Frames_224x224
    [DIDEMO]=${REPO_ROOT}/dataset/DiDeMo/test_frame_224x224
    [LSMDC]=${REPO_ROOT}/dataset/LSMDC/LSMDC_Videos
)

mkdir -p "${LOG_ROOT}" "${RESULTS_B_ROOT}"

if [ ! -f "${PANDA_CKPT}" ]; then
    echo "ERROR: PANDA_CKPT not found: ${PANDA_CKPT}" >&2
    exit 1
fi

run_cell () {
    local baseline=$1 ds=$2 setting=$3
    local manifest="${MANIFEST_DIR}/latency_subset_${ds}_t${setting}.json"
    if [ ! -f "${manifest}" ]; then
        echo "MISSING manifest: ${manifest}" >&2
        return 2
    fi
    local cand_file="${CAND_B_ROOT}/${baseline}/${ds,,}/${ds,,}_t${setting}_latency.json"
    if [ ! -f "${cand_file}" ]; then
        echo "MISSING Pass-B candidate JSON: ${cand_file}; run P4B_stage1_latency.sh first" >&2
        return 2
    fi
    local report_dir="${RESULTS_B_ROOT}/${baseline}/${ds,,}/setting${setting}"
    mkdir -p "${report_dir}"
    local log="${LOG_ROOT}/${baseline}_${ds}_t${setting}_stage2.console.log"
    local warmup_n=${WARMUP_N_USED_DEFAULT}
    [ "${baseline}" = "eercf" ] && warmup_n=1
    local extra=""
    [ "${ds}" = "ACTNET" ] && [ "${setting}" = "2" ] && extra="--max_queries 1000"
    echo "===== $(date -u +%FT%TZ) P4B-Stage2: ${baseline}/${ds}/t${setting} ====="
    if [ "${baseline}" = "eercf" ]; then
        # EERCF Stage-2 uses EERCF's native rerank (cold-load patch+frame from
        # P3.6 cache + get_similarity_logits + multi-level fusion), not X-Pool.
        # This is the correct apples-to-apples measurement of EERCF's rerank
        # cost; the shared-X-Pool rerank doesn't touch patch features and would
        # underestimate EERCF's true Stage-2 work.
        local eercf_dt
        case "${ds}" in
            MSRVTT) eercf_dt=msrvtt ;;
            ACTNET) eercf_dt=activity ;;
            DIDEMO) eercf_dt=didemo ;;
            LSMDC)  eercf_dt=lsmdc ;;
            *) echo "Unknown ds: ${ds}" >&2; return 2 ;;
        esac
        CUDA_VISIBLE_DEVICES=${DEVICE} python "${EERCF_NATIVE_PY}" \
            --eercf_dir "${EERCF_DIR}" \
            --init_model "${EERCF_CKPT}" \
            --datatype "${eercf_dt}" \
            --candidate_file "${cand_file}" \
            --frame_cache_dir "${EERCF_FRAME_CACHE}/${ds}" \
            --subset_manifest "${manifest}" \
            --warmup_n_used "${warmup_n}" \
            --wall_time_cap_s "${WALL_CAP_S}" \
            --summary_json "${report_dir}/perquery_summary.json" \
            --gpu 0 --seed "${SEED}" 2>&1 | tee "${log}"
    else
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH="${REPO_ROOT}/reranker/xpool" \
        python "${REPO_ROOT}/reranker/xpool/test_perquery.py" \
            --dataset_name "${ds}" \
            --huggingface \
            --videos_dir "${VIDEOS_DIR[${ds}]}" \
            --checkpoint "${PANDA_CKPT}" \
            --cache_dir "${CACHE_PARENT}/${ds}" \
            --candidate_file "${cand_file}" \
            --subset_manifest "${manifest}" \
            --warmup_n_used "${warmup_n}" \
            --wall_time_cap_s "${WALL_CAP_S}" \
            --latency_helpers_dir "${LATENCY_HELPERS_DIR}" \
            --report_dir "${report_dir}" \
            --summary_csv "${report_dir}/perquery_summary.csv" \
            --summary_json "${report_dir}/perquery_summary.json" \
            --seed "${SEED}" \
            ${extra} 2>&1 | tee "${log}"
    fi
    echo "----- $(date -u +%FT%TZ) cell ${baseline}/${ds}/t${setting} done -----"
}

for bl in ${BASELINES}; do
    for ds in ${DATASETS}; do
        for st in ${SETTINGS}; do
            run_cell "${bl}" "${ds}" "${st}" || echo "WARN: ${bl}/${ds}/t${st} failed (continuing)"
        done
    done
done
