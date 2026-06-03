#!/usr/bin/env bash
# P4 Stage-1 candidate export for ANN baselines (HNSW + IVF) over Panda-X-Pool features.
#
# Training-free baselines. For each (idx_type, dataset, setting):
#   - load Xpool-Panda video feature cache (parent dir; eval_ann resolves <ds> subdir)
#   - load CLIP text encoder weights from Panda-X-Pool ckpt
#   - build FAISS index (HNSW efSearch=128 / IVF nprobe=32)
#   - search top-100 -> candidate JSON
#
# Overrides via env: DEVICE, INDICES, DATASETS, SETTINGS.
set -u

REPO_ROOT=${REPO_ROOT:-/home/uqzzha35/Project/SemanticID/GRDR}
RUNTIME_ROOT=${RUNTIME_ROOT:-${REPO_ROOT}/output/evaluation_results/figures}
CAND_ROOT=${CAND_ROOT:-${RUNTIME_ROOT}/candidates}
LOG_ROOT=${LOG_ROOT:-${RUNTIME_ROOT}/logs/p4_ann_export}
OUT_ROOT=${OUT_ROOT:-${RUNTIME_ROOT}/output/evaluation_results/ann_baseline}
MANIFESTS_DIR=${MANIFESTS_DIR:-${RUNTIME_ROOT}/manifests}

DEVICE=${DEVICE:-0}
PANDA_CKPT=${PANDA_CKPT:-${REPO_ROOT}/reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}
CACHE_PARENT=${CACHE_PARENT:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool-Panda}
INDICES=${INDICES:-"hnsw ivf"}
DATASETS=${DATASETS:-"msrvtt actnet didemo lsmdc"}
SETTINGS=${SETTINGS:-"1 2"}
HNSW_EF_SEARCH=${HNSW_EF_SEARCH:-128}
IVF_NPROBE=${IVF_NPROBE:-32}
# Candidate budget K (the ANN operating point). ef_search/nprobe are held fixed.
NUM_CANDIDATES=${NUM_CANDIDATES:-100}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
else
    source "${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-semantictvr}"
    PYTHON=python
fi

mkdir -p "${CAND_ROOT}" "${LOG_ROOT}" "${OUT_ROOT}" "${MANIFESTS_DIR}"
LOG="${LOG_ROOT}/p4_ann_export.console.log"

if [ ! -f "${PANDA_CKPT}" ]; then
    echo "ERROR: PANDA_CKPT not found: ${PANDA_CKPT}" | tee -a "${LOG}"
    exit 1
fi

cd "${REPO_ROOT}"

echo "===== $(date -u +%FT%TZ) P4 ANN export DEVICE=${DEVICE} INDICES='${INDICES}' DATASETS='${DATASETS}' SETTINGS='${SETTINGS}' =====" | tee -a "${LOG}"

CELLS=()
for idx_type in ${INDICES}; do
    case "${idx_type}" in
        hnsw) knob_flag="--hnsw_ef_search ${HNSW_EF_SEARCH}" ;;
        ivf)  knob_flag="--ivf_nprobe ${IVF_NPROBE}" ;;
        *) echo "ERROR: unknown index type ${idx_type}" | tee -a "${LOG}"; exit 2 ;;
    esac
    for ds in ${DATASETS}; do
        for setting in ${SETTINGS}; do
            cand_dir="${CAND_ROOT}/${idx_type}/${ds}"
            mkdir -p "${cand_dir}"
            cell_log="${LOG_ROOT}/${idx_type}_${ds}_t${setting}.console.log"

            echo "----- $(date -u +%FT%TZ) ${idx_type}/${ds}/t${setting} -----" | tee -a "${LOG}"
            # Per-op candidate JSON: OUTPUT_JSON env override lets the caller pin a
            # per-op filename (e.g., ..._op32_candidates_t2.json) instead of the legacy
            # ..._100_candidates_t2.json that gets overwritten across operating points.
            output_json_flag=""
            [ -n "${OUTPUT_JSON:-}" ] && output_json_flag="--output_json ${OUTPUT_JSON}"
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" \
                "${REPO_ROOT}/baselines/ann_dense_retrieval/eval_ann.py" \
                --dataset "${ds}" \
                --setting "${setting}" \
                --index_type "${idx_type}" \
                ${knob_flag} \
                --num_candidates "${NUM_CANDIDATES}" \
                --checkpoint "${PANDA_CKPT}" \
                --cache_dir "${CACHE_PARENT}" \
                --output_dir "${OUT_ROOT}" \
                --candidate_dir "${cand_dir}" \
                ${output_json_flag} \
                --device "${DEVICE}" \
                2>&1 | tee "${cell_log}" | tee -a "${LOG}"
            rc=${PIPESTATUS[0]}
            CELLS+=("${idx_type}/${ds}/t${setting}=${rc}")
            echo "----- $(date -u +%FT%TZ) ${idx_type}/${ds}/t${setting} exit rc=${rc} -----" | tee -a "${LOG}"
        done
    done
done

MANIFEST="${MANIFESTS_DIR}/P4_ann_export.json"
{
    echo "{"
    echo "  \"created_at\": \"$(date -u +%FT%TZ)\","
    echo "  \"entry_point\": \"baselines/ann_dense_retrieval/eval_ann.py\","
    echo "  \"panda_ckpt\": \"${PANDA_CKPT}\","
    echo "  \"cache_parent\": \"${CACHE_PARENT}\","
    echo "  \"cand_root\": \"${CAND_ROOT}\","
    echo "  \"indices\": \"${INDICES}\","
    echo "  \"datasets\": \"${DATASETS}\","
    echo "  \"settings\": \"${SETTINGS}\","
    echo "  \"hnsw_ef_search\": ${HNSW_EF_SEARCH},"
    echo "  \"ivf_nprobe\": ${IVF_NPROBE},"
    echo "  \"num_candidates\": ${NUM_CANDIDATES},"
    echo "  \"device\": ${DEVICE},"
    echo "  \"cells\": [\"${CELLS[*]}\"]"
    echo "}"
} > "${MANIFEST}"

SENTINEL="${MANIFESTS_DIR}/P4_ann_export.done"
date -u +%FT%TZ > "${SENTINEL}"

echo "===== $(date -u +%FT%TZ) P4 ANN export done. Manifest: ${MANIFEST}; Sentinel: ${SENTINEL} =====" | tee -a "${LOG}"
cat "${MANIFEST}" | tee -a "${LOG}"
