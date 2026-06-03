#!/usr/bin/env bash
# P4 Track-2 Stage-2 rerank for ANN baselines (HNSW + IVF) -- local workstation.
#
# Same shape as P4_track2_rerank_local_tiger_avg.sh; difference is the candidate
# JSON path uses the ANN naming convention: <ds>_ann_<idx>_100_candidates_t<setting>.json.
#
# Uses reranker/xpool/test.py (effectiveness; NOT test_perquery.py).
#   --candidate_file <ann json>  --rerank_mode  --eval_checkpoint <Panda ckpt>
#   --video_cache_dir <CACHE_PARENT>  --use_cached_video_features
#   --expanded_pool   (Setting 2 only)
#
# Overrides via env: DEVICE, INDICES, DATASETS, SETTINGS, NUM_FRAMES, BATCH_SIZE.
set -u

DEVICE=${DEVICE:-0}
INDICES=${INDICES:-"hnsw ivf"}
DATASETS=${DATASETS:-"MSRVTT ACTNET DIDEMO LSMDC"}
SETTINGS=${SETTINGS:-"1 2"}
NUM_FRAMES=${NUM_FRAMES:-12}
BATCH_SIZE=${BATCH_SIZE:-32}

REPO_ROOT=${REPO_ROOT:-/home/uqzzha35/Project/SemanticID/GRDR}
PKG_ROOT=${PKG_ROOT:-${REPO_ROOT}/output/evaluation_results/figures}
CAND_ROOT=${CAND_ROOT:-${PKG_ROOT}/candidates}
RESULTS_ROOT=${RESULTS_ROOT:-${PKG_ROOT}/results}
LOG_ROOT=${LOG_ROOT:-${PKG_ROOT}/logs/p4_ann_rerank}
MANIFESTS_DIR=${MANIFESTS_DIR:-${PKG_ROOT}/manifests}

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}" "${MANIFESTS_DIR}" "${REPO_ROOT}/output/evaluation_results/rerank"
LOG="${LOG_ROOT}/p4_ann_rerank.console.log"

PANDA_CKPT=${PANDA_CKPT:-${REPO_ROOT}/reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}
CACHE_PARENT=${CACHE_PARENT:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool-Panda}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
else
    source "${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-xpool}"
    PYTHON=python
fi

if [ ! -f "${PANDA_CKPT}" ]; then
    echo "ERROR: PANDA_CKPT not found: ${PANDA_CKPT}" | tee -a "${LOG}"
    exit 1
fi

declare -A VIDEOS_DIR=(
    [MSRVTT]=${REPO_ROOT}/dataset/msrvtt_data/MSRVTT_Frames
    [ACTNET]=${REPO_ROOT}/dataset/ActivityNet/Activity_Frames_224x224
    [DIDEMO]=${REPO_ROOT}/dataset/DiDeMo/test_frame_224x224
    [LSMDC]=${REPO_ROOT}/dataset/LSMDC/LSMDC_Videos
)

ds_lower () { echo "$1" | tr '[:upper:]' '[:lower:]'; }

cd "${REPO_ROOT}"

echo "===== $(date -u +%FT%TZ) P4 ANN rerank (test.py) INDICES='${INDICES}' DATASETS='${DATASETS}' SETTINGS='${SETTINGS}' ckpt=${PANDA_CKPT} =====" | tee -a "${LOG}"

CELLS=()
for idx in ${INDICES}; do
    for ds in ${DATASETS}; do
        for setting in ${SETTINGS}; do
            dsl=$(ds_lower "${ds}")
            # Per-op candidate JSON path: CAND_PATH override wins; otherwise fall back
            # to the legacy `_100_` literal name (kept for back-compat with cells that
            # were exported before the per-op fix landed).
            cand_path="${CAND_PATH:-${CAND_ROOT}/${idx}/${dsl}/${dsl}_ann_${idx}_100_candidates_t${setting}.json}"
            if [ ! -f "${cand_path}" ]; then
                echo "WARN: missing candidate JSON ${cand_path}; skipping" | tee -a "${LOG}"
                CELLS+=("${idx}/${ds}/t${setting}=missing_cand")
                continue
            fi
            cache_dir="${CACHE_PARENT}/${ds}"
            if [ ! -d "${cache_dir}" ]; then
                echo "ERROR: ${ds} cache dir missing: ${cache_dir}" | tee -a "${LOG}"
                CELLS+=("${idx}/${ds}/t${setting}=missing_cache")
                continue
            fi
            report_dir="${RESULTS_ROOT}/${idx}/${dsl}/setting${setting}"
            mkdir -p "${report_dir}"
            result_csv="p4_ann_${idx}_${dsl}_t${setting}.csv"
            vdir="${VIDEOS_DIR[${ds}]}"
            pool_flag=""
            [[ "${setting}" == "2" ]] && pool_flag="--expanded_pool"

            echo "----- $(date -u +%FT%TZ) ${idx}/${ds}/t${setting} cand=$(basename ${cand_path}) cache_npz=$(find ${cache_dir} -maxdepth 1 -type f -name '*.npz' | wc -l) -----" | tee -a "${LOG}"

            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH="${REPO_ROOT}/reranker/xpool" \
            "${PYTHON}" "${REPO_ROOT}/reranker/xpool/test.py" \
                --dataset_name "${ds}" \
                --huggingface \
                ${pool_flag} \
                --videos_dir "${vdir}" \
                --num_frames "${NUM_FRAMES}" \
                --batch_size "${BATCH_SIZE}" \
                --eval_checkpoint "${PANDA_CKPT}" \
                --use_cached_video_features \
                --video_cache_dir "${CACHE_PARENT}" \
                --candidate_file "${cand_path}" \
                --rerank_mode \
                --result_file "${result_csv}" \
                --seed 42 \
                --no_tensorboard \
                2>&1 | tee -a "${LOG}"
            rc=${PIPESTATUS[0]}
            CELLS+=("${idx}/${ds}/t${setting}=${rc}")

            if [ -f "output/evaluation_results/rerank/${result_csv}" ]; then
                cp "output/evaluation_results/rerank/${result_csv}" "${report_dir}/result.csv"
                echo "  copied output/evaluation_results/rerank/${result_csv} -> ${report_dir}/result.csv" | tee -a "${LOG}"
            fi
            echo "----- $(date -u +%FT%TZ) ${idx}/${ds}/t${setting} exit rc=${rc} -----" | tee -a "${LOG}"
        done
    done
done

MANIFEST="${MANIFESTS_DIR}/P4_ann_rerank.json"
{
    echo "{"
    echo "  \"created_at\": \"$(date -u +%FT%TZ)\","
    echo "  \"entry_point\": \"reranker/xpool/test.py (effectiveness; NOT test_perquery.py)\","
    echo "  \"panda_ckpt\": \"${PANDA_CKPT}\","
    echo "  \"cache_parent\": \"${CACHE_PARENT}\","
    echo "  \"results_root\": \"${RESULTS_ROOT}\","
    echo "  \"indices\": \"${INDICES}\","
    echo "  \"datasets\": \"${DATASETS}\","
    echo "  \"settings\": \"${SETTINGS}\","
    echo "  \"cells\": [\"${CELLS[*]}\"]"
    echo "}"
} > "${MANIFEST}"

SENTINEL="${MANIFESTS_DIR}/P4_ann_rerank.done"
date -u +%FT%TZ > "${SENTINEL}"

echo "===== $(date -u +%FT%TZ) P4 ANN rerank done. Manifest: ${MANIFEST}; Sentinel: ${SENTINEL} =====" | tee -a "${LOG}"
cat "${MANIFEST}" | tee -a "${LOG}"
