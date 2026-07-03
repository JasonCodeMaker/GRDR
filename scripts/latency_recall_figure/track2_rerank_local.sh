#!/usr/bin/env bash
# Workstation-local X-Pool rerank for one Pass-A cell (single baseline/ds/setting/op).
#
# Why this exists: the prior P4_track2_rerank.sh is a Nectar/Track-2
# deployment wrapper (uv-env activation at /data/uv-envs/xpool/bin/activate,
# scratch root /home/ubuntu/scratch, hardcoded CAND_FILE map by codebook). It is
# unusable on this workstation. This script is the package-owned replacement
# (Self-contained scripts/ rule -- plan.html no-change-boundary).
#
# Source of the inner python invocation: same as the prior
# P1b_validate_panda_local_s2.sh (workstation-tested) +
# _vendored/P4_track2_rerank.sh body (lines 109-128) -- only the shell wrapper
# differs (workstation paths, per-cell args).
#
# Required env: BASELINE DATASET SETTING OP CAND_PATH OUT_DIR PANDA_CKPT
# Optional env: DEVICE (0) NUM_FRAMES (12) BATCH_SIZE (32) SEED (42)
#               MAX_QUERIES (unset; bounded smoke/debug only)
#               CONDA_SH (default workstation miniconda) CONDA_ENV (xpool)
#               CACHE_PARENT REPO_ROOT

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/uqzzha35/Project/SemanticID/GRDR}
CONDA_SH=${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-xpool}

BASELINE=${BASELINE:?BASELINE required}
DATASET=${DATASET:?DATASET required (MSRVTT|ACTNET|DIDEMO|PANDA)}
SETTING=${SETTING:?SETTING required (1|2)}
OP=${OP:?OP required (operating point numeric value)}
CAND_PATH=${CAND_PATH:?CAND_PATH required (Pass-A candidate JSON)}
OUT_DIR=${OUT_DIR:?OUT_DIR required (results_a/<m>/<ds>/setting<n>/<op>/)}
PANDA_CKPT=${PANDA_CKPT:-${REPO_ROOT}/reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}

DEVICE=${DEVICE:-0}
NUM_FRAMES=${NUM_FRAMES:-12}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-42}
CACHE_PARENT=${CACHE_PARENT:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool-Panda}
MAX_QUERIES=${MAX_QUERIES:-}

# Workstation videos_dir per dataset (only used as fallback when cache miss).
declare -A VIDEOS_DIR=(
    [MSRVTT]=/data2/uqzzha35/VideoRetrieval/msrvtt_data/MSRVTT_Frames
    [ACTNET]=/data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Frames_224x224
    [DIDEMO]=${REPO_ROOT}/dataset/DiDeMo
    [LSMDC]=${REPO_ROOT}/dataset/LSMDC
    [PANDA]=${REPO_ROOT}/data/panda
)
vdir="${VIDEOS_DIR[${DATASET}]:-${REPO_ROOT}/dataset/${DATASET}}"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/rerank.console.log"

if [ ! -f "${CAND_PATH}" ]; then
    echo "ERROR: CAND_PATH missing: ${CAND_PATH}" | tee -a "${LOG}"
    echo "rerank_status=missing_cand" > "${OUT_DIR}/rerank_status.txt"
    exit 2
fi
if [ ! -f "${PANDA_CKPT}" ]; then
    echo "ERROR: PANDA_CKPT missing: ${PANDA_CKPT}" | tee -a "${LOG}"
    echo "rerank_status=missing_ckpt" > "${OUT_DIR}/rerank_status.txt"
    exit 2
fi

result_csv="track2_${BASELINE}_${DATASET,,}_t${SETTING}_op${OP}.csv"
DST_CSV="${OUT_DIR}/result.csv"
max_query_args=()
if [ -n "${MAX_QUERIES}" ]; then
    max_query_args=(--max_queries "${MAX_QUERIES}")
fi

echo "===== $(date -u +%FT%TZ) track2_rerank_local: ${BASELINE}/${DATASET}/t${SETTING}/op${OP} =====" | tee -a "${LOG}"
echo "  cand=${CAND_PATH}" | tee -a "${LOG}"
echo "  ckpt=${PANDA_CKPT}" | tee -a "${LOG}"
echo "  out_dir=${OUT_DIR}" | tee -a "${LOG}"

# Activate workstation xpool conda env.
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"

if [ "${DATASET}" = "PANDA" ] && [ "${SETTING}" = "2" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH="${REPO_ROOT}/reranker/xpool" \
    python "${REPO_ROOT}/reranker/xpool/test_perquery.py" \
        --dataset_name "${DATASET}" \
        --huggingface \
        --videos_dir "${vdir}" \
        --num_frames "${NUM_FRAMES}" \
        --batch_size "${BATCH_SIZE}" \
        --checkpoint "${PANDA_CKPT}" \
        --cache_dir "${CACHE_PARENT}/PANDA" \
        --candidate_file "${CAND_PATH}" \
        --index_safe_candidates \
        --report_dir "${OUT_DIR}" \
        --summary_csv "${DST_CSV}" \
        --summary_json "${OUT_DIR}/xpool_eval.raw.json" \
        --seed "${SEED}" \
        "${max_query_args[@]}" \
        2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
else
    pool_flag=""
    [ "${SETTING}" = "2" ] && pool_flag="--expanded_pool"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH="${REPO_ROOT}/reranker/xpool" \
    python "${REPO_ROOT}/reranker/xpool/test.py" \
        --dataset_name "${DATASET}" \
        --huggingface \
        ${pool_flag} \
        --videos_dir "${vdir}" \
        --num_frames "${NUM_FRAMES}" \
        --batch_size "${BATCH_SIZE}" \
        --eval_checkpoint "${PANDA_CKPT}" \
        --use_cached_video_features \
        --video_cache_dir "${CACHE_PARENT}" \
        --candidate_file "${CAND_PATH}" \
        --rerank_mode \
        --result_file "${result_csv}" \
        --seed "${SEED}" \
        --no_tensorboard \
        2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}

    # X-Pool writes the result CSV under output/evaluation_results/rerank/.
    SRC_CSV="${REPO_ROOT}/output/evaluation_results/rerank/${result_csv}"
    if [ -f "${SRC_CSV}" ]; then
        cp -f "${SRC_CSV}" "${DST_CSV}"
        echo "  copied ${SRC_CSV} -> ${DST_CSV}" | tee -a "${LOG}"
    fi
fi

# Write a small JSON shape the aggregator (figure_data.csv) can read directly,
# extracting R@1/R@5/R@10 from the result.csv (X-Pool's column layout).
python3 - "${DST_CSV}" "${OUT_DIR}/xpool_eval.json" "${BASELINE}" "${DATASET}" "${SETTING}" "${OP}" "${CAND_PATH}" "${PANDA_CKPT}" <<'PY'
import sys, json, csv, os
dst_csv, out_json, baseline, ds, setting, op, cand, ckpt = sys.argv[1:]
metrics = {}
if os.path.exists(dst_csv):
    with open(dst_csv) as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        for k in ("R1", "R5", "R10", "R@1", "R@5", "R@10", "MedR", "MeanR"):
            if k in last:
                out_key = f"R@{k[1:]}" if k in ("R1", "R5", "R10") else k
                metrics[out_key] = float(last[k]) if last[k] else None
payload = {
    "metadata": {
        "method": baseline, "dataset": ds, "setting": int(setting),
        "operating_point": int(op),
        "ckpt": ckpt,
        "candidate_file": cand,
        "result_csv": dst_csv if os.path.exists(dst_csv) else None,
    },
    "metrics": metrics,
}
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote {out_json} keys={list(metrics.keys())}")
PY

# Write a terminal-state marker so the sweep wrapper can detect completion
# regardless of the rerank's internal rc.
echo "rerank_status=ok rc=${rc}" > "${OUT_DIR}/rerank_status.txt"
exit ${rc}
