#!/usr/bin/env bash
# GRDR (grdr_ref) Recall-Latency cell: stage-1 retrieval latency for one (ds,setting,op)
# on the 200-query subset manifest. Required env: BASELINE DATASET SETTING OP_VALUE.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

BASELINE=${BASELINE:?}; DATASET=${DATASET:?}; SETTING=${SETTING:?}; OP_VALUE=${OP_VALUE:?}
ds_lower="${DATASET,,}"
manifest="${MANIFEST_DIR}/latency_subset_${DATASET}_t${SETTING}.json"
[ -f "${manifest}" ] || { echo "MISSING manifest: ${manifest}" >&2; exit 2; }
cand_dir="${RECALL_LATENCY_ROOT}/grdr_ref/${ds_lower}"; mkdir -p "${cand_dir}"
cand_out="${cand_dir}/${ds_lower}_t${SETTING}_${OP_VALUE}_latency.json"
log_dir="${RECALL_LATENCY_ROOT}/_logs"; mkdir -p "${log_dir}"
log="${log_dir}/grdr_ref_${DATASET}_t${SETTING}_op${OP_VALUE}_stage1.console.log"
warmup_n=${WARMUP_N_USED_DEFAULT:-10}

if [ "${CONDA_DEFAULT_ENV:-}" != "${SEMANTICTVR_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
fi

# Indist mode: resolve per-dataset ckpt + matching codebook size (Jan-2026 ckpts use ds-specific C).
if [ "${EVAL_MODE:-zeroshot}" = "indist" ]; then
    GRDR_REF_CKPT="$(grdr_ckpt_for "${ds_lower}")"
    GRDR_CODE_NUM="$(grdr_code_num_for "${ds_lower}")"
    [ -z "${GRDR_REF_CKPT}" ] && { echo "ERROR: no indist GRDR ckpt for ${ds_lower}" >&2; exit 2; }
fi
CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" "${REPO_ROOT}/run.py" \
    --candidate_export --eval_checkpoint "${GRDR_REF_CKPT}" \
    --dataset "${ds_lower}" --setting "${SETTING}" \
    --code_num "${GRDR_CODE_NUM}" --max_length "${GRDR_MAX_LENGTH}" \
    --num_latent_tokens "${GRDR_NUM_LATENT_TOKENS}" \
    --use_pseudo_queries \
    --inference_reorder_by_access_score \
    --access_score_bucket_gamma "${GRDR_ACCESS_GAMMA}" --candidate_handoff_cap "$(( GRDR_HANDOFF_CAP_MULT * OP_VALUE ))" \
    --num_candidates "${OP_VALUE}" --batch_size 1 \
    --subset_manifest "${manifest}" \
    --warmup_n_used "${warmup_n}" --wall_time_cap_s "${WALL_CAP_S}" \
    --device "${DEVICE}" \
    --output_json "${cand_out}" --seed "${SEED}" 2>&1 | tee "${log}"
