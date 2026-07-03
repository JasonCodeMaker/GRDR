#!/usr/bin/env bash
# GRDR (grdr_ref) Recall-Stage cell: stage-1 candidate export for one (ds,setting,op).
# The volatile arm — GRDR config lives in _env.sh (GRDR_REF_CKPT + GRDR_* knobs).
# run.py --candidate_export writes to the hardcoded repo candidates/ path
# (trainer/evaluator.py); we rescue-copy it to CAND_OUT under candidates/GRDR/.
# Required env: DS_LOWER SETTING OP CAND_OUT.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

DS_LOWER=${DS_LOWER:?DS_LOWER required}
SETTING=${SETTING:?SETTING required}
OP=${OP:?OP required}
CAND_OUT=${CAND_OUT:?CAND_OUT required}
LOG_DIR=${LOG_DIR:-${RECALL_STAGE_ROOT}/_logs}
mkdir -p "$(dirname "${CAND_OUT}")" "${LOG_DIR}"
log="${LOG_DIR}/grdr_ref_${DS_LOWER}_t${SETTING}_op${OP}.console.log"

if [ "${CONDA_DEFAULT_ENV:-}" != "${SEMANTICTVR_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
fi

# In-distribution mode: resolve the per-dataset GRDR ckpt + its codebook size (the Jan-2026
# fallback ckpts use ds-specific C; passing 128 unconditionally silently fails to load DiDeMo/LSMDC).
if [ "${EVAL_MODE:-zeroshot}" = "indist" ]; then
    GRDR_REF_CKPT="$(grdr_ckpt_for "${DS_LOWER}")"
    GRDR_CODE_NUM="$(grdr_code_num_for "${DS_LOWER}")"
    [ -z "${GRDR_REF_CKPT}" ] && { echo "ERROR: no indist GRDR ckpt for ${DS_LOWER} (train it first)"; exit 2; }
    echo "  [indist] GRDR ckpt=${GRDR_REF_CKPT} code_num=${GRDR_CODE_NUM} max_length=${GRDR_MAX_LENGTH}"
fi

# Beam-aware export batch: beam-search generation memory ~ batch * num_beams, so cap
# batch*OP to a budget to avoid CUDA OOM at large beams (batch=128 OOMs once num_beams>=100;
# beam=50/batch=128 (=6400) fit a 48GB card). Target ~4000 for headroom; override EXPORT_BATCH_SIZE
# to force a fixed batch, or EXPORT_BEAM_BUDGET to retune the budget.
if [ -n "${EXPORT_BATCH_SIZE:-}" ]; then
    EXPORT_BS="${EXPORT_BATCH_SIZE}"
else
    EXPORT_BS=$(( ${EXPORT_BEAM_BUDGET:-4000} / OP ))
    [ "${EXPORT_BS}" -gt 128 ] && EXPORT_BS=128
    [ "${EXPORT_BS}" -lt 1 ] && EXPORT_BS=1
fi
echo "  export batch_size=${EXPORT_BS} (beam=${OP}, beam_budget=${EXPORT_BEAM_BUDGET:-4000})"

marker=$(mktemp)
trap 'rm -f "${marker}"' EXIT
touch "${marker}"

# run.py:188 overrides CUDA_VISIBLE_DEVICES from --device; the explicit CUDA_VISIBLE_DEVICES
# keeps the shell-set GPU consistent. NLT=4 + pseudo loads the multi-view encoder slots.
( cd "${REPO_ROOT}" && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON}" "${REPO_ROOT}/run.py" \
      --candidate_export \
      --eval_checkpoint "${GRDR_REF_CKPT}" \
      --model_name "${GRDR_MODEL_NAME}" \
      --dataset "${DS_LOWER}" --setting "${SETTING}" \
      --code_num "${GRDR_CODE_NUM}" --max_length "${GRDR_MAX_LENGTH}" \
      --num_latent_tokens "${GRDR_NUM_LATENT_TOKENS}" \
      --use_pseudo_queries \
      --inference_reorder_by_access_score \
      --access_score_bucket_gamma "${GRDR_ACCESS_GAMMA}" --candidate_handoff_cap "$(( GRDR_HANDOFF_CAP_MULT * OP ))" \
      --num_candidates "${OP}" \
      --device "${DEVICE}" \
      --batch_size "${EXPORT_BS}" \
      --output_json "${CAND_OUT}" --seed "${SEED}" \
) 2>&1 | tee "${log}"

if [ ! -s "${CAND_OUT}" ] || [ ! "${CAND_OUT}" -nt "${marker}" ]; then
    echo "ERROR: GRDR export did not produce a fresh candidate JSON at ${CAND_OUT}" >&2
    exit 2
fi
echo "  wrote fresh candidate JSON: ${CAND_OUT}"
