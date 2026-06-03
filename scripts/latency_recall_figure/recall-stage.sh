#!/usr/bin/env bash
# Recall-Stage: stage-1 candidate export (effectiveness; feeds CanHit@K).
# Honored env: BASELINES DATASETS SETTINGS OPERATING_POINTS DEVICE SEED DRY_RUN SKIP_EXISTING.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
source "${FUNC_DIR}/_cells.sh"

# Stage-1 export launchers (run.py / MM-SemanticTVR / EERCF import) need semantictvr.
if [ "${CONDA_DEFAULT_ENV:-}" != "${SEMANTICTVR_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"; echo "  conda env: ${CONDA_DEFAULT_ENV:-none}"
fi

mkdir -p "${RECALL_STAGE_ROOT}" "${CAND_GRDR_ROOT}" "${CAND_BASE_ROOT}" "${SENTINEL_DIR}"
run_stage_loop cell_recall_stage recall-stage
