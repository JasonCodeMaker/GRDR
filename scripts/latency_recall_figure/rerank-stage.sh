#!/usr/bin/env bash
# Rerank-Stage: X-Pool rerank of the stage-1 candidates (effectiveness; R@K).
# Each cell self-activates the xpool env (track2_rerank_local / ann_rerank).
# Honored env: BASELINES DATASETS SETTINGS OPERATING_POINTS DEVICE SEED DRY_RUN SKIP_EXISTING.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
source "${FUNC_DIR}/_cells.sh"

mkdir -p "${RERANK_STAGE_ROOT}" "${SENTINEL_DIR}"
run_stage_loop cell_rerank_stage rerank-stage
