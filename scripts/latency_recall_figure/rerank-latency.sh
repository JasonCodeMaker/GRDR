#!/usr/bin/env bash
# Rerank-Latency: X-Pool stage-2 per-query latency (efficiency).
# Reads the Recall-Latency JSONs; each cell stages the input and runs stage2 in an xpool subshell.
# Honored env: BASELINES DATASETS SETTINGS OPERATING_POINTS DEVICE SEED WALL_CAP_S DRY_RUN SKIP_EXISTING.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
source "${FUNC_DIR}/_cells.sh"

mkdir -p "${RERANK_LATENCY_ROOT}" "${SENTINEL_DIR}"
run_stage_loop cell_rerank_latency rerank-latency
