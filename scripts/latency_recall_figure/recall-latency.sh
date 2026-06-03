#!/usr/bin/env bash
# Recall-Latency: stage-1 retrieval latency on the 200-query subset (efficiency).
# ANN stage-1 latency is owned by Recall-Stage (A.2) and skipped here.
# Each cell self-activates its env.
# Honored env: BASELINES DATASETS SETTINGS OPERATING_POINTS DEVICE SEED WALL_CAP_S DRY_RUN SKIP_EXISTING.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
source "${FUNC_DIR}/_cells.sh"

mkdir -p "${RECALL_LATENCY_ROOT}" "${SENTINEL_DIR}"
run_stage_loop cell_recall_latency recall-latency
