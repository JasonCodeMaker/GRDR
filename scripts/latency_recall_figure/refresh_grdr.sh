#!/usr/bin/env bash
# High-frequency GRDR path: re-run ONLY the grdr_ref arm through all four stages,
# then re-aggregate the CSV with the cached baseline rows intact.
# Use after editing GRDR knobs in _env.sh or re-pointing output/checkpoints/GRDR/panda/latency_recall_best.
#   SKIP_EXISTING=1 (default) keeps stable baselines cached; pass SKIP_EXISTING=0 to force grdr re-run.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

FULL_METHODS="${BASELINES}"        # full method list for aggregation (keeps baseline rows)
echo "===== refresh_grdr: re-running grdr_ref; aggregating over '${FULL_METHODS}' ====="

# Force a fresh grdr_ref run regardless of cache (this is the point of refresh).
BASELINES="grdr_ref" SKIP_EXISTING="${SKIP_EXISTING_GRDR:-0}" \
    bash "${FUNC_DIR}/make_figure.sh" recall-stage rerank-stage recall-latency rerank-latency

# Aggregate over ALL methods so the baseline rows (cached) stay in the CSV.
BASELINES="${FULL_METHODS}" bash "${FUNC_DIR}/make_figure.sh" aggregate
echo "===== refresh_grdr done: ${RUNTIME_ROOT}/summaries/figure_data.csv ====="
