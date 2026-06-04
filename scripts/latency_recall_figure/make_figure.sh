#!/usr/bin/env bash
# Phase dispatcher for the latency-recall figure function.
#
# Phases: smoke | recall-stage | rerank-stage | recall-latency | rerank-latency
#         | aggregate | render | lint | all
#   recall-stage   : stage-1 candidate export (effectiveness -> CanHit@K)
#   rerank-stage   : X-Pool rerank          (effectiveness -> R@K)
#   recall-latency : stage-1 retrieval latency (efficiency)
#   rerank-latency : X-Pool stage-2 latency    (efficiency)
#   aggregate      : walk the four stage trees -> summaries/figure_data.{csv,json}
#   render         : draw per-dataset Panel A/B PNGs from figure_data.csv -> figures/
#   lint           : validate summaries/figure_data.csv against the column contract
#   all            : the four stages, then aggregate, then render (ends at the figures)
#
# Env honored (see _env.sh for the full list + defaults):
#   BASELINES DATASETS SETTINGS OPERATING_POINTS DEVICE SEED WALL_CAP_S
#   DRY_RUN SKIP_EXISTING RUNTIME_ROOT CAND_GRDR_ROOT CAND_BASE_ROOT
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

mkdir -p "${RUNTIME_ROOT}/summaries" "${SENTINEL_DIR}"

phase_recall_stage ()   { bash "${FUNC_DIR}/recall-stage.sh"; }
phase_rerank_stage ()   { bash "${FUNC_DIR}/rerank-stage.sh"; }
phase_recall_latency () { bash "${FUNC_DIR}/recall-latency.sh"; }
phase_rerank_latency () { bash "${FUNC_DIR}/rerank-latency.sh"; }

phase_aggregate () {
    local ag_methods="${BASELINES}" ag_datasets="${DATASETS}" ag_settings="${SETTINGS}" ag_ops="${OPERATING_POINTS}"
    ( \
        # shellcheck disable=SC1090
        source "${CONDA_SH}" && conda activate "${SEMANTICTVR_ENV}" && \
        "${PYTHON}" "${FUNC_DIR}/utils/aggregate_figure_csv.py" \
            --runtime_root "${RUNTIME_ROOT}" \
            --cand_grdr_root "${CAND_GRDR_ROOT}" \
            --cand_base_root "${CAND_BASE_ROOT}" \
            --out_csv "${RUNTIME_ROOT}/summaries/figure_data.csv" \
            --out_json "${RUNTIME_ROOT}/summaries/figure_data.json" \
            --methods "${ag_methods}" --datasets "${ag_datasets}" \
            --settings "${ag_settings}" --operating_points "${ag_ops}" \
    )
}

phase_render () {
    # Draw per-dataset Panel A (CanHit@100 vs stage1 ms) + Panel B (R@K vs total ms)
    # from the aggregated CSV. Needs matplotlib (semantictvr env).
    ( \
        # shellcheck disable=SC1090
        source "${CONDA_SH}" && conda activate "${SEMANTICTVR_ENV}" && \
        "${PYTHON}" "${FUNC_DIR}/utils/render_figures.py" \
            --csv "${RUNTIME_ROOT}/summaries/figure_data.csv" \
            --out_dir "${RUNTIME_ROOT}/figures" \
    )
}

phase_lint () {
    "${PYTHON}" "${FUNC_DIR}/utils/lint_figure_data.py" \
        --csv "${RUNTIME_ROOT}/summaries/figure_data.csv" \
        --runtime_root "${RUNTIME_ROOT}"
}

phase_smoke () {
    # 2-cell end-to-end plumbing check (grdr_ref + hnsw, MSRVTT, S2, op=100).
    local sc="BASELINES=grdr_ref hnsw|DATASETS=MSRVTT|SETTINGS=2|OPERATING_POINTS=100"
    BASELINES="grdr_ref hnsw" DATASETS="MSRVTT" SETTINGS="2" OPERATING_POINTS="100" phase_recall_stage
    BASELINES="grdr_ref hnsw" DATASETS="MSRVTT" SETTINGS="2" OPERATING_POINTS="100" phase_rerank_stage
    BASELINES="grdr_ref hnsw" DATASETS="MSRVTT" SETTINGS="2" OPERATING_POINTS="100" phase_recall_latency
    BASELINES="grdr_ref hnsw" DATASETS="MSRVTT" SETTINGS="2" OPERATING_POINTS="100" phase_rerank_latency
    BASELINES="grdr_ref hnsw" DATASETS="MSRVTT" SETTINGS="2" OPERATING_POINTS="100" phase_aggregate
    local sentinel="${SENTINEL_DIR}/smoke.done"
    local csv="${RUNTIME_ROOT}/summaries/figure_data.csv"
    local rows=0
    [ -f "${csv}" ] && rows=$(($(wc -l <"${csv}") - 1))
    local sha; sha=$(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null || echo unknown)
    {
        echo "status=ok"; echo "cells=2"; echo "scope=${sc}"
        echo "csv_rows=${rows}"; echo "commit_sha=${sha}"
        echo "completed_at=$(date -u +%FT%TZ)"
    } > "${sentinel}"
    echo "smoke sentinel: ${sentinel}"
}

phase_all () { phase_recall_stage; phase_rerank_stage; phase_recall_latency; phase_rerank_latency; phase_aggregate; phase_render; }

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <phase> [<phase> ...]" >&2
    echo "phases: smoke recall-stage rerank-stage recall-latency rerank-latency aggregate render lint all" >&2
    exit 2
fi
for phase in "$@"; do
    case "${phase}" in
        smoke)          phase_smoke ;;
        recall-stage)   phase_recall_stage ;;
        rerank-stage)   phase_rerank_stage ;;
        recall-latency) phase_recall_latency ;;
        rerank-latency) phase_rerank_latency ;;
        aggregate)      phase_aggregate ;;
        render)         phase_render ;;
        lint)           phase_lint ;;
        all)            phase_all ;;
        *) echo "unknown phase: ${phase}" >&2; exit 2 ;;
    esac
done
