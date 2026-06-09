#!/usr/bin/env bash
# Panda pool-scaling figure: phase dispatcher.
#   build      build_manifests.py -> nested seed-42 distractor manifests
#   stage1     per-(method, d) stage-1 candidate export
#   rerank     per-(method, d) X-Pool / native rerank -> R@K
#   aggregate  walk outputs -> summaries/figure_data.csv
#   all        build -> stage1 -> rerank -> aggregate
#
# Scope via env: METHODS, DISTRACTORS, DEVICE, SKIP_EXISTING, DRY_RUN.
# One cell = one (method, d). The render step is the notebook panda_pool_scaling.ipynb.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(all)
expand () { local p=("$@"); for x in "${p[@]}"; do [ "$x" = all ] && { echo "build stage1 rerank aggregate"; return; }; done; echo "${p[@]}"; }
RUN=$(expand "${PHASES[@]}")
echo "===== $(date -u +%FT%TZ) panda_figure phases='${RUN}' METHODS='${METHODS}' DISTRACTORS='${DISTRACTORS}' DEVICE=${DEVICE} ====="

for phase in ${RUN}; do
  case "${phase}" in
    build)
      source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
      "${PYTHON}" "${FUNC_DIR}/build_manifests.py" \
        --train_json "${PANDA_TRAIN_JSON}" --test_json "${PANDA_TEST_JSON}" \
        --out_dir "${MANIFEST_DIR}" --seed "${SEED}" --distractors ${DISTRACTORS}
      ;;
    stage1|rerank)
      script="${FUNC_DIR}/run_stage1.sh"; [ "${phase}" = rerank ] && script="${FUNC_DIR}/rerank.sh"
      for m in ${METHODS}; do
        for d in ${DISTRACTORS}; do
          echo "----- ${phase}: ${m} d=${d} -----"
          METHOD="${m}" D="${d}" bash "${script}" || echo "WARN: ${phase} ${m} d=${d} rc=$?"
        done
      done
      ;;
    aggregate)
      source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
      "${PYTHON}" "${FUNC_DIR}/aggregate.py" \
        --cand_root "${CAND_ROOT}" --rerank_root "${RERANK_ROOT}" \
        --manifest_dir "${MANIFEST_DIR}" --methods ${METHODS} \
        --distractors ${DISTRACTORS} --seed "${SEED}" \
        --out_csv "${SUMMARY_DIR}/figure_data.csv"
      ;;
    *) echo "Unknown phase: ${phase}" >&2; exit 2 ;;
  esac
  mkdir -p "${SENTINEL_DIR}"; date -u +%FT%TZ > "${SENTINEL_DIR}/${phase}.done"
done
echo "===== $(date -u +%FT%TZ) panda_figure done. CSV: ${SUMMARY_DIR}/figure_data.csv ====="
