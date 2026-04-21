#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/_common.sh

INDEX_TYPES_STR="${INDEX_TYPES:-flat hnsw ivf}"
read -r -a INDEX_TYPES_ARR <<<"$INDEX_TYPES_STR"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/ann_baseline}"
SUMMARY_CSV="${SUMMARY_CSV:-$OUTPUT_ROOT/ann_stage1_stage2_summary.csv}"
COMPACT_CSV="${COMPACT_CSV:-$OUTPUT_ROOT/ann_stage1_stage2_compact.csv}"

for index_type in "${INDEX_TYPES_ARR[@]}"; do
  echo ""
  echo "########################################################################"
  echo "Running ANN baseline with INDEX_TYPE=${index_type}"
  echo "########################################################################"
  INDEX_TYPE="$index_type" OUTPUT_DIR="$OUTPUT_ROOT/$index_type" bash scripts/baselines/eval_ann.sh
done

activate_conda_env "$ANN_BASELINE_ENV"
run_cmd python scripts/baselines/summarize_ann_results.py \
  --stage1_root "$OUTPUT_ROOT" \
  --candidate_dir candidates \
  --stage2_dir output/reranker \
  --output "$SUMMARY_CSV" \
  --compact-output "$COMPACT_CSV"
deactivate_conda_env

echo ""
echo "Summary CSV written to: $SUMMARY_CSV"
echo "Compact CSV written to: $COMPACT_CSV"
