#!/usr/bin/env bash
# Unified latency reporter on ANN and GRDR Stage-1 candidate JSONs.
# Reports per-query (T_text_encode, T_search, T_rerank, total) at identical K.
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/_common.sh

DEVICE=0
INDEX_TYPE=hnsw
MAX_QUERIES=100
NUM_RUNS=3
NUM_WARMUP=5
CANDIDATE_DIR=candidates
OUTPUT_ROOT=output/latency_unified

# (dataset:setting:K:GRDR_codebook), GRDR codelen is fixed at 3.
PAIRS=(
  "msrvtt:1:100:128"  "msrvtt:2:20:128"
  "actnet:1:100:128"  "actnet:2:50:128"
  "didemo:1:100:96"   "didemo:2:50:96"
  "lsmdc:1:100:200"   "lsmdc:2:20:200"
)
GRDR_CODELEN=3

mkdir -p "$OUTPUT_ROOT"
activate_conda_env "$XPOOL_ENV"

for spec in "${PAIRS[@]}"; do
  IFS=':' read -r dataset setting k code <<<"$spec"
  ann_json="${CANDIDATE_DIR}/${dataset}_ann_${INDEX_TYPE}_${k}_candidates_t${setting}.json"
  grdr_json="${CANDIDATE_DIR}/${dataset}_c${code}l${GRDR_CODELEN}_${k}_candidates_t${setting}.json"

  files=()
  labels=()
  if [[ -f "$ann_json" ]]; then
    files+=("$ann_json"); labels+=("ANN-${INDEX_TYPE^^}")
  else
    echo "skip ANN file (missing): $ann_json"
  fi
  if [[ -f "$grdr_json" ]]; then
    files+=("$grdr_json"); labels+=("GRDR-Stage1")
  else
    echo "skip GRDR file (missing): $grdr_json"
  fi
  if [[ ${#files[@]} -eq 0 ]]; then
    continue
  fi

  out_json="${OUTPUT_ROOT}/${dataset}_t${setting}_K${k}.json"
  echo ""
  echo "============================================================"
  echo "  ${dataset} setting=${setting} K=${k}"
  for f in "${files[@]}"; do echo "    - $f"; done
  echo "============================================================"
  run_cmd python -m baselines.ann_dense_retrieval.latency_report \
    --candidate_files "${files[@]}" \
    --labels "${labels[@]}" \
    --max_queries "$MAX_QUERIES" \
    --num_warmup "$NUM_WARMUP" \
    --num_runs_per_query "$NUM_RUNS" \
    --device "$DEVICE" \
    --output "$out_json"
done

deactivate_conda_env
echo ""
echo "All per-(dataset,setting) latency reports written under: $OUTPUT_ROOT"
