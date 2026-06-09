#!/usr/bin/env bash
# Stage-2 rerank for one (method, distractor_n): X-Pool Panda reranker over the
# <=BUDGET stage-1 candidates -> R@1/5/10. Rerank cost is bounded by the candidate
# count, independent of pool size. EERCF reranks natively (R@K read from its own
# candidate JSON). Output: ${RERANK_ROOT}/<method>/d<d>/rerank.json {"metrics": {...}}.
#
# Required env: METHOD D
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

METHOD=${METHOD:?METHOD required}
D=${D:?D required}
CAND="${CAND_ROOT}/${METHOD}/${METHOD}_d${D}_candidates.json"
OUT_DIR="${RERANK_ROOT}/${METHOD}/d${D}"
OUT_JSON="${OUT_DIR}/rerank.json"
mkdir -p "${OUT_DIR}"
LOG="${LOG_DIR}/rerank_${METHOD}_d${D}.log"; mkdir -p "${LOG_DIR}"

if [ "${SKIP_EXISTING}" -eq 1 ] && [ -s "${OUT_JSON}" ]; then echo "skip-existing: ${OUT_JSON}"; exit 0; fi
if [ ! -s "${CAND}" ]; then echo "rerank skip: no candidate JSON ${CAND} (stage1 pending/failed)"; exit 0; fi
if [ "${DRY_RUN}" -eq 1 ]; then echo "DRY rerank: ${METHOD} d=${D}"; exit 0; fi

case "${METHOD}" in
  eercf)
    # EERCF rerank is native; lift R@K straight from its candidate JSON metrics.
    "${PYTHON}" - "${CAND}" "${OUT_JSON}" "${METHOD}" "${D}" <<'PY'
import sys, json
cand, out, method, d = sys.argv[1:]
data = json.load(open(cand)); mc = data.get("metrics", {})
def g(*ks):
    for k in ks:
        if k in mc and mc[k] not in (None, ""):
            v = float(mc[k]); return v*100 if v <= 1.0 else v
    return None
m = {k: v for k, v in (("R@1", g("R@1","Recall@1")), ("R@5", g("R@5","Recall@5")),
                       ("R@10", g("R@10","Recall@10"))) if v is not None}
json.dump({"metadata": {"method": method, "distractor_n": int(d), "candidate_file": cand},
           "metrics": m}, open(out, "w"), indent=2)
print(f"wrote {out} keys={list(m.keys())}")
PY
    ;;
  *)
    source "${CONDA_SH}"; conda activate "${CONDA_ENV_XPOOL}"
    # Candidate-only rerank: load only the <=BUDGET candidate frames per query and pool/score
    # them directly. Cost is bounded by candidate count, independent of pool size; d=0 and d>0
    # use the same call (candidate frames are loaded straight from cache, no expanded pool).
    ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${DEVICE}" PYTHONPATH="${REPO_ROOT}/reranker/xpool" \
      "${PYTHON}" reranker/xpool/candidate_rerank.py \
        --candidate_file "${CAND}" \
        --eval_checkpoint "${XPOOL_CKPT}" \
        --video_cache_dir "${XPOOL_CACHE}" \
        --num_frames 4 --device 0 \
        --max_candidates "${BUDGET}" \
        --out_json "${OUT_JSON}" --seed "${SEED}" \
    ) 2>&1 | tee "${LOG}"
    ;;
esac
echo "ok rerank=ok" > "${OUT_DIR}/status.txt"
