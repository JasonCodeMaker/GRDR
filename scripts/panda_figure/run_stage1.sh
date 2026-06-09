#!/usr/bin/env bash
# Stage-1 candidate export for one (method, distractor_n) over the shared pool.
# Output: ${CAND_ROOT}/<method>/<method>_d<d>_candidates.json  (schema read by aggregate.py)
#
# Pool control per method (see README "Per-method wiring"):
#   d=0  -> test-only pool (Setting 1).  d>0 -> test + first-d distractors (nested manifest).
#
# GRDR is fully wired here. The six baselines call their entrypoints with the shared
# manifest; the ones whose entrypoint does not yet accept a Panda pool / manifest are
# marked status=pending (see README required-edits) instead of silently producing a
# wrong-pool result.
#
# Required env: METHOD D   (D = distractor count; one of $DISTRACTORS)
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

METHOD=${METHOD:?METHOD required}
D=${D:?D required (distractor count)}
MANIFEST="${MANIFEST_DIR}/panda_pool_d${D}.json"
CAND_OUT="${CAND_ROOT}/${METHOD}/${METHOD}_d${D}_candidates.json"
mkdir -p "$(dirname "${CAND_OUT}")" "${LOG_DIR}"
LOG="${LOG_DIR}/stage1_${METHOD}_d${D}.log"

if [ "${SKIP_EXISTING}" -eq 1 ] && [ -s "${CAND_OUT}" ]; then echo "skip-existing: ${CAND_OUT}"; exit 0; fi
if [ ! -f "${MANIFEST}" ] && [ "${D}" -ne 0 ]; then echo "ERROR: manifest missing ${MANIFEST} (run build first)"; exit 2; fi
if [ "${DRY_RUN}" -eq 1 ]; then echo "DRY stage1: ${METHOD} d=${D} -> ${CAND_OUT}"; exit 0; fi

mark_pending () {  # <reason> — record honest status so aggregate shows 'pending', not a fake row
    local rdir="${RERANK_ROOT}/${METHOD}/d${D}"; mkdir -p "${rdir}"
    echo "pending: $1" > "${rdir}/status.txt"
    echo "PENDING (${METHOD} d=${D}): $1 — see scripts/panda_figure/README.md"
    exit 0
}

case "${METHOD}" in
  grdr)
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
    # d=0 -> Setting 1 (test-only). d>0 -> Setting 2 + sub-sample train to d.
    pool_args=(--setting 1)
    [ "${D}" -ne 0 ] && pool_args=(--setting 2 --distractor_n "${D}")
    # NOTE (fidelity): --distractor_n uses Random(42).sample (statistically comparable but
    # not the *same* videos as the shared manifest). For exact-pool parity add
    # --distractor_manifest "${MANIFEST}" once run.py/evaluator.py support it (README).
    bs=$(( 4000 / BUDGET )); [ "${bs}" -lt 1 ] && bs=1; [ "${bs}" -gt 128 ] && bs=128
    ( cd "${REPO_ROOT}" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="${DEVICE}" "${PYTHON}" run.py \
        --candidate_export --eval_checkpoint "${GRDR_CKPT}" \
        --dataset panda "${pool_args[@]}" \
        --code_num "${GRDR_CODE_NUM}" --max_length "${GRDR_MAX_LENGTH}" \
        --num_latent_tokens "${GRDR_NUM_LATENT_TOKENS}" --use_pseudo_queries \
        --inference_reorder_by_access_score --access_score_bucket_gamma "${GRDR_ACCESS_GAMMA}" \
        --candidate_handoff_cap "${BUDGET}" --num_candidates "${BUDGET}" \
        --device "${DEVICE}" --batch_size "${bs}" \
        --output_json "${CAND_OUT}" --seed "${SEED}" \
    ) 2>&1 | tee "${LOG}"
    # run.py --candidate_export writes a hardcoded default path; rescue-copy if newer.
    setting_tag=1; [ "${D}" -ne 0 ] && setting_tag=2
    def="${REPO_ROOT}/candidates/panda_c${GRDR_CODE_NUM}l${GRDR_MAX_LENGTH}_${BUDGET}_candidates_t${setting_tag}.json"
    [ -f "${def}" ] && { [ ! -f "${CAND_OUT}" ] || [ "${def}" -nt "${CAND_OUT}" ]; } && cp -f "${def}" "${CAND_OUT}"
    ;;

  tiger|avg)
    # MM-SemanticTVR avg_train_retriever_t5.py --eval. Pool = test sIDs (+ first-d train
    # sIDs for d>0 via a per-leg truncated train index). c4096/l3, mode=none.
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
    if [ "${METHOD}" = tiger ]; then
      idx_type=standard
      ckpt="${TIGER_CKPT}/standard/t5-small_T1T2_t5small_none_c4096l3_v2.0/20260520_0810/best_model"
    else
      idx_type=text_guided
      ckpt="${AVG_CKPT}/text_guided/t5-small_T1T2_t5small_none_c4096l3_v2.0/20260520_1759/best_model"
    fi
    full_train_sid="${MM_TVR_DIR}/data/panda/none/${idx_type}_c4096_l3/panda_index_internvideo2_emb_train.json"
    eval_args=(--setting 1)
    if [ "${D}" -ne 0 ]; then
      leg="${SID_CACHE_DIR}/${METHOD}/train_sid_d${D}.json"; mkdir -p "$(dirname "${leg}")"
      "${PYTHON}" "${FUNC_DIR}/prep_leg_sids.py" \
        --full_train_index "${full_train_sid}" --manifest "${MANIFEST}" --out "${leg}"
      eval_args=(--setting 2 --train_index_file "${leg}")
    fi
    ( cd "${MM_TVR_DIR}" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PYTHON}" avg_train_retriever_t5.py --eval \
        --dataset panda --mode none --index_type "${idx_type}" \
        --code_book_size 4096 --code_book_num 3 \
        --eval_checkpoint "${ckpt}" \
        --num_candidates "${BUDGET}" "${eval_args[@]}" \
        --eval_batch_size "${TIGER_AVG_EVAL_BS:-4}" \
        --gpu_id "${DEVICE}" --seed "${SEED}" --no_wandb \
        --output_json "${CAND_OUT}" \
    ) 2>&1 | tee "${LOG}"
    ;;
  t2vindexer)
    mark_pending "T2VIndexer per-leg index over the pool; documented OOM risk at Panda c4096 (README §T2VIndexer)"
    ;;
  hnsw|ivf)
    # baselines/ann_dense_retrieval/eval_ann.py over the shared Panda pool (XPool-CLIP
    # mean-pooled features). d=0 -> Setting 1 (test-only); d>0 -> manifest distractor pool.
    source "${CONDA_SH}"; conda activate "${SEMANTICTVR_ENV}"
    ann_args=(--setting 1)
    [ "${D}" -ne 0 ] && ann_args=(--setting 2 --distractor_manifest "${MANIFEST}")
    # Generous search effort (this is effectiveness, not latency).
    ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${DEVICE}" \
      "${PYTHON}" baselines/ann_dense_retrieval/eval_ann.py \
        --dataset panda --index_type "${METHOD}" "${ann_args[@]}" \
        --num_candidates "${BUDGET}" \
        --checkpoint "${XPOOL_CKPT}" --cache_dir "${XPOOL_CACHE}" \
        --hnsw_ef_search 1024 --ivf_nlist 4096 --ivf_nprobe 256 \
        --device 0 --output_json "${CAND_OUT}" --batch_size 256 \
    ) 2>&1 | tee "${LOG}"
    ;;
  eercf)
    mark_pending "EERCF native dense over the pool (pool-monotonic single-pass shortcut); heavy long-pole (README §EERCF)"
    ;;
  *) echo "Unknown method: ${METHOD}"; exit 2 ;;
esac

[ -s "${CAND_OUT}" ] && echo "stage1 ok: ${CAND_OUT}" || echo "stage1 WARN: no candidate file at ${CAND_OUT}"
