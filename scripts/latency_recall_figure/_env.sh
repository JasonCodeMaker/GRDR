#!/usr/bin/env bash
# Single source of truth for every path the latency-recall figure function uses.
# Sourced by make_figure.sh, the four stage scripts, and every cell runner.
# All defaults are native (output/, candidates/, reranker/xpool/) or sibling
# PROJECT dirs (EERCF / MM-SemanticTVR / T2VIndexer). There are NO references
# into any research package directory.

# REPO_ROOT self-locates from this file (scripts/latency_recall_figure/_env.sh).
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FUNC_DIR=${FUNC_DIR:-${REPO_ROOT}/scripts/latency_recall_figure}
LATENCY_HELPERS_DIR=${LATENCY_HELPERS_DIR:-${FUNC_DIR}/utils}

# Eval mode switch (full block + resolvers below). Defined early so the output trees can branch.
# indist (default): per-dataset C=128/L=3, in-distribution — the canonical 'figures/' tree.
# zeroshot: Panda ckpts, zero-shot (legacy figure) — the 'figures_panda/' tree.
EVAL_MODE=${EVAL_MODE:-indist}
_indist_suffix=""; [ "${EVAL_MODE}" = "indist" ] && _indist_suffix="_indist"  # candidate subtree split (candidates_indist/)

# Runtime output root. The in-distribution figure is the canonical 'figures/' tree;
# the legacy Panda zero-shot tree was renamed to 'figures_panda/'.
_eval_tree_suffix=""; [ "${EVAL_MODE}" = "zeroshot" ] && _eval_tree_suffix="_panda"
RUNTIME_ROOT=${RUNTIME_ROOT:-${REPO_ROOT}/output/evaluation_results/figures${_eval_tree_suffix}}
SENTINEL_DIR=${SENTINEL_DIR:-${RUNTIME_ROOT}/manifests}
# Dataset-level inputs (latency subset manifests, EERCF query TSVs) are shared across
# eval modes — they describe the test/train slice, not the model ckpt. They physically
# live in the panda tree (where they were first generated); both modes read the same slice.
EVAL_INPUTS_ROOT=${EVAL_INPUTS_ROOT:-${REPO_ROOT}/output/evaluation_results/figures_panda}
MANIFEST_DIR=${MANIFEST_DIR:-${EVAL_INPUTS_ROOT}/manifests/latency}

# Candidate-set roots: GRDR (volatile) vs baselines (stable). Req 3. indist -> separate subtree
# so in-distribution candidates do not clobber the zero-shot ones for the same (ds, op).
CAND_GRDR_ROOT=${CAND_GRDR_ROOT:-${REPO_ROOT}/candidates${_indist_suffix}/GRDR}
CAND_BASE_ROOT=${CAND_BASE_ROOT:-${REPO_ROOT}/candidates${_indist_suffix}/baselines}

# Per-stage result roots (the 2x2 naming; replaces pass_a/pass_b).
RECALL_STAGE_ROOT=${RECALL_STAGE_ROOT:-${RUNTIME_ROOT}/recall-stage}
RERANK_STAGE_ROOT=${RERANK_STAGE_ROOT:-${RUNTIME_ROOT}/rerank-stage}
RECALL_LATENCY_ROOT=${RECALL_LATENCY_ROOT:-${RUNTIME_ROOT}/recall-latency}
RERANK_LATENCY_ROOT=${RERANK_LATENCY_ROOT:-${RUNTIME_ROOT}/rerank-latency}

# Shared X-Pool reranker checkpoint (used by every method's Stage-2). In place.
PANDA_CKPT=${PANDA_CKPT:-${REPO_ROOT}/reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}
CACHE_PARENT=${CACHE_PARENT:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool-Panda}

# ANN baselines (hnsw/ivf) sweep PAIRED (search-breadth, candidate-budget K)
# operating points: ef_search (HNSW) / nprobe (IVF) grows with K so Stage-1 search
# effort -- and thus Stage-1 latency -- rises across the curve instead of collapsing
# to a vertical line, while K also drives the candidate pool handed to Stage-2.
# Each op K maps to a knob value ("K:val" pairs). HNSW and IVF share the SAME K ladder.
#   HNSW: K aligned to ef_search, K=ef_search in {20,40,100,200} (matches the README + IVF
#         ladder for a like-for-like head-to-head; all four points sit within the <=310
#         compact-budget gate). ef_search is floored at K in eval_ann.py.
#   IVF:  K in {20,40,100,200}, nprobe geometric mirror 4/8/16/32 (max nprobe=32).
ANN_HNSW_EF_BY_K=${ANN_HNSW_EF_BY_K:-"20:20 40:40 100:100 200:200"}
ANN_IVF_NPROBE_BY_K=${ANN_IVF_NPROBE_BY_K:-"20:4 40:8 100:16 200:32"}

# GRDR (the method) — best ckpt stable alias under output/checkpoints/GRDR/panda. Req 6.
# Re-point latency_recall_best when a new GRDR champion lands; this default never changes.
GRDR_REF_CKPT=${GRDR_REF_CKPT:-${REPO_ROOT}/output/checkpoints/GRDR/panda/latency_recall_best/model-3-fit/best_model.pt}
GRDR_CODE_NUM=${GRDR_CODE_NUM:-4096}
GRDR_MAX_LENGTH=${GRDR_MAX_LENGTH:-3}
GRDR_NUM_LATENT_TOKENS=${GRDR_NUM_LATENT_TOKENS:-4}
GRDR_ACCESS_GAMMA=${GRDR_ACCESS_GAMMA:-0.50}
# Export handoff cap is per-op: candidate_handoff_cap = MULT * beam (project budget rule
# avg_candidates <= 3x beam_size). Drivers compute MULT*OP; do NOT use a fixed scalar here.
GRDR_HANDOFF_CAP_MULT=${GRDR_HANDOFF_CAP_MULT:-3}

# Baseline checkpoints — all native, model/dataset layout (output/checkpoints/Baseline/<model>/panda). Req 5.
BASE_CKPT_ROOT=${BASE_CKPT_ROOT:-${REPO_ROOT}/output/checkpoints/Baseline}
TIGER_AVG_CKPT_ROOT=${TIGER_AVG_CKPT_ROOT:-${BASE_CKPT_ROOT}}
EERCF_INIT_MODEL=${EERCF_INIT_MODEL:-${BASE_CKPT_ROOT}/eercf/panda/pytorch_model.bin.best.0}
EERCF_CACHE_ROOT=${EERCF_CACHE_ROOT:-${BASE_CKPT_ROOT}/eercf/panda/cached_video_features_p3d}
T2V_CKPT=${T2V_CKPT:-${BASE_CKPT_ROOT}/t2vindexer/panda/best_model.pt}

# EERCF inputs relocated into native. Sim-matrices are ckpt-specific (per eval mode);
# query sets are dataset-level (text captions, shared across eval modes via EVAL_INPUTS_ROOT).
MATRIX_ROOT=${MATRIX_ROOT:-${RUNTIME_ROOT}/matrices/eercf}
EERCF_QUERY_SET_ROOT=${EERCF_QUERY_SET_ROOT:-${EVAL_INPUTS_ROOT}/query_sets}

# Sibling PROJECT code/data (NOT research packages — allowed to stay external).
EERCF_DATA_ROOT=${EERCF_DATA_ROOT:-/home/uqzzha35/Project/SemanticID/EERCF/data}
EERCF_DIR=${EERCF_DIR:-/home/uqzzha35/Project/SemanticID/EERCF}
MM_TVR_DIR=${MM_TVR_DIR:-/home/uqzzha35/Project/SemanticID/MM-SemanticTVR}
T2V_DIR=${T2V_DIR:-/home/uqzzha35/Project/SemanticID/T2VIndexer-generativeSearch}
# Sibling DATA caches (workstation /data2; not packages).
EERCF_FRAME_CACHE=${EERCF_FRAME_CACHE:-/data2/uqzzha35/VideoRetrieval/eercf_cache/cached_frame_features_p3d}

# Conda. SEMANTICTVR_ENV runs stage-1 export + the CSV aggregator; XPOOL the rerank/latency.
CONDA_SH=${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}
SEMANTICTVR_ENV=${SEMANTICTVR_ENV:-semantictvr}
CONDA_ENV_XPOOL=${CONDA_ENV_XPOOL:-xpool}
EERCF_CONDA_ENV=${EERCF_CONDA_ENV:-semanticID}

# Sweep scope defaults.
BASELINES=${BASELINES:-"grdr_ref tiger avg t2vindexer eercf hnsw ivf"}
DATASETS=${DATASETS:-"MSRVTT ACTNET DIDEMO LSMDC"}
SETTINGS=${SETTINGS:-"2"}
OPERATING_POINTS=${OPERATING_POINTS:-""}
DEVICE=${DEVICE:-0}
SEED=${SEED:-42}
WALL_CAP_S=${WALL_CAP_S:-300}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
PYTHON=${PYTHON_BIN:-python}

# ---- In-distribution ckpt resolvers (EVAL_MODE=indist; EVAL_MODE defined near the top) ----
# In-distribution X-Pool video-feature cache parent (per-dataset subdirs MSRVTT/ACTNET/DIDEMO/LSMDC).
XPOOL_CACHE_INDIST=${XPOOL_CACHE_INDIST:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool}
# Per-dataset C=128/L=3 TIGER/AVG retriever ckpt root (written by scripts/baselines/train_gr_baseline.sh).
BASELINE_C128L3_ROOT=${BASELINE_C128L3_ROOT:-${REPO_ROOT}/output/checkpoints/Baseline_c128l3}
# GRDR MSRVTT champion (already C=128/L=3) reused as-is in indist mode.
GRDR_MSRVTT_INDIST_CKPT=${GRDR_MSRVTT_INDIST_CKPT:-${REPO_ROOT}/output/checkpoints/GRDR/msrvtt/bucket_candidate_k20/20260428163014-fit_bucket_l010_g10_k20_s42/model-3-fit/best_model.pt}

# Per-dataset X-Pool reranker ckpt (in-distribution); falls back to PANDA_CKPT for unknown ds.
xpool_ckpt_for () {
    case "${1,,}" in
        msrvtt) echo "${REPO_ROOT}/reranker/xpool/ckpt/msrvtt9k_model_best.pth" ;;
        actnet) echo "${REPO_ROOT}/reranker/xpool/ckpt/actnet_model_best.pth" ;;
        didemo) echo "${REPO_ROOT}/reranker/xpool/ckpt/didemo_model_best.pth" ;;
        lsmdc)  echo "${REPO_ROOT}/reranker/xpool/ckpt/lsmdc_model_best.pth" ;;
        *)      echo "${PANDA_CKPT}" ;;
    esac
}
# Per-dataset GRDR c128l3 ckpt: reuse MSRVTT champion; resolve latest model-3-fit for the rest,
# falling back to the flat <ds>/best_model/best_model.pt layout used by the Jan 2026 ckpts.
grdr_ckpt_for () {
    local ds="${1,,}"
    if [ "${ds}" = "msrvtt" ]; then echo "${GRDR_MSRVTT_INDIST_CKPT}"; return; fi
    # Per-dataset one-off override hook: GRDR_<DS>_INDIST_CKPT (e.g. GRDR_LSMDC_INDIST_CKPT).
    local ov="GRDR_${ds^^}_INDIST_CKPT"
    if [ -n "${!ov:-}" ]; then echo "${!ov}"; return; fi
    local f
    f=$(find "${REPO_ROOT}/output/checkpoints/GRDR/${ds}" -path '*/model-3-fit/best_model.pt' 2>/dev/null | sort | tail -1)
    if [ -z "${f}" ]; then
        f="${REPO_ROOT}/output/checkpoints/GRDR/${ds}/best_model/best_model.pt"
        [ -f "${f}" ] || f=""
    fi
    echo "${f}"
}
# Per-dataset codebook size of the resolved GRDR ckpt. The P3 (Jun-2026) retrain put all four
# datasets on a uniform C=128 (c128l3). Pass the matching --code_num or torch silently fails to
# load weights. (Pre-P3 fallback ckpts used ds-specific C: didemo=96, lsmdc=200.)
grdr_code_num_for () {
    local ds="${1,,}"
    # Per-dataset one-off override hook: GRDR_<DS>_CODE_NUM (e.g. GRDR_LSMDC_CODE_NUM=200).
    local ov="GRDR_${ds^^}_CODE_NUM"
    if [ -n "${!ov:-}" ]; then echo "${!ov}"; return; fi
    case "${ds}" in
        msrvtt) echo 128 ;;
        actnet) echo 128 ;;
        didemo) echo 128 ;;
        lsmdc)  echo 128 ;;
        *)      echo "${GRDR_CODE_NUM}" ;;
    esac
}
# Per-dataset TIGER/AVG c128l3 retriever ckpt dir (latest best_model under Baseline_c128l3/<baseline>/<ds>).
baseline_gr_ckpt_for () {
    find "${BASELINE_C128L3_ROOT}/$1/${2,,}" -path '*/best_model' -type d 2>/dev/null | sort | tail -1
}
# Per-dataset EERCF model (in-distribution); the P3D feature cache is shared (dataset-keyed subdirs).
eercf_ckpt_for () {
    echo "${BASE_CKPT_ROOT}/eercf/${1,,}/pytorch_model.bin.0"
}
# Per-dataset T2VIndexer c128l3 ckpt (newest exp under Model/logs/<ds>). Only c128l3-era
# exp dirs (>= T2V_CKPT_FLOOR) are eligible, so the stale Jan k30_c30 ckpt is never picked.
t2v_ckpt_for () {
    local floor="${T2V_CKPT_FLOOR:-exp_20260201}"
    find "${T2V_DIR}/Model/logs/${1,,}" -path '*/best_model.pt' 2>/dev/null \
        | awk -v f="${floor}" 'match($0,/exp_[0-9-]+/){e=substr($0,RSTART,RLENGTH); if(e>=f) print}' \
        | sort | tail -1
}

export REPO_ROOT FUNC_DIR LATENCY_HELPERS_DIR RUNTIME_ROOT EVAL_INPUTS_ROOT SENTINEL_DIR MANIFEST_DIR \
       EVAL_MODE XPOOL_CACHE_INDIST BASELINE_C128L3_ROOT GRDR_MSRVTT_INDIST_CKPT \
       CAND_GRDR_ROOT CAND_BASE_ROOT \
       RECALL_STAGE_ROOT RERANK_STAGE_ROOT RECALL_LATENCY_ROOT RERANK_LATENCY_ROOT \
       PANDA_CKPT CACHE_PARENT \
       ANN_HNSW_EF_BY_K ANN_IVF_NPROBE_BY_K \
       GRDR_REF_CKPT GRDR_CODE_NUM GRDR_MAX_LENGTH GRDR_NUM_LATENT_TOKENS \
       GRDR_ACCESS_GAMMA GRDR_HANDOFF_CAP_MULT \
       BASE_CKPT_ROOT TIGER_AVG_CKPT_ROOT EERCF_INIT_MODEL EERCF_CACHE_ROOT T2V_CKPT \
       MATRIX_ROOT EERCF_QUERY_SET_ROOT EERCF_DATA_ROOT EERCF_DIR MM_TVR_DIR T2V_DIR \
       EERCF_FRAME_CACHE \
       CONDA_SH SEMANTICTVR_ENV CONDA_ENV_XPOOL EERCF_CONDA_ENV \
       BASELINES DATASETS SETTINGS OPERATING_POINTS DEVICE SEED WALL_CAP_S \
       DRY_RUN SKIP_EXISTING PYTHON
