#!/usr/bin/env bash
# Single source of truth for the Panda pool-scaling figure pipeline.
# Sourced by make_panda_figure.sh, build_manifests, run_stage1.sh, rerank.sh, aggregate.
# Figure: y = X-Pool rerank Recall@{1,5,10}; x = search-pool size (test -> test + N train
# distractors). One curve per method. Single seed (42). Panda in-distribution (c4096/l3).

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FUNC_DIR=${FUNC_DIR:-${REPO_ROOT}/scripts/panda_figure}

# ---- The sweep axis: distractor counts d (pool size N = N_TEST + d) ----
# 6 points: test-only (d=0) + {400k, 800k, 1.2M, 1.6M, 2.0M} train distractors.
# Pools are NESTED prefixes of one seed-42 shuffle (400k subset 800k subset ... subset 2.0M).
N_TEST=${N_TEST:-5694}
DISTRACTORS=${DISTRACTORS:-"0 400000 800000 1200000 1600000 2000000"}

# ---- Fixed candidate budget (held constant across all pool sizes) ----
# GR methods (grdr/tiger/avg/t2vindexer): beam size = BUDGET; GRDR handoff cap = BUDGET.
# ANN (hnsw/ivf): top-K = BUDGET (search effort set generous; this is effectiveness, not latency).
# EERCF: native rerank top-k = BUDGET.
BUDGET=${BUDGET:-300}
SEED=${SEED:-42}

# ---- Methods on the figure ----
METHODS=${METHODS:-"grdr tiger avg t2vindexer eercf hnsw ivf"}

# ---- Output trees ----
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/output/evaluation_results/figures_panda_scaling}
CAND_ROOT=${CAND_ROOT:-${OUT_ROOT}/candidates}          # per-(method,d) stage-1 candidate JSON
RERANK_ROOT=${RERANK_ROOT:-${OUT_ROOT}/rerank}          # per-(method,d) rerank R@K outputs
SUMMARY_DIR=${SUMMARY_DIR:-${OUT_ROOT}/summaries}       # figure_data.csv
SENTINEL_DIR=${SENTINEL_DIR:-${OUT_ROOT}/manifests}     # phase sentinels
# Runtime state (id manifests + leg sID JSONs + logs) lives under var/research (not tracked).
STATE_ROOT=${STATE_ROOT:-${REPO_ROOT}/var/research/2026-06-01-panda-pool-scaling}
MANIFEST_DIR=${MANIFEST_DIR:-${STATE_ROOT}/manifests}
SID_CACHE_DIR=${SID_CACHE_DIR:-${STATE_ROOT}/sid_cache}
LOG_DIR=${LOG_DIR:-${STATE_ROOT}/logs}

# ---- Panda inputs ----
PANDA_TRAIN_JSON=${PANDA_TRAIN_JSON:-${REPO_ROOT}/data/panda/video_retreival_caption/panda_ret_train.json}
PANDA_TEST_JSON=${PANDA_TEST_JSON:-${REPO_ROOT}/data/panda/video_retreival_caption/panda_ret_test.json}

# ---- GRDR (the method): native Panda c4096/l3, NLT=4 champion ----
GRDR_CKPT=${GRDR_CKPT:-${REPO_ROOT}/output/checkpoints/GRDR/panda/latency_recall_best/model-3-fit/best_model.pt}
GRDR_CODE_NUM=${GRDR_CODE_NUM:-4096}
GRDR_MAX_LENGTH=${GRDR_MAX_LENGTH:-3}
GRDR_NUM_LATENT_TOKENS=${GRDR_NUM_LATENT_TOKENS:-4}
GRDR_ACCESS_GAMMA=${GRDR_ACCESS_GAMMA:-0.50}

# ---- Baseline checkpoints (native Panda c4096/l3) ----
BASE_CKPT_ROOT=${BASE_CKPT_ROOT:-${REPO_ROOT}/output/checkpoints/Baseline}
TIGER_CKPT=${TIGER_CKPT:-${BASE_CKPT_ROOT}/tiger/panda}
AVG_CKPT=${AVG_CKPT:-${BASE_CKPT_ROOT}/avg/panda}
T2V_CKPT=${T2V_CKPT:-${BASE_CKPT_ROOT}/t2vindexer/panda/best_model.pt}
EERCF_CKPT=${EERCF_CKPT:-${BASE_CKPT_ROOT}/eercf/panda/pytorch_model.bin.0}

# ---- Shared X-Pool Panda reranker (stage 2 for every method except EERCF) ----
XPOOL_CKPT=${XPOOL_CKPT:-${REPO_ROOT}/reranker/xpool/ckpt/panda_2150k_s42_model_best.pth}
XPOOL_CACHE=${XPOOL_CACHE:-${REPO_ROOT}/reranker/xpool/video_features_cache/Xpool-Panda}

# ---- Sibling project codebases (TIGER/AVG, T2VIndexer, EERCF) ----
MM_TVR_DIR=${MM_TVR_DIR:-/home/uqzzha35/Project/SemanticID/MM-SemanticTVR}
T2V_DIR=${T2V_DIR:-/home/uqzzha35/Project/SemanticID/T2VIndexer-generativeSearch}
EERCF_DIR=${EERCF_DIR:-/home/uqzzha35/Project/SemanticID/EERCF}

# ---- Conda envs ----
CONDA_SH=${CONDA_SH:-/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh}
SEMANTICTVR_ENV=${SEMANTICTVR_ENV:-semantictvr}   # GRDR stage-1 export + manifest build + aggregate
CONDA_ENV_XPOOL=${CONDA_ENV_XPOOL:-xpool}          # X-Pool rerank
EERCF_CONDA_ENV=${EERCF_CONDA_ENV:-semanticID}     # EERCF native eval

# ---- Run controls ----
DEVICE=${DEVICE:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
PYTHON=${PYTHON_BIN:-python}

export REPO_ROOT FUNC_DIR N_TEST DISTRACTORS BUDGET SEED METHODS \
       OUT_ROOT CAND_ROOT RERANK_ROOT SUMMARY_DIR SENTINEL_DIR \
       STATE_ROOT MANIFEST_DIR SID_CACHE_DIR LOG_DIR \
       PANDA_TRAIN_JSON PANDA_TEST_JSON \
       GRDR_CKPT GRDR_CODE_NUM GRDR_MAX_LENGTH GRDR_NUM_LATENT_TOKENS GRDR_ACCESS_GAMMA \
       BASE_CKPT_ROOT TIGER_CKPT AVG_CKPT T2V_CKPT EERCF_CKPT \
       XPOOL_CKPT XPOOL_CACHE MM_TVR_DIR T2V_DIR EERCF_DIR \
       CONDA_SH SEMANTICTVR_ENV CONDA_ENV_XPOOL EERCF_CONDA_ENV \
       DEVICE DRY_RUN SKIP_EXISTING PYTHON
