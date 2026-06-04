#!/usr/bin/env bash
# settle_output.sh — redesign output/ into checkpoints/ + evaluation_results/.
#
# REVIEW-FIRST. Dry-run by default: prints every action, touches nothing.
#   DRY_RUN=1 (default)  -> print planned actions only
#   DRY_RUN=0            -> actually perform them
#   COMPAT=symlink (default) -> leave backward-compat symlinks at every old path
#                               so ALL existing code keeps resolving with ZERO edits
#   COMPAT=none          -> no compat symlinks; you MUST then apply the path edits in
#                           scripts/dev/settle_output.PLAN to get a fully clean tree
#
# Target:
#   checkpoints/
#     GRDR/{msrvtt,actnet,didemo,lsmdc,panda}/...    (5 datasets; collections kept as sublevels)
#     Baseline/{msrvtt,actnet,didemo,lsmdc,panda}/...
#     _legacy/semantic-tvr-progressive/              (236 G, moved intact, NOT pruned)
#   evaluation_results/
#     rerank/  ann_baseline/  latency/  figures/     (by evaluation type)
#
# All moves are intra-/data2 renames (instant, no copy, no extra space). Nothing deleted.
set -uo pipefail

OUT="/data2/uqzzha35/semantic_id/output"
DRY_RUN="${DRY_RUN:-1}"
COMPAT="${COMPAT:-symlink}"

run() { if [ "$DRY_RUN" = "1" ]; then echo "DRY  $*"; else echo "RUN  $*"; "$@"; fi; }
mv_if() { [ -e "$OUT/$1" ] && run mv "$OUT/$1" "$OUT/$2"; }       # move src->dst if src exists
# compat symlink at old path: prints in dry-run; in a real run only creates if old path is now gone
link_if() {
  [ "$COMPAT" = "symlink" ] || return 0
  if [ "$DRY_RUN" = "1" ]; then echo "DRY  ln -s $2 $OUT/$1  (after move)"; return 0; fi
  [ -e "$OUT/$1" ] || ln -s "$2" "$OUT/$1"
}

echo "== settle_output.sh (DRY_RUN=$DRY_RUN COMPAT=$COMPAT) — target: $OUT =="
run mkdir -p "$OUT/checkpoints/GRDR" "$OUT/checkpoints/Baseline" "$OUT/checkpoints/_legacy" \
             "$OUT/evaluation_results/latency"

# ---------- checkpoints/GRDR (5 datasets) ----------
for ds in msrvtt actnet didemo lsmdc; do mv_if "GRDR/$ds" "checkpoints/GRDR/$ds"; done
mv_if "GRDR/panda"          "checkpoints/GRDR/panda"            # champions + latency_recall_best
mv_if "GRDR/multiview_waas" "checkpoints/GRDR/panda/multiview_waas"
mv_if "GRDR_P6/panda"       "checkpoints/GRDR/panda/p6"
mv_if "GRDR/bucket_candidate_k20/msrvtt" "checkpoints/GRDR/msrvtt/bucket_candidate_k20"  # msrvtt-only collection, incl current-best
[ -d "$OUT/GRDR/bucket_candidate_k20" ] && run rmdir "$OUT/GRDR/bucket_candidate_k20"
[ -d "$OUT/GRDR" ] && run rmdir "$OUT/GRDR"
[ -d "$OUT/GRDR_P6" ] && run rmdir "$OUT/GRDR_P6"

# ---------- checkpoints/Baseline (model -> dataset; 5 datasets) ----------
# Method names are unified to the code-canonical set: avg(=text_guided), tiger(=standard),
# videorqvae, eercf, t2vindexer. Small-dataset runs were laid out dataset/method; panda runs
# under baseline/panda-pretrain/<method>. Final layout: Baseline/<model>/<dataset>/<run>.
for m in avg tiger videorqvae eercf t2vindexer; do run mkdir -p "$OUT/checkpoints/Baseline/$m"; done
for ds in msrvtt actnet didemo lsmdc; do
  mv_if "baseline/$ds/text_guided" "checkpoints/Baseline/avg/$ds"        # text_guided == avg
  mv_if "baseline/$ds/standard"    "checkpoints/Baseline/tiger/$ds"      # standard == tiger
  mv_if "baseline/$ds/videorqvae"  "checkpoints/Baseline/videorqvae/$ds"
  [ -d "$OUT/baseline/$ds" ] && run rmdir "$OUT/baseline/$ds"
done
mv_if "baseline/panda-pretrain/avg"        "checkpoints/Baseline/avg/panda"
mv_if "baseline/panda-pretrain/tiger"      "checkpoints/Baseline/tiger/panda"
mv_if "baseline/panda-pretrain/eercf"      "checkpoints/Baseline/eercf/panda"
mv_if "baseline/panda-pretrain/t2vindexer" "checkpoints/Baseline/t2vindexer/panda"
[ -d "$OUT/baseline/panda-pretrain" ] && run rmdir "$OUT/baseline/panda-pretrain"
[ -d "$OUT/baseline" ] && run rmdir "$OUT/baseline"

# ---------- checkpoints/_legacy ----------
mv_if "semantic-tvr-progressive" "checkpoints/_legacy/semantic-tvr-progressive"
mv_if "bunya_to_ws.txt"          "checkpoints/_legacy/bunya_to_ws.txt"

# ---------- evaluation_results (by type) ----------
mv_if "reranker"             "evaluation_results/rerank"
mv_if "ann_baseline"         "evaluation_results/ann_baseline"
mv_if "ann_latency"          "evaluation_results/latency/ann_latency"
mv_if "ann_stage2_compare"   "evaluation_results/latency/ann_stage2_compare"
mv_if "latency_recall_figure" "evaluation_results/figures"
mv_if "xpool_t2v_eval"       "evaluation_results/rerank/xpool_t2v_eval"

# ---------- backward-compat symlinks (COMPAT=symlink) ----------
# Top-level old name -> new location. Keeps every current code path resolving.
link_if "GRDR"                   "checkpoints/GRDR"
link_if "baseline"               "checkpoints/Baseline"
link_if "semantic-tvr-progressive" "checkpoints/_legacy/semantic-tvr-progressive"
# (no compat symlink for GRDR_P6: 0 code refs, and its panda/ level collapses into p6/)
link_if "reranker"               "evaluation_results/rerank"
link_if "ann_baseline"           "evaluation_results/ann_baseline"
link_if "ann_latency"            "evaluation_results/latency/ann_latency"
link_if "ann_stage2_compare"     "evaluation_results/latency/ann_stage2_compare"
link_if "latency_recall_figure"  "evaluation_results/figures"
link_if "xpool_t2v_eval"         "evaluation_results/rerank/xpool_t2v_eval"
# Two INNER compat symlinks absorb the reshuffles so nested refs still resolve:
#   - GRDR save_path / current-best swap msrvtt<->bucket order (run.py:117,158; train.sh:19)
#   - Baseline panda-pretrain renamed to panda (_env.sh:56; stage2_latency.sh:35; import_eercf_matrix.py:44; avg_tiger_stage1_export.sh:58)
if [ "$COMPAT" = "symlink" ]; then
  run mkdir -p "$OUT/checkpoints/GRDR/bucket_candidate_k20"
  [ ! -e "$OUT/checkpoints/GRDR/bucket_candidate_k20/msrvtt" ] && \
    run ln -s "../msrvtt/bucket_candidate_k20" "$OUT/checkpoints/GRDR/bucket_candidate_k20/msrvtt"
  [ ! -e "$OUT/checkpoints/Baseline/panda-pretrain" ] && \
    run ln -s "panda" "$OUT/checkpoints/Baseline/panda-pretrain"
fi

echo "== done. Verify: ls -la $OUT ; tree -L 3 $OUT/checkpoints $OUT/evaluation_results =="
