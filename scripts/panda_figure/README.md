# Panda pool-scaling figure

Produces the **Panda** effectiveness-vs-scale figure for P7 of the
`2026-06-01-effectiveness-efficiency-figure` package.

- **y-axis** — stage-2 **rerank Recall@{1,5,10}** (X-Pool for every method except EERCF, which reranks natively).
- **x-axis** — **search-pool size**: test-only → test + N train distractors.
- **6 points** — distractor counts `d ∈ {0, 400k, 800k, 1.2M, 1.6M, 2.0M}` → pool sizes `N ∈ {5694, 405694, 805694, 1205694, 1605694, 2005694}` (Panda test = 5,694; train uniques = 2,150,540).
- **fixed budget = 300** held constant across all pool sizes: GRDR `candidate_handoff_cap=300` + beam 300; TIGER/AVG/T2VIndexer beam 300; ANN top-K 300; EERCF rerank top-300.
- **7 methods** — `grdr tiger avg t2vindexer eercf hnsw ivf`. Single seed (42). Panda in-distribution (**c4096/l3**, NLT=4 for GRDR) — *not* the c128l3 used for the 4 small datasets.

The curve shows graceful degradation: as the pool grows, stage-1 crowds the GT video out of the ≤300-candidate set, and rerank recall (upper-bounded by stage-1 visibility) falls.

## Pipeline

```
build_manifests.py   ── one seed-42 shuffle of train uniques → NESTED prefix manifests
                        (400k ⊂ 800k ⊂ … ⊂ 2.0M); every method reads the same manifest
run_stage1.sh        ── per (method, d): stage-1 retrieval over pool N → candidate JSON
rerank.sh            ── per (method, d): X-Pool / native rerank of ≤300 cands → R@K
aggregate.py         ── walk outputs → summaries/figure_data.csv (1 row per method×d)
panda_pool_scaling.ipynb ── figure_data.csv → 3-panel PNG/PDF (R@1/R@5/R@10 vs pool size)
```

Driver: `make_panda_figure.sh {build|stage1|rerank|aggregate|all}`.

### Output layout
```
output/evaluation_results/figures_panda_scaling/
  candidates/<method>/<method>_d<d>_candidates.json   stage-1 candidates
  rerank/<method>/d<d>/rerank.json                    {"metrics": {"R@1","R@5","R@10",…}}
  rerank/<method>/d<d>/status.txt                     ok | pending | oom | …
  summaries/figure_data.csv                           the data deliverable (15 cols)
  figures/panda_pool_scaling.{png,pdf}                rendered by the notebook
var/research/2026-06-01-panda-pool-scaling/
  manifests/panda_pool_d<d>.json                      shared distractor manifests (X-Pool schema)
  sid_cache/   logs/                                  GRDR full-corpus sID cache + per-cell logs
```

## Quick start
```bash
cd <repo-root>
bash scripts/panda_figure/make_panda_figure.sh build                 # manifests (done; cheap, CPU)
METHODS=grdr DISTRACTORS="0 400000" bash scripts/panda_figure/make_panda_figure.sh stage1 rerank aggregate  # GRDR smoke
bash scripts/panda_figure/make_panda_figure.sh all                   # full sweep (after wiring below)
# then run scripts/panda_figure/panda_pool_scaling.ipynb to render
```
Knobs (inline env): `METHODS`, `DISTRACTORS`, `DEVICE`, `SKIP_EXISTING` (1), `DRY_RUN`. All paths/ckpts/budget live in `_env.sh`.

## Status — what runs today vs. what needs wiring

| Component | State |
| --- | --- |
| `build_manifests.py` (nested manifests) | **Done + verified** (n_test=5694, nesting holds) |
| `aggregate.py` → `figure_data.csv` | **Done + verified** (schema + pool-size mapping) |
| `panda_pool_scaling.ipynb` | **Done + verified** (renders 3 panels from the CSV) |
| **GRDR** stage-1 + X-Pool rerank | **Wired** (runnable via `run.py --candidate_export --distractor_n`) — see fidelity note |
| TIGER / AVG / T2VIndexer / EERCF / HNSW / IVF stage-1 | **Pending** — each needs the Panda+pool edit below; `run_stage1.sh` marks them `status=pending` until then |

### Per-method wiring (the remaining eval-code edits)

These move eval numbers, so each lands with its own verification (smoke at d=0 + d=400k, assert R@1≤R@5≤R@10≤stage1_gt_visible and R@K monotone non-increasing in pool size).

- **GRDR fidelity** — `--distractor_n` uses `Random(42).sample` (statistically comparable but *not the same videos* as the shared manifest). For exact-pool parity, add `--distractor_manifest` to `run.py`/`trainer/evaluator.py:1050` (read `video_ids`, filter `raw_train_codes` by base-id; print the matched-count diagnostic to catch id-space mismatch). Also cache the full-corpus GRDR sIDs once into `sid_cache/` (no `.code` companion exists) so the 6 points are cheap decodes, not 6× `gen_sid`.
- **TIGER / AVG** — build the P5 **leg sID-JSON** (precomputed `MM-SemanticTVR/data/panda/none/{standard,text_guided}_c4096_l3` sIDs filtered to test+first-d) and point `avg_train_retriever_t5.py --eval --dataset panda --setting 1` at it; `--num_candidates 300`. Cheap (sIDs already on disk).
- **HNSW / IVF** — `baselines/ann_dense_retrieval/eval_ann.py` needs a Panda query/train loader + a `--distractor_manifest` pool (test + first-d). **Precondition:** 2.15M Panda train features must exist in the `Xpool-Panda` cache; load once, slice per d. top-300, ef_search/nprobe generous (effectiveness, not latency).
- **EERCF** — native dense over the pool. Use the **pool-monotonic shortcut**: score the full pool once, restrict the ranking to test+first-d per point (X-Pool scores are pairwise/pool-independent). Heavy long-pole (17h-class at full scale); may cap its max pool.
- **T2VIndexer** — per-leg index over the pool; **documented OOM at Panda c4096**. Attempt small pools, leave `status=oom` placeholder where it fails.

## Verification anchors
- d=0 (test-only) ≈ existing Panda **Setting-1** numbers; d≈2.15M ≈ existing Panda **Setting-2** numbers (cross-check GRDR/X-Pool against the current champion eval).
- Lint invariants (per row): `R@1≤R@5≤R@10≤stage1_gt_visible`, `avg_candidates≤300`; `pool_size` strictly increasing; per-method `R@K` monotone non-increasing in pool size (small noise tolerated).

## Notes
- This pipeline is intentionally separate from `scripts/latency_recall_figure/` (the 4-dataset effectiveness-vs-*latency* figure). It shares no code paths; the only overlap is the X-Pool Panda reranker ckpt + `Xpool-Panda` cache.
- The shared manifest is the fairness contract: every method retrieves from byte-identical pools at each x-point.
