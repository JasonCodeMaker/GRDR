# Latency–Recall Evaluation

Two-stage (recall → rerank) latency + effectiveness evaluation for GRDR and the
baselines. The pipeline ends at the aggregated CSV **and** its rendered figures:

```
output/evaluation_results/figures/summaries/figure_data.csv      <- the data
output/evaluation_results/figures/figures/<ds>_panel_AB.png      <- the figures
```

~120 rows = 7 methods × 4 datasets × Setting 2 × per-method operating points. The
`render` phase draws one Panel A/B PNG per dataset from the CSV (the `figure_data.json`
sibling is also written by `aggregate`).

## Eval modes & output trees

Two eval modes select per-dataset checkpoints **and** which output tree is written:

| `EVAL_MODE`        | checkpoints                   | output tree                                   |
| ------------------ | ----------------------------- | --------------------------------------------- |
| `indist` (default) | per-dataset C=128/L=3         | `output/evaluation_results/figures/`          |
| `zeroshot`         | Panda 2.15M, applied zero-shot| `output/evaluation_results/figures_panda/`    |

Dataset-level inputs (the 200-query latency-subset manifests and EERCF query TSVs) are
shared across modes and live in `figures_panda/` (`EVAL_INPUTS_ROOT`); they describe the
test slice, not the model. Candidate sets split by mode into `candidates_indist/` vs
`candidates/`.

Every script reads its paths from [`_env.sh`](_env.sh) (the single source of truth).
You normally only set a few env vars on the command line; the defaults handle the rest.

---

## Quick start

```bash
cd <repo-root>                       # paths self-locate from the repo root

# 1. Plumbing smoke (grdr_ref + hnsw, MSRVTT, op=100) — ~minutes, needs 1 GPU
bash scripts/latency_recall_figure/make_figure.sh smoke

# 2. Full CSV: all 7 methods × 4 datasets, every stage, then aggregate
bash scripts/latency_recall_figure/make_figure.sh all

# 3. Re-run ONLY GRDR and refresh the CSV (baselines stay cached)
bash scripts/latency_recall_figure/refresh_grdr.sh
```

`make_figure.sh all` is safe to re-run: cached cells are skipped (`SKIP_EXISTING=1`
by default), so it only fills in what is missing and re-aggregates.

---

## The four stages

The pipeline is a 2×2 of {stage-1 recall, stage-2 rerank} × {effectiveness, efficiency}.
Each stage is its own phase and writes its own subtree under `output/evaluation_results/figures/`.

|                       | Effectiveness                          | Efficiency                                |
| --------------------- | -------------------------------------- | ----------------------------------------- |
| **Stage-1 (retrieval)** | `recall-stage` → CanHit@20/50/100      | `recall-latency` → `stage1_latency_ms`    |
| **Stage-2 (rerank)**    | `rerank-stage` → R@1/5/10              | `rerank-latency` → `stage2_latency_ms`    |

`aggregate` walks the four subtrees and writes `summaries/figure_data.csv`; `lint`
validates that CSV against the column contract (auto-run at the end of `aggregate`).
`render` draws the per-dataset Panel A/B PNGs from that CSV into `figures/`.
`all` = the four stages, then `aggregate`, then `render`.

```
make_figure.sh <phase> ...
  phases: smoke | recall-stage | rerank-stage | recall-latency | rerank-latency
          | aggregate | render | lint | all
```

---

## Common tasks

**Evaluate GRDR only** (the high-frequency loop — use this after editing GRDR knobs
in `_env.sh` or re-pointing the champion checkpoint):

```bash
bash scripts/latency_recall_figure/refresh_grdr.sh
```

This re-runs `grdr_ref` through all four stages (forcing a fresh run), then aggregates
over the **full** method list so the cached baseline rows stay in the CSV.

**Evaluate one baseline** end-to-end (e.g. `tiger`), keeping GRDR + other baselines cached:

```bash
BASELINES=tiger bash scripts/latency_recall_figure/make_figure.sh \
  recall-stage rerank-stage recall-latency rerank-latency aggregate
```

**One cell only** (one method, one dataset, one operating point):

```bash
BASELINES=grdr_ref DATASETS=MSRVTT OPERATING_POINTS=100 \
  bash scripts/latency_recall_figure/make_figure.sh recall-stage
```

**Force a re-run** (ignore the cache for this invocation):

```bash
SKIP_EXISTING=0 BASELINES=avg bash scripts/latency_recall_figure/make_figure.sh recall-stage
```

**Dry run** (print the cells that would run, touch nothing):

```bash
DRY_RUN=1 bash scripts/latency_recall_figure/make_figure.sh all
```

**Pick a GPU** / **re-lint the existing CSV**:

```bash
DEVICE=1 BASELINES=grdr_ref bash scripts/latency_recall_figure/make_figure.sh recall-stage
bash scripts/latency_recall_figure/make_figure.sh lint
```

---

## Environment variables

Set these inline before the command. Full list + defaults live in [`_env.sh`](_env.sh).

| Var                | Default                                    | Meaning                                              |
| ------------------ | ------------------------------------------ | ---------------------------------------------------- |
| `BASELINES`        | `grdr_ref tiger avg t2vindexer eercf hnsw ivf` | Which methods to run (space-separated)           |
| `DATASETS`         | `MSRVTT ACTNET DIDEMO LSMDC`               | Which datasets                                       |
| `SETTINGS`         | `2`                                        | Eval setting (this study is Setting 2)               |
| `OPERATING_POINTS` | *(empty → per-method default below)*       | Override the operating-point sweep                   |
| `DEVICE`           | `0`                                        | GPU id                                               |
| `SEED`             | `42`                                       |                                                      |
| `SKIP_EXISTING`    | `1`                                        | `1` reuse cached cell outputs; `0` force re-run      |
| `DRY_RUN`          | `0`                                        | `1` print cells without running                      |
| `WALL_CAP_S`       | `300`                                      | Per-query latency wall cap                           |
| `EVAL_MODE`        | `indist`                                   | `indist` → `figures/` tree + per-dataset ckpts; `zeroshot` → `figures_panda/` tree + Panda ckpts |
| `RUNTIME_ROOT`     | `output/evaluation_results/figures` (indist)    | Where all outputs land (point elsewhere for a scratch run) |

---

## Methods and operating points

When `OPERATING_POINTS` is empty, each method sweeps its own knob:

| Method        | Operating points | Knob                          | Notes                                            |
| ------------- | ---------------- | ----------------------------- | ------------------------------------------------ |
| `grdr_ref`    | 20 50 100 200    | candidate budget              | **the method**; volatile, lives in `GRDR/`       |
| `tiger`,`avg` | 20 50 100 200    | candidate budget              | generative baselines                             |
| `t2vindexer`  | 20 50 100 200    | candidate budget              | indist: real candidates → **X-Pool reranked** (like tiger/avg); zeroshot: OOM placeholder |
| `eercf`       | 1 10 25 50       | rerank top-k                  | native sim-matrix; **no X-Pool rerank stage**    |
| `hnsw`        | 20 40 100 200    | budget `K` + paired `ef_search` | ANN; `ef_search = K` via `ANN_HNSW_EF_BY_K` (20/40/100/200)          |
| `ivf`         | 20 40 100 200    | budget `K` + paired `nprobe`    | ANN; `nprobe` paired to `K` via `ANN_IVF_NPROBE_BY_K` (4/8/16/32)    |

Method quirks worth knowing:
- **ANN operating point pairs the candidate budget `K` with the search-effort knob.**
  Each `K` maps to an `ef_search` (HNSW) / `nprobe` (IVF) value via `ANN_HNSW_EF_BY_K` /
  `ANN_IVF_NPROBE_BY_K` in `_env.sh`. HNSW uses `ef_search = K` (20/40/100/200); IVF uses
  `nprobe` 4/8/16/32 at K=20/40/100/200. Search effort therefore grows with the budget, so
  Stage-1 latency rises across the sweep instead of collapsing to a vertical line, while
  `K` also sizes the Stage-2 candidate pool. Setting `ef_search = K` makes the faiss export
  path agree with the per-query latency path, which floors `ef_search` at `K`
  (`ef_search = max(k, ef)` in `eval_ann.py`). IVF `nprobe` is independent of `K` and
  applies as set.
- **ANN stage-1 latency is cold-reload-dominated.** The per-query path resets the
  video-feature store each query, so the reported `stage1_latency_ms` is mostly the
  disk re-read + re-pool of each scored video (`video_load` ≫ `similarity`), not the
  index traversal. A deployed ANN would keep pooled vectors in RAM; treat these
  numbers as feature-I/O + search, not pure ANN compute.
- **ANN latency** (`hnsw`/`ivf`) is measured inside `recall-stage` (step A.2), so
  `recall-latency` deliberately skips them. ANN `rerank-latency` is skipped unless
  `ALLOW_ANN_RERANK_LATENCY=1`.
- **`eercf`** has no X-Pool `rerank-stage` (it ranks with its own sim-matrix); empty
  `rerank_source_path` lint warnings for EERCF rows are expected.
- **`t2vindexer`** is a Stage-1 generative retriever: in **indist** mode it produces real
  candidate lists that go through X-Pool `rerank-stage` exactly like `tiger`/`avg`, so its
  R@1/5/10 are reranked numbers. (Only in **zeroshot**/Panda mode does it OOM at c=4096/l=3
  and fall back to a placeholder row.)

---

## Where things live

```
scripts/latency_recall_figure/
  _env.sh                 single source of truth for all paths/knobs
  _cells.sh               the per-cell logic + loop runner (shared by the 4 stages)
  make_figure.sh          phase dispatcher (entry point)
  refresh_grdr.sh         GRDR-only refresh (high-frequency loop)
  recall-stage.sh  rerank-stage.sh  recall-latency.sh  rerank-latency.sh
  aggregate_figure_csv.py walk the 4 subtrees -> figure_data.csv/json
  render_figures.py       figure_data.csv -> per-dataset Panel A/B PNGs
  lint_figure_data.py     CSV column/value contract check
  GRDR/                   grdr_ref cell scripts (volatile — edited often)
  baselines/              tiger/avg/ann/eercf cell scripts (stable — cached)
  lib/                    shared latency helpers

output/evaluation_results/figures/      (RUNTIME_ROOT, EVAL_MODE=indist)
  recall-stage/ rerank-stage/ recall-latency/ rerank-latency/   per-stage artifacts
  summaries/figure_data.csv         <- the data deliverable
  figures/<ds>_panel_AB.png         <- the rendered figures
  manifests/                        stage sentinels (*.done)
output/evaluation_results/figures_panda/   (RUNTIME_ROOT, EVAL_MODE=zeroshot;
                                            also holds the shared query_sets/ + latency manifests)

candidates_indist/   (EVAL_MODE=indist) | candidates/   (EVAL_MODE=zeroshot)
  GRDR/<ds>/...                     grdr_ref candidate sets (volatile)
  baselines/<method>/<ds>/...       baseline candidate sets (stable)
```

### Checkpoints

| What                       | Path                                                             |
| -------------------------- | ---------------------------------------------------------------- |
| GRDR champion (the method) | `output/checkpoints/GRDR/panda/latency_recall_best/model-3-fit/best_model.pt` |
| Baseline generative ckpts  | `output/checkpoints/Baseline/{tiger,avg,eercf,t2vindexer}/panda/`    |
| Shared X-Pool reranker     | `reranker/xpool/ckpt/panda_2150k_s42_model_best.pth`             |

`latency_recall_best` is a **stable alias** — when a new GRDR champion lands, re-point
that one path (and `GRDR_REF_CKPT` in `_env.sh` never changes).

### Conda environments

The scripts activate the right env automatically: `semantictvr` (stage-1 export +
aggregate), `xpool` (rerank + rerank-latency), `semanticID` (EERCF). Override via
`SEMANTICTVR_ENV` / `CONDA_ENV_XPOOL` / `EERCF_CONDA_ENV` in `_env.sh`.
