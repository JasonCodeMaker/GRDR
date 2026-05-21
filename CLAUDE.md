# CLAUDE.md — GRDR

## Project
GRDR (Generative Recall, Dense Reranking) — a two-stage recall-rerank system for scalable text-to-video retrieval.
- Stage 1: Multi-View Video Tokenizer + T5-small generative retriever
- Stage 2: X-Pool dense reranker
- Targets: MSR-VTT, ActivityNet, DiDeMo, LSMDC
- Conda env: check `wandb/` requirements or `run.py` header

## Motivation and Goal

- The central systems bottleneck is stage-1 candidate generation, not stage-2 reranking.
- Dense dual-encoder retrieval is effective, but as a stage-1 index its storage footprint and query-time work grow with corpus size.
- Prior generative retrieval is efficient but weak on two fronts that matter for stage 1: access-path coverage and query-to-key reachability.
- GRDR’s goal is to serve as a scalable stage-1 semantic-ID index: assign multiple short IDs per video, learn them jointly with the query decoder under shared retrieval-aware supervision, and hand a compact high-recall candidate pool to a strong dense reranker.
- Preserve the recall-rerank split in future work: stage 1 optimizes storage, latency, and candidate coverage; stage 2 optimizes fine-grained ranking on the bounded pool.
- Recent MSR-VTT Setting 2 diagnostics showed that final X-Pool can improve simply from a wider candidate handoff. Treat those gains as candidate-budget effects unless compact Stage-1 reachability improves under the same measured budget.
- The current refinement target is access-path reachability under a fixed compact budget: reduce route misses and late overflows so the GT video enters the early deduplicated candidate set, not merely anywhere in a larger full candidate file.

## Global Optimization Objective

- Primary objective for future GRDR experiments: improve the Stage-1 candidate pool in a way that improves downstream X-Pool rerank, preferably by increasing GT visibility under a smaller or matched measured candidate budget.
- The optimal Stage-1 direction is a Pareto improvement: fewer candidates passed to X-Pool while covering more GT videos. A Stage-1 metric gain is only useful if it raises final rerank utility or preserves rerank while reducing candidate-pool size.
- `CanHit@100` / `CompactHit@100` means the GT video appears in the first 100 deduplicated expanded videos after semantic-ID expansion.
- `FullSetHit@All` means the GT video appears anywhere in the full candidate file passed to X-Pool; `route_miss = 1 - FullSetHit@All`.
- For any exact candidate set `C` handed to X-Pool, `XPool_R@10(C) <= GTVisible(C)`. If `C` is top-100, the visibility bound is `CanHit@100`; if `C` is the full beam export, it is `FullSetHit@All`.
- Do not equate route miss with `CanHit@100`. A run can reduce route miss while still failing compact Stage 1 if the GT appears only after rank 100.
- Training evaluation and candidate export should report `CanHit@20/50/100`, `FullSetHit@All`, `OverflowHit`, GT expanded-rank buckets, candidate-count stats, and a discounted rank metric such as `MeanLogDiscount`. Save-best can still use `CanHit@100`, but method selection must check downstream rerank utility under the exact candidate set handed to X-Pool.
- Do not treat candidate JSON `Recall@K` as GT-in-candidates. If sID recall is logged, label it explicitly as a semantic-ID diagnostic.
- X-Pool remains the downstream final-rank evaluator; optimize Stage 1 to raise compact reachability while keeping candidate pools bounded. X-Pool gains driven by larger average candidate count or `OverflowHit` are diagnostic only, not compact Stage-1 improvements.
- Preserve the Stage-1 storage contribution: the main GRDR index must not require a per-video dense or continuous embedding side table. Default compact methods should prefer semantic-ID-native signals such as beam score, route multiplicity, bucket size, prefix stats, margins, or codebook priors.

## MSR-VTT Setting 2 Rules

- Current compact budget gate: `avg_candidates_per_query <= 310`. Any Setting 2 row above this gate is excluded from compact-champion selection unless the user explicitly labels it as a diagnostic or large-pool ablation.
- The primary failure bucket to attack is route/access-path miss, but only under the compact budget gate. The desired movement is from absent or late GT to early top-100 GT, not from absent to late full-pool rescue.
- A valid Setting 2 champion must beat the current compact reference under the same budget gate, improve downstream X-Pool rerank or reduce candidate-pool size without rerank regression, and pass multi-seed validation. Do not promote a single-seed tie or a `+0.1` `CanHit@100` change.
- Required Setting 2 readout: `avg/p95/max candidates`, `CanHit@20/50/100`, `FullSetHit@All`, `OverflowHit`, `route_miss`, `MeanLogDiscount`, top-100-truncated X-Pool, full-candidate X-Pool, and whether any X-Pool gain is caused by larger pool size.

## Research Workflow

`WORKFLOW.md` at the repo root is the operating protocol for any `@WORKFLOW.md` invocation and overrides general harness defaults (e.g., "do not spawn agents unless asked", end-of-turn-summary style). Strictly follow it: when it says dispatch a subagent, dispatch; when it says emit a 10-minute status line, emit it; when it says schedule re-entry, schedule.

Before proposing or extending any new research idea, run a self-review first.
- Check the problem anchor and the exact bottleneck being solved
- Critically review the idea for weak assumptions, overclaimed conclusions, and likely reviewer attacks
- Run a novelty check before turning the idea into an experiment or paper claim
- For Review Loop workflows, do not save the loop transcript or full round-by-round review as a standalone doc. Distill only the actionable conclusions into `plan.html`, and record the important review judgments, concerns, and resolutions in `tracker.html`.
- For future experiments, do not include an A0/checkpoint reproduction run such as `V0: A0 reproduction` by default. Assume the current checkpoint and recorded performance in `AGENTS.md` are correct unless the user explicitly asks to revalidate the anchor or there is direct evidence of code/path drift.
- For inference/export changes, minimize per-query latency while preserving metric correctness. Reuse online generation outputs and offline caches where possible; do not add per-candidate T5 teacher-forcing or direct video-ANN search to the default inference path unless explicitly approved as an offline diagnostic.
- In research plans, call the ordered run/ablation section `Experiments List`, not `Decision Tree`.
- Do not pre-estimate run duration. `plan.html` rows, launcher manifests, allocation rows, and live-check rows record `est_time=unknown` until the run has executed at least 30 minutes of stable throughput; after that, derive ETA from observed throughput and update on every 10-minute report.

## Refinement Guardrails

For `/research-refine`, `/experiment-plan`, and `/research-refine-pipeline`, treat the current GRDR paper story as a compatibility constraint unless the user explicitly asks to replace it.
- Every refinement must explain why the design sharpens the current paper narrative, not just why it is novel in isolation.
- The paper's non-negotiable contribution spine is:
  1. Multi-view video encoder. In code this is the stage-1 `VideoRQVAE_V2` plus `VideoLatentEncoder_V2` path (`models/video_rqvae/videorqvae.py`, `models/video_rqvae/encoder.py`), which decomposes one pooled video feature into `num_latent_tokens` partial views. Caption-side routing depends on `token_idx` in `data/video_dataset.py`, and evaluation can use all latent views through MaxSim / `return_all=True` in `trainer/evaluator.py`.
  2. Progressive end-to-end co-training. In code this is the loop-wise training schedule in `run.py` and `trainer/trainer.py`: code length grows one layer at a time, checkpoints and semantic IDs are warm-started across loops, older codebooks are selectively frozen, and code drift is monitored to preserve prior layers.
  3. Contrastive stage plus main training stage. The training story must keep the pre-train contrastive-alignment stage and the main stage that combines CE, code prediction, reconstruction, and RQ losses. Do not collapse this into an unstructured single-stage recipe unless the user explicitly wants that redesign.
- Any refinement must remain compatible with:
  - stage-1 generative retrieval via hierarchical semantic IDs and constrained generation in `models/grdr.py` and `trainer/evaluator.py`
  - stage-2 X-Pool reranking via candidate JSON export and the `candidate_file` flow in `reranker/xpool`
  - multi-view token routing via `num_latent_tokens`, `token_idx`, and per-token code assignment
  - progressive code stability across loops and previously learned layers
- By default, push back on refinements that remove the multi-view decomposition, break the stage-1 to stage-2 handoff, discard progressive co-training, or weaken prior-layer code consistency without a strong reviewer-proof reason.
- When writing a proposal or experiment plan, explicitly state:
  - which of the three core contributions is dominant in the paper story
  - which modules stay unchanged
  - which modules are extended
  - how the contrastive stage, main stage, and X-Pool handoff remain valid
- When a refinement direction is explicitly judged failed, remove all worktrees created for that direction after preserving any needed notes in the owning research package. Do not leave stale branch worktrees under `.worktrees/` or elsewhere once the direction is abandoned.

## Research Output Contract

- The only valid in-repo location for new research material is `research/active/<YYYY-MM-DD>-<slug>/`.
- Every research package must contain `README.md`, `plan.html`, `tracker.html`, `results.html`, plus `docs/` and `scripts/`.
- Use `bash scripts/dev/new_research.sh <slug>` to create research packages; do not create ad hoc top-level research folders outside `research/`.
- Runtime state, supervisor JSON, local logs, and temporary CSVs must go under `var/research/<YYYY-MM-DD>-<slug>/`, not in tracked repo roots.
- When a research theme is complete or paused, move the whole package to `research/archive/<YYYY-MM-DD>-<slug>/`.
- Stable shared entrypoints stay in `scripts/`; one-off experiment scripts belong in the owning research package.

## Research Doc Style

Any new HTML doc under `research_html/packages/<pkg-id>/docs/` MUST start from the shared template and follow the style guide. This keeps every doc dashboard-wired (status-strip + package-nav) and visually consistent.

- **Skeleton:** copy `research_html/templates/doc-template.html`. Replace the six template variables (`$package_id`, `$doc_title`, `$eyebrow`, `$lead`, `$last_updated`, `$root_prefix`), delete the demo `<section data-section="primitives">`, then write the doc's real sections.
- **When to reach for each primitive:** see `research_html/templates/doc-style-guide.html`. It is the canonical reference for `pre.diagram`, `pre.code`, `.callout` (+`.warn`/`.ok`), `table.data-table`, `span.pill-mono`, `h2.stage-title`+`step-num`, `p.kv-mini`. Authors should re-read the style guide before creating a new doc.
- **Hard rules:** keep the shell verbatim (`data-status-strip`, `data-package-nav`, footer `<time data-field="last-updated">`, and the three trailing `<script>` tags). Do not invent new block classes; do not add page-local CSS beyond the primitive overrides at the top of the template. Bump the footer date with a short scope phrase on every meaningful edit.
- **Section composition is content-agnostic:** the template prescribes the shell and the primitives, not section count, section order, or section topics. A perf-fix doc can be one card; a full pipeline walk-through can be eight. Use only the primitives that earn their place.
- **The exemplar:** `research_html/packages/2026-05-16-panda-pseudo-queries-multiview/docs/training_pipeline.html` is the reference for what a fully-fleshed-out doc looks like under this style.

## Runtime Ops

- Launch every long-running bash script, dataset download, preprocessing pipeline, and experiment inside `tmux`.
- Prefer named `tmux` sessions/windows and report the attach command so the run can be monitored live.
- Enforce the Fact Propagation Contract on every per-turn live cycle: between the tracker live-check update and the §5 status line, run `python research_html/packages/<pkg-id>/scripts/propagate_facts.py`; for each event listed in the report, update its owning surfaces (`results.html`, `next-action.html`, `research_html/data/research-packages.js`, tracker Resume Block) in the same turn, then advance the cursor with `propagate_facts.py --bump`. A non-empty report at the Stop Gate is a workflow violation.

## Learnings Update Protocol

The dashboard's cross-package learnings index (`research_html/learnings.html`) is a derived view over `research_html/data/research-packages.js`. The data file is the canonical store; `learnings.html` re-renders on page load. This protocol fixes *when* to write to that data file and *how* to keep it trustworthy. It extends — does not replace — the Fact Propagation Contract above.

### Core principles

1. **Upstream surface is the witness, the data file is the index.** A `methodsTried[]` row is written to `research-packages.js` *only after* the corresponding row exists in the package's `results.html` with a stable section anchor, and the `evidencePath` resolves to a real file or anchor. Never invent a row from memory.
2. **Drafts are auto-detected; writes are user-acked at terminal transitions.** In-progress facts (E1, E2 below) update without user ack because the source-of-truth surface already exists. Terminal facts (E3–E6) require T1 user ack.
3. **Atomic per-turn closure.** Any turn that mutates a learnings-relevant field must, in the same turn, touch all of: upstream surface row → `research-packages.js` → tracker Resume Block `lastAction` → run `--lint-status`. A non-empty lint report is a Stop-Gate violation.

### Event trigger table

| Event | Trigger (where it originates) | User ack | Fields written in `research-packages.js` |
|---|---|---|---|
| **E1. Per-experiment verdict finalized** | `results.html` result-gate row gains `pass` / `fail` / `inconclusive` in `<td data-decision>` AND artifact verification recorded | none | Append one `methodsTried[]` row |
| **E2. In-progress live update** | tracker live-check update, plan revision, blocker change | none | `status`, `activeGate`, `primaryMetricVsGate`, `currentBlocker`, `openRuns`, `lastAction`, `lastUpdated` |
| **E3. Terminal status transition** | `next-action.html` chosen-route resolves to a terminal lane move (`archive_or_stop`, adoption) | **T1** | `category` (lane move), `status` (terminal value), `terminationMessage`; freeze `methodsTried[]` |
| **E4. Adoption** | CLAUDE.md "Current Best" edit, code merge into `models/` / `trainer/`, or a new in-progress package starts citing the win | **T1** | `adoptionPath` (specific anchor or path) |
| **E5. Supersession** | A newer success package replaces an older one | **T1** | On the *old* package: `status = SUPERSEDED`, `supersededBy = <new id>` |
| **E6. Reopen marked** | User explicitly states a fail package should be revisitable under a named condition | **T1** | `status = ARCHIVED_REOPENABLE`, `reopenTrigger = "<condition>"` |

### `methodsTried` row contract

Every row is exactly six fields, drawn verbatim from the witnessing `results.html` row:

```
{ method, hypothesis, gate, measured, verdict, evidencePath }
```

- `verdict` ∈ `{pass, fail, inconclusive}`. Diagnostic-only rows are `inconclusive`, not `pass`.
- `evidencePath` must resolve. Either a file under `var/research/...` / `output/...`, or an HTML anchor like `packages/<id>/results.html#<exp-anchor>`. If the anchor doesn't exist yet, write the row only after creating it.
- N upstream result-gate rows may collapse to 1 `methodsTried` row when they share a method (e.g., a 9-cell sweep summarized as one entry that links to the cell-level data). Prefer aggregation; do not let `methodsTried[]` mirror the full result-gate table.
- Single-seed `pass` is `inconclusive` until the gate's seed requirement is met.

### Atomic per-turn closure

When any event above fires, the same turn writes all four surfaces (or it doesn't write at all):

1. **Upstream witness** — `results.html` row, `next-action.html` chosen-route, or whichever surface owns the source of truth. Must exist with a stable anchor before step 2.
2. **`research_html/data/research-packages.js`** — the canonical row (append `methodsTried` for E1; update top-level fields for E2; update terminal fields for E3–E6).
3. **`research_html/packages/<id>/tracker.html`** — Resume Block `lastAction` = one-line description of the write (e.g., `"2026-05-12 added methodsTried[BARS_cap350] (verdict=fail)"`).
4. **Lint** — run `python research_html/scripts/learnings_lint.py lint-status` and `… lint-evidence`. A non-empty report at the Stop Gate is a workflow violation.

`learnings.html` is not in this list — it re-derives on load. Do not edit `learnings.html` directly.

### The dashboard-wide tool: `research_html/scripts/learnings_lint.py`

Single Python entry point that enforces this protocol across all packages. Reads `data/schema.js` + `data/research-packages.js` via the bundled `dump_packages.js` (node) helper. Subcommands:

| Command | What it does |
|---|---|
| `lint-status` | Schema lint per package: `(category, status)` legal; `_all` + status-specific required fields present; `forbidden` fields absent; `methodsTried` rows have the six fields and a legal verdict; cross-references (`supersededBy`, `promotedTo`) resolve; on-disk `packages/<id>/` ⇄ registry entries match. |
| `lint-evidence` | Every `methodsTried[].evidencePath` and `lastDecisionEvidencePath` resolves. File-missing is a warning; anchor-missing is an error (a typo or stale claim). |
| `scan-events [--pkg <id>]` | Runs the three draft writers: **E1** scans `results.html` `data-table="result-gate"` for finalized verdicts not yet in `methodsTried`; **E3** scans `next-action.html` `<div data-field="route">` for terminal-route language and proposes a `(category, status, terminationMessage)` block; **E4** scans `CLAUDE.md` for newly-cited package ids and proposes `adoptionPath`. Prints JSON drafts; does not write. |
| `draft-method <pkg-id> <anchor>` | Print one JSON `methodsTried` row drafted from `results.html#<anchor>`. If the anchor does not yet exist on the row, the draft says so — add the anchor first. |
| `draft-terminal <pkg-id>` | Print the JSON terminal block drafted from `next-action.html#chosen-route`. |
| `all [--pkg <id>]` | `lint-status` + `lint-evidence` + `scan-events`. Exit non-zero if any error was found. |

Add `--strict` to make warnings count toward the exit code (used by CI).

### Stop-Gate sequence (the contract for every learnings-relevant turn)

1. Make the upstream-witness edit (results.html / next-action.html / tracker.html / etc.).
2. Update `research_html/data/research-packages.js`.
3. Update tracker Resume Block `lastAction`.
4. Run `python research_html/scripts/learnings_lint.py all`. Fix every error before closing the turn.
5. If the turn includes a terminal status transition (E3–E6), confirm user ack is in hand.

The `scan-events` output should be reviewed every time a package's `results.html` or `next-action.html` changes — it tells the agent *which* draft writes the current state implies.

### Recovery: stale or wrong rows

- **Wrong verdict** (later evidence contradicts): edit the row in place; update `measured` and `verdict`; add a follow-up `methodsTried` row only if the new evidence comes from a *different* method or a re-run with new code. Do not append a "correction" row to the same method.
- **Stale evidencePath** (anchor removed or file moved): fix the upstream witness first (re-add the anchor in `results.html` or move the file), then update the row.
- **Mistaken adoption**: if `adoptionPath` was set but never landed in CLAUDE.md / code, clear it; if status is `ADOPTED`, downgrade to `ADOPTED_PENDING_ACK` with a `lastAction` note.

### What this protocol does NOT cover

- `learnings.html` rendering (purely derived; edit `assets/research.js` if the view itself needs to change).
- Per-experiment status tracking inside `experiments[]` — that's the tracker's resource-allocation table, not `methodsTried`.
- Brainstorm-direction notes — those live in `direction` and `contributionSpineFlag`, not in `methodsTried`.

## Current Best

Last updated: 2026-05-16

### MSR-VTT
- Variant: `fit_bucket_l010_g10_k20_s42` (seed 42)
- Ckpt: `output/GRDR/bucket_candidate_k20/msrvtt/20260428163014-fit_bucket_l010_g10_k20_s42/model-3-fit/best_model.pt`
- Setting 1 (beam 100, avg 130.73): CanHit@20/50/100/all = 52.8 / 74.5 / 86.5 / 91.0; XPool R@1/5/10 = 45.7 / 69.8 / 78.9
- Setting 2 (beam 15, avg 300.41): CanHit@20/50/100/all = 4.6 / 15.0 / 32.0 / 64.3; XPool R@1/5/10 = 17.4 / 32.2 / 40.6
- Setting 2 compact budget gate: `avg_candidates_per_query <= 310`; rows above are not comparable compact champions.

### Panda (in-distribution Setting 1) — pre-trained anchor for zero-shot use
- Variant: `panda_2150k_c4096l3_rq03_s42` (P14, 2.15M corpus, c=4096, l=3, seed 42, single-seed; multi-seed validation outstanding)
- Ckpt: `output/GRDR/panda/champion_pretrained_zeroshot_c4096l3_2150k_s42/model-3-fit/best_model.pt` (hardlinked from `var/research/2026-05-11-panda-scaleup-zeroshot-scalability/output/GRDR/panda_2150k_c4096l3/panda/20260515012011-panda_2150k_c4096l3_rq03_s42/model-3-fit/best_model.pt`; same inode, both paths live)
- Setting 1 seed 42 (beam 100, avg 103.93): CanHit@20/50/100/all = 80.70 / 90.52 / 95.57 / 95.77; MeanLogDiscount = 0.5493; train_test_collision = 0.477; sID utility = 0.975
- Compact-budget gate: avg_candidates_per_query 103.93 (best Pareto across all measured 2.15M ckpts; −10.42 vs 1.65M c=1024 anchor with +2.02 CanHit@100 lift)
- Promoted via `2026-05-11-panda-scaleup-zeroshot-scalability` package (success / ADOPTED 2026-05-16); supersedes the prior 818K c=512 P6 anchor for zero-shot use.

### Panda (in-distribution, prior 818K 3-seed anchor, retained for reproducibility)
- Variant: `panda_s1_p4_rq03_c512l3_s42` (P6 3-seed confirmed; seed 42 ckpt)
- Ckpt: `var/research/2026-05-05-panda-setting1-full-run/output/GRDR/panda_setting1_p4/panda/20260509164015-panda_s1_p4_rq03_c512l3_s42/model-3-fit/best_model.pt`
- Setting 1 seed 42 (beam 100, avg 125.95): CanHit@20/50/100/all = 75.54 / 86.95 / 92.80 / 94.01
- Setting 1 3-seed mean (42/220/3407, beam 100, avg 127.15): CanHit@100 = 92.42 ± 0.38; FullSetHit@All = 93.83
- Setting 2: not run.

### Zero-shot transfer (4-dataset, BARS-on, X-Pool rerank) — pre-trained 2.15M c=4096 l=3 s42 vs supervised target
| Dataset     | Setting | Pre-trained R@1 / R@5 / R@10 | Supervised target R@1 / R@5 / R@10 | Δ R@10 |
| ----------- | :-----: | ---------------------------: | ---------------------------------: | :----: |
| MSR-VTT     |    1    | 45.00 / 68.70 / 76.70        | 46.00 / 70.10 / 78.00              | −1.30  |
| ActivityNet |    1    | 32.90 / 60.30 / 71.00        | 33.70 / 63.70 / 76.60              | −5.60  |
| DiDeMo      |    1    | 39.48 / 66.20 / 74.28        | 39.90 / 65.80 / 74.20              | **+0.08** |
| LSMDC       |    1    | 21.80 / 34.20 / 39.80        | 23.50 / 39.40 / 46.20              | −6.40  |
| MSR-VTT     |    2    | 19.50 / 35.00 / 43.40        | 19.20 / 35.90 / 44.80              | −1.40  |
| ActivityNet |    2    | 8.50 / 22.50 / 32.60         | 19.20 / 41.10 / 51.80              | −19.20 |
| DiDeMo      |    2    | 19.24 / 33.70 / 38.38        | 15.50 / 29.70 / 36.10              | **+2.28** |
| LSMDC       |    2    | 2.70 / 4.70 / 5.90           | 2.10 / 4.80 / 5.90                 | **0.00** |
- 4-ds R@10 means: S1 65.44 vs 68.75 (gap −3.30); S2 30.07 vs 34.65 (gap −4.58). DiDeMo S2 exceeds supervised on every R@K; LSMDC S2 ties on R@10. ActivityNet S2 is the dominant remaining gap.
- All 8 cells compact (avg_candidates_per_query ≤ 310; largest LSMDC S2 = 214.64). Single-seed s42 caveat applies.

## graphify

A knowledge graph of this codebase lives in `graphify-out/`. It is rebuilt automatically when code changes.

### Before answering codebase questions
1. Check if `graphify-out/graph.json` exists
2. If it does, load the graph and query it before reading raw files — it is faster and shows cross-file connections
3. Use `/graphify query "<question>"` for broad questions, `/graphify explain "<concept>"` for single-node deep dives

### After making code changes
If you modified `.py` files, rebuild the graph:
```
python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
```
This only re-extracts changed files (AST-only for code, no LLM cost).

### Key graph facts (as of 2026-04-08)
- 1147 nodes, 2246 edges, 30 communities
- God nodes: Config (89 edges), T5Config (89), MLPLayers (70), VideoCapture (61), GRDR (45)
- The GRDR trainer intentionally reuses X-Pool's BaseTrainer and CandidateDataLoader
- Config is the main cross-community bridge (links datasets, evaluator, trainer, inference sim, CLIP transformer)
