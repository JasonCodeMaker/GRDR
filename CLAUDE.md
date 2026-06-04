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
- When a request spans multiple cells (e.g., Setting 1 + Setting 2, P1.a + P1.b, multi-dataset zero-shot transfer, beam-width sweeps), run every cell in scope before reporting back. State the full cell list at session start, update `results.html` and the canonical CSV incrementally as each cell finalizes, and list any skipped cells with explicit reason at close.

## Change Isolation Rule (check FIRST for any GRDR-related change)

Any edit that can move GRDR training or evaluation numbers MUST happen in an isolated git worktree, not in the main checkout. This is a hard precondition — check it before opening an editor on a GRDR file, and before dispatching any agent to implement a GRDR change.

- **In scope (worktree required):** any file under `models/`, `trainer/`, `data/` sampling/routing logic, `run.py` training schedule, loss/regularizer formulations, semantic-ID generation/export code, X-Pool reranker code that the published "Current Best" results depend on, hyperparameter defaults that ship as new behavior, and any dataset/config change that alters what the model sees during training or evaluation. Rule of thumb: if a training or evaluation run with the change could produce different numbers from a run without it, it is in scope.
- **Out of scope (main checkout is fine):** `research_html/` (dashboard, package pages, docs), top-level markdown docs, launcher comments / log strings, memory files, README and similar tracked-but-non-runtime artifacts.
- **Workflow:** create the worktree via the `superpowers:using-git-worktrees` skill (native worktree or `git worktree add .worktrees/<slug> -b <branch>`). Implement the change in the worktree. Verify effectiveness against the relevant Current-Best anchor (see `## Current Best` below) under the compact-budget gate and downstream X-Pool rerank as the ultimate witness — per `## Global Optimization Objective`. Only merge the worktree branch back after the verification passes; failed worktrees are torn down per `## Refinement Guardrails` ("remove all worktrees created for that direction").
- **Refusal:** if the user asks for a quick in-place GRDR edit on the main checkout, stop and surface this rule before editing.

## Code Alignment Rule (check FIRST for any experiment on Bunya or Nectar)

Before launching any experiment on a remote compute environment (Bunya, Nectar), verify that the remote checkout's code state matches the local workstation's code state. Drift between sites silently invalidates results — an experiment that runs on Bunya from a stale SHA is wasted compute and a misleading data point that can take days to detect.

- **In scope:** any wall-clock training run, evaluation run, semantic-ID generation pass, candidate export, captioning pipeline, or any other GPU/CPU job whose output is consumed as evidence. Includes both new launches and resumes/continuations of prior runs. Applies symmetrically to all sites — workstation, Bunya, Nectar — even if only two of the three are involved in a given experiment.
- **Out of scope:** light orchestration on a remote login node (mkdir, ls, sha256, file-transfer staging) that does not execute project code.
- **Verification protocol (run before each remote launch):**
  1. `git rev-parse HEAD` on local AND on the remote checkout (Bunya: `/scratch/user/uqzzha35/Project/<project>`; Nectar: the project mirror used by the run). The two SHAs must match.
  2. `git status --porcelain` on both sides. Both must be clean, OR the local working-tree diff must be explicitly synced to the remote (via the `bunya-file-transfer` skill for Bunya; the documented Nectar transfer path for Nectar) and re-verified by file hash.
  3. If a `.worktrees/<branch>` checkout is the runtime root per the Change Isolation Rule above, apply the protocol to the worktree path, not to the main checkout. The worktree's SHA is the witness.
- **On mismatch:** do NOT launch. Either fast-forward the lagging side (`git fetch && git checkout <sha>`), push the missing commits, or sync the working-tree diff. Re-verify after the sync. Only then launch.
- **Record the verified SHA** in the run's tracker live-card / launcher manifest / launch sentinel so post-hoc analysis can confirm which code produced the numbers. Untracked SHA = the run is not citable as evidence.

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

- The only valid in-repo location for a new research package is `research_html/packages/<YYYY-MM-DD>-<slug>/`. Use the `/research-package` skill to scaffold; do not create ad-hoc top-level research folders.
- Every package owns one decision per page: `index.html`, `plan.html`, `tracker.html`, `results.html`, `next-action.html`, plus `docs/`, `scripts/`, and `_agent/`.
- Package-owned scripts (launchers, eval drivers, sentinel writers, one-off ablations) live under `research_html/packages/<id>/scripts/`. Stable shared entrypoints stay in repo-root `scripts/`.
- Runtime state, supervisor JSON, local logs, candidate JSONs, and temporary CSVs must go under `var/research/<YYYY-MM-DD>-<slug>/`, not in tracked repo roots.
- Legacy `research/active/` and `research/archive/` directories are retained for pre-2026-05-11 packages only; do not add new content there.

## Brainstorm → Package Promotion Recipe

Three-phase lifecycle for a new idea: **explore → refine → promote**. Exploration is cheap and ungated; promotion is the gated commit that turns the idea into a trustworthy, runnable in-progress package. This is orchestration over the existing skills (`/research-package` scaffolds, `/research-op` mutates rows/fields) — not a new skill. Promotion is always user-triggered. Phases 2–3 adopt the convergence discipline of the superpowers `brainstorming` skill — divergent approaches, section-gated approval, and a spec self-review — applied at the `PILOT_READY` gate (not during early exploration, which stays frictionless). `idea-creator` diverges across *ideas* upstream; this recipe converges *one* idea into an approved design.

### Container model (hybrid)

While exploring and refining, a brainstorm lives as **two coupled artifacts**, not a full package:
1. **Content** — a doc-style HTML at `research_html/brainstorm/<YYYY-MM-DD>-<slug>.html`, following the doc-template + style guide (Rule 0 applies). This holds the substance.
2. **Dashboard handle** — one thin row in `research_html/data/research-packages.js` with `category: "brainstorm"`, `status: "EXPLORING"`, and `detailPath: "brainstorm/<slug>.html"` so the lane card opens the doc directly. No `packages/<id>/` directory exists yet. (`relativeDetailPath()` already reads a free-form `detailPath` per row, so no renderer change is needed.)

The thin row carries only brainstorm-legal fields: required `direction` + `contributionSpineFlag` (an id from `RESEARCH_CONTRIBUTION_SPINE` in `schema.js`); metric/gate fields (`activeGate`, `primaryMetricVsGate`, `methodsTried`, `openRuns`) are schema-forbidden for this category. Keep `name`, `problem`, `objective`, `motivation`, `lastUpdated` populated so the card renders.

### Phase 1 — explore

- Discuss the brief idea with the user; write the doc.
- Insert the thin lane row (status `EXPLORING`).
- Run `python research_html/scripts/learnings_lint.py lint-status`; fix any missing required field.

### Phase 2 — refine

Refine converges the idea to an approved design. Iterate the doc freely, but the `EXPLORING → PILOT_READY` bump is gated on three checks (adapted from the superpowers `brainstorming` skill):

- **Gate A — approaches considered (divergence).** The doc must carry a section with **≥2 candidate designs**, their trade-offs, and the chosen one *with reasoning* — not just a single `direction`. This forces converging on the best design rather than the first one imagined.
- **Gate B — section-gated approval (convergence).** Present the design one section at a time — *modules changed → train/eval data flow → metric & budget gate → risks* — and get user approval after each before advancing. Drive elicitation with `AskUserQuestion` multiple-choice batches (one decision per question), not a prose dump.
- **Gate C — spec self-review.** Run the research self-review under `## Research Workflow` (problem anchor, weak assumptions, reviewer attacks, novelty), *plus* the four superpowers checks: placeholder scan (no `TBD`/`TODO` left), internal consistency (sections don't contradict), scope check (one package or decompose), ambiguity check (metric/gate definitions admit one reading). Fix inline.

Only when all three pass — and the idea has a falsifiable hypothesis and a no-change boundary — bump the row `EXPLORING → PILOT_READY` (adds required `hypothesis` + `noChangeBoundary`) via `/research-op`. `PILOT_READY → PROMOTED` is a **hard gate**: do not scaffold a package until the user has explicitly approved the design, not merely until the hypothesis/no-change fields exist.

### Phase 3 — promote (user-triggered only)

A single atomic procedure; do not stop partway:
1. **Elicit** the in-progress trustworthiness fields the doc has not already settled, via `AskUserQuestion` multiple-choice batches (one decision per question — the CC-native form of the superpowers one-question-at-a-time rule), not a prose dump. Cover: hypothesis; primary metric + `metric-formula`/`metric-dataset`/`metric-protocol`/`metric-dedup`/`metric-cutoff`; baseline + `baseline-checkpoint`/`baseline-protocol`/`baseline-last-verified`; budget gate (Setting 2: `avg_candidates_per_query <= 310`); no-change boundary; seed plan; the `experiments[]` pipeline (`P\d+` rows that paint the timeline); `nextRoute`. Carry over `direction`→problem/objective, `hypothesis`, `noChangeBoundary`, `contributionSpineFlag`.
2. **Scaffold** `packages/<YYYY-MM-DD>-<slug>/` via `/research-package` as `category: in-progress`, `status: CONTEXT_LOADED`, `--scope all`, filling the elicited fields.
3. **Migrate the doc**: copy `research_html/brainstorm/<slug>.html` → `packages/<new-id>/docs/<slug>.html` (now the canonical, dashboard-wired copy). Freeze the original brainstorm file as a backup — add a banner at the top linking to the package doc and never edit it again. (If a frozen duplicate is unwanted, move instead and rely on git history.)
4. **Tear down the handle**: delete the thin brainstorm row from `research-packages.js`. Do not keep it as `PROMOTED` — the in-progress package is now the single home; provenance lives in the migrated doc + git history.
5. **GRDR ready-to-run gates** (these, not lint, define "runnable"): create the Change-Isolation worktree per `## Change Isolation Rule`; satisfy the `## Code Alignment Rule` SHA check before any remote launch; record the budget gate and name X-Pool downstream rerank as the ultimate witness. Advance toward `READY_TO_LAUNCH` only once these hold.
6. **Lint**: run `python research_html/scripts/learnings_lint.py all` and fix every error before closing the turn.

## Research Doc Style

Any new HTML doc under `research_html/packages/<pkg-id>/docs/` MUST start from the shared template and follow the style guide. This keeps every doc dashboard-wired (status-strip + package-nav) and visually consistent.

- **Rule 0 — analyze the topic before writing.** Open `research_html/templates/doc-style-guide.html` and work the three questions: (a) what is the hardest aspect of the topic to convey in prose? (b) what concrete entity grounds the abstraction (one named video, one query, one user)? (c) what invariant is hidden in the geometry (tensor shapes, codebook layout, route multiplicity)? Pick the visualization pattern that addresses (a). This is a hard precondition — drafting a doc without working Rule 0 is a workflow violation.
- **Two-part standard:** (1) *text primitives* (`pre.diagram`, `pre.code`, `.callout`, `table.data-table`, `pill-mono`, `stage-title`) for narrative, code, glossary, structured data; (2) *visualization patterns* (side-by-side architecture, function-level flow SVG, state-cell grid, discrete-ID chip display, candidate-flow lanes, benefits-at-a-glance table) for content that prose alone obscures. The style guide documents all six visualization patterns with reusable CSS scaffolds + color/animation conventions.
- **Visualization choice is topic-driven, not boilerplate.** Match the pattern to the topic. A 1-page perf-fix doc rarely needs more than one visualization; a multi-mechanism design doc justifies 3–5. Visualizations replace language where language fails — they are not decoration.
- **Skeleton:** copy `research_html/templates/doc-template.html`. Replace the six template variables (`$package_id`, `$doc_title`, `$eyebrow`, `$lead`, `$last_updated`, `$root_prefix`), delete the demo `<section data-section="primitives">`, then write the doc's real sections. If the topic needs visualization, copy the relevant scaffold CSS (`.arch-grid`, `.cb-grid`, `.sid-chip`, `.cand-flow`, animation keyframes) from the top of `doc-style-guide.html` into the doc's page-local `<style>` block — only the patterns you actually use.
- **Hard rules:** keep the shell verbatim (`data-status-strip`, `data-package-nav`, footer `<time data-field="last-updated">`, and the three trailing `<script>` tags). Page-local CSS is **permitted** for visualization scaffolds documented in the style guide; stay within the documented color palette (frozen blue, trained green, original red, proposed green, neutral cream, warm/cold/mixed for state). Animations must include `@media (prefers-reduced-motion: reduce)` fallback. Bump the footer date with a short scope phrase on every meaningful edit.
- **Section composition is content-agnostic:** the template prescribes the shell + primitives, not section count, section order, or section topics. A perf-fix doc can be one card; a full pipeline walk-through can be 8–10 numbered steps. Lead with the visualization, then table, then prose — visual → reference → narrative.
- **Reuse one concrete example across visualizations.** If the doc introduces `v_cook_42` in the arch diagram, also use it in the chip display, the candidate-flow lane, and the benefits framing. One mental model, threaded across patterns.
- **Render-check before declaring done.** Run `google-chrome --headless --screenshot` on the doc and visually confirm SVG, tables, chips, and animations render correctly. Tag-balance lint alone is insufficient.
- **The exemplars:**
  - **Visual exemplar** (mandatory reread before designing a new doc): `research_html/packages/2026-05-25-multiview-workload-aware-sid-index/docs/hierarchical_sid_design.html` — the canonical use of all six visualization patterns (side-by-side architecture, function-level flow SVG with override-vs-fallback branches, codebook fragmentation cell grid, sID chip display, candidate-flow dedup lanes, benefits-at-a-glance).
  - **Text exemplar**: `research_html/packages/2026-05-16-panda-pseudo-queries-multiview/docs/training_pipeline.html` — the reference for a fully-fleshed-out text-primitive-heavy doc.

## Runtime Ops

- Launch every long-running bash script, dataset download, preprocessing pipeline, and experiment inside `tmux`.
- Prefer named `tmux` sessions/windows and report the attach command so the run can be monitored live.
- Enforce the Fact Propagation Contract on every per-turn live cycle: between the tracker live-check update and the §5 status line, run `python research_html/packages/<pkg-id>/scripts/propagate_facts.py`; for each event listed in the report, update its owning surfaces (`results.html`, `next-action.html`, `research_html/data/research-packages.js`, tracker Resume Block) in the same turn, then advance the cursor with `propagate_facts.py --bump`. A non-empty report at the Stop Gate is a workflow violation.

### Background job hygiene

- Never spawn background processes with a trailing `&` from a `Bash run_in_background` call; the wrapper will self-detach and lose its status writer. Restructure the script to use an internal `sleep` loop instead.
- Polling for job completion must match on actual file state (sentinel file existence, output size, `sacct` State, exit code) — never on the echoed command string in a log tail, which false-matches as soon as the launcher prints the command it is about to run.
- `pgrep`-based liveness checks must require **N consecutive negatives** before declaring the process dead, to avoid the kill-then-relaunch race during fork.
- For any parallel CPU workload (numpy / faiss / sklearn / multiprocessing), prefix the launch with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Benchmark 1-worker vs N-worker on a small slice before scaling; if N-worker is not >1.5× faster, do not fan out.
- Launch wrappers must not use `set -e` around the sentinel-write step — an upstream failure must still flush the return-code sentinel so polling sees a terminal state.

### Bunya GPU allocation by phase

- **Smoke tests → interactive `salloc`.** Smokes are short-scale probes: env verification, throughput gates, single-step sanity, small-N pilots, S0-SMOKE-style cells, sentinel reruns. If the target node has multiple idle GPUs, request all of them on one salloc (e.g. `--gres=gpu:h100:2`) and fan smokes across the GPUs in parallel via `CUDA_VISIBLE_DEVICES=0` / `=1`. Release the salloc with `exit` as soon as the smoke phase finishes — do not hold an interactive lease idle while staging the next phase.
- **Full / long-running runs → `sbatch`.** Reserved for budgeted production cells: multi-loop GRDR training, multi-shard captioning, full-scale eval exceeding the interactive QoS time limit, and any cell whose result will be cited in a `results.html` row. Record the verified SHA in the sbatch script (per the Code Alignment Rule).
- The one-salloc-per-node ban still applies: a second interactive salloc cannot land on a host that already holds one of yours; pass `--exclude=<busy-nodes>` derived from `squeue --me`.

## Verification Discipline

Lint passing is necessary but not sufficient to declare a task complete. Before reporting any artifact-producing task as done:

- Enumerate the full scope up front. For multi-cell / multi-setting work (Setting 1 + Setting 2, dataset × beam × seed grids, P1.a + P1.b style sub-stages), list every cell that must complete and check each off explicitly. Do not stop after the first cell.
- Render the artifact and verify visible state matches intent. For HTML packages: confirm `next-action.html` is closed, `tracker.html` agent-zone collapses are closed by default, no intended content is hidden inside an unexpanded `<details>` block, and every referenced file resolves.
- For code changes affecting training/eval: run the narrowest available check (single-cell smoke, single-seed eval, lint) and confirm the change does not regress the existing behavior; if no automated check exists, state explicitly why prior behavior should remain unchanged.
- Surgical changes only. If a second edit seems needed beyond the requested scope, surface it as a question — do not silently fold it in.

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

Atomic per-turn closure: when any event above fires, the same turn touches all four surfaces or none.

1. **Upstream witness** — make the edit on the source-of-truth surface (`results.html` row, `next-action.html` chosen-route, etc.) with a stable anchor.
2. **`research_html/data/research-packages.js`** — append `methodsTried` (E1) or update top-level / terminal fields (E2–E6).
3. **`research_html/packages/<id>/tracker.html`** — Resume Block `lastAction` = one-line description of the write (e.g., `"2026-05-12 added methodsTried[BARS_cap350] (verdict=fail)"`).
4. **Lint** — run `python research_html/scripts/learnings_lint.py all`; fix every error before closing the turn. A non-empty report at the Stop Gate is a workflow violation.
5. If the turn includes a terminal status transition (E3–E6), confirm user ack is in hand.

`learnings.html` is not in this list — it re-derives on load; do not edit it directly. The `scan-events` output should be reviewed every time `results.html` or `next-action.html` changes — it tells the agent *which* draft writes the current state implies.

### Recovery: stale or wrong rows

- **Wrong verdict** (later evidence contradicts): edit the row in place; update `measured` and `verdict`; add a follow-up `methodsTried` row only if the new evidence comes from a *different* method or a re-run with new code. Do not append a "correction" row to the same method.
- **Stale evidencePath** (anchor removed or file moved): fix the upstream witness first (re-add the anchor in `results.html` or move the file), then update the row.
- **Mistaken adoption**: if `adoptionPath` was set but never landed in CLAUDE.md / code, clear it; if status is `ADOPTED`, downgrade to `ADOPTED_PENDING_ACK` with a `lastAction` note.

### What this protocol does NOT cover

- `learnings.html` rendering (purely derived; edit `assets/research.js` if the view itself needs to change).
- Per-experiment status tracking inside `experiments[]` — that's the tracker's resource-allocation table, not `methodsTried`.
- Brainstorm-direction notes — those live in `direction` and `contributionSpineFlag`, not in `methodsTried`.

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

### Key graph facts (durable)
- The GRDR trainer intentionally reuses X-Pool's `BaseTrainer` and `CandidateDataLoader`.
- `Config` is the main cross-community bridge (links datasets, evaluator, trainer, inference sim, CLIP transformer); expect it to appear as a god-node hub when querying the graph.
