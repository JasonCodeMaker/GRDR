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

## Storage Accounting SSOT (adopted 2026-06-09)

The single source of truth for all GRDR/baseline per-video storage numbers — across every package, figure, and the paper — is `scripts/panda_figure/storage_accounting_ssot.md`. Do not invent a different storage convention; cite this file.

- **Convention:** per-video index footprint, accounted **symmetrically**. Exclude on *every* method the corpus-independent query/tokenizer models (GRDR's T5 generator ↔ ANN's fine-tuned text encoder; GRDR's RQ-VAE codebooks ↔ ANN's video encoder) and any rebuildable search structure (GRDR's prefix trie ↔ ANN's IVF/HNSW graph). Count only the per-video payload, the query-time dictionary (PQ codebook; GRDR has none), and the int64 video id.
- **GRDR formula:** `S(N) = N·(V·L·b_code + b_id) = N·(4·3·2 + 8) = 32·N` bytes (26·N if 12-bit packed). Verified constants: `V=4` latent-token sIDs/video (every video registered under all 4 routes — `trainer/evaluator.py:587-592`), `L=3` codes/sID, `b_code=2 B` int16 (K=4096), `b_id=8 B`.
- **ANN formula:** IVF-PQ/OPQ `S(N)=N·(m+8)+codebook`; anchors `N·(dim·4+8)`.
- **Do NOT** count a fixed decoder offset (the old ~275 MB `D_decoder` is excluded), and **do NOT** size GRDR from the single-view MM-SemanticTVR json (6 B/vid) — that is a 4× undercount; the deployed index is multi-view (24 B of codes/video).

## MSR-VTT Setting 2 Rules

- Current compact budget gate: `avg_candidates_per_query <= 310`. Any Setting 2 row above this gate is excluded from compact-champion selection unless the user explicitly labels it as a diagnostic or large-pool ablation.
- The primary failure bucket to attack is route/access-path miss, but only under the compact budget gate. The desired movement is from absent or late GT to early top-100 GT, not from absent to late full-pool rescue.
- A valid Setting 2 champion must beat the current compact reference under the same budget gate, improve downstream X-Pool rerank or reduce candidate-pool size without rerank regression, and pass multi-seed validation. Do not promote a single-seed tie or a `+0.1` `CanHit@100` change.
- Required Setting 2 readout: `avg/p95/max candidates`, `CanHit@20/50/100`, `FullSetHit@All`, `OverflowHit`, `route_miss`, `MeanLogDiscount`, top-100-truncated X-Pool, full-candidate X-Pool, and whether any X-Pool gain is caused by larger pool size.

# Trustworthy Research Pipeline

This file is the agent operating context for any project that adopts the Trustworthy Research Pipeline. It is intentionally project-agnostic. A consuming project copies this file into its repo root and **prepends** project-specific sections (project name, motivation, optimization objective, contribution spine, current best, dataset / budget gates) above the protocols below. The protocols themselves are universal — do not edit them per project.

## What this pipeline produces

A trustworthy research record where every claim is gated by an explicit metric, every metric is backed by a verified artifact, every direction has one declared next route, and every adopted win or archived failure leaves a structured `methodsTried` trace the next session can learn from. The skills bundled with this repo install the HTML surfaces, Scope/Triage gates, orchestration, and mutation tooling that enforce this. `WORKFLOW.md` is the seven-step controller the agent follows inside any package.

`research_html/` is the shared context surface, not the authority by itself. For research-affecting tasks,
load the narrow owning layer: `outputs/_scope/transitions.jsonl` for intent, package pages for
plan/tracker/result witnesses, `outputs/<pkg>/` plus live process state for measurements, and
`research_html/data/research-packages.js` for dashboard index state. Derived pages such as `scope.html`,
`context.html`, `learnings.html`, lane pages, and `scope-projection.json/js` are read-only context unless
their owning skill says otherwise.

## The five protocols the agent obeys

The protocols form a stack — each one constrains the layers above it.

### 1. Research Workflow (`WORKFLOW.md` at the repo root)

The seven-step controller for any package: how to load context, when to dispatch a sub-agent, what to emit on the 10-minute live cycle, when to schedule re-entry, when to stop. `WORKFLOW.md` is the operating protocol for any `@WORKFLOW.md` invocation; it overrides general harness defaults (do not spawn agents unless asked, end-of-turn summary style, etc.) inside a research session.

Strictly follow `WORKFLOW.md`: when it says dispatch a sub-agent, dispatch; when it says emit a 10-minute status line, emit it; when it says schedule re-entry, schedule.

### 2. Research Output Contract

Research packages live under `research_html/packages/<YYYY-MM-DD>-<slug>/` and are created or materially
restructured through `/research-package`, never by ad-hoc folders. Materialization reads only committed
Scope state, not pending Triage proposals. Runtime logs, metrics, event manifests, checkpoints, and
temporary artifacts go under `outputs/<YYYY-MM-DD>-<slug>/`.

Current package canon: packages use `index.html`, `plan.html`, `tracker.html`, `results.html`,
`docs/index.html`, and `_agent/context.html`, with optional `implementation.html`, `analysis.html`,
conversion-only `brainstorm.html`, and package-local `scripts/`. `tracker.html` owns execution state and
`tracker.html#chosen-route`; standalone `launch.html`, `live.html`, and `next-action.html` are retired.
For detailed field ownership, load `skills/research-package/references/package-contract.md` only when a
package task needs it.

### 3. Fact Propagation Contract

Every artifact that lands during a research run (checkpoint, candidate JSON, sentinel, phase marker, chain-done) is a "locked fact" that the agent must propagate to every owning surface — `results.html`, `tracker.html#chosen-route`, registry status fields, tracker Resume Block — in the same turn the artifact is observed.

The mechanical check is `/research-op scan-events` (shipped with the `research-op` skill at `skills/research-op/scripts/research_op.py`):

```bash
# every per-turn live cycle
python skills/research-op/scripts/research_op.py --pkg <pkg-id> --op scan-events   # list newly-locked facts as JSON event lines
# … agent invokes --event <name> --payload <json> per event for atomic fan-out …
# The cursor advances on the next successful --event invocation (no separate --bump step).
```

The cursor lives at `<runtime-root>/manifests/.propagation_cursor` (epoch float). An empty report = nothing to propagate. A non-empty report at the Stop Gate is a workflow violation.

**Directive changes are locked facts too (`DIRECTIVE_CHANGE`).** A *user instruction that changes a package's constraints, plan, or scope* — "add a rule", "redesign experiment P1", "change the metric/baseline/roster" — is a locked fact on the same footing as an artifact event. It is not surfaced by `scan-events` (no artifact landed), so the agent must propagate it explicitly in the same turn: write the directive to its typed home (a binding rule → `/research-op insert --target package-invariant`; a plan/scope change → its owning surface), **and** update the tracker Resume Block `lastAction`/`workflow-state` **and** the registry `lastUpdated`. A directive that touches only one surface (e.g. a rule buried in a doc while the tracker and registry read unchanged) is a propagation violation — `learnings_lint.py lint-status` flags it as `directive-not-propagated`.

### 4. Learnings Update Protocol

The cross-package learnings index at `research_html/learnings.html` is a derived view over `research_html/data/research-packages.js`. The data file is the canonical store; `learnings.html` re-renders on page load. This protocol fixes *when* to write to the data file and *how* to keep it trustworthy.

**Core principles**

1. **Upstream surface is the witness, the data file is the index.** A `methodsTried[]` row is written to `research-packages.js` *only after* the corresponding row exists in the package's `results.html` with a stable section anchor, and the `evidencePath` resolves to a real file or anchor. Never invent a row from memory.
2. **Drafts are auto-detected; writes are user-acked at terminal transitions.** In-progress facts (`VERDICT_FINALIZED`, `STATUS_CHANGED`) update without user ack because the source-of-truth surface already exists. Terminal facts (`TERMINAL_TRANSITION`, `ADOPTION`, `SUPERSESSION`, `REOPEN`) require T1 user ack.
3. **Atomic per-turn closure.** Any turn that mutates a learnings-relevant field must, in the same turn, touch all of: upstream surface row → `research-packages.js` → tracker Resume Block `lastAction` → run `learnings_lint.py`. A non-empty lint report is a Stop-Gate violation.

**Event trigger table**

Learnings event names (`LEARNINGS_EVENT` constant — SSOT: this file): `DIRECTIVE_CHANGE`, `VERDICT_FINALIZED`, `STATUS_CHANGED`, `TERMINAL_TRANSITION`, `ADOPTION`, `SUPERSESSION`, `REOPEN`.

| Event | Trigger (where it originates) | User ack | Fields written in `research-packages.js` |
| --- | --- | --- | --- |
| **`DIRECTIVE_CHANGE`** | A user instruction changes the package's constraints / plan / scope (add a binding rule, redesign an experiment, change metric / baseline / roster) — not an artifact event, so `scan-events` will not surface it | none | Write the directive to its typed home (`bindingRules[]` via `--target package-invariant`, or the owning surface) + `lastAction`, `lastUpdated` |
| **`VERDICT_FINALIZED`** | `results.html` result-gate row gains `PASS` / `FAIL` / `INCONCLUSIVE` / `DIAGNOSTIC` AND artifact verification recorded | none | Append one `methodsTried[]` row |
| **`STATUS_CHANGED`** | tracker live-check, plan revision, blocker change | none | `status`, `activeGate`, `primaryMetricVsGate`, `currentBlocker`, `openRuns`, `lastAction`, `lastUpdated` |
| **`TERMINAL_TRANSITION`** | `tracker.html#chosen-route` resolves to a terminal lane move (`TERMINATE`, adoption) | **T1** | `category` (lane move), `status` (terminal value), `terminationMessage`; freeze `methodsTried[]` |
| **`ADOPTION`** | `CLAUDE.md` "Current Best" edit, code merge into `models/` / `trainer/`, or a new in-progress package starts citing the win | **T1** | `adoptionPath` (specific anchor or path) |
| **`SUPERSESSION`** | A newer success package replaces an older one | **T1** | On the *old* package: `status = WIN_SUPERSEDED`, `supersededBy = <new id>` |
| **`REOPEN`** | User explicitly states a fail package should be revisitable under a named condition | **T1** | `status = ARCHIVED_CONDITIONAL`, `reopenTrigger = "<condition>"` |

**`methodsTried` row contract**

Every row is exactly six fields, drawn verbatim from the witnessing `results.html` row:

```
{ method, hypothesis, gate, measured, verdict, evidencePath }
```

- `verdict` ∈ `{PASS, FAIL, INCONCLUSIVE, DIAGNOSTIC}`. Diagnostic-only rows use `DIAGNOSTIC` (not `INCONCLUSIVE`). Single-seed or ambiguous results use `INCONCLUSIVE`.
- `evidencePath` must resolve. Either a file under `outputs/...` / `output/...`, or an HTML anchor like `packages/<id>/results.html#<exp-anchor>`. If the anchor doesn't exist yet, write the row only after creating it.
- N upstream result-gate rows may collapse to 1 `methodsTried` row when they share a method (e.g., a 9-cell sweep summarized as one entry that links to the cell-level data). Prefer aggregation.
- Single-seed `PASS` is `INCONCLUSIVE` until the gate's seed requirement is met. Runs producing only diagnostic evidence (no hypothesis test) use `DIAGNOSTIC`.

**The dashboard-wide tool: `research_html/scripts/learnings_lint.py`**

| Command | What it does |
| --- | --- |
| `lint-status` | Schema lint per package: `(category, status)` legal; required fields present; forbidden fields absent; `methodsTried` rows have the six fields and a legal verdict; cross-references (`supersededBy`, `promotedTo`) resolve; on-disk `packages/<id>/` ⇄ registry entries match. |
| `lint-evidence` | Every `methodsTried[].evidencePath` and `lastDecisionEvidencePath` resolves. File-missing is a warning; anchor-missing is an error. |
| `scan-events [--pkg <id>]` | Runs the three draft writers (`VERDICT_FINALIZED` / `TERMINAL_TRANSITION` / `ADOPTION`). Prints JSON drafts; does not write. |
| `draft-method <pkg-id> <anchor>` | Print one JSON `methodsTried` row drafted from `results.html#<anchor>`. |
| `draft-terminal <pkg-id>` | Print the JSON terminal block drafted from `tracker.html#chosen-route` (legacy packages may fall back to `next-action.html#chosen-route`). |
| `all [--pkg <id>]` | All three lints + scan. Exit non-zero if any error was found. |

Add `--strict` to make warnings count toward the exit code (CI mode).

**Stop-Gate sequence (the contract for every learnings-relevant turn)**

1. Make the upstream-witness edit (`results.html` / `tracker.html#chosen-route` / `tracker.html`).
2. Update `research_html/data/research-packages.js`.
3. Update tracker Resume Block `lastAction`.
4. Run `python research_html/scripts/learnings_lint.py all`. Fix every error before closing the turn.
5. If the turn includes a terminal status transition (`TERMINAL_TRANSITION` / `ADOPTION` / `SUPERSESSION` / `REOPEN`), confirm user ack is in hand.

### 5. Refinement Guardrails

Treat the consuming project's contribution spine as a compatibility constraint unless the user explicitly asks to replace it. Every refinement must explain why the design sharpens the current research story, not just why it is novel in isolation.

Each consuming project declares its **non-negotiable contribution spine** as cards in `RESEARCH_PROJECT_PROFILE` in `research_html/data/research-packages.js` (or as a numbered list in the project's own CLAUDE.md). By default, push back on refinements that:

- Remove a spine component without a strong reviewer-proof reason
- Break the Stage-1 → Stage-2 handoff (if the project has one)
- Discard a co-training or progressive-training schedule
- Weaken prior-state code/data consistency

When a refinement direction is explicitly judged failed, remove all worktrees created for that direction after preserving any needed notes in the owning research package.

## The state model that ties protocols 2-4 together

`research_html/data/schema.js` declares the `(category, status)` state machine and the required-field rules each cell must satisfy. The card renderer and `learnings_lint.py` both import from it.

**Naming convention:** Package *category* (lane) values are lowercase-kebab (`in-progress`, `success`, `fail`) — they are URL/CSS/attribute facets. Package *status* values are SCREAMING_SNAKE — they are state-machine positions. Never recase the lane values; never use lowercase for status values.

```
category=in-progress → status ∈ { CONTEXT_LOADED, IMPLEMENTING, IMPLEMENTATION_REVIEW,
                                  DECISION_ADJUDICATION, READY_TO_LAUNCH, EXPERIMENT_RUNNING,
                                  LIVE_ANALYSIS, RESULT_ANALYSIS, NEXT_ACTION_READY,
                                  BLOCKED, STOPPED }
category=success     → status ∈ { ADOPTED_UNCONFIRMED, ADOPTED, WIN_SUPERSEDED }
category=fail        → status ∈ { ARCHIVED, ARCHIVED_CONDITIONAL }
```

`STOPPED` is a terminal-within-lane state: it requires `terminationMessage` and is exempt from the `activeGate`/`primaryMetricVsGate`/`nextRoute` trio. `DECISION_ADJUDICATION` is a transient active state that keeps the full trio.

Brainstorm is **not** a package category. Pre-package, pre-SSOT ideas live on the dashboard brainstorm
lane (`research_html/data/brainstorms.js`); they become a package only at conversion (`/research-brainstorm`
→ a ratified Direction → `create_from_scope`), which freezes the source idea(s) into the package's
`brainstorm.html` provenance sub-page.

Field requirements key off `(category, status)`:

- `category=in-progress` (except `STOPPED`): requires `activeGate`, `primaryMetricVsGate`, `nextRoute`.
- `category=in-progress`, `status=STOPPED`: requires `terminationMessage`; exempt from the trio above.
- `category=success`: requires `terminationMessage`, `methodsTried`, `adoptionPath`.
- `category=fail`: requires `terminationMessage`, `methodsTried`; `reopenTrigger` iff `status=ARCHIVED_CONDITIONAL`.

Terminal transitions (any status change that crosses a lane boundary) require user ack per Trust rule T1.

## Cross-cutting agent rules

- **Build context first.** Read the invocation, project profile, Scope SSOT, package state, active plan,
  results, docs, and runtime evidence required by the task before work.
- **Use the source-routing model.** Load the SSOT or package witness that owns the decision; use derived
  `research_html` pages for in-context learning, not as mutation targets or final proof.
- **Runtime truth wins.** Validate live runs, logs, outputs, summaries, and artifact roots before changing state. Recalled content is unverified (T3).
- **Consult Learnings before new directions.** Open `research_html/learnings.html` before proposing a new direction, refinement, or experiment idea, and before converting a brainstorm idea into a package.
- **Surgical changes.** Touch only what the task requires. Match existing style. Do not refactor adjacent code.
- **No A0 reproduction by default.** Trust the recorded checkpoint and `AGENTS.md` / `CLAUDE.md` unless the user explicitly asks to revalidate the anchor.
- **All long-running work goes in `tmux`.** Named sessions/windows so the run can be monitored live; report the attach command.
- **ETA discipline.** Do not pre-estimate run duration. Plan rows, launcher manifests, allocation rows, and live-check rows record `est_time=unknown` until the run has executed at least 30 minutes of stable throughput; after that, derive ETA from observed throughput and update on every 10-minute report.

## Per-project customization

A consuming project's CLAUDE.md should prepend (above this file's content) sections for:

- **Project** — one-paragraph description (system, datasets, agent stack).
- **Motivation and Goal** — the central bottleneck the project attacks.
- **Global Optimization Objective** — the primary objective and the success rule (e.g., "metric X must improve under budget Y").
- **Project-specific rules** — non-negotiable dataset / budget / evaluation constraints.
- **Refinement Guardrails — Contribution Spine** — the project's non-negotiable spine components (mirrored into `RESEARCH_PROJECT_PROFILE.cards`).
- **Current Best** — the live anchor record (checkpoint path, metric values, validation seeds).

These project-specific sections are written by the user. The five protocols above stay verbatim.
