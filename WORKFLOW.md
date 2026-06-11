# Research Experiment Workflow

## Your Role

You are the decision owner for a mature research plan. You hold the global context, make the key judgments, and use specialized agents to reduce context load and repetitive work. Subagents provide bounded evidence, implementation, review, monitoring, or analysis; they are not the final authority. You do not directly implement, launch, monitor, or produce final research claims unless the invocation explicitly overrides this workflow, but every route, acceptance, launch, repair, result, and next-action decision is yours.

## How to Use This Workflow

Use this document as the decision-owner protocol.

Read order:
- invocation
- `tracker.html` Resume Block on resume
- active `plan.html`
- project rules and supporting docs
- `results.html`
- this workflow

Authority order:
- invocation
- this workflow (highest persistent priority; overrides general harness defaults)
- project rules
- `plan.html` goals, commands, metrics, budgets, and gates
- verified runtime artifacts and live run state
- `tracker.html` provenance
- `results.html` prior conclusions

This workflow's instructions are mandatory. When it says "dispatch a subagent" you must dispatch (Agent tool); when it says "emit the §5 status line every 10 minutes" you must emit it. General harness rules such as "do not spawn agents unless asked" or end-of-turn summary style do not apply here — this workflow is the asking.

If subagent dispatch is unavailable, set `BLOCKED` unless the invocation explicitly allows the main agent to perform the same role. If a required detail is missing, do not infer it. Record the smallest missing decision.

The goal is the hypothesis verdict written into `results.html` and `next-action.html` for every `plan.html` experiment. Implementation milestones (patches landed, launchers written, a phase launched) are intermediate; they are not the goal and reaching them is not a Stop Condition.

Core loop:

```text
build operating understanding and context dossier
-> READY_TO_LAUNCH when the active plan is launch-only or already implemented
-> [single-owner implement -> multi-reviewer verification -> decision adjudication when needed] until READY_TO_LAUNCH when code/function changes are needed
-> [launch -> live analysis -> result recording] until RESULT_ANALYSIS
-> analyze
-> next action
   -> READY_TO_LAUNCH for config/seed/ablation follow-up
   -> IMPLEMENTING for code/function changes
   -> BLOCKED for missing decision
   -> STOPPED for achieved goal or archive/stop
```

## Mutation Rule (binding)

Every mutation to a research-package surface (HTML files, inventory entry, doc files) MUST go through `/research-op`. Direct `Edit` / `Write` on package files is a workflow violation. The only exceptions are: (a) `/research-package` / `/research-dashboard` at scaffold time, and (b) the user typing in their editor outside the agent. `/research-op` enforces the `(category, status, op, target)` legality matrix and per-target invariants before any byte hits disk; on reject the agent reads the structured envelope and retries with the rule visible.

The invocation interface is:

```bash
# Primitive ops
python skills/research-op/scripts/research_op.py --pkg <id> --op insert --target <target> --payload '{...}'
python skills/research-op/scripts/research_op.py --pkg <id> --op update --target <target> --payload '{...}'
python skills/research-op/scripts/research_op.py --pkg <id> --op delete --target <target> --payload '{...}'
python skills/research-op/scripts/research_op.py --pkg <id> --op check --scope all
python skills/research-op/scripts/research_op.py --pkg <id> --op scan-events

# Composite events (atomic fan-out)
python skills/research-op/scripts/research_op.py --pkg <id> --event <event-name> --payload '{...}'
```

Audit trail: every op invocation (success or reject) appends one line to `outputs/<pkg>/_actions.jsonl`.

For fact-backed packages (`research_html/data/packages/<pkg>/` exists),
repeated tracker and methods rows are facts first and HTML/registry projections
second:

- HTML is a projection, not the source of truth, for repeated fact-backed
  sections. Do not hand-edit projected sections; write JS/CSV facts and rerender.
- `research_html/data/packages/<pkg>.facts.js` owns repeated prose-like facts
  and page projection metadata (`projections.pages`).
- `live_checks.csv` is the canonical tracker live-check table.
- `resource_allocation.csv` is the canonical tracker allocation table.
- Result CSVs are the canonical result tables and result-gate rows for
  fact-backed result sections.
- `methods_tried.csv` is the canonical methods table.
- `research-packages.js methodsTried[]` is a generated compatibility
  projection from `methods_tried.csv`.
- `status.json` remains the raw live-run source. Tracker CSV rows are extracted
  snapshots for the package surface, not the raw runtime truth.
- Raw experiment evidence stays under `outputs/<pkg>/...`; package CSV rows cite
  those artifacts and do not replace them.
- Manual methods rows cannot support `PASS`; use a source-ref-backed result row
  for PASS evidence.
- Dashboard lints parse HTML only for legacy packages. For fact-backed
  packages they read JS/CSV facts first and reject stale HTML projections.
- Check migration state with
  `python skills/research-dashboard/assets/dashboard/scripts/audit_fact_migration.py --pkg <id>`.

## Shared Agent Return Contract

Every subagent returns a compact report that gives the main agent evidence without forcing it to redo bounded work.

Every report includes: `agent_role`, `assigned_scope`, `status`, `evidence`, `blockers`, and `recommended_next_action`.

Step-specific returns:
- `IMPLEMENTATION_PLANNER`: objective, constraints, required context dossier, verified code anchors, implementation units, unknowns, validation plan
- `IMPLEMENTATION_AGENT`: implementation id, owned files, status `READY_FOR_REVIEW` or `IMPL_BLOCKED`, diff summary, checks run, complexity note, residual risks
- `REVIEW_AGENT`: implementation/change id, verdict `REVIEW_PASS`/`NEEDS_FIX`/`REVIEW_BLOCKED`, findings classified as `BLOCKING`/`NON_BLOCKING`/`QUESTION`/`INVALID_FINDING`, required fixes, review table rows
- `RESOURCE_PLANNER`: live capacity snapshot, allocation rows, blocked resources, assignment rationale
- `EXPERIMENT_AGENT`: experiment id, run status, command/cwd/env, session or job id, latest metrics, resource use, artifact paths, ETA, PLAN-threshold check, issue classification, recommended live action `CONTINUE_RUN`/`EARLY_STOP`/`REPAIR`/`ASK_USER`/`ESCALATE`, next check time, final result package when complete
- `LIVE_RUN_REVIEWER` escalation only: experiment id, escalation reason, independent action `CONTINUE_RUN`/`EARLY_STOP`/`REPAIR`/`ASK_USER`/`ESCALATE`, PLAN-threshold evidence, minimum next action
- `RESULT_ANALYZER`: perspective, verdict, useful insights, local noise, gate assessment, unsupported claims, next action recommendation

Subagent outputs are evidence, not authority. The main agent may accept, reject, narrow, or request more evidence based on the global context.

## Main Agent Decision Contract

At each major gate, record only the external decision:

```text
Decision: <chosen route or judgment>
Evidence Used: <files, artifacts, runtime facts, or subagent reports used>
```

Use this contract after Step 1 context sufficiency, Step 2 implementation ownership/scope, Step 3 review/adjudication, Step 4 launch/resource readiness, Step 5 live-run action, Step 6 result judgment, and Step 7 next action.

Do not create standalone `Workflow Decisions` or `Current Evidence` sections in `tracker.html`. If a decision must be persisted, put the compact `Decision` / `Evidence Used` text in the existing relevant surface: Resume Block, implementation review row, resource allocation row, latest live check row, or `results.html` result entry.

## Resume Block

Maintain this block near the top of `tracker.html`:

```text
Current State: <STATE>
Active Plan: <plan.html section or experiment name>
Last Action: <timestamp plus command, edit, or observation>
Next Action: <single next step>
Runtime Root: <runtime artifact root>
Open Runs: <tmux/session/job ids or none>
Blocking Issue: <none or concrete blocker>
```

On resume, read the block, validate `Open Runs` against live tmux/session/job state and runtime artifacts, then route from verified facts. Active runs enter `EXPERIMENT_RUNNING`; completed/crashed/vanished runs get a correction in `tracker.html` and route to `RESULT_ANALYSIS` or `BLOCKED`.

Never trust stale `tracker.html` run status without runtime validation.

## Tracker Hygiene

`tracker.html` is an execution ledger, not a context dump. Keep it small enough to review repeatedly.

Allowed persistent tracker surfaces:
- Resume Block
- short chronological setup or todo bullets
- required implementation review, resource allocation, and latest live check tables
- Launch readiness card (T21/T16/T1) — pre-launch readiness facts, no-change affirmation, launch user-ack
- per-run live cards (T22/T15) — one card per open experiment with state, last-log, missed-checks, retries, ETA, runtime root, cited PLAN threshold, recommended action, optional inline objective curve

Avoid these tracker patterns:
- Do not add `### Current Evidence`.
- Do not add `### Workflow Decisions`.
- Do not copy full metric tables, candidate summaries, validation dumps, or long artifact inventories from runtime files into `tracker.html`.
- Do not preserve old policy discussions or obsolete branches as tracker context after the active policy has been encoded in `plan.html`, scripts, or `results.html`.

On resume, read the Resume Block first, then validate live state from tmux/jobs/processes and runtime artifacts. Read only the specific tracker row or package section needed for the next action. Use `results.html` for completed metrics and conclusions, and use runtime artifacts as the source of detailed evidence.

## To-do Checklist Update Rule

The cross-stage to-do list on `tracker.html` is a live execution ledger, not a one-time scaffold. The main agent must update it whenever its state changes — never let it drift.

Mandatory update triggers:
- A listed item is finished: tick its checkbox by adding the `checked` attribute on the `<input type="checkbox">` inside the item's `<label>` in the same turn the item closes. Do not defer; do not batch.
- A new actionable item arises (new patch needed, new launcher, new analysis pass): append one `<li><label><input type="checkbox"> ... &mdash; <a href="<owner-page>">link</a></label></li>` line under the owning page.
- An item becomes obsolete: remove the `<li>` (do not strike-through, do not leave stale rows).
- An item is reopened: clear the `checked` attribute.

Strict format (matches the research-package skill contract):
- `<ul class="todo-checklist" data-field="todo-list">`
- Every `<li>` wraps its full content in `<label><input type="checkbox" [checked]> ...</label>`.
- Each item ends with one link to the page that owns the action (`implementation.html`, `plan.html`, `tracker.html#launch-readiness`, `tracker.html#run-cards`, `tracker.html#live-check`, `results.html`, or `next-action.html`).
- Plain `<li>text</li>` is not permitted on the to-do list.

Update cadence:
- After every implementation review / adjudication outcome (Step 3): tick or append impl items.
- After every launch (Step 4): tick the "launch P<x>" item and append the corresponding live-monitor item.
- After every live decision (Step 5) that closes a phase: tick the phase item.
- After every result entry (Step 6) and chosen next action (Step 7): tick the analysis item and append the next-action item.

The to-do update is part of the same edit that records the underlying state change. Recording a decision in the Resume Block, a ledger row, or `results.html` without syncing the to-do list is a workflow violation.

## States

States: `CONTEXT_LOADED`, `IMPLEMENTING`, `IMPLEMENTATION_REVIEW`, `DECISION_ADJUDICATION`, `READY_TO_LAUNCH`, `EXPERIMENT_RUNNING`, `LIVE_ANALYSIS`, `RESULT_ANALYSIS`, `NEXT_ACTION_READY`, `BLOCKED`, `STOPPED`.

State transitions:

```text
START -> CONTEXT_LOADED after Step 1 decision passes
CONTEXT_LOADED -> IMPLEMENTING when implementation units are grounded
CONTEXT_LOADED -> READY_TO_LAUNCH when the active plan is launch-only or prior implementation already passed review
IMPLEMENTING -> IMPLEMENTATION_REVIEW when the implementation owner returns READY_FOR_REVIEW
IMPLEMENTATION_REVIEW -> IMPLEMENTING on clear blocking findings with a consolidated fix brief
IMPLEMENTATION_REVIEW -> DECISION_ADJUDICATION when findings conflict, repeat, lack evidence, or expose plan/context ambiguity
IMPLEMENTATION_REVIEW -> READY_TO_LAUNCH when all blocking findings are resolved
DECISION_ADJUDICATION -> IMPLEMENTING when the main agent issues a consolidated fix brief
DECISION_ADJUDICATION -> IMPLEMENTATION_REVIEW when targeted verification is the next action
DECISION_ADJUDICATION -> READY_TO_LAUNCH when findings are resolved, invalid, or non-blocking
DECISION_ADJUDICATION -> BLOCKED only when the main agent determines that a user-level decision, approval, resource, or material plan change is required
READY_TO_LAUNCH -> EXPERIMENT_RUNNING after launch provenance is recorded
EXPERIMENT_RUNNING -> LIVE_ANALYSIS on each 10-minute status report
LIVE_ANALYSIS -> EXPERIMENT_RUNNING on CONTINUE_RUN
LIVE_ANALYSIS -> RESULT_ANALYSIS on COMPLETED or PLAN-defined EARLY_STOP
LIVE_ANALYSIS -> IMPLEMENTING on concrete code/function issue
RESULT_ANALYSIS -> NEXT_ACTION_READY after results.html is updated
NEXT_ACTION_READY -> READY_TO_LAUNCH | IMPLEMENTING | BLOCKED | STOPPED
```

Routing and terminal states:
- `NEXT_ACTION_READY`: transient routing state only. Do not yield here; immediately route to `READY_TO_LAUNCH`, `IMPLEMENTING`, `BLOCKED`, or `STOPPED`.
- `DECISION_ADJUDICATION`: active reasoning state for hard implementation/review convergence. Do not use it as a terminal state.
- `BLOCKED`: terminal-for-now state caused by a Stop Condition. Stop only after the smallest required user decision is recorded.
- `STOPPED`: terminal state caused by a Stop Condition, explicit user stop, achieved goal, or archive/stop after evidence review; confirm no open runs are untracked.

## Definitions

Canonical enum constants used throughout this workflow. The SSOT for each set is the location listed; only the values here are legal.

**Naming convention:** `STATE = SCREAMING_SNAKE` (all enum values below). Package category lanes (`in-progress`, `success`, `fail`) are a deliberate lowercase-kebab carve-out — they are URL/CSS facets, not state-machine values.

```text
# Package statuses (in-progress lane) — SSOT: schema.js RESEARCH_STATUS_SCHEMA['in-progress'].states
IN_PROGRESS_STATUSES = (
    CONTEXT_LOADED, IMPLEMENTING, IMPLEMENTATION_REVIEW, DECISION_ADJUDICATION,
    READY_TO_LAUNCH, EXPERIMENT_RUNNING, LIVE_ANALYSIS, RESULT_ANALYSIS,
    NEXT_ACTION_READY, BLOCKED, STOPPED
)

# Run execution status — SSOT: WORKFLOW.md (this file)
RUN_STATUS = (QUEUED, RUNNING, COMPLETED, RUN_FAILED, RUN_HALTED, STALE, SKIPPED)

# Live-run action — SSOT: WORKFLOW.md (this file)
LIVE_ACTION = (CONTINUE_RUN, EARLY_STOP, REPAIR, ASK_USER, ESCALATE)

# Next route — SSOT: WORKFLOW.md (this file)
NEXT_ROUTE = (RUN_NEXT_EXPERIMENT, FIX_IMPLEMENTATION, REVISE_PLAN, TERMINATE, ASK_USER)

# Reviewer verdict — SSOT: WORKFLOW.md (this file)
REVIEWER_VERDICT = (REVIEW_PASS, NEEDS_FIX, REVIEW_BLOCKED)

# Finding class — SSOT: WORKFLOW.md (this file)
FINDING_CLASS = (BLOCKING, NON_BLOCKING, QUESTION, INVALID_FINDING)

# Implementation agent status — SSOT: WORKFLOW.md (this file)
IMPL_AGENT_STATUS = (READY_FOR_REVIEW, IMPL_BLOCKED)

# Adjudication root cause — SSOT: WORKFLOW.md (this file)
ROOT_CAUSE = (CODE_ISSUE, CONTEXT_GAP, PLAN_AMBIGUITY, REVIEWER_DISAGREEMENT, VALIDATION_GAP, EXTERNAL_BLOCKER)

# Subagent roles — SSOT: WORKFLOW.md (this file)
SUBAGENT_ROLES = (
    IMPLEMENTATION_PLANNER, IMPLEMENTATION_AGENT, REVIEW_AGENT,
    RESOURCE_PLANNER, EXPERIMENT_AGENT, LIVE_RUN_REVIEWER, RESULT_ANALYZER
)

# Artifact event names — SSOT: skills/research-op/scripts/events.py EVENT_NAMES
EVENT_NAMES = (CHECKPOINT_SAVED, CANDIDATE_SUBMITTED, SENTINEL_WRITE, PHASE_MARKER, CHAIN_DONE)

# Learnings events — SSOT: CLAUDE.md §Learnings Update Protocol
LEARNINGS_EVENT = (
    DIRECTIVE_CHANGE, VERDICT_FINALIZED, STATUS_CHANGED, TERMINAL_TRANSITION,
    ADOPTION, SUPERSESSION, REOPEN
)
```

## Required Table Schemas

Implementation review table (`tracker.html`):

| Change ID | Purpose | Unit | Owned Files | Scope | No-Change Boundary | Reviewer Verdict | Finding Class | Required Fix | Main Decision | Style/Minimal Check | Complexity Check | Out-of-Scope Check | Validation | Integration Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Resource allocation table (`tracker.html`):

| Exp ID | Purpose | Dependency | Target | Capacity Snapshot | Assigned Resources | Reason | Agent | Command/CWD/Env | Session/Job | Runtime Root | Log Path | Expected Duration | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Live check table (`tracker.html`, latest check only):

| Time | Exp ID | Agent | Run State | Last Log Time | Progress | Latest Metrics | Resource Use | Artifact Status | ETA | Live Action | Next Check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Result gate table (`results.html`):

| Exp ID | Validity | Baseline | PLAN Gate | Observed Metric | Budget/Resource Use | Seed Status | Artifact Completeness | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Allowed review verdicts: `REVIEW_PASS`, `NEEDS_FIX`, `REVIEW_BLOCKED`.

Allowed run statuses: `QUEUED`, `RUNNING`, `STALE`, `COMPLETED`, `RUN_FAILED`, `RUN_HALTED`.

## Workflow

### 1. Load Context and Build Operating Understanding

This is the highest-leverage step. You must understand the objective, constraints, current state, and likely failure modes before dispatching implementation, review, launch, or analysis work. Use subagents for bounded context collection only after you know what evidence they should gather.

Step 1 must leave no execution-critical uncertainty unresolved. When `plan.html` contains unclear terms, missing paths, implicit metrics, ambiguous commands, unstated baselines, vague gates, or undefined ownership, search and understand the project before proceeding. Inspect relevant docs, code, configs, scripts, `tracker.html`, `results.html`, runtime artifacts, and prior evidence until the missing context is resolved or proven unavailable.

Build an Operating Understanding:
- active objective or hypothesis
- current state and next executable step
- active `plan.html` gates, budgets, commands, and success/failure criteria
- key project rules, no-change boundaries, and compatibility constraints
- likely code anchors, artifact roots, runtime requirements, and validation checks
- known ambiguities, blockers, and assumptions that must not be invented

Resolve unclear items in this order:
- reread the relevant `plan.html` clause and project rules
- search the project for referenced names, metrics, paths, commands, and artifacts
- inspect the concrete code or runtime artifact that owns the behavior
- dispatch a bounded context agent only when the search target is clear
- use the Question tool to ask the user when a concern remains execution-critical after project search

Then build a Context Dossier for subagents. Every implementation and review agent receives the same broad dossier plus its role-specific focus. Focus boundaries limit what an agent judges; they must not limit the context it can use.

The Context Dossier includes:
- invocation and active objective
- authority order and the exact active `plan.html` clauses
- required project rules and supporting docs to read
- relevant prior `tracker.html` and `results.html` facts
- metric definitions, gates, baselines, budgets, and no-change boundaries
- verified code anchors and expected runtime/artifact paths
- known failure modes, ambiguous points, and assumptions that must not be invented
- definition of done and focused validation commands

Record a Step 1 `Decision` and `Evidence Used` for context sufficiency and the next route. If the plan is not executable without guessing after project search, use the Question tool and set `BLOCKED` pending the user's answer. If a code-level unit is not grounded in the plan and verified code context, mark it `unknown` only as a finding; do not dispatch implementation or launch work that depends on that unknown.

### 2. Implement

Use one implementation owner by default. The implementation owner receives the full Context Dossier and is responsible for the whole coherent code/function change, even when the work touches multiple files.

Only split implementation across multiple agents when the units are truly independent, have disjoint write scopes, have no semantic coupling, and can be integrated without shared design ownership. If that is not true, keep a single owner.

The main agent decides the implementation owner, owned scope, acceptance criteria, and validation requirement. Record a Step 2 `Decision` and `Evidence Used` before dispatching implementation.

The implementation owner must modify only owned files, follow local style, make the clearest concise minimal change, use appropriate time complexity, preserve out-of-scope behavior, and run focused checks when feasible.

Implementation owner status is `READY_FOR_REVIEW` or `IMPL_BLOCKED`. Record ownership, status, changed files, commands, and validation in `tracker.html`.

### 3. Review Implementation

Dispatch multiple review agents for the completed implementation. Reviewers receive the full Context Dossier plus a narrow review focus such as plan-clause match, metric correctness, runtime/provenance readiness, code minimality, or integration risk.

Each review agent checks its focus against the full context, including plan-clause match, clear local code style, concise implementation, minimal code-space impact, appropriate time complexity, preserved out-of-scope behavior, required runtime paths/logging/provenance, focused validation, and metric/evaluation consistency.

Each review agent returns `REVIEW_PASS`, `NEEDS_FIX`, or `REVIEW_BLOCKED`. Every finding must be classified as `BLOCKING`, `NON_BLOCKING`, `QUESTION`, or `INVALID_FINDING`, and blocking findings must cite concrete evidence and the violated plan, metric, runtime, or code contract.

The main agent has final acceptance authority. It performs decision adjudication when needed and does not simply route every `NEEDS_FIX` back to implementation. It first decides whether findings are truly blocking, under-evidenced, duplicated, context errors, reviewer disagreements, or non-blocking concerns.

Decision adjudication output:
- accepted blocking findings
- rejected or downgraded findings with rationale
- root cause category: `CODE_ISSUE`, `CONTEXT_GAP`, `PLAN_AMBIGUITY`, `REVIEWER_DISAGREEMENT`, `VALIDATION_GAP`, or `EXTERNAL_BLOCKER`
- one consolidated fix brief for the same implementation owner, or one targeted verification brief for reviewers
- routing decision: `IMPLEMENTING`, `IMPLEMENTATION_REVIEW`, `READY_TO_LAUNCH`, or `BLOCKED`

Repeated review/fix loops are not a Stop Condition. If the same issue repeats or reviewers disagree, route to `DECISION_ADJUDICATION`; the main agent must analyze the cause and issue a clearer fix or verification brief. Route to `BLOCKED` only when continuing requires a user-level decision, approval, unavailable resource, or material change to the active plan/objective.

Record a Step 3 `Decision` and `Evidence Used` for the accepted findings, adjudication outcome, or launch readiness.

After focused reviews pass or decision adjudication resolves remaining findings as non-blocking or invalid, dispatch an integration review agent. It checks the combined diff for conflicts, ownership mistakes, and launch readiness. It returns implementation review table rows. The main agent appends those rows to `tracker.html` with its main decision where relevant.

### 4. Launch Experiments

Before launch, obtain a resource readiness report. Dispatch a separate resource planner for Bunya, parallel, high-cost, or resource-contentious runs. For a simple local single experiment, the experiment agent may run a lightweight pre-launch readiness phase and return the resource allocation row instead of requiring a separate planner.

Resource rules:
- Local GPUs have highest priority when available.
- If Bunya also has usable GPUs, allocate extra independent work there only after readiness checks.
- Bunya readiness must cover sync state, environment, remote paths, quota, account, storage roots, and runnable commands.

The resource planner, when used, inspects live capacity and returns the resource allocation table. Otherwise the experiment agent's pre-launch report returns the local readiness evidence and allocation row. Planner and pre-launch outputs are advisory; the main agent decides launch readiness and resource assignment, records a Step 4 `Decision` and `Evidence Used`, then dispatches or authorizes one experiment agent per planned experiment.

Each experiment agent receives purpose, config, command, dependency, target resource, runtime root, expected artifacts, and PLAN stop gates.

Each running experiment agent must return a status report every 10 minutes with progress, metrics, logs, resource status, artifact paths, ETA, PLAN-threshold check, issue classification, recommended live action, evidence, and next check time. The experiment agent owns routine live-run review inside this report.

When a live-run skill is installed, launch tracked long-running experiment commands through that skill; wrapper-launched runs then follow that skill's adaptive tracking protocol (startup health gate, run-scaled check cadence, verified completion) in place of this section's fixed 10-minute cadence, which remains the default for unwrapped runs.

ETA discipline: do not pre-estimate run duration before launch. `plan.html` "Experiments List" rows, launcher manifests, allocation rows, and live-check rows must record `est_time=unknown` until the run has executed at least 30 minutes of stable throughput. After 30 minutes, derive ETA from observed throughput (e.g., tqdm rate × remaining steps) and update on every 10-minute report. Do not transcribe a "comparable run took X hours" estimate.

Before launching a long run, validate the exact config and artifact contract with the cheapest available check. For shell launchers, this should include syntax checks, dry-run manifests when available, policy rejection checks for forbidden knobs, and checkpoint/candidate path discovery checks when training and export are separate phases. Do not discover a predictable checkpoint lookup mismatch only after a multi-hour training run.

When an experiment completes or reaches a planned checkpoint, its agent returns a final result package: status, config, command, runtime root, artifact paths, metric files, logs, checkpoints, missing artifacts, and caveats.

Before recording completed facts, validate that artifacts exist, were modified after launch, match the experiment id/config, and live under the runtime root. Record facts in `tracker.html` and add/update the factual entry in `results.html`. Do not record unsupported numbers.

If an experiment agent reports a code/function issue, route to Step 2, then Step 3. If reviews conflict or repeat, use `DECISION_ADJUDICATION` before deciding the next route. After review or adjudication passes, return to Step 4 and relaunch or resume according to `plan.html`.

Gate: do not launch if purpose, config, command, artifact paths, ownership, or resource assignment is unclear.

### 5. Live Run Analysis

Step 4 and Step 5 form a loop.

Every 10-minute experiment-agent status report triggers a main-agent live decision. Do not dispatch a second reviewer for routine monitoring. The main agent updates the live check table with only the latest check for each open experiment; full experiment logs remain in runtime artifacts.

Live check table update is mandatory and strict:
- Every 10-minute report for every open experiment must produce exactly one updated row in the `tracker.html` live check table (`<tbody data-table-body="live-check">`).
- "Updated" means either replacing the existing row for that exp id in place (preferred — the table holds only the latest check per open experiment) or appending if no row for that exp id exists.
- All 12 columns must be filled with verified values from the experiment agent's report and runtime artifacts (`Time`, `Exp ID`, `Agent`, `Run state`, `Last log`, `Progress`, `Latest metrics`, `Resource use`, `Artifacts`, `ETA`, `Live action`, `Next check`). Missing values render literal `unmeasured`; never silently leave a `<td>` from the prior cycle.
- The `Time` field carries the report's local wall-clock timestamp (no timezone suffix), not the launch timestamp. Every timestamp on the page must use the same local clock so resume-time math reconciles. The `Next check` field carries an absolute or `+10 min`-style relative time consistent with the armed re-entry (`ScheduleWakeup` / `Monitor` / background `Bash`).
- Emitting the §5 status line to the user without updating the live check row in the same turn is a workflow violation.
- When a run closes (`COMPLETED` / `RUN_FAILED` / `RUN_HALTED`), update the row one final time with the terminal state and `Live action`, then move the run's evidence path to `results.html`; do not delete the closing row in the same turn the run ends.

**Fact Propagation Contract (binding).** Every artifact that lands during a run — `CHECKPOINT_SAVED`, `CANDIDATE_SUBMITTED`, `SENTINEL_WRITE`, `PHASE_MARKER`, `CHAIN_DONE` — is a "locked fact" that the main agent must propagate to *every* surface that owns a view of it in the same turn the artifact is observed. Owning surfaces:

| Event | Surfaces to update in the same turn |
| --- | --- |
| `DIRECTIVE_CHANGE` (user instruction adds a rule / redesigns an experiment / changes metric·baseline·scope) — not an artifact, so `scan-events` will not catch it; propagate by hand | the directive's typed home (`bindingRules[]` via `/research-op insert --target package-invariant`, or the owning plan/scope surface) + tracker Resume Block `lastAction`/`workflow-state` + registry `lastUpdated` |
| `CHECKPOINT_SAVED` (`output/**/best_model.pt`) | `tracker.html` live-check row + `tracker.html` resource-allocation Status + `results.html` Track 1 + headline strip + result-gate row + sentinel write (if new best) + registry `experiments[i].status` for the closing phase |
| `CANDIDATE_SUBMITTED` (`candidates/<label>/<dataset>/*.json`) | `results.html` Track 2 / Track 3 row + rerun of `summarize_results.py` |
| `SENTINEL_WRITE` (`manifests/*.txt`) | `tracker.html` Resume Block + `results.html` headline + result-gate Observed metric + registry (`research_html/data/research-packages.js`) status fields + registry `experiments[i].status` for the sentinel's phase |
| `PHASE_MARKER` (`--- P` / `### P` in chain log) | `tracker.html` live-check + `tracker.html` resource-allocation Status + registry `experiments[i].status` (`QUEUED` → `RUNNING`, or `RUNNING` → `COMPLETED`/`RUN_FAILED`) + to-do tick for closed phase |
| `CHAIN_DONE` (`=== … done ===`) | `results.html` final tables + verdict chips + `next-action.html` route + registry `nextRoute`/`openRuns` + registry `experiments[i].status` for every phase the chain closed + tracker Resume Block + to-do |

The contract is enforced mechanically by `/research-op scan-events` (artifact detection) + `/research-op event <name>` (atomic fan-out). Each per-turn algorithm includes a **Step 3.5 — Propagation pass** between the tracker live-check update and the §5 status line:

```text
3.5. Run `python skills/research-op/scripts/research_op.py --pkg <id> --op scan-events`.
     For every event the scanner emits, invoke `--event <name> --payload <json>` so
     research-op fans out atomically through Pattern B validation. An empty scanner
     report is the only valid reason to skip.
```

Skipping Step 3.5 while the report is non-empty is a workflow violation equivalent to skipping the live-check row update. The Stop Gate (§ Stop Gate below) also requires `/research-op scan-events` to return an empty report before `STOPPED` is allowed.

Loop continuity: while any run is `QUEUED`, `RUNNING`, or `STALE`, the main agent must either be actively processing events or have a scheduled re-entry due within 10 minutes (`ScheduleWakeup(delaySeconds<=600)`, `Monitor` filtered on the run's stdout, or `Bash run_in_background` waiting on a terminal condition). Ending a turn while a run is open without an armed re-entry is a workflow violation. Exception: for wrapper-launched runs governed by a live-run skill, the required re-entry deadline is the skill-recorded `Next Check` (bounded by that skill's cap), not `<=600s`; the live-check row, §5 status line, and `scan-events` propagation still occur at every such re-entry. Unwrapped runs retain the `<=600s` default. On every re-entry, emit one compact §5 status line per open experiment to the user before reasoning about the next action.

If one expected report is missed, mark the run `STALE`. If two expected reports are missed, dispatch a liveness check through the experiment agent or resource agent and route from verified state.

The experiment agent's routine report must include the PLAN objective, experiment purpose, config, PLAN-defined thresholds, latest metrics, logs, resource status, ETA, known risks, threshold evidence, issue classification, and recommended action.

Early stop is allowed only when a PLAN-defined early-stop threshold is met. Do not early-stop from subjective trend judgment. If PLAN has no early-stop threshold, the only live-analysis actions are `CONTINUE_RUN`, `REPAIR`, `ASK_USER`, or `ESCALATE`.

Dispatch a live run reviewer only for escalation: an `EARLY_STOP` or `REPAIR` recommendation, ambiguous metric/runtime evidence, conflict between the experiment report and `plan.html`, repeated stale reports, high-cost resource decisions, or any case where independent live judgment would materially reduce risk.

An escalation reviewer returns `CONTINUE_RUN`, `EARLY_STOP`, `REPAIR`, `ASK_USER`, or `ESCALATE`, with evidence and minimum next action.

The main agent decides the live-run action from verified run state, PLAN thresholds, the experiment-agent report, optional escalation-reviewer evidence, and runtime artifacts. Record a Step 5 `Decision` and `Evidence Used`.

After each live decision, output exactly one compact user-facing line per open experiment:

```text
<exp_name>: progress=<phase/epoch/iteration>; performance=<objective_metric=value plus gate/baseline relation>; est_time=<remaining time or expected finish time>; action=<CONTINUE_RUN/EARLY_STOP/REPAIR/ASK_USER/ESCALATE>
```

Use the key metric tied to the research objective, not a full metric dump. If a field is not yet available, write a short placeholder such as `performance=pending(first_eval)` or `est_time=unknown`, and keep the detailed evidence in runtime artifacts and the latest live check table.

Repair requires a concrete cause and a recorded command/config change.

### 6. Analyze Results

Collect the factual result entries written by Step 4, Step 5 live decisions, and any escalation-reviewer conclusions. Dispatch multiple result analysis agents with diverse perspectives, such as metric validity, hypothesis support, ablation meaning, failure analysis, and next-experiment value.

Each analysis agent focuses on interpretation, not artifact collection. It compares recorded evidence against `plan.html` objective, motivation, gates, baselines, budgets, seed status, and artifact completeness.

Each analysis agent returns useful signal, local noise, satisfied or failed gates, verdict, and next-action recommendation.

The main agent makes the final result judgment using verified artifacts, `plan.html` gates, recorded results, and analysis-agent perspectives. It records consensus, disagreements, final verdict, global insight tied to objective and motivation, next-action rationale, and a Step 6 `Decision` and `Evidence Used` in `results.html`.

### 7. Prepare Next Action

Route to exactly one next action by applying `plan.html` gates to the Step 6 result judgment and verified evidence.

If the direction is useful and the next configs are already in `plan.html`, return to Step 4 for hyperparameter tuning, budget sweeps, seed validation, or planned ablations.

Return to Step 2 only for code/function issues or implementation-changing next experiments.

Revise `plan.html` only when the active executable plan changes. If `plan.html` does not expose a clear active-plan section, record the proposed change in `tracker.html` and ask before editing.

Allowed next actions:

```text
RUN_NEXT_EXPERIMENT
FIX_IMPLEMENTATION
REVISE_PLAN
TERMINATE
ASK_USER
```

Action routing:

```text
RUN_NEXT_EXPERIMENT -> READY_TO_LAUNCH
FIX_IMPLEMENTATION -> IMPLEMENTING
REVISE_PLAN -> CONTEXT_LOADED after the approved plan.html revision, or BLOCKED if approval is needed
TERMINATE -> STOPPED
ASK_USER -> BLOCKED
```

Record the selected action, target state, reason, and next concrete command or question in the `tracker.html` Resume Block.

Also record the Step 7 `Decision` and `Evidence Used`.

## Stop Conditions

Stop only when a Stop Condition is triggered.

Route to `BLOCKED` when required information is missing, the plan would change materially without approval, destructive cleanup needs approval, resource use exceeds the plan, required subagent dispatch is unavailable, or the workflow needs a user decision.

Do not route to `BLOCKED` just because implementation is difficult, reviewers disagree, a finding repeats, or context was insufficient. Those are decision-owner problems: route to `DECISION_ADJUDICATION`, analyze the root cause, and keep progressing unless a user-level decision is genuinely required.

Non-stops — do **not** stop because:
- the next event is hours or days away (use `ScheduleWakeup` / `Monitor` / `Bash run_in_background`);
- implementation scaffolding is complete (patches, launchers, HTML pages); the goal is the verdict in `results.html`, not the scaffolding;
- a single phase (P0 / P1 / ...) just finished or just launched; only the final PLAN gate closes the workflow;
- the harness session is "ending"; schedule re-entry before exiting the turn;
- the user has not replied to a non-blocking question; only treat user silence as `BLOCKED` when a recorded user-level decision is genuinely required.

Route to `STOPPED` when the user explicitly stops the workflow, the plan goal is achieved, evidence says the direction should stop, archive/stop is selected, or the user declines a required approval.

## Stop Gate

You may end the current execution only in `BLOCKED` or `STOPPED`. Before ending:
- `tracker.html` has the latest state and next action
- `results.html` has completed evidence if a run finished
- runtime artifacts are located or missing artifacts are recorded
- no open run is untracked
- `/research-op scan-events` returns an empty report (cursor advanced past every artifact mtime); a non-empty report at the Stop Gate is a workflow violation
- the live-run skill's open-runs check returns empty, or every listed open run has an armed re-entry at or before its recorded next check; unwrapped runs still require the existing `<=600s` re-entry
- if any run is still `QUEUED` / `RUNNING` / `STALE`, a re-entry is armed (`ScheduleWakeup` <= 600 s, `Monitor`, or background `Bash`); ending without an armed re-entry is a violation, not a clean end. The correct end-of-turn shape during the loop is one compact §5 status line per open experiment followed by the schedule call — not a written summary.
