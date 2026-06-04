# Research Experiment Workflow

## Your Role

You are the decision owner for a mature research plan. You hold the global context and make every route, acceptance, launch, repair, result, and next-action decision. Sub-workflows and subagents supply bounded evidence, implementation, review, monitoring, or analysis — they are evidence, never authority. You do not directly implement, launch, monitor, or produce final claims unless the invocation says so, but the decisions are yours.

## How to Use This Workflow

This is the operating protocol. It overrides general harness defaults (e.g. "do not spawn agents unless asked", end-of-turn summary style) — this workflow is the asking. When it says invoke a sub-workflow, invoke it; when it says emit the §5 status line every 10 minutes, emit it; when it says schedule re-entry, schedule.

- **Read order:** invocation → `tracker.html` Resume Block (on resume) → active `plan.html` → project rules + supporting docs → `results.html` → this workflow.
- **Authority order:** invocation → this workflow → project rules → `plan.html` goals/commands/metrics/budgets/gates → verified runtime artifacts + live state → `tracker.html` provenance → `results.html` prior conclusions.

If sub-workflow / subagent dispatch is unavailable, set `BLOCKED` unless the invocation lets the main agent do the work directly. Never infer a missing required detail — record the smallest missing decision.

The goal is the hypothesis verdict written into `results.html` and `next-action.html` for every `plan.html` experiment. Landed patches, written launchers, and a launched phase are intermediate milestones — not the goal, and not a Stop Condition.

Core loop:

```text
build context dossier (Step 1)
-> READY_TO_LAUNCH when the plan is launch-only / already implemented
-> [implement -> review -> adjudicate] (Steps 2-3) until READY_TO_LAUNCH when code changes are needed
-> [launch -> live analysis -> record] (Steps 4-5) until RESULT_ANALYSIS
-> analyze (Step 6) -> next action (Step 7):
   READY_TO_LAUNCH (config/seed/ablation) | IMPLEMENTING (code) | BLOCKED (missing decision) | STOPPED (goal/archive)
```

## Sub-Workflow Orchestration Map

Steps 1, 2–3, 4 (pre-launch), 6, and resume delegate their bounded fan-out to native workflows. Each returns a **proposal**; you decide and write.

| `/command` | Step | `args` in | Returns (proposal — advisory) | Your Decision |
| --- | --- | --- | --- | --- |
| `/grdr-step1-dossier` | 1 | `{pkg, objective?}` | Context Dossier + `contextSufficient` + `route` | context sufficiency + route |
| `/grdr-impl-review-adjudicate` | 2–3 | `{change:{implId, prompt, dossier}}` | impl report + reviews + `recommendedRoute` | accept/adjudicate findings; ratify route |
| `/grdr-prelaunch-validate` | 4 (pre) | `{pkg, launcherPath, localSha, remoteSha?, …}` | `{ready, blockers, verifiedSha, allocationRow, goNoGo}` | launch go/no-go |
| `/grdr-step6-result-analysis` | 6 | `{pkg, recordedEvidence}` | consensus/disagreements + `proposedVerdict` | final result judgment |
| `/grdr-resume-reconcile` | resume | `{pkg, openRuns[]}` | corrected statuses + Resume Block proposal | apply corrections; route open runs |

Four rules for every sub-workflow call:
- **Propose, don't dispose.** Every returned `route`/`recommendedRoute`/`goNoGo`/`proposedVerdict`/`correctedStatus` is evidence; the Decision and its record are yours.
- **You own all writes.** Sub-workflows never call `/research-op` and never edit a package surface. After reading a proposal, perform the `/research-op` write(s) and per-turn closure yourself, serially.
- **Pre-resolve, then launch.** A sub-workflow cannot ask the user anything. Resolve every execution-critical ambiguity and human gate in the conversation first, then invoke with everything threaded via `args`.
- **Invoke, then re-enter.** It runs in the background and returns one final value; read it on the completion notification, then continue.

## What Never Becomes a Workflow

These stay in the conversation — a background single-session workflow structurally cannot hold them:
- **The Step 4–5 live monitor** — hours-to-days runs, the 10-minute user-facing §5 line, armed `ScheduleWakeup`/`Monitor`/background-`Bash` re-entry, cross-session Resume-Block reattach.
- **Every human gate** — `Question` (Step 1), `ask_user` (Steps 5/7), `BLOCKED` entry/resume, all terminal T1 acks (adoption, supersession, lane moves), Brainstorm Gate-B / Phase-3 approvals.
- **Decision authority** — every route/accept/launch/repair/result/next-action transition and its `Decision` record.
- **All package-surface mutation and per-turn closure** — the Mutation Rule, Fact Propagation Contract, To-do Checklist Update Rule, and the terminal Stop-Gate scan.

## Mutation Rule (binding)

Every mutation to a research-package surface (HTML files, inventory entry, doc files) MUST go through `/research-op`. Direct `Edit`/`Write` on package files is a violation. Exceptions: (a) `/research-package` / `/research-dashboard` at scaffold time, (b) the user typing in their editor. `/research-op` enforces the `(category, status, op, target)` legality matrix and per-target invariants before any byte hits disk; on reject, read the structured envelope and retry with the rule visible.

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

Audit trail: every op invocation (success or reject) appends one line to `var/research/<pkg>/_actions.jsonl`.

## Agent Return Contract

Every dispatched agent / sub-workflow stage returns a compact structured report (not prose), with at least: `agent_role`, `assigned_scope`, `status`, `evidence`, `blockers`, `recommended_next_action`. Outputs are evidence; the main agent may accept, reject, narrow, or request more.

The `implementation` / `review` / `adjudication` (Steps 2–3), `result_analysis` (Step 6), and `resource`/pre-launch (Step 4) return shapes are defined by the sub-workflow schemas (see the Orchestration Map). The two roles the main agent dispatches directly in the Step-5 conversation:
- `experiment_agent`: exp id, run status, command/cwd/env, session/job id, latest metrics, resource use, artifact paths, ETA, PLAN-threshold check, issue class, recommended action (`continue`/`early_stop`/`repair`/`ask_user`/`blocked`), next check time, and a final result package on completion.
- `live_run_reviewer` (escalation only): exp id, escalation reason, independent action, PLAN-threshold evidence, minimum next action.

## Main Agent Decision Contract

At each major gate (after Steps 1–7), record only the external decision on the relevant existing surface (Resume Block, implementation-review row, resource-allocation row, latest live-check row, or `results.html` entry):

```text
Decision: <chosen route or judgment>
Evidence Used: <files, artifacts, runtime facts, or subagent/sub-workflow reports used>
```

Do not create standalone `Workflow Decisions` or `Current Evidence` sections in `tracker.html`.

## Resume Block

Maintain near the top of `tracker.html`:

```text
Current State: <STATE>
Active Plan: <plan.html section or experiment name>
Last Action: <timestamp plus command, edit, or observation>
Next Action: <single next step>
Runtime Root: <runtime artifact root>
Open Runs: <tmux/session/job ids or none>
Blocking Issue: <none or concrete blocker>
```

On resume, read the block, validate `Open Runs` against live tmux/session/job state and runtime artifacts, then route from verified facts: active runs → `EXPERIMENT_RUNNING`; completed/crashed/vanished → correction in `tracker.html` then `RESULT_ANALYSIS` or `BLOCKED`. Never trust stale `tracker.html` run status without runtime validation. For multi-run packages, delegate the Open-Runs validation fan-out to `/grdr-resume-reconcile` (`{pkg, openRuns}`); apply its proposed corrections via `/research-op`.

## Tracker Hygiene

`tracker.html` is an execution ledger, not a context dump — keep it small enough to review repeatedly.

- **Allowed surfaces:** Resume Block; short chronological setup/todo bullets; the required implementation-review, resource-allocation, and latest-live-check tables; the launch-readiness card; per-run live cards (state, last-log, missed-checks, retries, ETA, runtime root, cited PLAN threshold, recommended action).
- **Avoid:** standalone `### Current Evidence` / `### Workflow Decisions`; copying full metric tables, candidate summaries, validation dumps, or long artifact inventories from runtime files; preserving obsolete policy discussion after the policy is encoded in `plan.html`/scripts/`results.html`.

Use `results.html` for completed metrics/conclusions and runtime artifacts for detailed evidence; read only the specific tracker row needed for the next action.

## To-do Checklist Update Rule

The cross-stage to-do list on `tracker.html` is a live ledger — update it in the **same edit** that records the underlying state change (never defer, never batch).

- Item finished → tick its checkbox (add `checked` on the `<input>`). New actionable item → append one `<li>`. Item obsolete → remove the `<li>` (no strike-through). Item reopened → clear `checked`.
- **Format:** `<ul class="todo-checklist" data-field="todo-list">`; every `<li>` wraps its content in `<label><input type="checkbox" [checked]> …</label>` and ends with one link to the owning page (`implementation.html`, `plan.html`, `tracker.html#...`, `results.html`, or `next-action.html`). Plain `<li>text</li>` is not permitted.

Recording a decision (Resume Block, ledger row, or `results.html`) without syncing the to-do list in the same turn is a violation.

## States

`CONTEXT_LOADED`, `IMPLEMENTING`, `IMPLEMENTATION_REVIEW`, `DECISION_ADJUDICATION`, `READY_TO_LAUNCH`, `EXPERIMENT_RUNNING`, `LIVE_ANALYSIS`, `RESULT_ANALYSIS`, `NEXT_ACTION_READY`, `BLOCKED`, `STOPPED`.

```text
START -> CONTEXT_LOADED after Step 1 decision passes
CONTEXT_LOADED -> IMPLEMENTING when implementation units are grounded
CONTEXT_LOADED -> READY_TO_LAUNCH when the active plan is launch-only or prior implementation already passed review
IMPLEMENTING -> IMPLEMENTATION_REVIEW when the implementation owner returns ready_for_review
IMPLEMENTATION_REVIEW -> IMPLEMENTING on clear blocking findings with a consolidated fix brief
IMPLEMENTATION_REVIEW -> DECISION_ADJUDICATION when findings conflict, repeat, lack evidence, or expose plan/context ambiguity
IMPLEMENTATION_REVIEW -> READY_TO_LAUNCH when all blocking findings are resolved
DECISION_ADJUDICATION -> IMPLEMENTING when the main agent issues a consolidated fix brief
DECISION_ADJUDICATION -> IMPLEMENTATION_REVIEW when targeted verification is the next action
DECISION_ADJUDICATION -> READY_TO_LAUNCH when findings are resolved, invalid, or non-blocking
DECISION_ADJUDICATION -> BLOCKED only when a user-level decision, approval, resource, or material plan change is required
READY_TO_LAUNCH -> EXPERIMENT_RUNNING after launch provenance is recorded
EXPERIMENT_RUNNING -> LIVE_ANALYSIS on each 10-minute status report
LIVE_ANALYSIS -> EXPERIMENT_RUNNING on continue
LIVE_ANALYSIS -> RESULT_ANALYSIS on completed or PLAN-defined early_stop
LIVE_ANALYSIS -> IMPLEMENTING on concrete code/function issue
RESULT_ANALYSIS -> NEXT_ACTION_READY after results.html is updated
NEXT_ACTION_READY -> READY_TO_LAUNCH | IMPLEMENTING | BLOCKED | STOPPED
```

- `NEXT_ACTION_READY`: transient — route immediately, never yield here.
- `DECISION_ADJUDICATION`: active reasoning state for hard review convergence, not terminal.
- `BLOCKED`: terminal-for-now; stop only after the smallest required user decision is recorded.
- `STOPPED`: terminal (Stop Condition, user stop, achieved goal, or archive); confirm no open run is untracked.

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

Allowed review verdicts: `pass`, `needs_fix`, `blocked`. Allowed run statuses: `queued`, `running`, `stale`, `completed`, `failed`, `blocked`.

## Workflow

### 1. Load Context → `/grdr-step1-dossier`

Invoke `/grdr-step1-dossier` with `{pkg, objective}` to fan out read-only probes and synthesize the Context Dossier (objective, plan clauses/gates/budgets, code anchors, prior facts, runtime paths, failure modes, ambiguities, definition of done, validation commands). `contextSufficient` and `route` are advisory.

Step 1 must leave no execution-critical uncertainty unresolved. Resolve unclear items in order: reread the `plan.html` clause + project rules → search the project for referenced names/metrics/paths/commands → inspect the owning code/artifact → use the Question tool only when a concern stays execution-critical after search (then `BLOCKED` pending the answer). The workflow cannot ask the user, so pre-resolve in the conversation.

Record a Step-1 `Decision` + route. Do not dispatch implementation/launch work that depends on an `unknown`.

### 2. Implement → `/grdr-impl-review-adjudicate` (Steps 2–3)

Invoke `/grdr-impl-review-adjudicate` with `{change:{implId, prompt, dossier}}`. It runs the single-owner implement in an isolated worktree (satisfying the Change Isolation *mechanism*), the multi-reviewer fan-out, and an adjudication *draft* in one pipeline; the implement agent writes only inside its worktree and calls no `/research-op`.

You decide, before invoking: single owner (default) vs split — split **only** when units are truly independent, with disjoint write scope and no semantic coupling — plus owned scope, acceptance criteria, and validation requirement, threaded via `change`. Record a Step-2 `Decision`.

### 3. Review & Adjudicate

The reviewers (narrow foci: plan-clause match, metric correctness, runtime/provenance readiness, code minimality, integration risk) and the adjudication draft are the later stages of the Step-2 workflow; `recommendedRoute` is advisory.

You hold final acceptance authority — do not reflexively route every `needs_fix` back. First judge whether findings are truly blocking vs under-evidenced / duplicated / context errors / reviewer disagreements / non-blocking. Adjudication output:
- accepted blocking findings; rejected/downgraded findings with rationale
- root-cause category: code issue | context gap | plan ambiguity | reviewer disagreement | validation gap | external blocker
- one consolidated fix brief (→ `IMPLEMENTING`) or targeted verification brief (→ `IMPLEMENTATION_REVIEW`); else `READY_TO_LAUNCH` or `BLOCKED`

Repeated review/fix loops are not a Stop Condition → route to `DECISION_ADJUDICATION` and issue a clearer brief. Then run a combined-diff integration check (conflicts, ownership, launch readiness), append the implementation-review table rows, and record a Step-3 `Decision`.

### 4. Launch Experiments → `/grdr-prelaunch-validate` (pre-launch)

Invoke `/grdr-prelaunch-validate` with `{pkg, launcherPath, localSha, remoteSha?, expectedCheckpointPath?, candidateExportPath?, target}` for the cheapest-check validation (launcher syntax, forbidden-knob policy, checkpoint/candidate path discovery, Code-Alignment SHA match) + a resource-readiness snapshot. **You make the go/no-go and perform the launch + all monitoring in the conversation** — the workflow does not launch or monitor.

- **Resource rules:** local GPUs first; allocate independent work to Bunya only after readiness (sync state, env, remote paths, quota, account, storage roots, runnable commands).
- **ETA discipline:** record `est_time=unknown` until ≥30 min of stable throughput, then derive from observed throughput and update every 10-min report. Never transcribe a "comparable run took X" estimate.
- **Gate:** do not launch if purpose, config, command, artifact paths, ownership, or resource assignment is unclear.

Record a Step-4 `Decision` + the verified SHA, then authorize one `experiment_agent` per planned experiment (each receives purpose, config, command, dependency, target, runtime root, expected artifacts, PLAN stop gates). On a reported code issue, route to Step 2→3. On completion, validate artifacts (exist, modified after launch, match exp id/config, under the runtime root) before recording any fact.

### 5. Live Run Analysis — stays in the conversation

A multi-day run with a 10-minute user-facing line, armed re-entry, and cross-session reattach cannot live in a background workflow. Run this loop on the main-agent loop (`ScheduleWakeup` / `Monitor` / background-`Bash`); you may offload a single bounded status-snapshot to a one-shot agent, but the cadence, the user-facing line, and per-cycle closure remain yours.

Each 10-minute `experiment_agent` report (PLAN objective/thresholds, latest metrics, logs, resource status, ETA, risks, issue class, recommended action) triggers one main-agent live decision. Do not dispatch a second reviewer for routine monitoring.

**Live-check row (mandatory, strict).** Each report produces exactly one updated row in the live-check table (`<tbody data-table-body="live-check">`) — replace the existing row for that exp id in place, or append if none. All 12 columns filled with verified values (`Time`, `Exp ID`, `Agent`, `Run state`, `Last log`, `Progress`, `Latest metrics`, `Resource use`, `Artifacts`, `ETA`, `Live action`, `Next check`); missing values render literal `unmeasured`, never a stale `<td>`. `Time` is the report's local wall-clock; `Next check` is absolute or `+10 min` consistent with the armed re-entry. On close (`completed`/`failed`/`blocked`), update the row once more with the terminal state, then move the evidence path to `results.html`. Emitting the §5 line without updating the row in the same turn is a violation.

**Fact Propagation Contract (binding).** Every artifact that lands — checkpoint save, candidate JSON, sentinel write, phase marker, chain-done — is a locked fact the main agent must propagate to *every* owning surface in the same turn it is observed:

| Event | Surfaces to update in the same turn |
| --- | --- |
| Checkpoint save (`output/**/best_model.pt`) | `tracker.html` live-check row + resource-allocation Status + `results.html` Track 1 + headline strip + result-gate row + sentinel write (if new best) + registry `experiments[i].status` for the closing phase |
| Candidate JSON (`candidates/<label>/<dataset>/*.json`) | `results.html` Track 2 / Track 3 row + rerun of `summarize_results.py` |
| Sentinel (`manifests/*.txt`) | `tracker.html` Resume Block + `results.html` headline + result-gate Observed metric + registry status fields + registry `experiments[i].status` for the sentinel's phase |
| Phase marker (`--- P` / `### P` in chain log) | `tracker.html` live-check + resource-allocation Status + registry `experiments[i].status` (`queued`→`running`, or `running`→`completed`/`failed`) + to-do tick for the closed phase |
| Chain done (`=== … done ===`) | `results.html` final tables + verdict chips + `next-action.html` route + registry `nextRoute`/`openRuns` + registry `experiments[i].status` for every phase the chain closed + tracker Resume Block + to-do |

Each per-turn cycle includes **Step 3.5 — Propagation pass** between the live-check update and the §5 line:

```text
3.5. Run `python skills/research-op/scripts/research_op.py --pkg <id> --op scan-events`.
     For every event the scanner emits, invoke `--event <name> --payload <json>` so
     research-op fans out atomically through Pattern B validation. An empty scanner
     report is the only valid reason to skip.
```

**Loop continuity.** While any run is `queued`/`running`/`stale`, either be processing events or have a re-entry armed within 10 minutes (`ScheduleWakeup(delaySeconds<=600)`, `Monitor` on the run's stdout, or `Bash run_in_background` on a terminal condition). Ending a turn with an open run and no armed re-entry is a violation. On every re-entry, emit one §5 line per open experiment before reasoning. One missed report → mark `stale`; two missed → dispatch a liveness check and route from verified state.

**Early stop** only when a PLAN-defined threshold is met (never from subjective trend); otherwise live actions are `continue`, `repair`, `ask_user`, or `blocked`. Dispatch a `live_run_reviewer` only for escalation (an `early_stop`/`repair` recommendation, ambiguous evidence, report-vs-plan conflict, repeated stale reports, high-cost decisions). Repair requires a concrete cause and a recorded command/config change. Record a Step-5 `Decision`.

After each live decision, output exactly one compact line per open experiment:

```text
<exp_name>: progress=<phase/epoch/iteration>; performance=<objective_metric=value plus gate/baseline relation>; est_time=<remaining or expected finish>; action=<continue/early_stop/repair/ask_user/blocked>
```

Use the key objective metric, not a dump; if a field is unavailable use `performance=pending(first_eval)` / `est_time=unknown` and keep detail in the live-check table.

### 6. Analyze Results → `/grdr-step6-result-analysis`

Invoke `/grdr-step6-result-analysis` with `{pkg, recordedEvidence}` to fan out diverse perspectives (metric validity, hypothesis support, ablation meaning, failure analysis, next-experiment value) over already-recorded evidence and return a synthesis (consensus, disagreements, `proposedVerdict`, global insight, next-action recommendation). The verdict is advisory — a single-seed pass is inconclusive until the gate's seed requirement is met.

You make the final result judgment against `plan.html` objective/motivation/gates/baselines/budgets/seed-status/artifact-completeness, and record consensus, disagreements, final verdict, global insight, next-action rationale, and a Step-6 `Decision` in `results.html` (via `/research-op`).

### 7. Prepare Next Action

Route to exactly one next action by applying `plan.html` gates to the Step-6 judgment:

```text
run_next_experiment_from_step4 -> READY_TO_LAUNCH   # next configs already in plan.html (tuning, sweeps, seeds, planned ablations)
fix_implementation             -> IMPLEMENTING       # code/function issue or implementation-changing experiment
revise_plan                    -> CONTEXT_LOADED after approved plan.html revision, else BLOCKED if approval needed
archive_or_stop                -> STOPPED
ask_user                       -> BLOCKED
```

Revise `plan.html` only when the active executable plan changes; if there is no clear active-plan section, record the proposed change in `tracker.html` and ask before editing. Record the selected action, target state, reason, and next concrete command/question in the Resume Block, plus a Step-7 `Decision`.

## Stop Conditions

Stop only on a Stop Condition.

Route to `BLOCKED` when required information is missing, the plan would change materially without approval, destructive cleanup needs approval, resource use exceeds the plan, required dispatch is unavailable, or a user decision is needed. Do **not** `BLOCKED` merely because implementation is hard, reviewers disagree, a finding repeats, or context was thin — those route to `DECISION_ADJUDICATION`.

Do **not** stop because: the next event is hours/days away (arm `ScheduleWakeup`/`Monitor`/`Bash run_in_background`); scaffolding is complete (the goal is the `results.html` verdict); a single phase just finished/launched (only the final PLAN gate closes the workflow); the session is "ending" (schedule re-entry first); or a non-blocking question is unanswered.

Route to `STOPPED` when the user stops the workflow, the plan goal is achieved, evidence says stop, archive/stop is selected, or the user declines a required approval.

## Stop Gate

You may end execution only in `BLOCKED` or `STOPPED`. Before ending:
- `tracker.html` has the latest state and next action; `results.html` has completed evidence if a run finished
- runtime artifacts are located or missing artifacts recorded; no open run is untracked
- `/research-op scan-events` returns an empty report (cursor past every artifact mtime) — a non-empty report here is a violation
- if any run is still `queued`/`running`/`stale`, a re-entry is armed (`ScheduleWakeup` ≤ 600 s, `Monitor`, or background `Bash`). The correct end-of-turn shape during the loop is one §5 line per open experiment followed by the schedule call — not a written summary.
