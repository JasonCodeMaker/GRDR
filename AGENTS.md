# AGENTS.md — GRDR

## Project
GRDR (Generative Recall, Dense Reranking) — a two-stage recall-rerank system for scalable text-to-video retrieval.
- Stage 1: Multi-View Video Tokenizer + T5-small generative retriever
- Stage 2: X-Pool dense reranker
- Targets: MSR-VTT, ActivityNet, DiDeMo, LSMDC, Panda
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

## Research Workflow

Before proposing or extending any new research idea, run a self-review first.
- Check the problem anchor and the exact bottleneck being solved
- Critically review the idea for weak assumptions, overclaimed conclusions, and likely reviewer attacks
- Run a novelty check before turning the idea into an experiment or paper claim
- For Review Loop workflows, do not save the loop transcript or full round-by-round review as a standalone doc. Distill only the actionable conclusions into `PLAN.md`, and record the important review judgments, concerns, and resolutions in `TRACKER.md`.
- For future experiments, do not include an A0/checkpoint reproduction run such as `V0: A0 reproduction` by default. Assume the current checkpoint and recorded performance in `AGENTS.md` are correct unless the user explicitly asks to revalidate the anchor or there is direct evidence of code/path drift.
- For inference/export changes, minimize per-query latency while preserving metric correctness. Reuse online generation outputs and offline caches where possible; do not add per-candidate T5 teacher-forcing or direct video-ANN search to the default inference path unless explicitly approved as an offline diagnostic.
- In research plans, call the ordered run/ablation section `Experiments List`, not `Decision Tree`.

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

## Trustworthy Research Pipeline For Codex

This file is the Codex-facing operating contract for the Trustworthy Research Pipeline. It translates
the shared protocol in `CLAUDE.md` and the package controller in `WORKFLOW.md` into actions Codex can
take inside this toolbox repo or inside a target research project.

Do not treat this file as a weaker copy of `CLAUDE.md`. `CLAUDE.md` remains the durable shared research
contract; this file is the thin Codex bootloader that tells Codex where to start, which skill owns the
task, which source-of-truth layer to load, and when to stop for user ratification.

## First Decision: Where Am I?

Before acting, classify the current working directory:

- **Toolbox repo**: this repository, whose git root is the directory containing this file, `README.md`,
  `CLAUDE.md`, `WORKFLOW.md`, `skills/`, `lib/`, and `tests/`. The parent workspace is not the repo.
- **Target research project**: a consuming ML/research repo where the pipeline has been attached. It may
  contain copied or merged `AGENTS.md`, `CLAUDE.md`, `WORKFLOW.md`, `research_html/`, `outputs/_scope/`,
  and project source/config/data files.

If the user asks to change the pipeline implementation or protocol, work in the toolbox repo. If the
user asks to run research, initialize a project, inspect experiments, or update package state, work in
the target research project and resolve pipeline scripts through installed Codex skills.

## Required Read Order

For any non-trivial target-project task, read only the relevant files in this order:

1. User request and any active project-specific section at the top of `AGENTS.md` or `CLAUDE.md`.
2. `CLAUDE.md` for the project-level operating contract and project-specific guardrails.
3. `WORKFLOW.md` before any research-package implementation, launch, monitoring, result analysis, or
   package state transition.
4. The relevant skill body under `$HOME/.codex/skills/<skill-name>/SKILL.md`, or `skills/<skill-name>/`
   when editing this toolbox.
5. The smallest live authority set for the task: Scope for intent, package pages for plans/verdicts,
   runtime artifacts for measurements, and `research_html/data/research-packages.js` for dashboard index
   state.

Runtime artifacts and live process state override remembered summaries. If a required fact is missing,
record the gap and stop at the smallest useful user decision instead of inventing intent.

Source hierarchy: Scope owns intent; package `plan.html` owns executable gates; runtime artifacts own
measurements; package `results.html` owns verdicts; `research-packages.js` owns dashboard index state.
Derived pages (`scope.html`, `context.html`, `learnings.html`, lane pages, `scope-projection.json/js`) are
read surfaces unless their owning skill says otherwise. For detailed surface ownership, read only the
relevant skill/reference, especially `research-dashboard`, `research-package`, `research-op`, and
`research-scope`.

## Attaching The Pipeline To A Target Project

When setting up a target research repo for Codex:

1. Install toolbox skills by symlinking `skills/research-*` into `$HOME/.codex/skills`; do not copy skill
   directories.
2. Copy or merge `AGENTS.md`, `CLAUDE.md`, and `WORKFLOW.md` into the target repo root.
3. If the target already has `AGENTS.md` or `CLAUDE.md`, merge the pipeline protocol without overwriting
   existing user/project instructions.
4. Prepend target-specific context above the reusable protocol sections:
   - project objective and motivation;
   - datasets, baselines, metrics, gates, and success criteria;
   - compute constraints and available machines;
   - non-goals, safety constraints, reviewer concerns, and current best checkpoint.
5. Create `outputs/_scope/` and `outputs/_selfevolve/`, then run `/research-dashboard`.
6. If no committed Project node exists, run `/research-onboard` for an existing repo or `/research-scope`
   when the user already knows the exact Project objective.

The pipeline is not active for research execution until a Project node is committed in the Scope SSOT.
Onboarding and scoping may propose; only a human-ratified transition commits the objective.

## Codex Skill And Script Resolution

In a target research project, do not assume this toolbox source tree is vendored into the repo.

- Resolve `skills/<name>/scripts/...` through the installed Codex skill first:
  `$HOME/.codex/skills/<name>/scripts/...`.
- If a command in `CLAUDE.md`, `WORKFLOW.md`, or a skill body uses a relative `skills/...` path, adapt it
  to the installed skill path when running from the target project.
- When editing this toolbox repo itself, use the local `skills/<name>/...` path and run the toolbox tests.

## Scope And Triage Contract

Scope is the versioned intent store for Project -> Direction -> Task. Codex must preserve this boundary:

- Pending ideas, Project objectives, Directions, Tasks, and scope revisions go through Triage first.
- Codex may draft proposals and show them to the user, but it must not silently commit
  `outputs/_scope/transitions.jsonl`.
- A committed Scope transition requires explicit human ratification and the gated writer documented by
  `research-scope` / `research-op`.
- Dashboard Scope projection files are read-only derived views. Do not hand-edit them to change intent.

If Scope and package/dashboard surfaces disagree, treat Scope as the intent authority and package/runtime
artifacts as evidence authority; repair through the appropriate gated operation rather than ad-hoc edits.

## Research Package Operation Contract

For package work, Codex is the decision owner described by `WORKFLOW.md`:

- Read the package Resume Block, active plan, project rules, results, and relevant runtime artifacts before
  acting.
- Use `/research-op` for research-package surface mutations. Direct edits to package HTML, inventory rows,
  or package docs are violations unless the relevant skill explicitly owns scaffolding or large structural
  setup.
- Treat user instructions that change constraints, plan, metric, baseline, or scope as locked facts. Record
  them in the typed home and propagate status in the same turn.
- Run the required lint/check command after learnings-relevant or package-state mutations, and fix errors
  before claiming the turn is complete.
- Put long-running experiments, training, preprocessing, downloads, and remote jobs in named `tmux`
  sessions unless the user explicitly asks for a different runner.
- For long-running experiment commands, use the project live-run skill (`research-exp-live`) when
  available: launch through its wrapper and read routine run state from structured runtime artifacts
  (`status.json`), not raw scrollback; raw logs are bounded debug fallback.

Do not declare a win from chat memory. Claims need metric gates, evidence paths, and the package/result
surface required by `CLAUDE.md` and `WORKFLOW.md`.

## Toolbox Maintenance Contract

When modifying this toolbox repo:

- Keep changes surgical and preserve project-agnostic protocol bodies unless the task explicitly asks to
  change them.
- Read `README.md` and the relevant `skills/*/SKILL.md` before changing setup, dashboard, Scope, package,
  or operation behavior.
- Update tests with behavior changes and run `python3.13 -m pytest tests/` before claiming toolbox
  behavior changes are complete. For documentation-only changes, at minimum run a syntax/consistency check
  appropriate to the touched files.
- Do not modify target-project artifacts while working on toolbox internals unless the user explicitly asks
  for an end-to-end consuming-project test.

## Completion Standard

Before final response, Codex must be able to state:

- which context it operated in: toolbox repo or target research project;
- which protocol or skill controlled the work;
- what files or surfaces changed;
- what validation ran, or why validation was not applicable;
- whether any human ratification is still required.
