# AGENT.md — GRDR

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

## Research Output Contract

- The only valid in-repo location for new research material is `research/active/<YYYY-MM-DD>-<slug>/`.
- Every research package must contain `README.md`, `PLAN.md`, `TRACKER.md`, `RESULTS.md`, plus `docs/` and `scripts/`.
- Use `bash scripts/dev/new_research.sh <slug>` to create research packages; do not create ad hoc top-level research folders outside `research/`.
- Runtime state, supervisor JSON, local logs, and temporary CSVs must go under `var/research/<YYYY-MM-DD>-<slug>/`, not in tracked repo roots.
- When a research theme is complete or paused, move the whole package to `research/archive/<YYYY-MM-DD>-<slug>/`.
- Stable shared entrypoints stay in `scripts/`; one-off experiment scripts belong in the owning research package.

## Runtime Ops

- Launch every long-running bash script, dataset download, preprocessing pipeline, and experiment inside `tmux`.
- Prefer named `tmux` sessions/windows and report the attach command so the run can be monitored live.

## Current Best

Last updated: 2026-05-11

### MSR-VTT
- Variant: `fit_bucket_l010_g10_k20_s42` (seed 42)
- Ckpt: `output/GRDR/bucket_candidate_k20/msrvtt/20260428163014-fit_bucket_l010_g10_k20_s42/model-3-fit/best_model.pt`
- Setting 1 (beam 100, avg 130.73): CanHit@20/50/100/all = 52.8 / 74.5 / 86.5 / 91.0; XPool R@1/5/10 = 45.7 / 69.8 / 78.9
- Setting 2 (beam 15, avg 300.41): CanHit@20/50/100/all = 4.6 / 15.0 / 32.0 / 64.3; XPool R@1/5/10 = 17.4 / 32.2 / 40.6
- Setting 2 compact budget gate: `avg_candidates_per_query <= 310`; rows above are not comparable compact champions.

### Panda (Setting 1 only)
- Variant: `panda_s1_p4_rq03_c512l3_s42` (P6 3-seed confirmed; seed 42 ckpt)
- Ckpt: `var/research/2026-05-05-panda-setting1-full-run/output/GRDR/panda_setting1_p4/panda/20260509164015-panda_s1_p4_rq03_c512l3_s42/model-3-fit/best_model.pt`
- Setting 1 seed 42 (beam 100, avg 125.95): CanHit@20/50/100/all = 75.54 / 86.95 / 92.80 / 94.01
- Setting 1 3-seed mean (42/220/3407, beam 100, avg 127.15): CanHit@100 = 92.42 ± 0.38; FullSetHit@All = 93.83
- Setting 2: not run.

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
