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

## Research Workflow

Before proposing or extending any new research idea, run a self-review first.
- Check the problem anchor and the exact bottleneck being solved
- Critically review the idea for weak assumptions, overclaimed conclusions, and likely reviewer attacks
- Run a novelty check before turning the idea into an experiment or paper claim

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

Last updated: 2026-04-28

- Variant: `fit_bucket_l010_g10_k20_s42`
- Seed: `42`
- Checkpoint:
  `output/GRDR/bucket_candidate_k20/msrvtt/20260428163014-fit_bucket_l010_g10_k20_s42/model-3-fit/best_model.pt`
- Resolved checkpoint path:
  `/data2/uqzzha35/semantic_id/output/GRDR/bucket_candidate_k20/msrvtt/20260428163014-fit_bucket_l010_g10_k20_s42/model-3-fit/best_model.pt`

Current best Stage 1 candidates:
- Setting 1 selected beam: `100`
  - avg_candidates_per_query: `130.73`
  - candidate Recall@10: `0.627`
  - candidate file:
    `var/research/2026-04-28-best-ckpt-full-eval/candidates/best_s42_t1_beam100.json`
- Setting 2 selected beam: `15`
  - avg_candidates_per_query: `300.41`
  - candidate Recall@10: `0.568`
  - candidate file:
    `var/research/2026-04-28-best-ckpt-full-eval/candidates/best_s42_t2_beam15.json`

Current best Stage 2 results:
- Setting 1, beam 100:
  - XPool R@1/R@5/R@10: `45.7 / 69.8 / 78.9`
  - result file:
    `var/research/2026-04-28-best-ckpt-full-eval/results/best_s42_t1_beam100_candidates.csv`
- Setting 2, beam 15:
  - XPool R@1/R@5/R@10: `17.4 / 32.2 / 40.6`
  - result file:
    `var/research/2026-04-28-best-ckpt-full-eval/results/best_s42_t2_beam15_candidates.csv`

Summary artifact:
- `var/research/2026-04-28-best-ckpt-full-eval/results/summary_manual.tsv`

README MSR-VTT comparison:
- Setting 1 improves R@10 over README (`78.9` vs `78.0`) but is lower on R@1/R@5.
- Setting 2 matches README R@1/R@5 and improves R@10 (`40.6` vs `39.7`).

Treat this as the current best until a new run improves the matched setting and evaluation budget.

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
