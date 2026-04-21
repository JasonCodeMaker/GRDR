# CLAUDE.md — GRDR

## Project
GRDR (Generative Recall, Dense Reranking) — a two-stage recall-rerank system for scalable text-to-video retrieval.
- Stage 1: Multi-View Video Tokenizer + T5-small generative retriever
- Stage 2: X-Pool dense reranker
- Targets: MSR-VTT, ActivityNet, DiDeMo, LSMDC
- Conda env: check `wandb/` requirements or `run.py` header

## Motivation and Goal

- Repository-scale TVR is bottlenecked by the stage-1 index: anything missed there cannot be recovered downstream.
- Dense retrieval remains useful for reranking, but as a stage-1 index it scales poorly in both memory and query latency.
- Earlier generative retrieval fixes efficiency but often exposes only one semantic access path per video and learns IDs without enough retrieval-aware text supervision.
- GRDR’s project goal is therefore a multi-view semantic-ID stage-1 engine: multiple access paths per video, shared-vocabulary joint key-decoder training, prefix-constrained decoding over valid IDs, and dense reranking only on the compact candidate pool.
- Keep this division of labor explicit in planning: GRDR replaces the stage-1 candidate generator, not the need for a strong stage-2 matcher.

## Refinement Guardrails

For `/research-refine`, `/experiment-plan`, and `/research-refine-pipeline`, keep refinements compatible with the current GRDR architecture and paper story unless the user explicitly requests a replacement.
- The current contribution spine to preserve is:
  1. Multi-view video encoder: `VideoRQVAE_V2` with `VideoLatentEncoder_V2` turns one pooled video feature into `num_latent_tokens` partial views; `token_idx` routes captions to a view and evaluation can use full-view MaxSim.
  2. Progressive end-to-end co-training: `run.py` and `trainer/trainer.py` grow code length loop by loop, warm-start checkpoints and semantic IDs, freeze prior codebooks when needed, and monitor code drift.
  3. Contrastive stage plus main training stage: keep the pre-train contrastive alignment stage and the main stage with CE, code, reconstruction, and RQ objectives as part of the method story.
- Every refinement should explain why it sharpens this paper narrative.
- Keep compatibility with hierarchical semantic-ID generation in `models/grdr.py` / `trainer/evaluator.py` and with stage-2 X-Pool reranking through the candidate JSON / `candidate_file` interface in `reranker/xpool`.
- By default, challenge refinements that remove multi-view structure, break the GRDR-to-X-Pool handoff, or discard progressive co-training without a strong reason.
- If a refinement direction is confirmed failed, remove the related worktrees after any necessary notes are copied back into the owning research package. Keep `.worktrees/` clean instead of accumulating abandoned branches.

## Research Output Contract

- Create new research packages with `bash scripts/dev/new_research.sh <slug>`.
- Keep tracked research docs under `research/active/<YYYY-MM-DD>-<slug>/` and archive them under `research/archive/<YYYY-MM-DD>-<slug>/`.
- Keep runtime state, local logs, supervisor JSON, and temporary analysis artifacts under `var/research/<YYYY-MM-DD>-<slug>/`.
- Do not add new top-level research directories outside `research/`.
- Keep `scripts/` root limited to stable shared entrypoints; one-off experiment scripts belong in the owning research package.

## graphify

A knowledge graph of this codebase lives in `graphify-out/`. It is rebuilt automatically when code changes.

### Before answering codebase questions
1. Check if `graphify-out/graph.json` exists
2. If it does, load the graph and query it before reading raw files — it is faster and shows cross-file connections
3. Use `/graphify query "<question>"` for broad questions, `/graphify explain "<concept>"` for single-node deep dives

### After making code changes
If you modified `.py` files, rebuild the graph:
```
/graphify . --update
```
This only re-extracts changed files (AST-only for code, no LLM cost).

### Key graph facts (as of 2026-04-08)
- 1147 nodes, 2246 edges, 30 communities
- God nodes: Config (89 edges), T5Config (89), MLPLayers (70), VideoCapture (61), GRDR (45)
- The GRDR trainer intentionally reuses X-Pool's BaseTrainer and CandidateDataLoader
- Config is the main cross-community bridge (links datasets, evaluator, trainer, inference sim, CLIP transformer)
