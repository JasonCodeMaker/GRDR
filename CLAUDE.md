# CLAUDE.md — GRDR

## Project
GRDR (Generative Recall, Dense Reranking) — a two-stage text-to-video retrieval system.
- Stage 1: Multi-View Video Tokenizer + T5-small generative retriever
- Stage 2: X-Pool dense reranker
- Targets: MSR-VTT, ActivityNet, DiDeMo, LSMDC
- Conda env: check `wandb/` requirements or `run.py` header

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
