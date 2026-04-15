# AGENT.md — GRDR

## Project
GRDR (Generative Recall, Dense Reranking) — a two-stage text-to-video retrieval system.
- Stage 1: Multi-View Video Tokenizer + T5-small generative retriever
- Stage 2: X-Pool dense reranker
- Targets: MSR-VTT, ActivityNet, DiDeMo, LSMDC
- Conda env: check `wandb/` requirements or `run.py` header

## Research Workflow

Before proposing or extending any new research idea, run a self-review first.
- Check the problem anchor and the exact bottleneck being solved
- Critically review the idea for weak assumptions, overclaimed conclusions, and likely reviewer attacks
- Run a novelty check before turning the idea into an experiment or paper claim

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
