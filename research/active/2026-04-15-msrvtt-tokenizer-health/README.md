# 2026-04-15-msrvtt-tokenizer-health

Status: active

## Summary

This package records a code-grounded health check of the current MSRVTT stage-1 video tokenizer checkpoint at `output/GRDR/msrvtt/best_model/`.
It exists to test whether the tokenizer already shows token redundancy or partial collapse, instead of relying on architecture intuition alone.

## Package Map

- `PLAN.md`: scope of the tokenizer-health pass
- `TRACKER.md`: package progress
- `RESULTS.md`: current conclusions and confirmed readout
- `docs/`: metric contract and interpretation notes
- `scripts/`: standalone package-specific utilities

## On-Demand Usage

Run the standalone checker from the training environment when needed:

```bash
conda run -n semantictvr python \
  research/active/2026-04-15-msrvtt-tokenizer-health/scripts/tokenizer_health_check.py \
  --checkpoint output/GRDR/msrvtt/best_model/best_model.pt \
  --dataset msrvtt \
  --features_root dataset/features \
  --code_num 128 \
  --code_length 3 \
  --num_latent_tokens 4 \
  --embed_dim 512 \
  --split both
```

## Runtime State

Machine-local state for this package belongs in `var/research/2026-04-15-msrvtt-tokenizer-health/`.
