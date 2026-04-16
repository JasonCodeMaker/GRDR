# Plan

## Objective

Use the existing best MSRVTT checkpoint to verify whether the current video tokenizer already exhibits measurable token redundancy or slot-level collapse.

## Work Items

- Define a compact metric contract for tokenizer-collapse diagnostics on unique videos.
- Add a standalone `tokenizer_health_check.py` script under this package.
- Run the script on `output/GRDR/msrvtt/best_model/` and record the observed behavior in a small `RESULTS.md` report.
- Keep the script on-demand and standalone; do not wire it into the main training pipeline.
