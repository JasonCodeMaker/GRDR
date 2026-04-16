# Tracker

| Run ID | Milestone | Purpose | Variant | Metrics | Priority | Status | Notes |
| ------ | --------- | ------- | ------- | ------- | -------- | ------ | ----- |
| T001 | tokenizer-health-check | Validate slot redundancy on current MSRVTT best checkpoint | `output/GRDR/msrvtt/best_model` | slot cosine, duplicate rates, active codes, top-share | HIGH | DONE | Standalone script added and rerun confirmed partial collapse / slot redundancy on train and test videos |
