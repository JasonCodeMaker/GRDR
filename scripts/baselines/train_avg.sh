#!/usr/bin/env bash
# AVG baseline (text-guided RQ-VAE sID) end-to-end training at C=128, L=3.
# Usage: DATASET=msrvtt DEVICE=0 bash scripts/baselines/train_avg.sh
# All work is in train_gr_baseline.sh; this wrapper just pins INDEX_TYPE=text_guided.
set -uo pipefail
INDEX_TYPE=text_guided bash "$(dirname "${BASH_SOURCE[0]}")/train_gr_baseline.sh"
