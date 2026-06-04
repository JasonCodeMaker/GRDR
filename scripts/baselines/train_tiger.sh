#!/usr/bin/env bash
# TIGER baseline (standard RQ-VAE sID) end-to-end training at C=128, L=3.
# Usage: DATASET=msrvtt DEVICE=0 bash scripts/baselines/train_tiger.sh
# All work is in train_gr_baseline.sh; this wrapper just pins INDEX_TYPE=standard.
set -uo pipefail
INDEX_TYPE=standard bash "$(dirname "${BASH_SOURCE[0]}")/train_gr_baseline.sh"
