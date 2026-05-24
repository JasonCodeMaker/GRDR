#!/usr/bin/env bash
# Lightweight smoke: drives smoke_test_flags.py which exercises every new code
# path (per_slot_init, return_all_slots, multiview_all_slot_ce train_step,
# build_per_video_loader, build_loss_weights propagation) on synthetic features.
# Avoids the multi-hour text-kmeans pre-processing that the full run.py pipeline
# would trigger on a fresh cache_dir; the heavy end-to-end check belongs in the
# B*/F2/K1 launchers proper, not the smoke gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$ROOT/var/research/2026-05-24-multiview-fix-pipeline}"
LOG_DIR="$RUNTIME_ROOT/logs"
DEVICE="${DEVICE:-0}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

FULL_LOG="$LOG_DIR/smoke_test_flags.log"
exec > >(tee -a "$FULL_LOG") 2>&1
source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1

conda activate semantictvr
DEVICE="$DEVICE" python "$SCRIPT_DIR/smoke_test_flags.py"
conda deactivate
