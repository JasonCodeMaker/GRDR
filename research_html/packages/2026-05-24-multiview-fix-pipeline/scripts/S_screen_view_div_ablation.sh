#!/usr/bin/env bash
# S single-loop screening: ablate (view_div_high_weight, slot_orthogonality_weight,
# per_slot_init) over a small panda subset (~100K vids), code_length=1, 1 loop only.
# Writes screening/mech_ablation.json with per-variant CanHit@100 (proxy: best fit
# val metric since 1-loop chain has no model-3-fit).
set -euo pipefail

PACKAGE_ID="2026-05-24-multiview-fix-pipeline"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$ROOT/var/research/$PACKAGE_ID}"
LOG_DIR="$RUNTIME_ROOT/logs"
FEATURES_ROOT="${FEATURES_ROOT:-$ROOT/dataset/features}"
SUBSET_DIR="${SUBSET_DIR:-$RUNTIME_ROOT/data/panda_100k}"
SCREEN_DIR="${SCREEN_DIR:-$RUNTIME_ROOT/screening}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-1}"

mkdir -p "$LOG_DIR" "$SCREEN_DIR" "$RUNTIME_ROOT/manifests"
cd "$ROOT"

FULL_LOG="$LOG_DIR/S_screen_view_div_s${SEED}.log"
exec > >(tee -a "$FULL_LOG") 2>&1
source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Build the 100k subset if missing
if [[ ! -f "$SUBSET_DIR/panda_500k_ret_train.json" ]]; then
    log "building 100k panda subset at $SUBSET_DIR"
    conda activate semantictvr
    python "$SCRIPT_DIR/build_500k_subset.py" \
        --source-train data/panda/video_retreival_caption/panda_ret_train.json \
        --source-addition data/panda/video_retreival_caption/panda_ret_train_addition.json \
        --n 100000 --seed "$SEED" --out-dir "$SUBSET_DIR"
    conda deactivate
fi

source "$SCRIPT_DIR/_subset_swap.sh"
subset_swap_activate "$SUBSET_DIR"
trap 'subset_swap_restore' EXIT

conda activate semantictvr
python "$SCRIPT_DIR/_screen_runner.py" \
    --device "$DEVICE" --seed "$SEED" \
    --features-root "$FEATURES_ROOT" \
    --runtime-root "$RUNTIME_ROOT" \
    --screen-dir "$SCREEN_DIR"
conda deactivate
log "done S screen"
