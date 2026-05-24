#!/usr/bin/env bash
# B2 baseline: NLT=1, with pseudo queries, c=4096, l=3, Panda 500K subset.
# Ablates the +pseudo contribution; pairs with B1 (no-pseudo) and B3 (NLT=4 broken).
set -euo pipefail

PACKAGE_ID="2026-05-24-multiview-fix-pipeline"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$ROOT/var/research/$PACKAGE_ID}"
LOG_DIR="$RUNTIME_ROOT/logs"
FEATURES_ROOT="${FEATURES_ROOT:-$ROOT/dataset/features}"
SUBSET_DIR="${SUBSET_DIR:-$RUNTIME_ROOT/data/panda_500k}"
SAVE_PATH="${SAVE_PATH:-$RUNTIME_ROOT/output/GRDR/b2_nlt1_pseudo}"
CACHE_DIR="${CACHE_DIR:-$RUNTIME_ROOT/cache/b2}"
WANDB_DIR="${WANDB_DIR:-$RUNTIME_ROOT/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_DIR/wandb}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$CACHE_DIR/wandb_config}"

DEVICE="${DEVICE:-0}"
case "$DEVICE" in 0|1) ;; *) echo "DEVICE must be 0 or 1; got $DEVICE" >&2; exit 2 ;; esac

CODE_NUM=4096
MAX_LENGTH=3
NUM_LATENT_TOKENS=1
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
TRAIN_NUM_CANDIDATES="${TRAIN_NUM_CANDIDATES:-100}"
BEAMS="${BEAMS:-100}"
PRETRAIN_LR="${PRETRAIN_LR:-2e-4}"
MAIN_LR="${MAIN_LR:-5e-5}"
FIT_LR="${FIT_LR:-2e-5}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
MAIN_EPOCHS="${MAIN_EPOCHS:-1}"
FIT_EPOCHS="${FIT_EPOCHS:-1}"
SEED="${SEED:-42}"

W2_CL_LOSS="${W2_CL_LOSS:-0.2}"
W2_CE_LOSS="${W2_CE_LOSS:-0.5}"
W2_CODE_LOSS="${W2_CODE_LOSS:-0.8}"
W2_CL_DD_LOSS="${W2_CL_DD_LOSS:-0.1}"
W2_RQ_LOSS="${W2_RQ_LOSS:-0.3}"
W3_CE_LOSS="${W3_CE_LOSS:-1}"
W3_CODE_LOSS="${W3_CODE_LOSS:-1}"
W3_RQ_LOSS="${W3_RQ_LOSS:-0}"
W3_BUCKET_ROUTE_LOSS="${W3_BUCKET_ROUTE_LOSS:-0.10}"
ROUTE_BUCKET_GAMMA="${ROUTE_BUCKET_GAMMA:-1.0}"

EXP_NAME="${EXP_NAME:-b2_nlt1_pseudo_s${SEED}}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$RUNTIME_ROOT/candidates/$EXP_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-$PACKAGE_ID}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-B2_s${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
EXPORT_ONLY="${EXPORT_ONLY:-0}"
FULL_LOG="$LOG_DIR/B2_${EXP_NAME}.log"
MANIFEST="$RUNTIME_ROOT/manifests/${EXP_NAME}_manifest.json"
TRIGGER_FILE="$RUNTIME_ROOT/manifests/B2_seed${SEED}_canhit100.txt"
CHECKPOINT_PATH=""

mkdir -p "$LOG_DIR" "$CANDIDATE_DIR" "$SAVE_PATH" "$CACHE_DIR" "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$RUNTIME_ROOT/manifests"
cd "$ROOT"

exec > >(tee -a "$FULL_LOG") 2>&1
source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

source "$SCRIPT_DIR/_subset_swap.sh"
subset_swap_activate "$SUBSET_DIR"
trap 'subset_swap_restore' EXIT

train_b2() {
    log "train B2 NLT=1 +pseudo exp=$EXP_NAME"
    conda activate semantictvr
    export WANDB_MODE WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_CONSOLE=off
    python run.py \
        --device "$DEVICE" --model_name t5-small --dataset panda \
        --features_root "$FEATURES_ROOT" --cache_dir "$CACHE_DIR" \
        --code_num "$CODE_NUM" --max_length "$MAX_LENGTH" \
        --batch_size "$TRAIN_BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
        --num_candidates "$TRAIN_NUM_CANDIDATES" --setting 1 \
        --num_latent_tokens "$NUM_LATENT_TOKENS" \
        --use_pseudo_queries \
        --pretrain_lr "$PRETRAIN_LR" --main_lr "$MAIN_LR" --fit_lr "$FIT_LR" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" --main_epochs "$MAIN_EPOCHS" --fit_epochs "$FIT_EPOCHS" \
        --save_path "$SAVE_PATH" --exp_name "$EXP_NAME" --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME" \
        --w2_cl_loss "$W2_CL_LOSS" --w2_ce_loss "$W2_CE_LOSS" --w2_code_loss "$W2_CODE_LOSS" \
        --w2_cl_dd_loss "$W2_CL_DD_LOSS" --w2_rq_loss "$W2_RQ_LOSS" \
        --w3_ce_loss "$W3_CE_LOSS" --w3_code_loss "$W3_CODE_LOSS" --w3_rq_loss "$W3_RQ_LOSS" \
        --w3_bucket_route_loss "$W3_BUCKET_ROUTE_LOSS" --route_bucket_gamma "$ROUTE_BUCKET_GAMMA" \
        --enable_fit
    conda deactivate
}

latest_checkpoint() {
    find "$SAVE_PATH/panda" -path "*${EXP_NAME}/model-${MAX_LENGTH}-fit/best_model.pt" -type f 2>/dev/null \
        | sort | tail -n 1
}

export_one_beam() {
    local checkpoint_path="$1"; local beam="$2"
    conda activate semantictvr
    python run.py \
        --eval --device "$DEVICE" --model_name t5-small --dataset panda \
        --features_root "$FEATURES_ROOT" --cache_dir "$CACHE_DIR" \
        --code_num "$CODE_NUM" --max_length "$MAX_LENGTH" \
        --batch_size "$EVAL_BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
        --num_candidates "$beam" --setting 1 --num_latent_tokens "$NUM_LATENT_TOKENS" \
        --use_pseudo_queries \
        --eval_checkpoint "$checkpoint_path" --candidate_output_dir "$CANDIDATE_DIR" \
        --save_path "$SAVE_PATH" --exp_name "${EXP_NAME}_beam${beam}" \
        --wandb_project "$WANDB_PROJECT" --wandb_run_name "${WANDB_RUN_NAME}_beam${beam}"
    conda deactivate
}

export_beams() {
    local checkpoint_path; checkpoint_path="$(latest_checkpoint)"
    [[ -n "$checkpoint_path" ]] || { log "no checkpoint"; exit 1; }
    CHECKPOINT_PATH="$checkpoint_path"
    for beam in $BEAMS; do
        log "### EXPORT BEGIN beam=$beam ###"
        export_one_beam "$checkpoint_path" "$beam"
        log "### EXPORT END beam=$beam ###"
    done
}

write_manifest() {
    python - "$MANIFEST" <<PY
import json, sys
from pathlib import Path
m = Path(sys.argv[1])
payload = {
    "package": "$PACKAGE_ID", "cell": "B2",
    "code_num": int("$CODE_NUM"), "max_length": int("$MAX_LENGTH"),
    "num_latent_tokens": int("$NUM_LATENT_TOKENS"), "use_pseudo_queries": True,
    "seed": int("$SEED"), "exp_name": "$EXP_NAME", "device": int("$DEVICE"),
    "features_root": "$FEATURES_ROOT", "subset_dir": "$SUBSET_DIR",
    "save_path": "$SAVE_PATH", "candidate_dir": "$CANDIDATE_DIR",
    "beams": [int(v) for v in "$BEAMS".split()],
    "runtime_root": "$RUNTIME_ROOT", "full_log": "$FULL_LOG",
    "trigger_file": "$TRIGGER_FILE",
}
if "$CHECKPOINT_PATH": payload["checkpoint"] = "$CHECKPOINT_PATH"
payload["candidate_jsons"] = sorted(str(p) for p in Path("$CANDIDATE_DIR").glob("panda_c*l*_candidates_t1.json"))
m.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

main() {
    log "B2 runtime root: $RUNTIME_ROOT"
    write_manifest
    [[ "${DRY_RUN:-0}" == "1" ]] && { log "dry run"; return 0; }
    if [[ "$EXPORT_ONLY" == "1" ]]; then export_beams; write_manifest; log "done B2 export-only"; return 0; fi
    log "### TRAIN BEGIN ###"; train_b2; log "### TRAIN END ###"
    export_beams; write_manifest
    log "done B2"
}
main "$@"
