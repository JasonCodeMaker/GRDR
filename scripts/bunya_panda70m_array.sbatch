#!/bin/bash --login
#SBATCH --partition=general
#SBATCH --qos=debug
#SBATCH --account=a_eecs_ds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="${ROOT:?set ROOT to the Panda-70M dataset root}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the synced repo root}"
ENV_PREFIX="${ENV_PREFIX:?set ENV_PREFIX to the conda env path}"
SPLIT_DIR="${SPLIT_DIR:?set SPLIT_DIR to the prepared split CSV directory}"
SPLIT_PREFIX="${SPLIT_PREFIX:?set SPLIT_PREFIX to the split CSV prefix}"
RUN_NAME="${RUN_NAME:?set RUN_NAME for this array run}"

BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$REPO_ROOT/config/panda70m/panda70m_10m_noaudio_balanced.yaml}"
COOKIE_FILE="${COOKIE_FILE:-$ROOT/cookie.txt}"
PROCESSES_COUNT="${PROCESSES_COUNT:-1}"
THREAD_COUNT="${THREAD_COUNT:-2}"
SLEEP_REQUESTS="${SLEEP_REQUESTS-}"
SLEEP_INTERVAL="${SLEEP_INTERVAL-}"
MAX_SLEEP_INTERVAL="${MAX_SLEEP_INTERVAL-}"

TASK_ID="$(printf '%03d' "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}")"
TASK_CSV="$SPLIT_DIR/$SPLIT_PREFIX.part_${TASK_ID}.csv"
TASK_ROOT="$ROOT/downloads/$RUN_NAME/task_${TASK_ID}"
RUNTIME_CONFIG="$ROOT/logs/slurm/$RUN_NAME/runtime/task_${TASK_ID}.json"

mkdir -p "$TASK_ROOT" "$(dirname "$RUNTIME_CONFIG")"

module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

WRITE_CONFIG_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/bunya_panda70m_parallel.py" write-runtime-config
  --base-config-path "$BASE_CONFIG_PATH"
  --output-path "$RUNTIME_CONFIG"
  --cookie-file "$COOKIE_FILE"
  --processes-count "$PROCESSES_COUNT"
  --thread-count "$THREAD_COUNT"
)

if [[ -n "$SLEEP_REQUESTS" ]]; then
  WRITE_CONFIG_CMD+=(--sleep-requests "$SLEEP_REQUESTS")
fi
if [[ -n "$SLEEP_INTERVAL" ]]; then
  WRITE_CONFIG_CMD+=(--sleep-interval "$SLEEP_INTERVAL")
fi
if [[ -n "$MAX_SLEEP_INTERVAL" ]]; then
  WRITE_CONFIG_CMD+=(--max-sleep-interval "$MAX_SLEEP_INTERVAL")
fi

"${WRITE_CONFIG_CMD[@]}"

DOWNLOAD_CMD="$(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/bunya_panda70m_parallel.py" print-download-cmd \
    --root "$ROOT" \
    --csv-path "$TASK_CSV" \
    --config-path "$RUNTIME_CONFIG" \
    --output-dir "$TASK_ROOT"
)"

echo "[$(date -Is)] task=$TASK_ID csv=$TASK_CSV"
echo "$DOWNLOAD_CMD"
eval "$DOWNLOAD_CMD"
