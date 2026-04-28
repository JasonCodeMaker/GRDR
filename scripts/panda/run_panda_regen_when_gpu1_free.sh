#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_LOG="dataset/Panda-70M-10M/video_retreival_caption/panda_10m_ret_train_addition.regen_model.wait.log"
RUN_LOG="dataset/Panda-70M-10M/video_retreival_caption/panda_10m_ret_train_addition.regen_model.log"
IDLE_MEMORY_THRESHOLD_MIB=512
IDLE_CONFIRMATION_CHECKS=2
IDLE_CONFIRMATION_SLEEP=30

is_gpu1_idle() {
    local uuid
    local memory_used
    local active

    uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d ' ')
    memory_used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    active=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | grep -cFx "${uuid}" || true)

    if [[ -z "$memory_used" ]]; then
        memory_used=999999
    fi

    echo "${uuid} ${active} ${memory_used}"
    [[ "$active" -eq 0 && "$memory_used" -le "$IDLE_MEMORY_THRESHOLD_MIB" ]]
}

while true; do
    ts=$(date "+%Y-%m-%d %H:%M:%S %Z")
    read -r uuid active memory_used < <(is_gpu1_idle)
    echo "${ts} waiting_for_gpu1 active_compute_procs=${active} memory_used_mib=${memory_used}" | tee -a "$WAIT_LOG"
    if [[ "$active" -eq 0 && "$memory_used" -le "$IDLE_MEMORY_THRESHOLD_MIB" ]]; then
        confirmed=1
        while [[ "$confirmed" -lt "$IDLE_CONFIRMATION_CHECKS" ]]; do
            sleep "$IDLE_CONFIRMATION_SLEEP"
            ts=$(date "+%Y-%m-%d %H:%M:%S %Z")
            read -r uuid active memory_used < <(is_gpu1_idle)
            echo "${ts} confirming_gpu1_idle check=${confirmed}/${IDLE_CONFIRMATION_CHECKS} active_compute_procs=${active} memory_used_mib=${memory_used}" | tee -a "$WAIT_LOG"
            if [[ "$active" -ne 0 || "$memory_used" -gt "$IDLE_MEMORY_THRESHOLD_MIB" ]]; then
                confirmed=0
                break
            fi
            confirmed=$((confirmed + 1))
        done
    else
        confirmed=0
    fi

    if [[ "${confirmed:-0}" -eq "$IDLE_CONFIRMATION_CHECKS" ]]; then
        echo "${ts} gpu1_is_free starting_regeneration" | tee -a "$WAIT_LOG"
        export CUDA_VISIBLE_DEVICES=1
        /data2/uqzzha35/VideoRetrieval/Panda-70M-10M/captioning/.conda-env/bin/python \
            scripts/regenerate_panda_flagged_queries.py \
            --batch-size 8 \
            --num-workers 4 2>&1 | tee "$RUN_LOG"
        break
    fi
    sleep 300
done
