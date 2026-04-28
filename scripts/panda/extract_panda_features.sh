#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
conda activate internvideo2

CHECKPOINT=${CHECKPOINT:-/home/uqzzha35/Project/SemanticID/MM-SemanticTVR/data_process/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt}

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

python scripts/extract_panda_features.py \
    --split "${SPLIT:-test}" \
    --gpu_id 0 \
    --mode "${MODE:-both}" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "${BATCH:-36}" \
    --num_workers "${WORKERS:-8}" \
    --text_batch_size "${TEXT_BATCH:-256}" \
    --max_txt_l "${MAX_TXT_L:-40}"
