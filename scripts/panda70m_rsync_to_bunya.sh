#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/data2/uqzzha35/VideoRetrieval/Panda-70M-10M}"
REMOTE_HOST="${REMOTE_HOST:-uqzzha35@bunya.rcc.uq.edu.au}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/project/openps/uqzzha35/Panda-70M-10M}"
REMOTE_PREFIX="${REMOTE_PREFIX:-federated/local_main}"
FRAMES_OUTPUT_NAME="${FRAMES_OUTPUT_NAME:-train_10m_4f_s256_q4}"
RAW_OUTPUT_NAME="${RAW_OUTPUT_NAME:-train_10m_noaudio_raw}"
RSYNC_FLAGS="${RSYNC_FLAGS:---archive --human-readable --partial --append-verify --info=progress2}"
MODE="${1:-all}"

local_frames_root="$LOCAL_ROOT/frames/$FRAMES_OUTPUT_NAME"
local_download_root="$LOCAL_ROOT/downloads/$RAW_OUTPUT_NAME"
remote_base="$REMOTE_HOST:$REMOTE_ROOT/$REMOTE_PREFIX"

run_rsync() {
  local src="$1"
  local dst="$2"
  # This may trigger MFA on first contact with Bunya.
  # Run it interactively so Codex can stop and wait for the user if needed.
  # shellcheck disable=SC2086
  rsync $RSYNC_FLAGS "$src" "$dst"
}

sync_markers() {
  run_rsync "$local_frames_root/_processed/" "$remote_base/frames/$FRAMES_OUTPUT_NAME/_processed/"
}

sync_frame_shards() {
  run_rsync "$local_frames_root/shards/" "$remote_base/frames/$FRAMES_OUTPUT_NAME/shards/"
}

sync_download_metadata() {
  # shellcheck disable=SC2086
  rsync $RSYNC_FLAGS \
    --include='*/' \
    --include='*.parquet' \
    --include='*_stats.json' \
    --exclude='*' \
    "$local_download_root/" \
    "$remote_base/downloads/$RAW_OUTPUT_NAME/"
}

case "$MODE" in
  markers)
    sync_markers
    ;;
  frames)
    sync_frame_shards
    ;;
  metadata)
    sync_download_metadata
    ;;
  all)
    sync_markers
    sync_frame_shards
    sync_download_metadata
    ;;
  *)
    echo "Usage: $0 [markers|frames|metadata|all]" >&2
    exit 2
    ;;
esac
