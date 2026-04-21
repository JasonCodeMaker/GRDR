#!/usr/bin/env bash
set -euo pipefail

ROOT="${PANDA70M_ROOT:-/data2/uqzzha35/VideoRetrieval/Panda-70M-10M}"
PYTHON_BIN="${PANDA70M_PYTHON:-$ROOT/.venv/bin/python}"
HELPER="${PANDA70M_HELPER:-$ROOT/scripts/download_panda70m_10m.py}"
COOKIE_FILE="${PANDA70M_COOKIE_FILE:-$ROOT/cookie.txt}"
USER_AGENT="${PANDA70M_USER_AGENT:-Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.7680.164 Safari/537.36}"
SLEEP_REQUESTS="${PANDA70M_SLEEP_REQUESTS:-10}"
SLEEP_INTERVAL="${PANDA70M_SLEEP_INTERVAL:-120}"
MAX_SLEEP_INTERVAL="${PANDA70M_MAX_SLEEP_INTERVAL:-300}"

wait_for_session_completion() {
    local session_name="$1"
    local session_dir="$ROOT/logs/tmux/$session_name"
    local workers=(download postprocess ratelimit)

    while true; do
        local ready=1
        local worker
        for worker in "${workers[@]}"; do
            local status_path="$session_dir/$worker.status"
            if [[ ! -f "$status_path" ]]; then
                ready=0
                break
            fi
            local status
            status="$(tr -d '[:space:]' < "$status_path")"
            if [[ "$status" != "0" ]]; then
                echo "[$session_name] $worker exited with status $status" >&2
                exit 1
            fi
        done
        if [[ "$ready" -eq 1 ]]; then
            echo "[$session_name] completed successfully"
            return 0
        fi
        sleep 30
    done
}

verify_processed_completeness() {
    local session_name="$1"
    local raw_output_name="$2"
    local frames_output_name="$3"
    local audit_payload

    if ! audit_payload="$(
        "$PYTHON_BIN" "$HELPER" audit-shards \
            --root="$ROOT" \
            --raw-output-name="$raw_output_name" \
            --frames-output-name="$frames_output_name"
    )"; then
        echo "[$session_name] failed to audit shard completeness" >&2
        return 1
    fi

    if ! AUDIT_PAYLOAD="$audit_payload" "$PYTHON_BIN" - "$session_name" <<'PY'
import json
import os
import sys

session_name = sys.argv[1]
payload = json.loads(os.environ["AUDIT_PAYLOAD"])
total_stats = int(payload.get("total_stats_shards", 0) or 0)
total_processed = int(payload.get("total_processed_shards", 0) or 0)
retry_count = int(payload.get("retry_shard_count", 0) or 0)
report_path = payload.get("report_path")

if retry_count != 0 or total_stats != total_processed:
    print(
        f"[{session_name}] shard completeness check failed: "
        f"total_stats_shards={total_stats}, total_processed_shards={total_processed}, "
        f"retry_shard_count={retry_count}, report_path={report_path}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
    then
        echo "$audit_payload" >&2
        return 1
    fi

    echo "[$session_name] shard completeness verified"
}

stop_session_if_exists() {
    local session_name="$1"
    local session_dir="$ROOT/logs/tmux/$session_name"
    local name
    for name in download postprocess ratelimit; do
        local pgid_path="$session_dir/$name.pgid"
        local pid_path="$session_dir/$name.pid"
        if [[ -f "$pgid_path" ]]; then
            local pgid
            pgid="$(tr -d '[:space:]' < "$pgid_path")"
            if [[ -n "$pgid" ]]; then
                kill -TERM -- "-$pgid" 2>/dev/null || true
            fi
        elif [[ -f "$pid_path" ]]; then
            local pid
            pid="$(tr -d '[:space:]' < "$pid_path")"
            if [[ -n "$pid" ]]; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        fi
    done
    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
    fi
}

launch_split() {
    local session_name="$1"
    local csv_name="$2"
    local raw_output_name="$3"
    local frames_output_name="$4"

    echo "Launching $session_name using $csv_name"
    "$PYTHON_BIN" "$HELPER" launch-tmux \
        --root="$ROOT" \
        --csv-name="$csv_name" \
        --raw-output-name="$raw_output_name" \
        --frames-output-name="$frames_output_name" \
        --session-name="$session_name" \
        --cookie-file="$COOKIE_FILE" \
        --user-agent="$USER_AGENT" \
        --sleep-requests="$SLEEP_REQUESTS" \
        --sleep-interval="$SLEEP_INTERVAL" \
        --max-sleep-interval="$MAX_SLEEP_INTERVAL"
}

main() {
    stop_session_if_exists "panda70m_10m_repair"
    stop_session_if_exists "panda70m_test"
    stop_session_if_exists "panda70m_val"

    launch_split "panda70m_test" "panda70m_test.csv" "test_noaudio_raw" "test_4f_s256_q4"
    wait_for_session_completion "panda70m_test"
    verify_processed_completeness "panda70m_test" "test_noaudio_raw" "test_4f_s256_q4"

    launch_split "panda70m_val" "panda70m_val.csv" "val_noaudio_raw" "val_4f_s256_q4"
    wait_for_session_completion "panda70m_val"
    verify_processed_completeness "panda70m_val" "val_noaudio_raw" "val_4f_s256_q4"

    launch_split "panda70m_10m_repair" "panda70m_training_10m.csv" "train_10m_noaudio_raw" "train_10m_4f_s256_q4"
    echo "Training session resumed: tmux attach -t panda70m_10m_repair"
}

main "$@"
