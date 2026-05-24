#!/usr/bin/env bash
# Source-only helper. Swap the canonical panda train/addition JSONs to a subset
# variant for the duration of the launcher, restore on exit. Path swap keeps
# data/video_dataset.py's hardcoded paths intact (per package no-touch boundary).
#
# Reference-counted + flock-protected so concurrent launchers (e.g. B-row +
# F-row on the same workstation) cannot corrupt the canonical state.
# Activate increments the counter; only the first activate captures the
# original target and installs the swap. Restore decrements; only the final
# restore (count -> 0) reinstalls the captured original. Counter + captured
# state live under $RUNTIME_ROOT/data/ so each package owns its own lock.
#
# Usage from a launcher script:
#   SUBSET_DIR="$RUNTIME_ROOT/data/panda_500k"
#   source "$SCRIPT_DIR/_subset_swap.sh"
#   subset_swap_activate "$SUBSET_DIR"
#   trap 'subset_swap_restore' EXIT
#   # ... run training ...
#
# The subset dir must contain panda_500k_ret_train.json + panda_500k_ret_train_addition.json.

_PANDA_DIR_DEFAULT="data/panda/video_retreival_caption"
_PANDA_TRAIN="$_PANDA_DIR_DEFAULT/panda_ret_train.json"
_PANDA_ADD="$_PANDA_DIR_DEFAULT/panda_ret_train_addition.json"

_subset_state_dir() {
    # RUNTIME_ROOT is set by the launcher before sourcing this file.
    echo "${RUNTIME_ROOT:?RUNTIME_ROOT must be set}/data"
}

subset_swap_activate() {
    local subset_dir="$1"
    [[ -d "$subset_dir" ]] || { echo "[subset_swap] missing $subset_dir" >&2; return 2; }
    local sub_train="$subset_dir/panda_500k_ret_train.json"
    local sub_add="$subset_dir/panda_500k_ret_train_addition.json"
    [[ -f "$sub_train" ]] || { echo "[subset_swap] missing $sub_train" >&2; return 2; }
    [[ -f "$sub_add" ]] || { echo "[subset_swap] missing $sub_add" >&2; return 2; }

    local state_dir; state_dir="$(_subset_state_dir)"
    mkdir -p "$state_dir"
    local count_file="$state_dir/.subset_swap.count"
    local state_file="$state_dir/.subset_swap.state"
    local lock_file="$state_dir/.subset_swap.lock"

    exec 9>"$lock_file"
    flock 9
    local n=0
    [[ -s "$count_file" ]] && n="$(cat "$count_file")"
    if [[ "$n" -eq 0 ]]; then
        # First swap-in: capture the original train + addition forms.
        local train_kind="" train_target="" add_kind="" add_target=""
        if [[ -L "$_PANDA_TRAIN" ]]; then
            train_kind="link"
            train_target="$(readlink "$_PANDA_TRAIN")"
            # Reject the pathological case where the canonical already points
            # into a subset under this RUNTIME_ROOT — that would mean a prior
            # launcher leaked state.
            if [[ "$train_target" == *"$state_dir/"* ]]; then
                echo "[subset_swap] ERROR: canonical $_PANDA_TRAIN already points into runtime ($train_target); refusing to swap" >&2
                flock -u 9; exec 9>&-; return 1
            fi
            rm -f "$_PANDA_TRAIN"
        elif [[ -f "$_PANDA_TRAIN" ]]; then
            train_kind="file"
            mv "$_PANDA_TRAIN" "$state_dir/.panda_ret_train.json.fullbak"
        fi
        if [[ -L "$_PANDA_ADD" ]]; then
            add_kind="link"
            add_target="$(readlink "$_PANDA_ADD")"
            rm -f "$_PANDA_ADD"
        elif [[ -f "$_PANDA_ADD" ]]; then
            add_kind="file"
            mv "$_PANDA_ADD" "$state_dir/.panda_ret_train_addition.json.fullbak"
        fi
        ln -s "$(realpath "$sub_train")" "$_PANDA_TRAIN"
        ln -s "$(realpath "$sub_add")" "$_PANDA_ADD"
        # Persist captured state so restore() works across process boundaries.
        {
            printf 'TRAIN_KIND=%s\n' "$train_kind"
            printf 'TRAIN_TARGET=%s\n' "$train_target"
            printf 'ADD_KIND=%s\n' "$add_kind"
            printf 'ADD_TARGET=%s\n' "$add_target"
        } > "$state_file"
        echo 1 > "$count_file"
        echo "[subset_swap] active (refcount=1): $sub_train + $sub_add"
    else
        echo $((n + 1)) > "$count_file"
        echo "[subset_swap] already active (refcount=$((n + 1))); reusing existing swap"
    fi
    flock -u 9
    exec 9>&-
}

subset_swap_restore() {
    local state_dir; state_dir="$(_subset_state_dir)"
    local count_file="$state_dir/.subset_swap.count"
    local state_file="$state_dir/.subset_swap.state"
    local lock_file="$state_dir/.subset_swap.lock"

    [[ -f "$count_file" ]] || { echo "[subset_swap] restore: no active swap"; return 0; }
    exec 9>"$lock_file"
    flock 9
    local n=0
    [[ -s "$count_file" ]] && n="$(cat "$count_file")"
    n=$((n - 1))
    if [[ "$n" -le 0 ]]; then
        # Final restore: tear down the subset symlinks and restore captured form.
        [[ -L "$_PANDA_TRAIN" || -e "$_PANDA_TRAIN" ]] && rm -f "$_PANDA_TRAIN"
        [[ -L "$_PANDA_ADD" || -e "$_PANDA_ADD" ]] && rm -f "$_PANDA_ADD"
        local train_kind="" train_target="" add_kind="" add_target=""
        if [[ -f "$state_file" ]]; then
            # shellcheck disable=SC1090
            source "$state_file"
            train_kind="$TRAIN_KIND"; train_target="$TRAIN_TARGET"
            add_kind="$ADD_KIND"; add_target="$ADD_TARGET"
        fi
        if [[ "$train_kind" == "link" && -n "$train_target" ]]; then
            ln -s "$train_target" "$_PANDA_TRAIN"
        elif [[ -f "$state_dir/.panda_ret_train.json.fullbak" ]]; then
            mv "$state_dir/.panda_ret_train.json.fullbak" "$_PANDA_TRAIN"
        fi
        if [[ "$add_kind" == "link" && -n "$add_target" ]]; then
            ln -s "$add_target" "$_PANDA_ADD"
        elif [[ -f "$state_dir/.panda_ret_train_addition.json.fullbak" ]]; then
            mv "$state_dir/.panda_ret_train_addition.json.fullbak" "$_PANDA_ADD"
        fi
        rm -f "$count_file" "$state_file"
        echo "[subset_swap] restored (refcount=0)"
    else
        echo "$n" > "$count_file"
        echo "[subset_swap] decremented (refcount=$n); other launcher still holds swap"
    fi
    flock -u 9
    exec 9>&-
}
