#!/usr/bin/env bash
# Shared helpers for repo-local training and evaluation scripts.

_repo_root_from_common() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
}

_default_features_root() {
  local repo_root
  repo_root="$(_repo_root_from_common)"
  if [[ -d "$repo_root/dataset/features" ]]; then
    printf '%s\n' "$repo_root/dataset/features"
    return
  fi
  if [[ -d "$repo_root/data_process/datasets/features" ]]; then
    printf '%s\n' "$repo_root/data_process/datasets/features"
    return
  fi
  printf '%s\n' "$repo_root/dataset/features"
}

SEMANTIC_FEATURES_ROOT="${SEMANTIC_FEATURES_ROOT:-$(_default_features_root)}"
SEMANTICTVR_ENV="${SEMANTICTVR_ENV:-semantictvr}"
ANN_BASELINE_ENV="${ANN_BASELINE_ENV:-$SEMANTICTVR_ENV}"
XPOOL_ENV="${XPOOL_ENV:-xpool}"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
}

source_conda_sh() {
  if declare -F conda >/dev/null 2>&1; then
    return 0
  fi

  local conda_sh_candidates=()
  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_sh_candidates+=("$(cd -- "$(dirname -- "$CONDA_EXE")/.." && pwd)/etc/profile.d/conda.sh")
  fi
  conda_sh_candidates+=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/data2/uqzzha35/miniconda3/etc/profile.d/conda.sh"
  )
  if [[ -n "${ROOTMINIFORGE:-}" ]]; then
    conda_sh_candidates+=("$ROOTMINIFORGE/etc/profile.d/conda.sh")
  fi

  local candidate
  for candidate in "${conda_sh_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      return 0
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi

  echo "Unable to locate conda.sh. Set CONDA_EXE or source conda before running." >&2
  exit 1
}

activate_conda_env() {
  local env_name="$1"
  source_conda_sh
  run_cmd conda activate "$env_name"
}

deactivate_conda_env() {
  if declare -F conda >/dev/null 2>&1; then
    conda deactivate || true
  fi
}
