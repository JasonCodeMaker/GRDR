#!/usr/bin/env bash
# Source-only helper. Activate the project's conda env on either the
# workstation or a Bunya compute node. Idempotent; call once at script start.
# Bunya recipe follows the MEMORY note: "Bunya conda activation recipe"
# (drain inherited base env, module load miniforge/24.11.3-0, source the
# miniforge conda.sh, activate semantictvr, PYTHONNOUSERSITE=1).
if [[ -d /scratch/project/openps ]] || [[ -n "${SLURM_JOB_ID:-}" ]]; then
    while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate 2>/dev/null || break; done
    module load miniforge/24.11.3-0
    source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
    conda activate semantictvr
    export PYTHONNOUSERSITE=1
else
    source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
    conda activate semantictvr
fi
