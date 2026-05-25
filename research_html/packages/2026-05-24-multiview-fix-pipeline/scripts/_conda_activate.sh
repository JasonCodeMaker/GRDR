#!/usr/bin/env bash
# Source-only helper. Activate the project's conda env on either the
# workstation or a Bunya compute node. Idempotent; call once at script start.
# Bunya recipe follows the MEMORY note: "Bunya conda activation recipe"
# (drain inherited base env, module load miniforge/24.11.3-0, source the
# miniforge conda.sh, activate semantictvr, PYTHONNOUSERSITE=1).
if [[ -d /scratch/project/openps ]] || [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Bunya compute node: grdr-stage1-gpu is the canonical env (has faiss-gpu
    # built for Hopper SM_90). Activating via direct path because the env lives
    # under /scratch/project/openps and isn't auto-registered in (base) conda's
    # env list. faiss-gpu needs (a) imkl/2025.1.0 module for MKL libs and (b)
    # the env's bundled nvidia/cu13 cudart in LD_LIBRARY_PATH.
    while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate 2>/dev/null || break; done
    module load miniforge/24.11.3-0
    source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
    BUNYA_ENV_PATH="${BUNYA_ENV_PATH:-/scratch/project/openps/uqzzha35/conda/envs/grdr-stage1-gpu}"
    conda activate "$BUNYA_ENV_PATH"
    module load imkl/2025.1.0
    # Order: env-lib (libstdc++.so.6.0.34 with GLIBCXX_3.4.30) + bundled cudart
    # (faiss links against libcudart.so.13). System /lib64 libstdc++ is too old
    # and there's no standalone cuda/13.0.0 module on Bunya.
    export LD_LIBRARY_PATH="$BUNYA_ENV_PATH/lib:$BUNYA_ENV_PATH/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONNOUSERSITE=1
else
    source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
    conda activate "${WORKSTATION_CONDA_ENV:-semantictvr}"
fi
