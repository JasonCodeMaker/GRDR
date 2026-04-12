#!/bin/bash --login
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/project/openps/uqzzha35/conda/envs/panda70m-v2d}"
VIDEO2DATASET_VERSION="${VIDEO2DATASET_VERSION:-1.0.0}"

module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
  mamba create -y -p "$ENV_PREFIX" python=3.11 nodejs
fi

conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
PYTHON_BIN="${ENV_PREFIX}/bin/python"

"$PYTHON_BIN" -m pip install --upgrade --no-user pip
"$PYTHON_BIN" -m pip install --no-user --upgrade --force-reinstall \
  "video2dataset==${VIDEO2DATASET_VERSION}" gdown pyyaml yt-dlp

"$PYTHON_BIN" - <<'PY'
import shutil
import sys

import gdown
import video2dataset
import yaml

print("PYTHON_OK")
print("sys.executable", sys.executable)
print("video2dataset", getattr(video2dataset, "__file__", None))
print("node", shutil.which("node"))
print("gdown", getattr(gdown, "__file__", None))
print("yaml", getattr(yaml, "__file__", None))
PY
