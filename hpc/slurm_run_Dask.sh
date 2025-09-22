#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=HPCP_Dask
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

# ---------------- Config via env (optional) ----------------
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"

# Python entrypoint for the Dask suite
ENTRY="${ENTRY:-src.dask.main}"

# Pass any CLI args your Dask runner accepts (sizes, tiles, workers, etc.)
# Tip: your code already accepts multi-size strings like "14,64,256,1024"
ARGS="${ARGS:---sizes 14,64,256,1024 --idx -1}"

# Poetry/cache knobs
export POETRY_VIRTUALENVS_IN_PROJECT=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_DIR/.cache/pip}"
export POETRY_CACHE_DIR="${POETRY_CACHE_DIR:-$REPO_DIR/.cache/pypoetry}"

# Threading & Dask tuning (safe defaults)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# Dask envs you might want (tweak if your runner reads them)
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="50ms"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="30s"
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.85
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.90
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.98

echo "[SLURM] Repo dir : $REPO_DIR"
echo "[SLURM] Image    : $IMAGE"
echo "[SLURM] Entry    : $ENTRY"
echo "[SLURM] Args     : $ARGS"
echo "[SLURM] CPUS     : ${SLURM_CPUS_PER_TASK:-NA}"

RUNCTL="$(command -v apptainer || true)"
if [[ -z "$RUNCTL" ]]; then
  RUNCTL="$(command -v singularity)"
fi

"$RUNCTL" exec --cleanenv \
  --bind "$REPO_DIR":/workspace \
  "$IMAGE" bash -lc "
    set -euo pipefail
    cd /workspace

    poetry --version
    poetry install --no-interaction --no-ansi

    echo '[RUN] poetry run python -m $ENTRY $ARGS'
    poetry run python -m \"$ENTRY\" $ARGS
"
