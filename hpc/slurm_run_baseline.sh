#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 04:00:00
#SBATCH --job-name=HPCP_Baseline
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1   # uncomment if you want a GPU available to the container

set -euo pipefail

# ---------------- Config via env (optional) ----------------
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"

# Python entrypoint (module path)
ENTRY="${ENTRY:-src.baseline.main}"

# CLI args to your baseline runner (tweak as needed)
# --idx -1 == process *all* stacks (your codebase now treats -1 as 'all')
ARGS="${ARGS:---sizes 14,64,256,1024 --idx -1 --repeats 3 --nmu 42}"

# Poetry/cache knobs
export POETRY_VIRTUALENVS_IN_PROJECT=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_DIR/.cache/pip}"
export POETRY_CACHE_DIR="${POETRY_CACHE_DIR:-$REPO_DIR/.cache/pypoetry}"

# Math threading (pin BLAS/OpenMP to SLURM allocation)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "[SLURM] Repo dir : $REPO_DIR"
echo "[SLURM] Image    : $IMAGE"
echo "[SLURM] Entry    : $ENTRY"
echo "[SLURM] Args     : $ARGS"
echo "[SLURM] CPUS     : ${SLURM_CPUS_PER_TASK:-NA}"

# Prefer apptainer if present, else singularity
RUNCTL="$(command -v apptainer || true)"
if [[ -z "$RUNCTL" ]]; then
  RUNCTL="$(command -v singularity)"
fi

"$RUNCTL" exec --cleanenv \
  --bind "$REPO_DIR":/workspace \
  "$IMAGE" bash -lc "
    set -euo pipefail
    cd /workspace

    # Optional: fail fast if vendor fast path is required
    # export HPCP_REQUIRE_VENDOR=\${HPCP_REQUIRE_VENDOR:-0}

    # Install deps (cached into /workspace/.venv)
    poetry --version
    poetry install --no-interaction --no-ansi

    echo '[RUN] poetry run python -m $ENTRY $ARGS'
    poetry run python -m \"$ENTRY\" $ARGS
"
