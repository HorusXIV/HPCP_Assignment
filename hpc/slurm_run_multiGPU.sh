#!/bin/bash -l

#SBATCH --job-name=HPCP_MultiGPU
#SBATCH --partition=performance
#SBATCH --time=24:00:00               # walltime (DD-HH:MM:SS or HH:MM:SS)
#SBATCH --nodes=2                     # set >1 for multi-node runs
#SBATCH --ntasks-per-node=1           # usually 1 task that will spawn mpi ranks
#SBATCH --cpus-per-task=8             # CPUs per task (adjust for your workload)
#SBATCH --mem=32G                    # memory per node
#SBATCH --gres=gpu:2

set -euo pipefail

# Repository and container defaults (overridable via sbatch --export)
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
# Use same `ENTRY` naming as other scripts for consistency
ENTRY="${ENTRY:-src.multiGPU.main}"
PY_ENV_ACTIVATE="${PY_ENV_ACTIVATE:-}"  # optional: path to venv/conda activate script

# Centralized log dir (can be overridden via env/SBATCH export)
LOG_DIR="${LOG_DIR:-${REPO_DIR}/src/multiGPU/results/logs}"

info() { printf '[SLURM] %s\n' "$*"; }
die() { printf '[SLURM][ERROR] %s\n' "$*" >&2; exit 1; }

info "Job ${SLURM_JOB_ID:-<no-id>} on ${SLURM_JOB_NODELIST:-<no-nodelist>}"
info "Repo dir: ${REPO_DIR}"
info "Image:    ${IMAGE}"
info "Entry:    ${ENTRY}"
info "Log dir:  ${LOG_DIR}"

# Setup OpenMP / thread limits
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTHONUNBUFFERED=1

# Ensure repo workspace and logs exist so we can capture srun outputs
cd "$REPO_DIR"
mkdir -p "${LOG_DIR}"

# Helper: report allocated GPUs (slurm sets CUDA_VISIBLE_DEVICES on some systems)
echo "[SLURM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set>}"
if command -v nvidia-smi &>/dev/null; then
  info "nvidia-smi summary:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
fi

## Environment setup options (choose one strategy):
## 1) Use site modules (recommended on clusters)
##    module load python/3.11 cuda/12.2 openmpi/4.1
## 2) Use a container image (Singularity/Apptainer)
## 3) Activate a project virtualenv/conda

if [[ -n "${PY_ENV_ACTIVATE}" ]]; then
  info "Activating Python environment: ${PY_ENV_ACTIVATE}"
  # shellcheck disable=SC1091
  source "${PY_ENV_ACTIVATE}"
fi

# If using site modules, uncomment and adapt these lines to your cluster
# module purge
# module load python/3.11 cuda/12.2 openmpi/4.1

# Choose launch method: container or direct. Accept 1/0 or true/false-like values.
USE_SINGULARITY=${USE_SINGULARITY:-1}
# Normalize to lowercase in a portable way
_use_singularity_lc=$(printf '%s' "${USE_SINGULARITY}" | tr '[:upper:]' '[:lower:]')
case "${_use_singularity_lc}" in
  1|true|yes) USE_SINGULARITY=1 ;;
  0|false|no) USE_SINGULARITY=0 ;;
  *) USE_SINGULARITY=1 ;;
esac

if [[ "$USE_SINGULARITY" -eq 1 ]]; then
  info "Running inside Singularity image: ${IMAGE}"

  # Launch with srun on the host and have srun invoke singularity per task.
  # This avoids requiring `srun` inside the container image.
  NTASKS=${NTASKS:-${SLURM_NTASKS:-1}}
  # Normalize SLURM CPU env vars to avoid srun mismatch errors where
  # SLURM_CPUS_PER_TASK may differ from SLURM_TRES_PER_TASK (e.g. cpu:8).
  if [[ -n "${SLURM_TRES_PER_TASK:-}" ]]; then
    if [[ "${SLURM_TRES_PER_TASK}" =~ cpu:([0-9]+) ]]; then
      export SLURM_CPUS_PER_TASK="${BASH_REMATCH[1]}"
      export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
      export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    fi
  fi
  info "Debug env before srun: SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK:-<unset>} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>} OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>} NTASKS=${NTASKS}"

  # Configure job-local poetry venv and cache inside the bound workspace to avoid races
  export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
  export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

  info "Preparing Poetry environment once into ${POETRY_VIRTUALENVS_PATH}"
  singularity exec --cleanenv --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
    bash -lc "set -euo pipefail; cd /workspace; mkdir -p ${POETRY_VIRTUALENVS_PATH%/*}; if command -v poetry &>/dev/null; then poetry install --no-interaction --no-ansi; fi"

  srun --mpi=pmix -n ${NTASKS} --output=${LOG_DIR}/slurm-%j-%t.out --error=${LOG_DIR}/slurm-%j-%t.err \
    singularity exec --cleanenv --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
    bash -lc "set -euo pipefail; cd /workspace; poetry run python -m ${ENTRY}"
else
  info "Running directly on node"
  # Ensure Python environment is ready (virtualenv/conda) or modules are loaded
  if command -v poetry &>/dev/null; then
    # Configure job-local poetry venv to avoid concurrent installs into the same cache
    export POETRY_VIRTUALENVS_PATH="${REPO_DIR}/.venv"
    export POETRY_CACHE_DIR="${REPO_DIR}/.cache/pypoetry"
    poetry install --no-interaction --no-ansi
  fi

  # For MPI-enabled launches, use srun. For single-node multi-GPU, set NTASKS to number of GPUs/ranks
  NTASKS=${NTASKS:-${SLURM_NTASKS:-1}}
  # Normalize SLURM CPU env vars as above before srun
  if [[ -n "${SLURM_TRES_PER_TASK:-}" ]]; then
    if [[ "${SLURM_TRES_PER_TASK}" =~ cpu:([0-9]+) ]]; then
      export SLURM_CPUS_PER_TASK="${BASH_REMATCH[1]}"
      export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
      export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    fi
  fi

  info "Debug env before srun: SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK:-<unset>} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>} OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>} NTASKS=${NTASKS}"
  info "srun --mpi=pmix -n ${NTASKS} poetry run python -m ${ENTRY}"
  srun --mpi=pmix -n ${NTASKS} --output=${LOG_DIR}/slurm-%j-%t.out --error=${LOG_DIR}/slurm-%j-%t.err \
    poetry run python -m ${ENTRY}
fi

EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
  die "job exit code: ${EXIT_CODE}"
else
  info "job exit code: ${EXIT_CODE}"
fi
exit ${EXIT_CODE}
