#!/bin/bash -l
#SBATCH --job-name=HPCP_MultiGPU
#SBATCH --partition=performance
#SBATCH --time=24:00:00               # walltime (DD-HH:MM:SS or HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4             
#SBATCH --mem=24G                    
#SBATCH --gres=gpu:2
#SBATCH --output=src/multiGPU/results/logs/slurm-%j-%t.out 
#SBATCH --error=src/multiGPU/results/logs/slurm-%j-%t.err

set -euo pipefail

# Repository and container defaults (overridable via sbatch --export)
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"

ENTRY="${ENTRY:-src.multiGPU.main}"
PY_ENV_ACTIVATE="${PY_ENV_ACTIVATE:-}"

# Centralized log dir (can be overridden via env/SBATCH export)
LOG_DIR="${LOG_DIR:-${REPO_DIR}/src/multiGPU/results/logs}"

info() { printf '[SLURM] %s\n' "$*"; }
die() { printf '[SLURM][ERROR] %s\n' "$*" >&2; exit 1; }

info "Job ${SLURM_JOB_ID:-<no-id>} on ${SLURM_JOB_NODELIST:-<no-nodelist>}"
info "Repo dir: ${REPO_DIR}"
info "Image:    ${IMAGE}"
info "Entry:    ${ENTRY}"
info "Log dir:  ${LOG_DIR}"

# Compact SLURM environment summary for debugging
info "SLURM summary: JOB_ID=${SLURM_JOB_ID:-<no-id>} NNODES=${SLURM_NNODES:-<unset>} NTASKS=${SLURM_NTASKS:-<unset>} NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-<unset>} NODELIST=${SLURM_JOB_NODELIST:-<unset>}"

# GPUs-per-node default (can be overridden via env or sbatch --export)
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

# Setup OpenMP / thread limits
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTHONUNBUFFERED=1

# Ensure repo workspace and logs exist so we can capture srun outputs
cd "$REPO_DIR"
mkdir -p "${LOG_DIR}"

# Helper: report allocated GPUs (slurm sets CUDA_VISIBLE_DEVICES on some systems)
info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set>}"
if command -v nvidia-smi &>/dev/null; then
  info "nvidia-smi summary:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
fi


export CUPY_DUMP_CUDA_SOURCE_ON_ERROR=1
export CUPY_NVRTC_LOGLEVEL=info
export CUPY_CACHE_DIR="/workspace/.cupy_cache"
export CUPY_COMPILE_WITH_PTX=1


# Launch with srun on the host and have srun invoke singularity per task.
# This avoids requiring `srun` inside the container image.
# Compute a safe NTASKS: prefer SLURM_NTASKS if provided, otherwise derive from
if [[ -n "${SLURM_NTASKS:-}" ]]; then
  NTASKS=${NTASKS:-${SLURM_NTASKS}}
else
  NODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
  NTASKS=${NTASKS:-$(( NODES * GPUS_PER_NODE ))}
fi
info "Computed NTASKS=${NTASKS} (GPUS_PER_NODE=${GPUS_PER_NODE})"
# Normalize SLURM CPU env vars to avoid srun mismatch errors where
# SLURM_CPUS_PER_TASK may differ from SLURM_TRES_PER_TASK (e.g. cpu:8).
if [[ -n "${SLURM_TRES_PER_TASK:-}" ]]; then
  if [[ "${SLURM_TRES_PER_TASK}" =~ cpu:([0-9]+) ]]; then
    export SLURM_CPUS_PER_TASK="${BASH_REMATCH[1]}"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  fi
fi

# Configure job-local poetry venv and cache inside the bound workspace to avoid races
export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

info "Preparing Poetry environment into ${POETRY_VIRTUALENVS_PATH}"
singularity exec --cleanenv --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
  bash -lc "
    set -euo pipefail
    cd /workspace
    mkdir -p ${POETRY_VIRTUALENVS_PATH%/*}
    if command -v poetry &>/dev/null
      then poetry install --no-interaction --no-ansi
    fi"

srun --mpi=pmix -n ${NTASKS} \
  singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
  poetry run python -m ${ENTRY} --input-dir data/np32 --max-samples 1000

EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
  die "job exit code: ${EXIT_CODE}"
else
  info "job exit code: ${EXIT_CODE}"
fi
exit ${EXIT_CODE}
