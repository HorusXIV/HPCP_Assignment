#!/bin/bash -l
##
## SLURM multi-GPU launcher (enhanced)
##
## Usage (submit with `sbatch hpc/slurm_run_multiGPU.sh`)
## Override environment variables if needed, e.g.:
##   sbatch --export=REPO_DIR=/path/to/repo,IMAGE=/path/to/image hpc/slurm_run_multiGPU.sh
##

#SBATCH --job-name=HPCP_MultiGPU
#SBATCH --partition=performance
#SBATCH --time=24:00:00               # walltime (DD-HH:MM:SS or HH:MM:SS)
#SBATCH --nodes=1                     # set >1 for multi-node runs
#SBATCH --ntasks-per-node=1           # usually 1 task that will spawn mpi ranks
#SBATCH --cpus-per-task=8             # CPUs per task (adjust for your workload)
#SBATCH --mem=128G                    # memory per node
## request GPUs per node (change 4 to number of gpus per node you need)
#SBATCH --gres=gpu:4
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

set -euo pipefail

# Repository and container defaults (overridable via sbatch --export)
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
ENTRY_MODULE="${ENTRY_MODULE:-src.multiGPU.main}"
PY_ENV_ACTIVATE="${PY_ENV_ACTIVATE:-}"  # optional: path to venv/conda activate script

echo "[SLURM] Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "[SLURM] Repo dir: ${REPO_DIR}"
echo "[SLURM] Image:    ${IMAGE}"
echo "[SLURM] Entry:    ${ENTRY_MODULE}"

# Setup OpenMP / thread limits
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"

# Helper: report allocated GPUs (slurm sets CUDA_VISIBLE_DEVICES on some systems)
echo "[SLURM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set>}"
if command -v nvidia-smi &>/dev/null; then
  echo "[SLURM] nvidia-smi summary:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
fi

## Environment setup options (choose one strategy):
## 1) Use site modules (recommended on clusters)
##    module load python/3.11 cuda/12.2 openmpi/4.1
## 2) Use a container image (Singularity/Apptainer)
## 3) Activate a project virtualenv/conda

if [[ -n "${PY_ENV_ACTIVATE}" ]]; then
  echo "[SLURM] Activating Python environment: ${PY_ENV_ACTIVATE}"
  # shellcheck disable=SC1091
  source "${PY_ENV_ACTIVATE}"
fi

# If using site modules, uncomment and adapt these lines to your cluster
# module purge
# module load python/3.11 cuda/12.2 openmpi/4.1

# Choose launch method: container or direct
USE_SINGULARITY=${USE_SINGULARITY:-1}

if [[ "$USE_SINGULARITY" -eq 1 ]]; then
  echo "[SLURM] Running inside Singularity image: ${IMAGE}"

  singularity exec --cleanenv --nv --bind "$REPO_DIR":/workspace "$IMAGE" bash -lc "
    set -euo pipefail
    cd /workspace
    # Ensure dependencies are installed in image or via poetry
    if command -v poetry &>/dev/null; then
      poetry install --no-interaction --no-ansi
    fi
    # Prefer srun for MPI-aware execution; `srun` will propagate SLURM env vars
    srun --mpi=pmix --ntasks=
      \\"${SLURM_NTASKS:-1}\\" poetry run python -m \"${ENTRY_MODULE}\"
  "
else
  echo "[SLURM] Running directly on node"
  # Ensure Python environment is ready (virtualenv/conda) or modules are loaded
  if command -v poetry &>/dev/null; then
    poetry install --no-interaction --no-ansi
  fi

  # For MPI-enabled launches, use srun. For single-node multi-GPU, set NTASKS to number of GPUs/ranks
  NTASKS=${NTASKS:-${SLURM_NTASKS:-1}}
  echo "[SLURM] srun --mpi=pmix -n ${NTASKS} poetry run python -m ${ENTRY_MODULE}"
  srun --mpi=pmix -n ${NTASKS} poetry run python -m ${ENTRY_MODULE}
fi

EXIT_CODE=$?
echo "[SLURM] job exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
