#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=Performance_Dask
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Use the directory you submit the job from as the project root
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"


ENTRY="${ENTRY:-src/dask/main.py}"

echo "[SLURM] Using image: $IMAGE"
echo "[SLURM] Repo dir:   $REPO_DIR"
echo "[SLURM] Entry:      $ENTRY"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

singularity exec --cleanenv \
  --bind "$REPO_DIR":/workspace \
  "$IMAGE" bash -lc "
    set -euo pipefail
    cd /workspace
    poetry install --no-interaction --no-ansi
    poetry run python \"$ENTRY\"
"
