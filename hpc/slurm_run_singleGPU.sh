#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=Performance_SingleGPU
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
# Uncomment if you need GPUs:
#SBATCH --gres=gpu:1

set -euo pipefail

# Use the directory you submit the job from as the project root
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"

ENTRY="${ENTRY:-src/singleGPU/main.py}"

echo "[SLURM] Using image: $IMAGE"
echo "[SLURM] Repo dir:   $REPO_DIR"
echo "[SLURM] Entry:      $ENTRY"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

singularity exec --cleanenv --nv \
  --bind "$REPO_DIR":/workspace \
  "$IMAGE" bash -lc "
    set -euo pipefail
    cd /workspace
    poetry install --no-interaction --no-ansi
    poetry run python \"$ENTRY\"
"
