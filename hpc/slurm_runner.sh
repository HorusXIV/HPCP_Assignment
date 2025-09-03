#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=Cluster_CKW_Data
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2


set -euo pipefail

# Absolute path to your project on the host
PROJECT_DIR=/home2/lukas.breiter/poetry_python/CICMC-L4-2

IMAGE="${IMAGE:-containers/python_poetry.sif}"

REPO_DIR="${REPO_DIR:-$PWD}"

module load singularity

echo "[SLURM] Using image: $IMAGE"
echo "[SLURM] Working dir: $REPO_DIR"

singularity exec --cleanenv \
  --bind "$REPO_DIR":/workspace \
  "$IMAGE" bash -lc '
    set -euo pipefail \
    cd /workspace \
    poetry install --no-interaction --no-ansi
    poetry run python run_full_pipeline.py \
  "