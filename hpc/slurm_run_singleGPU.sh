#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=Performance_SingleGPU
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

set -euo pipefail

# Repo and environment
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
ENV_DIR="${ENV_DIR:-$REPO_DIR/env/HPCP}"
ENTRY="${ENTRY:-src.singleGPU.main}"

# Activate environment
if [ -f "$ENV_DIR/bin/activate" ]; then
  source "$ENV_DIR/bin/activate"
else
  echo "[SLURM][ERROR] Conda env not found at $ENV_DIR"
  exit 1
fi

cd "$REPO_DIR"

# Optional: set OMP threads to match SLURM allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# Run single-GPU benchmark
python -m "$ENTRY" \
    --data-dir "data/np32" \
    --sizes "4096x4096" \
    --tile "256" \
    --nmu 42 \
    --bench-root "benchmark_results" \
    --device "${CUDA_VISIBLE_DEVICES:-0}" \
    ${ARGS:-}
