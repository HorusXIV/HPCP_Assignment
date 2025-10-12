#!/bin/bash -l
#SBATCH --partition=performance
#SBATCH --time=02:00:00
#SBATCH --job-name=baseline_4096
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/baseline_4096_%j.out
#SBATCH --error=logs/baseline_4096_%j.err

set -euo pipefail

# ============================================================
# Baseline DEM Benchmark - Size 512, 3 Repeats
# ============================================================

# Configuration
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
ENTRY="${ENTRY:-src.baseline.main}"

# Benchmark parameters
DATA_DIR="${DATA_DIR:-data/np32}"
IDX="${IDX:-0}"         # Which frame: "0", "1", "all", etc.
SIZES="4096"
REPEATS="3"
NMU="42"

# Threading - use all allocated CPUs for baseline
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONUNBUFFERED=1

# Poetry cache
export POETRY_VIRTUALENVS_IN_PROJECT=1
export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

# Create logs directory
mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/benchmark_out/baseline"

echo "============================================================"
echo "Baseline DEM Benchmark - Size 512x512"
echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURMD_NODENAME}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Memory      : 32GB"
echo "Container   : ${IMAGE}"
echo "Work dir    : ${REPO_DIR}"
echo ""
echo "Benchmark Configuration:"
echo "  Data dir  : ${DATA_DIR}"
echo "  Frame     : ${IDX}"
echo "  Size      : ${SIZES}x${SIZES}"
echo "  Repeats   : ${REPEATS}"
echo "  NMU       : ${NMU}"
echo "============================================================"
echo ""

# Verify container exists
if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: Container not found: ${IMAGE}"
    exit 1
fi

# Install dependencies (once)
echo "[INFO] Installing dependencies..."
singularity exec --cleanenv \
    --bind "${REPO_DIR}:/workspace" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd /workspace
        poetry install --no-interaction --no-ansi
    "

echo ""
echo "[INFO] Running baseline benchmark..."
echo ""

# Run baseline benchmark
singularity exec --cleanenv \
    --bind "${REPO_DIR}:/workspace" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd /workspace
        poetry run python -m ${ENTRY} \
            --data-dir ${DATA_DIR} \
            --idx ${IDX} \
            --sizes ${SIZES} \
            --repeats ${REPEATS} \
            --nmu ${NMU} \
            --outdir benchmarking/baseline \
            --save-benchmark first
    "

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "ERROR: Baseline benchmark failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "============================================================"
echo "Results:"
echo "  CSV:      benchmark_out/baseline/wallclock/wallclock.csv"
echo "  Markdown: benchmark_out/baseline/wallclock/wallclock.md"
echo "  NPZ:      data/output/baseline/ (first repeat)"
echo ""
echo "To view results:"
echo "  cat benchmark_out/baseline/wallclock/wallclock.md"
echo "============================================================"