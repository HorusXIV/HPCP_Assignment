#!/bin/bash -l
#SBATCH --partition=performance
#SBATCH --time=04:00:00
#SBATCH --job-name=dask
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=logs/dask_%j.out
#SBATCH --error=logs/dask_%j.err

set -euo pipefail

# ============================================================
# Dask Distributed DEM Runner
# ============================================================

# Configuration
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
ENTRY="${ENTRY:-src.dask.main}"

# Arguments (customize as needed)
SIZES="${SIZES:-4096}"        # Frame size (256, 512, 1024, 2048)
TILE="${TILE:-512}"          # Tile size (64, 128, 256)
WORKERS="${WORKERS:-16}"      # Number of Dask workers (recommend 50% of CPUs)
FRAMES="${FRAMES:-0}"      # Which frames: "all" or "0"
NMU="${NMU:-42}"            # Regularization parameter

# CRITICAL: Single-threaded workers to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Disable vendor's nested parallelism
export HPCP_INNER_PROCS=1

# Poetry cache
export POETRY_VIRTUALENVS_IN_PROJECT=1
export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

# Dask configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="60s"
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.80
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.85
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95

# Use SLURM temporary directory if available
if [[ -n "${TMPDIR:-}" ]]; then
    export DASK_TEMPORARY_DIRECTORY="${TMPDIR}"
fi

# Create logs directory
mkdir -p "${REPO_DIR}/logs"

echo "============================================================"
echo "Dask Distributed DEM Runner"
echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURMD_NODENAME}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Memory      : 48GB"
echo "Container   : ${IMAGE}"
echo "Work dir    : ${REPO_DIR}"
echo ""
echo "Configuration:"
echo "  Size      : ${SIZES}x${SIZES}"
echo "  Tile      : ${TILE}x${TILE}"
echo "  Workers   : ${WORKERS}"
echo "  Frames    : ${FRAMES}"
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
echo "[INFO] Running Dask..."
echo ""

# Run Dask
singularity exec --cleanenv \
    --bind "${REPO_DIR}:/workspace" \
    "${IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd /workspace
        poetry run python -m ${ENTRY} \
            --sizes ${SIZES} \
            --tile ${TILE} \
            --n-workers ${WORKERS} \
            --idx ${FRAMES} \
            --nmu ${NMU}
    "

echo ""
echo "============================================================"
echo "Dask run complete!"
echo "Check benchmarking/dask/ for results"
echo "============================================================"