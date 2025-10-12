#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 24:00:00
#SBATCH --job-name=Performance_SingleGPU
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --chdir=${SLURM_SUBMIT_DIR}
#SBATCH --output=logs/singlegpu_%j.out
#SBATCH --error=logs/singlegpu_%j.err

set -euo pipefail

# ================
# Paths & entries
# ================
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
ENTRY="${ENTRY:-src.singleGPU.main}"

# Benchmark params (tweak via environment when sbatching)
DATA_DIR="${DATA_DIR:-data/np32}"
IDX="${IDX:-0}"                 # which frame: 0 / 1 / all
SIZES="${SIZES:-4096}"          # accepts 4096 or "4096x4096" depending on your code
TILE="${TILE:-256}"
NMU="${NMU:-42}"
BENCH_ROOT="${BENCH_ROOT:-benchmarking/singlegpu}"
SAVE_BENCH="${SAVE_BENCH:-first}"  # first / always / never

# Threading hygiene
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# Poetry caches inside the container mount
export POETRY_VIRTUALENVS_IN_PROJECT=1
export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

# Logs & output dirs
mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/${BENCH_ROOT}"

# -------------------------------
# Find container runtime (+ --nv)
# -------------------------------
module purge >/dev/null 2>&1 || true
module load apptainer  >/dev/null 2>&1 || \
module load singularity >/dev/null 2>&1 || true

if [[ -x /usr/bin/apptainer ]]; then
  RUNCTL=/usr/bin/apptainer
elif [[ -x /usr/bin/singularity ]]; then
  RUNCTL=/usr/bin/singularity
else
  RUNCTL="$(command -v apptainer || command -v singularity || true)"
fi

if [[ -z "${RUNCTL:-}" ]]; then
  echo "[FATAL] apptainer/singularity not found on PATH"; exit 127
fi

# Verify image exists
if [[ ! -f "${IMAGE}" ]]; then
  echo "[FATAL] Container not found: ${IMAGE}"; exit 2
fi

echo "============================================================"
echo "Single-GPU DEM Benchmark"
echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs        : ${SLURM_GPUS_PER_TASK:-1}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-2}"
echo "Memory      : ${SLURM_MEM_PER_NODE:-128G}"
echo "Container   : ${IMAGE}"
echo "Work dir    : ${REPO_DIR}"
echo ""
echo "Benchmark Configuration:"
echo "  Data dir  : ${DATA_DIR}"
echo "  Frame     : ${IDX}"
echo "  Size      : ${SIZES}"
echo "  Tile      : ${TILE}"
echo "  NMU       : ${NMU}"
echo "  Bench out : ${BENCH_ROOT}"
echo "============================================================"
echo ""

# -----------------------------
# Ensure Poetry in the image
# -----------------------------
"${RUNCTL}" exec --nv --cleanenv \
  --bind "${REPO_DIR}:/workspace" \
  "${IMAGE}" bash -lc '
    set -euo pipefail
    export PATH="$HOME/.local/bin:$PATH"
    cd /workspace
    if ! command -v poetry >/dev/null 2>&1; then
      echo "[INFO] poetry not found; installing user-local..."
      python -m pip install --user --upgrade pip
      python -m pip install --user "poetry<2"
    fi
    poetry --version
    poetry config virtualenvs.in-project true
    echo "[INFO] Installing project dependencies..."
    poetry install --no-interaction --no-ansi
  '

echo ""
echo "[INFO] Running single-GPU benchmark..."
echo ""

# -----------------------------
# Run (GPU passthrough via --nv)
# -----------------------------
DEVICE_ARG="${CUDA_VISIBLE_DEVICES:-0}"

"${RUNCTL}" exec --nv --cleanenv \
  --bind "${REPO_DIR}:/workspace" \
  "${IMAGE}" bash -lc "
    set -euo pipefail
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    export OMP_NUM_THREADS=${OMP_NUM_THREADS}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}
    export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}
    cd /workspace
    echo '[RUN] poetry run python -m ${ENTRY} ...'
    poetry run python -m \"${ENTRY}\" \
        --data-dir \"${DATA_DIR}\" \
        --idx \"${IDX}\" \
        --sizes \"${SIZES}\" \
        --tile \"${TILE}\" \
        --nmu \"${NMU}\" \
        --bench-root \"${BENCH_ROOT}\" \
        --device \"${DEVICE_ARG}\" \
        ${ARGS:-}
  "

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
  echo ""
  echo "ERROR: Single-GPU benchmark failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo ""
echo "============================================================"
echo "Single-GPU Benchmark Complete!"
echo "============================================================"
echo "Results directory: ${BENCH_ROOT}"
echo "Latest bench json: $(find "${REPO_DIR}/${BENCH_ROOT}" -name 'bench_*.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
echo "============================================================"
