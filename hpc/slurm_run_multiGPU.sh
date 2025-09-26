#!/bin/bash -l
#SBATCH --job-name=HPCP_MultiGPU_OPT
#SBATCH --partition=performance
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4            # 1 MPI rank per GPU
#SBATCH --gpus-per-task=1              # bind one GPU per rank
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread           # prefer physical cores for tighter binding
#SBATCH --mem=64G
#SBATCH --output=src/multiGPU/results_Test/logs/opt-%j.out
#SBATCH --error=src/multiGPU/results_Test/logs/opt-%j.err

set -euo pipefail

# -------------------------------
# Tunables (override via sbatch --export VAR=...)
# -------------------------------
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
ENTRY="${ENTRY:-src.multiGPU.main}"
INPUT_DIR="${INPUT_DIR:-data/np32}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/src/multiGPU/results_Test}"

PROFILE="${PROFILE:-0}"              # set 1 to enable Nsight Systems
NSYS_OPTS="${NSYS_OPTS:-cuda,nvtx,osrt,cublas,cusolver}"  # Enhanced for GPU kernel analysis

MULTIGPU_NVTX="${MULTIGPU_NVTX:-0}" # NVTX phase annotation toggle (Python + CuPy)
# CUDA streams overlap toggles (see PERFORMANCE_OPTIMIZATIONS.md)
MULTIGPU_STREAMS="${MULTIGPU_STREAMS:-1}"
MULTIGPU_STREAMS_DEPTH="${MULTIGPU_STREAMS_DEPTH:-2}"

# Nsight Systems output and temp dirs 
NSYS_OUT_DIR_HOST="${NSYS_OUT_DIR_HOST:-$LOG_ROOT/nsys}"
NSYS_TMP_DIR_HOST="${NSYS_TMP_DIR_HOST:-$LOG_ROOT/.nsys-tmp}"
# Map host paths to container-binded paths (REPO_DIR -> /workspace)
NSYS_OUT_DIR_CTR="/workspace${NSYS_OUT_DIR_HOST#$REPO_DIR}"
NSYS_TMP_DIR_CTR="/workspace${NSYS_TMP_DIR_HOST#$REPO_DIR}"

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT/logs"

# Performance tuning environment variables for high-resolution image processing
export MULTIGPU_BATCH_SIZE="${MULTIGPU_BATCH_SIZE:-0}"  # 0=adaptive, >0=override
export MULTIGPU_STABLE_PINV="${MULTIGPU_STABLE_PINV:-1}"  # 1=enable for large images
export MULTIGPU_KEEP_DEVICE="${MULTIGPU_KEEP_DEVICE:-1}"  # Keep arrays on device
export MULTIGPU_VECTOR_DISABLE="${MULTIGPU_VECTOR_DISABLE:-0}"  # Fallback to scalar path
export MULTIGPU_STREAMS="${MULTIGPU_STREAMS}"
export MULTIGPU_STREAMS_DEPTH="${MULTIGPU_STREAMS_DEPTH}"

# MPI / comm environment optimized for high-resolution image processing
export UCX_TLS=${UCX_TLS:-sm,self,cuda_copy,cuda_ipc,rc}  # reordered for intra-node first
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}
# Enhanced NCCL settings for large data transfers
export NCCL_ALGO=${NCCL_ALGO:-Tree,Ring}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-32}
# Ensure deterministic GPU ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CuPy / CUDA runtime tuning for high-resolution image processing
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/workspace/.cupy_cache}"
export CUPY_ACCELERATORS="cub"   
export CUPY_COMPILE_WITH_PTX=1
export CUPY_DUMP_CUDA_SOURCE_ON_ERROR=1
export PYTHONUNBUFFERED=1

# CuPy memory pool is configured programmatically in Python (see gpu_kernels.py)

# Normalize inconsistent Slurm CPU env vars (avoid srun fatal mismatch)
# Some environments export stale SLURM_CPUS_PER_TASK that disagrees with the
# TRES description (e.g. SLURM_CPUS_PER_TASK=10 while #SBATCH requests 2).
if [[ -n "${SLURM_TRES_PER_TASK:-}" ]] && [[ "${SLURM_TRES_PER_TASK}" =~ cpu:([0-9]+) ]]; then
  _tres_cpus="${BASH_REMATCH[1]}"
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "${_tres_cpus}" ]]; then
    echo "[OPT][${SLURM_JOB_ID:-noid}] Normalizing SLURM_CPUS_PER_TASK ${SLURM_CPUS_PER_TASK} -> ${_tres_cpus}" >&2
  fi
  export SLURM_CPUS_PER_TASK="${_tres_cpus}"
fi

# Threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

info(){ printf '[OPT][%s] %s\n' "${SLURM_JOB_ID:-noid}" "$*"; }
die(){ info "ERROR: $*"; exit 1; }

mkdir -p "$NSYS_OUT_DIR_HOST" "$NSYS_TMP_DIR_HOST"
info "Job on nodes: ${SLURM_JOB_NODELIST:-<none>}"
info "MPI tasks: ${SLURM_NTASKS:-unset}  GPUs/task: ${SLURM_GPUS_PER_TASK:-1}  CPUs/task: ${SLURM_CPUS_PER_TASK:-?}"

# GPU inventory and topology
if [[ "${SLURM_PROCID:-0}" == "0" ]] && command -v nvidia-smi &>/dev/null; then
  info "GPU inventory:"
  nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
  info "GPU topology for large image processing optimization:"
  nvidia-smi topo -m || true
fi

# Prepare poetry env inside container (performed once per job; races avoided by lock dir)
export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

# Simple file lock to avoid redundant installation storms
singularity exec --cleanenv --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
    bash -lc 'cd /workspace
    if command -v poetry &>/dev/null
      then poetry install --no-interaction --no-ansi
    fi'


PROFILE_CMD=""
if [[ "$PROFILE" == "1" ]]; then
  # Probe for nsys inside the container (not on host) to avoid srun spam.
  if singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" bash -lc 'command -v nsys >/dev/null 2>&1'; then
    # Create output and temp directories inside container
    mkdir -p "$NSYS_OUT_DIR_HOST" "$NSYS_TMP_DIR_HOST"
    
    # Use comprehensive nsys profile options for GPU kernel analysis
    PROFILE_CMD="nsys profile -t ${NSYS_OPTS} --force-overwrite=true --cuda-memory-usage=true --output=${NSYS_OUT_DIR_CTR}/nsys_rank%q{SLURM_PROCID}"
    
  # Set TMPDIR inside the container (not on the host) for nsys temp files
  # Use Singularity's env passthrough to avoid Slurm creating host paths
  export SINGULARITYENV_TMPDIR="${NSYS_TMP_DIR_CTR}"
    
    info "Nsight Systems detected; profiling enabled (${NSYS_OPTS}) with GPU metrics. Output -> ${NSYS_OUT_DIR_HOST}"
  else
    info "WARNING: nsys not found in container; continuing WITHOUT profiling. Set PROFILE=0 or rebuild image with Nsight Systems CLI."
    PROFILE_CMD=""
  fi
fi

# CPU/GPU binding strategy optimized for high-resolution image processing
SRUN_BIND="--cpu-bind=cores --gpu-bind=closest --distribution=block:block --mpi=pmix --kill-on-bad-exit=1"

srun ${SRUN_BIND} \
  --export=ALL,SINGULARITYENV_TMPDIR=${NSYS_TMP_DIR_CTR:-},MULTIGPU_NVTX=${MULTIGPU_NVTX},MULTIGPU_BATCH_SIZE=${MULTIGPU_BATCH_SIZE},MULTIGPU_STABLE_PINV=${MULTIGPU_STABLE_PINV},MULTIGPU_KEEP_DEVICE=${MULTIGPU_KEEP_DEVICE},MULTIGPU_VECTOR_DISABLE=${MULTIGPU_VECTOR_DISABLE},MULTIGPU_STREAMS=${MULTIGPU_STREAMS},MULTIGPU_STREAMS_DEPTH=${MULTIGPU_STREAMS_DEPTH} \
  singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
  bash -lc "cd /workspace && ${PROFILE_CMD} poetry run python -m ${ENTRY} --input-dir ${INPUT_DIR} --max-samples ${MAX_SAMPLES}"


SRUN_EXIT_CODE=$?

if [[ $SRUN_EXIT_CODE -ne 0 ]]; then
    die "srun failed with exit code $SRUN_EXIT_CODE"
fi

info "Completed with exit code $SRUN_EXIT_CODE"
info "NVTX profiling (MULTIGPU_NVTX=${MULTIGPU_NVTX}); PROFILE=${PROFILE} (nsys ${PROFILE_CMD:+enabled}${PROFILE_CMD:+' '}${PROFILE_CMD:-disabled})."
info "If you need nsys: (a) load a host module and run host-side nsys wrapping srun, or (b) extend the container with Nsight Systems CLI (see docs/NVTX_PROFILING.md)."

# Post-run: If any .qdstrm remain (e.g., packaging skipped due to abrupt exit), convert to .nsys-rep
if [[ "$PROFILE" == "1" ]]; then
  info "Post-processing Nsight traces in ${NSYS_OUT_DIR_HOST}"
  singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
    bash -lc "shopt -s nullglob; cd ${NSYS_OUT_DIR_CTR}; for q in *.qdstrm; do out=\"${q%.qdstrm}.nsys-rep\"; echo 'Converting' \"$q\" '->' \"$out\"; nsys convert --input \"$q\" --output \"$out\" || true; done"
fi