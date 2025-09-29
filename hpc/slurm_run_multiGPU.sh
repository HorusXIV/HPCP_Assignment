#!/bin/bash -l
#SBATCH --job-name=HPCP_MultiGPU
#SBATCH --partition=performance
#SBATCH --time=24:00:00
#SBATCH --nodes=1                 # adjust as needed   
#SBATCH --ntasks-per-node=4       # depend on results / nodes
#SBATCH --gpus-per-task=1         # 1 GPU per MPI rank
#SBATCH --cpus-per-task=3         # Enough to keep GPUs fed
#SBATCH --hint=nomultithread      # disable hyperthreading
#SBATCH --mem=32G                 # rather big memory for large images
#SBATCH --signal=USR1@60          # preemptive signal for cleanup
#SBATCH --output=src/multiGPU/logs/opt-%j.out
#SBATCH --error=src/multiGPU/logs/opt-%j.err

# could add #SBATCH --exclusive

set -euo pipefail

# Tunables (override via: sbatch --export VAR=...)
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-$REPO_DIR/containers/python_poetry.sif}"
ENTRY="${ENTRY:-src.multiGPU.main}"
INPUT_DIR="${INPUT_DIR:-data/np32}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/src/multiGPU/logs}"

PROFILE="${PROFILE:-0}"              # 1 = enable Nsight Systems (nsys)
NSYS_OPTS="${NSYS_OPTS:-cuda,nvtx,osrt,cublas,cusolver}"

MULTIGPU_NVTX="${MULTIGPU_NVTX:-0}" # NVTX ranges in kernels
MULTIGPU_STREAMS="${MULTIGPU_STREAMS:-1}" # overlap compute/transfers
MULTIGPU_STREAMS_DEPTH="${MULTIGPU_STREAMS_DEPTH:-2}"

# Nsight Systems output and temp dirs
NSYS_OUT_DIR_HOST="${NSYS_OUT_DIR_HOST:-$LOG_ROOT/nsys}"
NSYS_TMP_DIR_HOST="${NSYS_TMP_DIR_HOST:-$LOG_ROOT/.nsys-tmp}"

NSYS_OUT_DIR_CTR="/workspace${NSYS_OUT_DIR_HOST#$REPO_DIR}"
NSYS_TMP_DIR_CTR="/workspace${NSYS_TMP_DIR_HOST#$REPO_DIR}"

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT/rank_logs"

export MULTIGPU_BATCH_SIZE="${MULTIGPU_BATCH_SIZE:-0}"  # 0=auto, >0=override
export MULTIGPU_STABLE_PINV="${MULTIGPU_STABLE_PINV:-1}"
export MULTIGPU_KEEP_DEVICE="${MULTIGPU_KEEP_DEVICE:-1}"
export MULTIGPU_VECTOR_DISABLE="${MULTIGPU_VECTOR_DISABLE:-0}"
export MULTIGPU_STREAMS="${MULTIGPU_STREAMS}"
export MULTIGPU_STREAMS_DEPTH="${MULTIGPU_STREAMS_DEPTH}"

export UCX_TLS=${UCX_TLS:-sm,self,cuda_copy,cuda_ipc,rc}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}
export NCCL_ALGO=${NCCL_ALGO:-Tree,Ring}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-32}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/workspace/.cupy_cache}"
export CUPY_ACCELERATORS="cub"   
export CUPY_COMPILE_WITH_PTX=1
export CUPY_DUMP_CUDA_SOURCE_ON_ERROR=1
export PYTHONUNBUFFERED=1

# Normalize SLURM_CPUS_PER_TASK from TRES if mismatched
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

# Brief GPU inventory/topology on rank 0
if [[ "${SLURM_PROCID:-0}" == "0" ]] && command -v nvidia-smi &>/dev/null; then
  info "GPU inventory:"
  nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
  info "GPU topology for large image processing optimization:"
  nvidia-smi topo -m || true
fi

# Prepare poetry env inside container (idempotent per job)
export POETRY_VIRTUALENVS_PATH="/workspace/.venv"
export POETRY_CACHE_DIR="/workspace/.cache/pypoetry"

singularity exec --cleanenv --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
  bash -lc 'set -e; cd /workspace; \
    if command -v poetry &>/dev/null; then \
      if [ ! -f .venv/.installed ]; then \
        poetry install --no-interaction --no-ansi; \
        touch .venv/.installed; \
      fi; \
    fi'


PROFILE_CMD=""
if [[ "$PROFILE" == "1" ]]; then
  if singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" bash -lc 'command -v nsys >/dev/null 2>&1'; then
    mkdir -p "$NSYS_OUT_DIR_HOST" "$NSYS_TMP_DIR_HOST"
    PROFILE_CMD="nsys profile -t ${NSYS_OPTS} --force-overwrite=true --cuda-memory-usage=true --output=${NSYS_OUT_DIR_CTR}/nsys_rank%q{SLURM_PROCID}"
    export SINGULARITYENV_TMPDIR="${NSYS_TMP_DIR_CTR}"
    info "Nsight Systems enabled (${NSYS_OPTS}); output: ${NSYS_OUT_DIR_HOST}"
  else
    info "nsys not found in container; profiling disabled"
    PROFILE_CMD=""
  fi
fi

# CPU/GPU binding strategy
SRUN_BIND="--cpu-bind=cores --gpu-bind=closest --distribution=block:block --mpi=pmix --kill-on-bad-exit=1"

srun ${SRUN_BIND} \
  --export=ALL,SINGULARITYENV_TMPDIR=${NSYS_TMP_DIR_CTR:-},MULTIGPU_NVTX=${MULTIGPU_NVTX},MULTIGPU_BATCH_SIZE=${MULTIGPU_BATCH_SIZE},MULTIGPU_STABLE_PINV=${MULTIGPU_STABLE_PINV},MULTIGPU_KEEP_DEVICE=${MULTIGPU_KEEP_DEVICE},MULTIGPU_VECTOR_DISABLE=${MULTIGPU_VECTOR_DISABLE},MULTIGPU_STREAMS=${MULTIGPU_STREAMS},MULTIGPU_STREAMS_DEPTH=${MULTIGPU_STREAMS_DEPTH} \
  singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
  bash -lc "cd /workspace && ${PROFILE_CMD} poetry run python -m ${ENTRY} --input-dir ${INPUT_DIR}"


SRUN_EXIT_CODE=$?

if [[ $SRUN_EXIT_CODE -ne 0 ]]; then
    die "srun failed with exit code $SRUN_EXIT_CODE"
fi

info "Completed with exit code $SRUN_EXIT_CODE"
info "NVTX profiling (MULTIGPU_NVTX=${MULTIGPU_NVTX}); PROFILE=${PROFILE}"

# Post-run: If any .qdstrm remain (e.g., packaging skipped due to abrupt exit), convert to .nsys-rep
if [[ "$PROFILE" == "1" ]]; then
  info "Post-processing Nsight traces in ${NSYS_OUT_DIR_HOST}"
  singularity exec --nv --bind "$REPO_DIR":/workspace "$IMAGE" \
    bash -lc "shopt -s nullglob; cd ${NSYS_OUT_DIR_CTR}; for q in *.qdstrm; do out=\"\${q%.qdstrm}.nsys-rep\"; echo 'Converting' \"\$q\" '->' \"\$out\"; nsys convert --input \"\$q\" --output \"\$out\" || true; done"
fi