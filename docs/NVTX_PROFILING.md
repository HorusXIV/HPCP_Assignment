# NVTX Profiling & Nsight Workflow

This project includes lightweight NVTX instrumentation for the multi-GPU (MPI) path to help you correlate host phases, MPI collectives, and GPU work in NVIDIA Nsight Systems / Nsight Compute.

## Overview
NVTX ranges are inserted in `src/multiGPU/main.py` and inside GPU kernels (`gpu_kernels.py`) when enabled. They are disabled by default and introduce near-zero overhead when off.

Key high-level phase labels:
- `INIT_MPI` – MPI initialization & communicator setup
- `RANK_GPU_BIND` – Rank-to-GPU binding
- `ENUM_INPUTS` / `BCAST_INPUTS` – Input discovery & distribution
- `PROCESS_FILE:<filename>` – Wrapper for per-file processing (encloses all below)
- `LOAD_FILE` – Rank 0 file I/O & preparation
- `BCAST_COUNTS`, `BCAST_DTYPES` – Small control broadcasts
- `SCATTER_DN`, `SCATTER_EDN` – Distributed scatterv operations
- `GPU_COMPUTE` – Batched device-side reconstruction loops
- `GATHER_DEM` – gatherv aggregation
- `POST_GATHER_BARRIER` – Barrier after gather (if any)
- `SAVE_RESULTS` – Compressed output save
- `SHUTDOWN` – Logging & finalization

Fine-grained device ranges (inside `gpu_kernels.demmap_pos` when `MULTIGPU_NVTX=1`):
- `BATCH_PREP` – Host->device prep per batch
- `SVD` – CuPy batched SVD
- `LAMBDA_SELECT` – Regularization parameter selection
- `BUILD_W` – Host reconstruction matrices
- `RECON_VECTOR` / `RECON_FALLBACK_LOOP` – Vectorized reconstruction or per-sample fallback

## Enabling NVTX
1. Install optional profiling dependency (once):
```bash
poetry install --with profiling
```
2. Export environment variable (shell or inside Slurm script):
```bash
export MULTIGPU_NVTX=1
```
3. (Optional) Enable Nsight Systems profiling: set `PROFILE=1` in the Slurm optimized script or run `nsys` manually.

The Slurm script `hpc/slurm_run_multiGPU_optimized.sh` accepts:
```bash
sbatch --export=ALL,MULTIGPU_NVTX=1,PROFILE=1 hpc/slurm_run_multiGPU_optimized.sh
```

## Running Locally with Nsight Systems
Example (single node, interactive):
```bash
MULTIGPU_NVTX=1 nsys profile -t cuda,nvtx,osrt -o nsys_run \
  poetry run python -m src.multiGPU.main --input-dir data/np32 --max-samples 512
```

Afterward open `nsys_run.qdrep` in Nsight Systems GUI.

## Coloring
Distinct hexadecimal RGB colors are assigned to major phases for visual grouping. Adjust if needed by editing `src/multiGPU/main.py` color arguments.

## Disabling Quickly
Unset or set to zero:
```bash
unset MULTIGPU_NVTX  # or
export MULTIGPU_NVTX=0
```
No code changes required; instrumentation becomes a no-op.

## Overhead
- Disabled (`MULTIGPU_NVTX=0`): only an env check (fast path) per range entry.
- Enabled: NVTX push/pop (microseconds) compared to millisecond+ GPU work; negligible for current batch sizes.

## Verifying NVTX Presence
If the optional `nvtx` Python package is not installed, ranges silently degrade to no-ops even if `MULTIGPU_NVTX=1`. Confirm in a Python shell:
```python
from src.common.nvtx import nvtx_available
print(nvtx_available())  # True when active
```

## Nsight Systems Tips
- Filter timeline by NVTX category to declutter.
- Correlate `SCATTER_*` and `GATHER_DEM` with MPI timelines to spot imbalance.
- Zoom into `GPU_COMPUTE` to see internal batch phases (`SVD`, `RECON_VECTOR`).
- If batches are irregularly sized, check free memory heuristic vs actual GPU memory occupancy.

## Nsight Compute (Kernel Detail)
Launch Nsight Compute on a representative kernel while NVTX is enabled for context:
```bash
MULTIGPU_NVTX=1 ncu --set full --target-processes all \
  poetry run python -m src.multiGPU.main --input-dir data/np32 --max-samples 256
```
Then correlate kernel names with surrounding NVTX ranges from Systems.

## Extending Instrumentation
Use the helper:
```python
from src.common.nvtx import nvtx_range
with nvtx_range("NEW_PHASE"):
    ...
```
Keep labels concise (< 48 chars) and reuse colors for logical groups.

## Troubleshooting
| Symptom | Cause | Action |
|---------|-------|--------|
| No NVTX ranges in nsys timeline | `MULTIGPU_NVTX` unset or `nvtx` pkg missing | Set `MULTIGPU_NVTX=1` and install profiling extras |
| Ranges appear only for some ranks | Ranks launched without exported env var | Ensure `--export=ALL,MULTIGPU_NVTX=1` in `srun` |
| Excessive timeline clutter | Over-instrumentation | Disable fine-grained (comment or regroup) |
| ImportError for `nvtx` | Optional dep not installed | `poetry install --with profiling` |

## Change Summary
- Added optional dependency group `profiling` with `nvtx`.
- Added env-gated helper `nvtx_range` with color support.
- Instrumented major multi-GPU phases & GPU batch internals.
- Slurm script now exposes `MULTIGPU_NVTX` toggle.

Happy profiling!
