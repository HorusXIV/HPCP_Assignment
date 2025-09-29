multiGPU module
==============================

Purpose
-------
`src.multiGPU` contains GPU-oriented numerical kernels and small MPI
helpers to run the project's DEM inversion workloads on multi-GPU clusters.

Contents
--------
- `gpu_kernels.py` — GPU-first numerical kernels (expects CuPy at import). This
  module is GPU-only: `demmap_pos` will raise `RuntimeError` when no CUDA
  devices are available, rather than silently falling back to a CPU path.
- `mpi_manager.py` — MPI helpers, scatter/gather, and GPU mapping.
- `main.py` — CLI entrypoint that initializes MPI, maps GPUs, and runs
  batched processing per rank.

Essential CuPy / CUDA fixes
---------------------------
If the module fails due to CuPy/CUDA errors, follow these minimal steps:

1) Confirm GPUs and drivers are visible

```pwsh
nvidia-smi
```

2) Install a CuPy binary wheel that matches your CUDA driver

Activate the Poetry venv and install the matching wheel with pip inside
that venv (recommended when Poetry cannot fetch the wheel directly):

multiGPU README
===============

Overview
--------
`src.multiGPU` provides a compact multi-GPU implementation of the DEM (Differential Emission Measure) reconstruction pipeline for HPC clusters. It uses:
- MPI (via `mpi4py`) for rank orchestration and data distribution
- CuPy for GPU-accelerated linear algebra (SVD-based solver)
- Optional CUDA streams + pinned memory for overlap of compute and transfers

Key components
- `gpu_kernels.py`: Batched, GPU-first kernels (CuPy), adaptive batch sizing, optional NVTX ranges and fused kernels.
- `mpi_manager.py`: Rank/GPU mapping, robust scatter/gather helpers, barriers.
- `main.py`: CLI orchestrator; enumerates inputs, scatters work, runs kernels, gathers/saves outputs.
- `logging.py`: Rank-aware logging with minimal console noise.
- `preempt.py`: Best‑effort preemption handlers for schedulers.
- `checkpoint.py`: Atomic checkpoint saves with optional async write.

Prerequisites
-------------
Hardware
- NVIDIA GPUs with recent CUDA support (tested with CUDA 12.x; CUDA 11.x also works with matching CuPy wheel)
- Nodes with NVLink preferred for intra-node P2P; IB- or high-speed fabric for inter-node (optional)

Software
- Python 3.11+ (project uses Poetry for dependency management)
- CUDA driver compatible with your chosen CuPy wheel (e.g., CUDA 12.x)
- CuPy: install a prebuilt wheel matching your CUDA version (e.g., `cupy-cuda12x`)
- mpi4py + MPI runtime (e.g., OpenMPI/PMIx on cluster)
- Optional: Singularity/Apptainer to run the provided container (`containers/python_poetry.def`)
- Optional: Nsight Systems CLI (`nsys`) inside container for profiling

Cluster Middleware
- Slurm (the provided launcher `hpc/slurm_run_multiGPU.sh` assumes Slurm + Singularity)

Installation
------------
Option A — Container-first (recommended on clusters)
1) Build the container (Singularity/Apptainer). Example (on a Linux host with Singularity):
   - See `containers/python_poetry.def` to build an SIF image.
2) Submit jobs with `hpc/slurm_run_multiGPU.sh`. The script mounts the repo into `/workspace`, sets up Poetry in-container, and runs the orchestrator.

Option B — Host install (development)
1) Install Poetry and create the venv:
   ```pwsh
   poetry install
   ```
2) Install a matching CuPy wheel inside the Poetry venv (example for CUDA 12.x):
   ```pwsh
   poetry run python -m pip install "cupy-cuda12x==13.6.0"
   ```
   For CUDA 11.x, use `cupy-cuda11x`. Verify CuPy and GPUs from the same env:
   ```pwsh
   poetry run python -c "import cupy as cp; print(cp.__version__, cp.cuda.runtime.getDeviceCount())"
   ```
3) Ensure `mpi4py` can locate your MPI runtime (on Windows, prefer WSL or Linux for MPI/GPU work).

Execution
---------
Data expectations
- Preferred input: `dn` and `edn` arrays shaped `(n_pixels, n_filters)`; `edn` may be `(1, n_filters)` to broadcast.
- Alternate: `bands` layout `(n_filters, ny, nx)`; the loader flattens to `(n_pixels, n_filters)` and can subsample for quick tests.

Entrypoint options
1) Slurm launcher (recommended)
   - Submit with defaults:
     ```bash
     sbatch hpc/slurm_run_multiGPU.sh
     ```
   - Useful overrides (pass via `--export` or `export` before sbatch):
     - `ENTRY=src.multiGPU.main` (Python `-m` module to run)
     - `INPUT_DIR=data/np32`
     - `PROFILE=1` (enable Nsight Systems if `nsys` is in the container)
     - `MULTIGPU_NVTX=1` (NVTX ranges in kernels)
     - `MULTIGPU_STREAMS=1` (async D2H overlap; default on)
     - `MULTIGPU_BATCH_SIZE=<N>` (override adaptive batch; 0 = auto)

Environment toggles (complete)
------------------------------
These variables can be exported before `sbatch` or passed via `--export` to `hpc/slurm_run_multiGPU.sh`.

- Job/entry selection
  - `ENTRY` (str): Python module to run (default: `src.multiGPU.main`).
  - `INPUT_DIR` (path): Folder with `.npz` inputs (default: `data/np32`).
  - `LOG_ROOT` (path): Root for rank logs and Nsight traces (default: `src/multiGPU/logs`).

- Multi-GPU runtime (code-level)
  - `MULTIGPU_BATCH_SIZE` (int): 0 = auto (adaptive by free mem). >0 forces a fixed batch size.
  - `MULTIGPU_NVTX` (0/1): Enable Python-level NVTX ranges (requires `poetry install --with profiling`).
  - `MULTIGPU_STREAMS` (0/1): Enable CUDA streams + pinned host buffers for async D2H overlap (default 1).
  - `MULTIGPU_STREAMS_DEPTH` (int): Pinned ring buffer depth for async copies (default 2; 2–3 typical).
  - `MULTIGPU_NO_FUSE` (0/1): Disable `cp.fuse()` wrappers in kernels (debug/determinism; default 0).
  - `MULTIGPU_VERBOSE` (0/1): Extra info logs in kernel paths (default 0).
  - `MULTIGPU_PREEMPT` (0/1): Register preemption signal handlers (best-effort checkpoint hook; default 0).
  - `MULTIGPU_SAVE_COMPRESSED` (0/1): Save aggregated outputs with compression (default 0 = plain `.npz`).
  - Logging controls: `MULTIGPU_LOG_LEVEL` (e.g., `INFO`), `MULTIGPU_QUIET` (1 to suppress per-rank files at WARNING+), `MULTIGPU_RANK_FILES` (1 to force per-rank logs).

- Profiling (Nsight Systems via Slurm script)
  - `PROFILE` (0/1): Enable `nsys profile` in the container.
  - `NSYS_OPTS` (csv): Trace domains (default `cuda,nvtx,osrt,cublas,cusolver`).
  - `NSYS_OUT_DIR_HOST` / `NSYS_TMP_DIR_HOST`: Host output and temp directories for Nsight files.

- Communication/NCCL/UCX (Slurm script)
  - `UCX_TLS`, `UCX_NET_DEVICES`: UCX transport selection.
  - `NCCL_DEBUG`, `NCCL_P2P_LEVEL`, `NCCL_*`: NCCL tuning variables (defaults are safe).
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`: Stable PCI ordering.

2) Manual run (single node, already on a GPU host)
   - Poetry, local filesystem:
     ```pwsh
     # PowerShell
     $env:MULTIGPU_NVTX = "1"
     $env:MULTIGPU_STREAMS = "1"
     poetry run python -m src.multiGPU.main --input-dir data/np32
     ```
   - With `srun` binding GPUs to ranks (1 rank/GPU):
     ```bash
     srun --ntasks-per-node=4 --gpus-per-task=1 \
          --cpu-bind=cores --gpu-bind=closest --mpi=pmix \
          poetry run python -m src.multiGPU.main --input-dir data/np32
     ```

CLI arguments (from `main.py`)
- `--input-dir <path>`: Folder containing `.npz` files. Rank 0 enumerates and broadcasts the list.
- `--max-samples <N>`: Optional cap for pixels per file (quick sampling when flattening `bands`).

Outputs
- Aggregated per-file output saved by rank 0 to `data/results_multiGPU/dem_all_<input>.npz` (toggle compression with `MULTIGPU_SAVE_COMPRESSED=1`).

Behavior on systems without GPUs
--------------------------------
This module is designed for GPU execution. When `cp.cuda.runtime.getDeviceCount()`
reports 0, `src.multiGPU.gpu_kernels.demmap_pos` raises:

> RuntimeError: No CUDA device available. multiGPU.demmap_pos requires a GPU.

This explicit error helps catch misconfigured environments early (e.g., missing
`--nv` in Singularity, empty `CUDA_VISIBLE_DEVICES`, or running locally without a GPU).

Operational environment variables (selected)
- GPU kernel controls (see `gpu_kernels.py`):
  - `MULTIGPU_BATCH_SIZE` (int): 0 = adaptive (default). Set > 0 to override.
  - `MULTIGPU_NVTX` (0/1): Add NVTX ranges in kernels.
  - `MULTIGPU_STREAMS` (0/1): Enable CUDA streams + pinned async D2H.
  - `MULTIGPU_STREAMS_DEPTH` (int): Ring-buffer depth for pinned staging (default 2).
  - `MULTIGPU_NO_FUSE` (0/1): Disable cp.fuse wrappers for debugging.
  - `MULTIGPU_VERBOSE` (0/1): Extra logs (batching, memory pool).
  - `MULTIGPU_PREEMPT` (0/1): Register preemption handlers.
- NCCL/UCX tuning (in `slurm_run_multiGPU.sh`):
  - `UCX_TLS`, `NCCL_DEBUG`, `NCCL_P2P_LEVEL`, `NCCL_ALGO`, `NCCL_*CHANNELS`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

Performance considerations
--------------------------
Batch size and memory
- Adaptive batching uses current free GPU memory and dominant tensor sizes (SVD + lambda search) to choose batch size.
- Override with `MULTIGPU_BATCH_SIZE` for predictable behavior (e.g., performance sweeps).

Overlap compute and transfers
- `MULTIGPU_STREAMS=1` enables asynchronous device-to-host copies into pinned buffers using a copy stream while the compute stream proceeds with the next batch.
- Tune `MULTIGPU_STREAMS_DEPTH` (2–3 typically sufficient) for deeper overlap.

CuPy and memory pools
- CuPy’s default memory pool is leveraged implicitly; heavy runs benefit from fewer allocs and better reuse. The workspace manager can free blocks between very large runs.

Profiling and visibility
- Set `MULTIGPU_NVTX=1` for NVTX ranges (works with Nsight Systems/Compute).
- Enable `PROFILE=1` in the Slurm script to run under `nsys` inside the container. Traces are written to `data/results_multiGPU/nsys/` and converted to `.nsys-rep` after the run.

NVTX legend
-----------
Enable with `MULTIGPU_NVTX=1`. You’ll see the following ranges in Nsight:
- GPU kernel phases (colors shown when Python-level NVTX is active):
  - `DEM_SOLVE_INIT` — 0x455A64 (setup: device copies, matrices, batch size)
  - `STREAMS_INIT` — 0x00796B (create compute/copy streams)
  - `PINNED_POOL_INIT` — 0x303F9F (allocate pinned host buffers)
  - `BATCH[b0:b1)` — 0xFF6F00 (per-batch envelope) containing:
    - `BATCH_PREP` (batch slicing, response prep)
    - `SVD` (batched rectangular SVD)
    - `LAMBDA_SELECT` (per-sample regularization search)
    - `RECONSTRUCTION_CALC` (DEM reconstruction + predictions)
    - `DEVICE_TO_HOST` (D2H copies; may be async when streams enabled)
- MPI collectives (appear around data distribution/aggregation):
  - `MPI.Scatterv`
  - `MPI.Gatherv`
  - `MPI.Barrier`

Notes
- If the `nvtx` Python package isn’t present, tags are no-ops; Nsight CUDA-level tags may still appear when supported by CuPy.
- Colors are applied to the top-level Python NVTX ranges listed above; inner CUDA ranges use default colors.

Binding and locality
- The launcher uses `--cpu-bind=cores` and `--gpu-bind=closest` with `--distribution=block:block`, which works well for 1 rank/GPU on a single node. On multi-node runs, ensure network fabrics are configured (UCX/NCCL) and consider IB vs. SHM path selection.

Scalability notes
- Work is embarrassingly parallel across pixels. Most scaling is intra-node and linear with GPUs per node; inter-node scaling is dominated by IO/gather and storage bandwidth.
- For very large images (millions of pixels), prefer larger batches (when memory allows) to amortize SVD setup; the heuristic increases batch size conservatively.

Troubleshooting
---------------
Quick checks
- `cupy` not found: Install the correct wheel into the Poetry venv or container. Verify from the same env.
- `getDeviceCount() == 0`: The job likely has no GPUs or `CUDA_VISIBLE_DEVICES` is empty; confirm Slurm allocation and container `--nv` passthrough.
- NVRTC/header errors: Use a `-devel` CUDA base or bind toolkit headers (`/usr/local/cuda/include`); set `CUDA_HOME`/`CPATH` if necessary.
- MPI errors in dev: Use the Slurm launcher or run on Linux/WSL with a proper MPI installation; Windows native MPI + GPUs is not a typical dev path.

Appendix: module map
--------------------
- `main.py` — enumerates inputs, scatters per-rank rows, runs `gpu_kernels.demmap_pos`, gathers/saves.
- `gpu_kernels.py` — adaptive batching, batched SVD, lambda selection, optional streams/NVTX/fused ops.
- `mpi_manager.py` — `init_mpi()`, `get_local_rank_info()`, `scatterv_array()`, `gatherv_array()`, and GPU binding helpers.
- `logging.py` — safe per-rank logs + quiet console.
- `preempt.py` — signal handlers to run a user callback and barrier.
- `checkpoint.py` — atomic saves with pruning; async write option.
  the module raises `RuntimeError` to avoid silent fallbacks in

Vendor parity notes
-------------------
The function `src.multiGPU.gpu_kernels.demmap_pos` is now a faithful port of the vendor algorithm used in `src.baseline.vendor.demmap_pos` and its dependencies:

- Self-normalized L pass: When `dem_norm0` is not provided (or is all ones) and `glc` has no selected filters, we execute an initial GSVD using `L0 = diag(1/sqrt(dlogt))`, compute a provisional DEM, and use it to derive the constraint weights. We then smooth these weights with a 5-point moving average over the interior and clamp with a minimum of 1e-8, matching vendor behavior.
- EM loci weighting: When `glc` selects filters, we compute the minimum of the EM loci across selected channels to form the initial constraint weights, apply the same smoothing and clamping, and proceed.
- dem_norm0: If provided and not trivial (not all ones/zero), it is honored directly as the weighting for L.
- Regularization parameter: We use the same discrepancy-principle search (`dem_reg_map`) with identical geometric sampling of `mu`, identical coefficient formation, and selection criterion.
- Reconstruction and positivity loop: The GSVD factors and reconstruction follow the same equations and we iterate the `reg_tweak` by `rgt_fact` until positivity or `max_iter`.
- edem and elogt: Vertical and horizontal errors are computed identically using `kdag@kdag.T` and the half-maximum width of `kdagk` interpolated over the fixed `ltt` grid.
- rscl: When enabled, we rescale DEM and edem by the mean ratio of observed to predicted counts and recompute `dn_reg` and `chisq`.

Implementation notes that lead to identical results:
- We use `safe_svd` and `safe_pinv` to sanitize inputs (NaN/inf clipping, value bounds) before SVD, analogous to the vendor's robustness. This does not change outputs for finite inputs but prevents numerical pathologies on corner cases.
- Shapes: `dem_inv_gsvd` returns `U_T` as `U.T` to match vendor indexing; we slice `U_T[:nf,:nf]` consistently like the vendor code.
- Floating precision: All internal computations are in float64 to preserve numerical parity; callers can cast outputs as needed.

There are no runtime toggles to switch to the vendor implementation; this code path is self-contained and functionally equivalent.
