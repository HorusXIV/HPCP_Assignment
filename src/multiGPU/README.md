multiGPU module — Operator Guide
================================

This document describes how to run, monitor, and collect results for experiments in `src/multiGPU`. It is written for a technical audience that needs operational clarity, not troubleshooting.

What this module does
---------------------
`src.multiGPU` reconstructs Differential Emission Measure (DEM) maps using a GPU-first solver. MPI distributes rows (pixels) across ranks; each rank drives one GPU and internally batches work based on free device memory. The orchestration overlaps communication and computation across files:

- Nonblocking gather of current results on a dedicated communicator
- Immediate broadcast/scatter of the next file on a separate communicator
- Prefetch of the next file’s local slices so GPUs can start the next compute without waiting for root
- Optional async saving on rank 0

Experiment initialization
-------------------------
Inputs
- Expected inputs are `.npz` files in a directory (default: `data/np32`). The loader accepts either:
  - `dn` and `edn` shaped `(n_pixels, n_filters)`; `edn` may be `(1, n_filters)` and will broadcast.
  - `bands` shaped `(n_filters, ny, nx)`; it is flattened to `(n_pixels, n_filters)`.

Run options
1) Slurm (recommended)
   - Submit the provided launcher (one rank per GPU, 3 CPUs per Task, 4 GPUs all on one Node):
     ```bash
     sbatch hpc/slurm_run_multiGPU.sh
     ```
   - Override key parameters by exporting or passing via `--export` to start an Profile run:
     ```bash
     sbatch --export=ALL,MULTIGPU_LOG_LEVEL=INFO,MULTIGPU_VERBOSE=1,PROFILE=1 hpc/slurm_run_multiGPU.sh
     ```

2) Local (single node, dev environment)
   - From a GPU host with the Poetry environment:
     ```pwsh
     poetry run python -m src.multiGPU.main --input-dir data/np32
     ```
   - CLI flags:
     - `--input-dir <path>`: folder containing `.npz` files (required)
     - `--max-samples <N>`: optional cap for pixels when flattening `bands`

What happens per run
- Rank 0 enumerates `.npz` files in `INPUT_DIR`, broadcasts the list, then for each file i:
  1. Loads/partitions file i (and, once i has started, also preloads file i+1).
  2. Broadcasts counts/dtypes/spatial for i, then nonblockingly scatters dn/edn to all ranks.
  3. Each rank computes on its GPU with internal batch pipelining (H2D → compute → D2H overlap), using pinned host buffers and prioritized copy streams where supported.
  4. Nonblocking gather of DEM rows for file i to rank 0 on a separate communicator.
  5. While gather(i) is in flight, rank 0 broadcasts/scatters file i+1 so all ranks can prefetch their local slices and be ready to compute immediately on the next loop.
  6. Rank 0 waits for gather(i), reshapes (if applicable), and saves (optionally in a background thread).

Logging mechanism
-----------------
Locations and format
- Root: `src/multiGPU/logs/`
  - Per-rank files: `src/multiGPU/logs/rank_logs/rankNNN.log`
  - Rank-0 console output: emitted to stdout (captured by Slurm `.out`).
- Format (all handlers):
  ```
  YYYY-MM-DD HH:MM:SS,mmm - rank=<r> - <LEVEL> - <logger> - <message>
  ```

How to read logs
- Rank-0 console messages that include `extra={"general": True}` are designed as milestones (start/finish per image, total counts).
- When `MULTIGPU_VERBOSE=1`, additional `[metrics]` lines are emitted from the GPU path, including batch sizing and memory estimates.
- Preemption registration (if enabled) and NVTX availability are logged once per run.

Results storage
---------------
Output location and naming
- Per input file, rank 0 saves a single `.npz` under:
  - `data/results_multiGPU/dem_all_<input_basename>.npz`
- Contents:
  - `dem_all`: stacked DEM rows (shape `(n_pixels, nt)`), where `nt` is the temperature grid length used by the run.
- Compression:
  - Plain `.npz` by default; set `MULTIGPU_SAVE_COMPRESSED=1` to use `np.savez_compressed`.

Environmental configuration
---------------------------
Set as standard environment variables. Defaults are shown in parentheses.

Job selection and paths
- `ENTRY` (`src.multiGPU.main`): Python module executed by the launcher.
- `INPUT_DIR` (`data/np32`): Folder with `.npz` inputs.
- `LOG_ROOT` (`src/multiGPU/logs`): Root for per-rank logs and profiler outputs.

Runtime (code-level)
- `MULTIGPU_BATCH_SIZE` (`0`): 0 = adaptive by free GPU memory; `> 0` forces fixed batch size.
- `MULTIGPU_BATCH_MEM_FRAC` (`0.7`): Fraction of free memory targeted by the batch planner.
- `MULTIGPU_NVTX` (`0/1`): Enable Python NVTX ranges (requires `nvtx` package).
- `MULTIGPU_VERBOSE` (`0/1`): Emit extra metrics during planning and OOM retries.
- `MULTIGPU_PREEMPT` (`0/1`): Register preemption handlers (best-effort save + barrier).
- `MULTIGPU_SAVE_COMPRESSED` (`0/1`): Save outputs with compression when `1`.
- `MULTIGPU_ASYNC_SAVE` (`1`): Save on rank 0 in a background thread (copies the buffer to ensure safety).
- `MULTIGPU_PIPELINE_FILES` (`1`): Enable cross-file pipelining (preload + prefetch next file).
- `MULTIGPU_PIPELINE_COMM` (`1`): Use split MPI communicators to separate scatter/bcast and gather traffic.
- `MULTIGPU_GATHER_DTYPE` (`float64`): Downcast DEM before gather to reduce network volume (set to `float32` for 2× bandwidth reduction if acceptable).

Logging
- `MULTIGPU_LOG_LEVEL` (`WARNING`): Root log level name (e.g., `INFO`, `DEBUG`).
- `MULTIGPU_QUIET` (`1`): When level ≥ WARNING, suppress per-rank log files unless overridden.
- `MULTIGPU_RANK_FILES` (`0/1`): Force per-rank file logging.

Memory pools (advanced)
- `MULTIGPU_POOL_LIMIT_FRACTION` (unset): If set in `(0.1 … 0.95)`, soft-limit CuPy device pool by fraction of total.
- `MULTIGPU_POOL_LIMIT_BYTES` (unset): Soft-limit CuPy device pool by bytes.
- `MULTIGPU_PINNED_POOL_LIMIT_BYTES` (`1 GiB`): Soft-limit pinned host memory pool.

Performance notes and best practices
------------------------------------
- Overlap across files: Results gathering for file i is overlapped with broadcasting/scattering file i+1, so non-root ranks can start GPU compute sooner on the next iteration.
- Overlap within a file: H2D staging, compute, and D2H copies are pipelined with multiple CUDA streams. Where supported, transfer streams use higher priority to keep copies progressing under load.
- Pinned memory: Host buffers for D2H and MPI are pinned to improve transfer throughput.
- Network volume: Use `MULTIGPU_GATHER_DTYPE=float32` to halve gather bandwidth if it meets accuracy requirements.

Safety and constraints
----------------------
- Collective ordering: All ranks post nonblocking collectives in a consistent sequence. Split communicators (`MULTIGPU_PIPELINE_COMM=1`) reduce head-of-line blocking and simplify ordering.
- MPI progress: Most stacks progress nonblocking collectives adequately; if you observe stalls, enable your MPI’s async progress or increase host-side MPI activity during prefetch phases.
- Memory footprint: Cross-file prefetch holds next-file slices in memory per rank. Ensure sufficient host memory is available (rank 0 may transiently hold gathered results plus save buffers).

Profiling (via Slurm launcher)
- `PROFILE` (`0/1`): If `1`, run under `nsys profile` inside the container.
- `NSYS_OPTS` (`cuda,nvtx,osrt,cublas,cusolver`): Trace domains.
- `NSYS_OUT_DIR_HOST`, `NSYS_TMP_DIR_HOST`: Host directories for Nsight outputs.
- `PYPROFILE` (`0/1`): If `1`, record a Python profile at `$LOG_ROOT/profile.html` instead of system traces.

Operational notes
-----------------
- GPU is required: if no CUDA device is visible, the solver raises a clear `RuntimeError`.
- Batching is adaptive and OOM-aware; the kernel may downshift batch size automatically and will log the changes when verbose mode is enabled.

Architecture notes
------------------
The GPU kernel implementation has been modularized for maintainability.

- Public API: import directly from `src.multiGPU.kernels`:
  - `demmap_pos`, `dem_inv_gsvd`, `dem_reg_map`, `safe_svd`, `safe_pinv`,
    `estimate_batch_plan`, `nvtx_range`, `verbose_enabled`.
- Modular package: `src/multiGPU/kernels/`
  - `demmap_pos.py`: batched CuPy implementation of DEM reconstruction
  - `dem_inv_gsvd.py`: GSVD-equivalent factorization via SVD
  - `dem_reg_map.py`: discrepancy-principle lambda selection
  - `linalg.py`: SVD and pseudo-inverse helpers with input sanitization
  - `memory.py`: batch sizing and memory estimation utilities
  - `utils.py`: NVTX ranges, verbosity flag, pinned host memory allocator