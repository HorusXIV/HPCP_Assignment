multiGPU module — Operator Guide
================================

This document describes how to run, monitor, and collect results for experiments in `src/multiGPU`. It is written for a technical audience that needs operational clarity, not troubleshooting.

What this module does
---------------------
`src.multiGPU` reconstructs Differential Emission Measure (DEM) maps using a GPU-first solver. MPI is used to distribute rows (pixels) across ranks; each rank runs one GPU and internally batches work based on free device memory.

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
- Rank 0 enumerates `.npz` files in `INPUT_DIR`, broadcasts the list, then for each file:
  1. Loads arrays and scatters rows across ranks.
  2. Each rank runs the GPU kernel once; internal batch size is chosen adaptively.
  3. Rank 0 gathers DEMs and writes a single output file per input.

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

Logging
- `MULTIGPU_LOG_LEVEL` (`WARNING`): Root log level name (e.g., `INFO`, `DEBUG`).
- `MULTIGPU_QUIET` (`1`): When level ≥ WARNING, suppress per-rank log files unless overridden.
- `MULTIGPU_RANK_FILES` (`0/1`): Force per-rank file logging.

Memory pools (advanced)
- `MULTIGPU_POOL_LIMIT_FRACTION` (unset): If set in `(0.1 … 0.95)`, soft-limit CuPy device pool by fraction of total.
- `MULTIGPU_POOL_LIMIT_BYTES` (unset): Soft-limit CuPy device pool by bytes.
- `MULTIGPU_PINNED_POOL_LIMIT_BYTES` (`1 GiB`): Soft-limit pinned host memory pool.

Profiling (via Slurm launcher)
- `PROFILE` (`0/1`): If `1`, run under `nsys profile` inside the container.
- `NSYS_OPTS` (`cuda,nvtx,osrt,cublas,cusolver`): Trace domains.
- `NSYS_OUT_DIR_HOST`, `NSYS_TMP_DIR_HOST`: Host directories for Nsight outputs.
- `PYPROFILE` (`0/1`): If `1`, record a Python profile at `$LOG_ROOT/profile.html` instead of system traces.

Operational notes
-----------------
- GPU is required: if no CUDA device is visible, the solver raises a clear `RuntimeError`.
- Batching is adaptive and OOM-aware; the kernel may downshift batch size automatically and will log the changes when verbose mode is enabled.
