multiGPU package
===============

Purpose
-------
This package contains modular, low-level building blocks to run the baseline
`vendor` DEM inversion algorithms on multi-CPU and multi-GPU clusters using
explicit MPI for distribution and CuPy/numba for GPU acceleration. The
implementation keeps CPU fallbacks so it can run on developer laptops.

Layout
------
- `src/multiGPU/main.py` - Entry point. Initializes MPI (if present), binds
  local ranks to GPUs, splits input data across processes, and runs the
  per-rank processing loop. CLI arguments include `--input`, `--max-samples`
  and `--block-size`.

- `src/multiGPU/gpu_kernels.py` - GPU-aware numerical kernels. Exposes
  `dem_pix` (single-pixel inversion) and `demmap_pos` (batch wrapper).
  Prefer CuPy for linear algebra; fall back to NumPy/SciPy if CuPy is not
  available. Recent changes add a batched SVD path using CuPy to amortize
  decomposition costs across many pixels.

- `src/multiGPU/mpi_manager.py` - Lightweight MPI helpers. Provides:
  - `init_mpi()` to initialize `mpi4py` safely (falls back to serial mode),
  - utilities to discover local ranks and map them to GPUs,
  - `scatterv_array` / `gatherv_array` to distribute 2D arrays row-wise.

- `src/multiGPU/io.py` - Small IO helpers for loading `.npz` files and
  normalizing input shapes (AIA `bands` layouts are supported via
  heuristics).

- `src/multiGPU/utils.py` - Misc helpers (env parsing, mkdir wrappers).

Quick start (developer/laptop)
------------------------------
1. Create and activate the project's Poetry environment (project root):

# multiGPU (detailed)

This package contains modular components to run the baseline `vendor` DEM
inversion algorithms on single-node or multi-node clusters using explicit
MPI for distribution and CuPy/numba for GPU acceleration. It intentionally
keeps CPU fallbacks so development and CI can run on machines without GPUs.

Contents & layout
-----------------
- `src/multiGPU/main.py` — CLI entry point and orchestrator. Initializes
  MPI (if available), binds local ranks to GPUs, shards data across ranks
  and invokes the per-rank processing loop. Key CLI options: `--input`,
  `--max-samples`, `--block-size`, `--save-dir` (see Usage).
- `src/multiGPU/gpu_kernels.py` — Numeric kernels and GPU-aware algorithms
  (exposes `dem_pix` and `demmap_pos`). Prefer CuPy; fallback to NumPy/
  SciPy when GPU libraries are unavailable.
- `src/multiGPU/mpi_manager.py` — Lightweight MPI helpers: safe init,
  local-rank detection, scatterv/gatherv helpers and GPU mapping helpers.
- `src/multiGPU/io.py` — Input helpers and shape normalization (handles
  `bands` -> flattened `dn` layout heuristics).
- `src/multiGPU/checkpoint.py` — Checkpoint manager (atomic saves,
  optional async writes, per-rank shard naming). Useful for long runs.
- `src/multiGPU/preempt.py` — Preemption/signal handlers that call a
  user-supplied checkpoint callback and attempt to synchronize ranks.
- `hpc/slurm_run_multiGPU.sh` — Example SLURM launcher (container-aware,
  srun-based, GPU-aware). Adapt the SBATCH directives for your cluster.

Quick start (developer laptop)
------------------------------
1. Create and activate the Poetry environment from repo root:

```pwsh
poetry install
poetry shell
```

2. Run a serial smoke test (no MPI):

```pwsh
poetry run python -m src.multiGPU.main --input data/np32/20170906_12_00_12.npz --max-samples 200
```

Notes about input files
----------------------
- Preferred: `dn` of shape `(n_pixels, n_filters)` and `edn` of shape
  `(n_pixels, n_filters)` (or `(1, n_filters)` to be broadcast).
- Supported: `bands` layout `(n_filters, ny, nx)` — the script will flatten
  to `(n_pixels, n_filters)` and may sample down with `--max-samples`.

CLI highlights
--------------
- `--input <path>`: path to the `.npz` input file.
- `--max-samples N`: limit number of pixels sampled for rapid development.
- `--block-size B`: number of pixels processed per GPU batch (tune by GPU memory).
- `--save-dir DIR`: optional directory where each rank can write outputs or
  checkpoints (useful for resuming and debugging).

Outputs and persistence
-----------------------
`demmap_pos` returns per-pixel results:
- `dem` : `(n_pixels, n_temp_bins)` — recovered DEM.
- `edem`: `(n_pixels, n_temp_bins)` — uncertainties.
- `elogt`: `(n_pixels, n_temp_bins)` — effective temperature widths.
- `chisq`: `(n_pixels,)` — per-pixel chi-square.
- `dn_reg`: `(n_pixels, n_filters)` — reconstructed DN from DEM.

By default `main.py` aggregates results to the rank 0 process and prints a
summary. For production use you should persist the gathered arrays to disk.
Recommended pattern (atomic write)

```python
# in main.py after gather
out = dict(dem=dem_all, edem=edem_all, chisq=chisq_all)
tmp = os.path.join(save_dir, f"tmp_{jobid}_{rank}.npz")
np.savez_compressed(tmp, **out)
os.replace(tmp, os.path.join(save_dir, f"results_{jobid}.npz"))
```

Checkpointing and preemption (recommended)
-----------------------------------------
This repo includes `CheckpointManager` and `register_preempt_handlers`:

- Use `CheckpointManager` to perform atomic and optionally asynchronous
  checkpoint saves. Example:

```python
from src.multiGPU.checkpoint import CheckpointManager
from src.multiGPU.preempt import register_preempt_handlers

ck = CheckpointManager(outdir=args.save_dir or './checkpoints', keep=5, comm=comm)

def save_cb():
    state = { 'step': step, 'model': model_state_dict, 'opt': opt_state_dict }
    ck.save(state, step=step, async_write=False)

register_preempt_handlers(save_cb, comm=comm)
```

Behavior on preemption:
- Signal handler calls your `save_cb` (best-effort; runs in a thread),
  then attempts `comm.Barrier()` and exits with code 2. On restart your
  `main.py` should check `ck.latest()` and `ck.load()` to resume.

SLURM example
-------------
An example launcher is included as `hpc/slurm_run_multiGPU.sh`. Key
recommendations:

- Request GPUs per node with `#SBATCH --gres=gpu:<N>` (N = GPUs per node).
- If you want one rank per GPU, set `--ntasks-per-node` equal to GPUs per
  node and launch `srun -n <total_ranks> ...` (or use the SBATCH `--ntasks`).
- Use `srun --mpi=pmix` to launch the Python entrypoint so SLURM
  propagates environment variables like `CUDA_VISIBLE_DEVICES`.
- For containerized runs, use `singularity exec --nv` and bind the repo as
  shown in the script.

Example submit (adapt `--gres` and partition):

```pwsh
sbatch --export=REPO_DIR=/path/to/repo,IMAGE=/path/to/image.sif hpc/slurm_run_multiGPU.sh
```

Batched GPU/SVD tuning notes
---------------------------
- The implementation performs batched SVDs using CuPy to amortize the
  decomposition cost across multiple pixels. Tune `--block-size` to fit
  GPU memory. Estimation heuristic: batch_mem ≈ batch_size * nt * nf * 8
  bytes * safety_factor (0.6).
- For higher throughput consider:
  - Increasing batch size until GPU memory pressure, then back off.
  - Overlapping H2D transfers and computation using multiple CUDA streams.
  - Using cuSOLVER batched routines (via CuPy wrappers) for better perf.

Testing
-------
- `tests/test_gpu_batched_svd.py` exercises the batched path; it is
  skipped automatically if `cupy` is not present.
- Add CPU-only unit tests that call kernels with small deterministic
  inputs and assert shapes / finite numerics to enable fast CI runs.

Troubleshooting
---------------
- `RuntimeWarning: 'src.multiGPU.main' found in sys.modules ...` when you
  run `python -m src.multiGPU.main` indicates `src/multiGPU/__init__.py`
  imported `main` at package import time. This package avoids that.
- CuPy warnings about CUDA path: ensure `CUDA_PATH` or drivers are
  installed on the compute nodes. Match CuPy wheel to CUDA version.
- If GPU path fails at runtime the code falls back to CPU. Check logs
  for stack traces and use `nvidia-smi` to confirm device visibility.

Further work and suggestions
---------------------------
- Move `dem_reg_map` fully onto GPU (experimental) to reduce host-device
  round-trips; requires careful numeric verification.
- Add a `--save-output` flag in `main.py` to atomically persist results
  per-rank and an aggregation script to merge outputs post-job.
- Add CI workflows: CPU-only tests on every PR, optional GPU job on
  a GPU-enabled runner (if available) that installs a matching CuPy wheel.

If you want I can update `main.py` to integrate `CheckpointManager` and
`register_preempt_handlers` with a `--save-dir` flag and add example
sbatch metadata that includes a preemption window.
