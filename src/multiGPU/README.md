multiGPU module — concise guide
==============================

Purpose
-------
`src.multiGPU` contains GPU-oriented numerical kernels and small MPI
helpers to run the project's DEM inversion workloads on multi-GPU clusters.

Contents
--------
- `gpu_kernels.py` — GPU-first numerical kernels (expects CuPy at import).
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

```pwsh
poetry shell
python -m pip install "cupy-cuda12x==13.6.0"
```

Replace `cupy-cuda12x` with the correct `cupy-cuda11x` variant if your
system uses CUDA 11.x.

3) Fix NVRTC / header errors

If you see errors about missing headers (e.g. `cuda_fp16.h`), use a
`-devel` CUDA base image or provide the CUDA toolkit headers to the
runtime (bind-mount `/usr/local/cuda/include` into the container and set
`CUDA_HOME`/`CPATH` accordingly).

4) Verify from the same Python environment

```pwsh
python -c "import cupy; print('cupy', cupy.__version__); print('cuda', cupy.cuda.runtime.getDeviceCount())"
```

Expected: CuPy imports and `getDeviceCount()` returns >= 1. If `0`,
confirm the scheduler allocated GPUs and `CUDA_VISIBLE_DEVICES` is set.

Starting on Slurm
-----------------
Use the provided launcher `hpc/slurm_run_multiGPU.sh`. Key points:

- Request GPUs per node: `#SBATCH --gres=gpu:<N>`
- One rank per GPU: set `--ntasks-per-node` equal to GPUs per node
- Use `srun --mpi=pmix` so SLURM propagates `CUDA_VISIBLE_DEVICES`
- For containers, use `singularity exec --nv` (or equivalent) to expose GPUs

Example submit (adapt REPO_DIR and IMAGE):

```pwsh
sbatch --export=REPO_DIR=/path/to/repo,IMAGE=/path/to/image.sif hpc/slurm_run_multiGPU.sh
```

Minimal troubleshooting checklist
---------------------------------
- `No module named 'cupy'`: ensure you installed the wheel into the
  Poetry venv used by this project.
- `cupy` imports but `getDeviceCount()` is 0: ensure the job reserves GPUs
  and `CUDA_VISIBLE_DEVICES` is set in the job environment.
- NVRTC header errors: use `-devel` CUDA image or provide toolkit headers.

Notes / recommended improvement
------------------------------
- Currently `gpu_kernels.py` imports `cupy` at module import time and will
  raise if CuPy is absent. That behavior helps avoid silent CPU fallbacks
  in production, but it makes local testing harder. Consider a small
  refactor to lazily import CuPy inside GPU-specific functions so the
  module can be imported on CPU-only machines without a test shim.

If you want, I can add `TESTING_PI.md` with detailed Pi/CI run steps or
implement the lazy-import refactor.
multiGPU package
===============

Purpose
-------
This package contains modular, low-level building blocks to run the baseline
`vendor` DEM inversion algorithms on single-node or multi-node clusters using
explicit MPI for distribution and CuPy/numba for GPU acceleration.

Current status (important)
--------------------------
- The code in `src/multiGPU/gpu_kernels.py` currently imports `cupy` at
  module import time and will raise ImportError if `cupy` is not installed.
  This means the module is not importable on systems without CuPy unless a
  test shim or refactor is used. Although many functions are written to
  perform CPU work, the current implementation intentionally fails early to
  avoid silently falling back to CPU when the package is used in a true
  multi-GPU context.
- `src/multiGPU/mpi_manager.py` attempts to import `cupy` at module import
  time as well (used to query device counts). `init_mpi()` and other helpers
  do support serial operation when `mpi4py` is absent (they return
  `comm=None, rank=0, size=1`).

Layout (what's actually in the package)
--------------------------------------
- `src/multiGPU/main.py` — CLI entry point and orchestrator. Initializes
  MPI (if available), maps local ranks to GPUs, shards data across ranks,
  and invokes the per-rank processing loop. Note: `main.py` calls
  `mmpi._require_cupy()` in the GPU execution path and may raise if CuPy
  is not importable when a GPU is required.
- `src/multiGPU/gpu_kernels.py` — Numeric kernels and GPU-aware algorithms.
  The module prefers CuPy for performance. Key functions:
  - `safe_svd(A, ...)` — runs SVD on device via CuPy and converts outputs
    to NumPy arrays; raises `RuntimeError` if the GPU SVD fails.
  - `safe_pinv(A, ...)` — pseudo-inverse wrapper built on `safe_svd`.
  - `dem_reg_map(...)` — regularization parameter search; written to use
    CuPy arrays when `GPU_AVAILABLE` is True but relies on CuPy APIs.
  - `demmap_pos(...)` — batched wrapper that drives a GPU-accelerated
    batched SVD path when `GPU_AVAILABLE` is True; it raises a
    `RuntimeError` if the GPU path fails so calls don't silently continue
    on CPU in multi-GPU mode.
  - `dem_pix(...)` — intentionally raises in this multiGPU module; the
    single-node baseline implementation provides single-pixel evaluation.
- `src/multiGPU/mpi_manager.py` — Lightweight MPI helpers. Provides:
  - `init_mpi()` — returns `(comm, rank, size)`; falls back to serial when
    `mpi4py` is unavailable.
  - `get_local_rank_info()` — determines node-local rank using `comm`.
  - `scatterv_array()` / `gatherv_array()` — helpers for row-wise array
    distribution using MPI byte-wise collectives.
  - `set_device_for_local_rank()` / `map_rank_to_gpu()` / `bind_gpu()` —
    small helpers to map ranks to GPUs and set `CUDA_VISIBLE_DEVICES`.
- `src/multiGPU/io.py` — input helpers and shape normalization.
- `src/multiGPU/checkpoint.py` — checkpoint manager for atomic saves.
- `src/multiGPU/preempt.py` — preemption handlers for graceful checkpointing.

Quick start (developer laptop)
------------------------------
1. Create and activate the Poetry environment from repo root:

```pwsh
poetry install
poetry shell
```

2a. If you have GPUs and CuPy installed, run a serial smoke test (no MPI):

```pwsh
poetry run python -m src.multiGPU.main --input data/np32/20170906_12_00_12.npz \
  --max-samples 200
```

2b. On a Raspberry Pi or any CPU-only system (no CuPy), you have two
options:

- Use the test shim approach in unit tests: tests may inject a small
  `cupy` shim (NumPy-backed) into `sys.modules` before importing
  `src.multiGPU` modules. This permits importing `gpu_kernels` for unit
  testing without installing CuPy or requiring GPUs. See
  `tests/test_multiGPU_shim.py` for an example shim and usage.
- Refactor `src/multiGPU/gpu_kernels.py` to lazily import CuPy within
  GPU-specific functions; this makes the module importable on CPU-only
  systems without shims. (Recommended for long-term maintainability.)

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
`demmap_pos` returns per-pixel results (same shapes as the baseline):
- `dem` : `(n_pixels, n_temp_bins)` — recovered DEM.
- `edem`: `(n_pixels, n_temp_bins)` — uncertainties.
- `elogt`: `(n_pixels, n_temp_bins)` — effective temperature widths.
- `chisq`: `(n_pixels,)` — per-pixel chi-square.
- `dn_reg`: `(n_pixels, n_filters)` — reconstructed DN from DEM.

By default `main.py` gathers per-rank outputs and prints a summary. For
production runs persist gathered arrays to disk using atomic writes.

Checkpointing and preemption
---------------------------
Use `CheckpointManager` and `register_preempt_handlers` to handle
preemption and to perform atomic checkpoint saves. See `src/multiGPU/checkpoint.py`
and `src/multiGPU/preempt.py` for examples.

SLURM example
-------------
See `hpc/slurm_run_multiGPU.sh` for a working example launcher. Key items:

- Request GPUs per node with `#SBATCH --gres=gpu:<N>`.
- For one rank per GPU set `--ntasks-per-node` equal to GPUs per node and
  launch with `srun -n <total_ranks> ...`.
- Use `srun --mpi=pmix` to have SLURM propagate environment variables like
  `CUDA_VISIBLE_DEVICES` correctly.

Batched GPU/SVD tuning notes
---------------------------
- `demmap_pos` contains a batched CUDA/SVD path that tries to amortize
  SVD costs using CuPy's batched operations. Tune `--block-size` to fit
  GPU memory. As a rough heuristic: batch_mem ≈ batch_size * nt * nf * 8
  bytes * safety_factor (0.6).
- The GPU path attempts retries with smaller batches in case of
  allocation/memory failures; however, when the GPU path ultimately fails
  the module raises `RuntimeError` to avoid silent fallbacks in
  multi-GPU production runs.

Testing (how to run on CPU-only hardware)
----------------------------------------
- Unit tests that need to import `gpu_kernels.py` without CuPy can use a
  simple NumPy-backed shim inserted into `sys.modules['cupy']` before
  importing. Example test added: `tests/test_multiGPU_shim.py` (this
  injects a minimal shim and exercises `dem_reg_map`, `safe_svd` error
  handling, and `mpi_manager` GPU-mapping). The shim is deliberately
  minimal and only implements the small CuPy API surface the package
  uses.
- Run the shim-based tests on a Raspberry Pi / CPU-only runner without
  installing CuPy:

```pwsh
poetry install
poetry run pytest -q tests/test_multiGPU_shim.py
```

- For CI: run CPU-only tests by default on each PR. If you have a GPU
  runner available, provide an optional job that installs the appropriate
  CuPy wheel and runs the full GPU tests.

Developer notes & recommended refactor
-------------------------------------
- Current behavior: the package intentionally imports `cupy` at module
  import time and will raise if CuPy is not installed. This is a
  defensive choice to prevent accidental silent CPU fallbacks in
  production multi-GPU workflows, but it makes testing and local
  imports harder on CPU-only machines.
- Recommendation: lazily import CuPy inside functions that require it
  (or centralize a runtime-check helper). This small refactor improves
  testability (removes the need for a shim) while preserving the
  ability to raise loudly when running in GPU-only production mode.

Troubleshooting (updated)
-------------------------
- If `cupy` is not installed you will see ImportError when importing
  `src.multiGPU.gpu_kernels` or `src.multiGPU.mpi_manager` — on CPU-only
  development machines prefer using the test shim or apply the lazy
  import refactor described above.
- If the GPU path fails at runtime inside `demmap_pos`, the function
  will log and raise `RuntimeError` rather than silently falling back to
  CPU in multi-GPU mode. This is intentional for correctness on
  multi-node runs.

Further work and suggestions
---------------------------
- Move `dem_reg_map` fully onto GPU (experimental) to reduce host-device
  round-trips; requires numeric verification.
- Add a `--save-output` flag in `main.py` to atomically persist per-rank
  outputs, and a separate aggregation utility to merge results.
- Add CI workflows: CPU-only tests on every PR; optional GPU job on a
  GPU-enabled runner that installs a matching CuPy wheel and runs GPU
  path tests.

If you'd like, I can:
- create `TESTING_PI.md` with concrete Pi/CI instructions,
- refactor `gpu_kernels.py` to lazy-import CuPy and update tests to drop
  the shim, or
- add more unit tests for `demmap_pos` CPU fallback and `scatterv_array`/
  `gatherv_array` using a fake `comm` object.
