# HPCP\_Assignment

This project combines a real scientific computing application with HPC techniques. It uses CPU vector units, GPUs, MPI, and Dask in a structured way. The focus is on correctness, analysis, and engineering discipline—speedup numbers are secondary.

## Table of contents

- [HPCP\_Assignment](#hpcp_assignment)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Local setup (Poetry)](#local-setup-poetry)
  - [Linting \& tests](#linting--tests)
  - [Continuous Integration (GitHub Actions)](#continuous-integration-github-actions)
  - [HPC: Singularity + Slurm](#hpc-singularity--slurm)
    - [Build the Singularity image](#build-the-singularity-image)
    - [Run on Slurm](#run-on-slurm)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
  - [NVTX Profiling](#nvtx-profiling)
  - [License](#license)

---

## Project structure

```
HPCP_Assignment/
├─ .github/
│  └─ workflows/
│     └─ ci.yml                     # CI: lint + tests via Poetry (push + PRs to main)
├─ containers/
│  └─ python_poetry.def             # Singularity/Apptainer definition (Python + Poetry + Nsight CLI)
├─ hpc/
│  ├─ slurm_run_baseline.sh         # Baseline CPU/GPU job (optional)
│  ├─ slurm_run_singleGPU.sh        # Run src/singleGPU
│  ├─ slurm_run_multiGPU.sh         # Run src/multiGPU (MPI, multi-GPU, NVTX/NSight ready)
│  └─ slurm_run_Dask.sh             # Run src/dask (Dask cluster/client)
├─ src/
│  ├─ baseline/                     # CPU baseline
│  ├─ common/                       # shared helpers (nvtx, gpu, paths, …)
│  ├─ dask/                         # Dask implementation
│  ├─ multiGPU/                     # Multi-GPU MPI implementation
│  └─ singleGPU/                    # Single-GPU implementation
├─ tests/
│  ├─ test_baseline_helpers.py
│  ├─ test_dask_main_callables.py
│  ├─ test_io_helpers_csv.py
│  ├─ test_multiGPU_determinism.py
│  ├─ test_multiGPU_shim.py
│  ├─ test_reporting_smoke.py
│  ├─ test_tiles.py
│  └─ test_vendor_smoke.py
├─ pyproject.toml                   # Poetry config + deps
├─ poetry.lock                      # generated after `poetry install`
├─ README.md                        # this file
└─ .gitignore
```
---

## Prerequisites

* [**Python** 3.12+](https://www.python.org/downloads/)
* [**Poetry** 1.7+](https://python-poetry.org/docs/)
* (HPC) **Singularity**
* (GPU on HPC) NVIDIA drivers on the host + `singularity exec --nv` support
* (MPI on HPC) Site-provided MPI modules if you use host MPI

---

## Local setup (Poetry)

```bash
# Clone and enter the project
git clone https://github.com/HorusXIV/HPCP_Assignment
cd HPCP_Assignment

# Install dependencies (creates .venv in project)
poetry install

# Optional: activate the virtualenv
poetry env activate

# Run one of the entry points locally
poetry run python src/singleGPU/main.py
poetry run python src/multiGPU/main.py
poetry run python src/dask/main.py
```

---

## Linting & tests

```bash
# Lint (ruff recommended)
poetry run ruff check .

# Tests
poetry run pytest -q
```

---

## Continuous Integration (GitHub Actions)

`.github/workflows/ci.yml`:

* Triggers on **every push** and **PRs targeting `main`**
* Uses **Poetry** to install deps
* Runs **ruff**
* Runs **pytest**

Runs on self-hosted runner.

---

## HPC: Singularity + Slurm

This repo includes:

* `containers/python_poetry.def` — definition for a Python 3.12 + Poetry image
* Three Slurm job scripts that execute your code **inside** the image:

  * `hpc/slurm_run_singleGPU.sh`
  * `hpc/slurm_run_multiGPU.sh`
  * `hpc/slurm_run_Dask.sh`

Each script:

* Binds your repo into `/workspace` inside the container
* Does `poetry install`
* Runs the corresponding `src/**/main.py`

### Build the Singularity image

On FHNW’s Slurm you typically build via `srun`. From the project root:

```bash
srun -p performance --mem=16G singularity build --fakeroot containers/python_poetry.sif containers/python_poetry.def
```

### Run on Slurm

Submit one of the included scripts (edit `#SBATCH` lines if needed):

```bash
# Single GPU
sbatch hpc/slurm_run_singleGPU.sh

# Multi-GPU
sbatch hpc/slurm_run_multiGPU.sh

# dask workload
sbatch hpc/slurm_run_Dask.sh
```

Environment toggles for hpc/slurm_run_multiGPU.sh (what they change):

- Job/entry selection
  - `REPO_DIR` (path): Repository root bound into the container (default: `$SLURM_SUBMIT_DIR`).
  - `IMAGE` (path): Singularity image to use (default: `$REPO_DIR/containers/python_poetry.sif`).
  - `ENTRY` (str): Python module to run with `-m` (default: `src.multiGPU.main`).
  - `INPUT_DIR` (path): Directory with `.npz` inputs (default: `data/np32`).
  - `LOG_ROOT` (path): Host log root for rank logs and Nsight traces (default: `src/multiGPU/logs`).
  - `MAX_SAMPLES` (int): Reserved; currently not passed to the Python entry. Use `--max-samples` CLI if needed.

- Profiling
  - `PROFILE` (0/1): Run under Nsight Systems (`nsys profile ...`) if available in the container.
    - When `PROFILE=1` and `MULTIGPU_NVTX` is not explicitly set by you, the launcher automatically sets `MULTIGPU_NVTX=1` for rich timeline ranges.
  - `NSYS_OPTS` (csv): Nsight trace domains (default: `cuda,nvtx,osrt,cublas,cusolver`).
  - `NSYS_OUT_DIR_HOST` (path): Host directory for Nsight outputs (default: `$LOG_ROOT/nsys`).
  - `NSYS_TMP_DIR_HOST` (path): Host tmp dir for `.qdstrm` (default: `$LOG_ROOT/.nsys-tmp`).
  - `PYPROFILE` (0/1): Use lightweight Python profiler (pyinstrument) instead of plain Python; writes `$LOG_ROOT/profile.html`.

- Runtime behavior (MULTIGPU_*)
  - `MULTIGPU_BATCH_SIZE` (int): 0 = auto (adaptive by free GPU memory). >0 forces a fixed batch size.
  - `MULTIGPU_NVTX` (0/1): Enable Python-level NVTX ranges inside the app (requires `poetry install --with profiling`).
  - `MULTIGPU_VERBOSE` (0/1): Extra verbose logging (rank metrics, memory/batch info). Default 0.
  - `MULTIGPU_LOG_LEVEL` (str): Root log level (`INFO`, `WARNING`, …). Default `WARNING`.
  - `MULTIGPU_QUIET` (0/1): When level is `WARNING+`, suppress per-rank files unless `MULTIGPU_RANK_FILES=1`. Default 1.
  - `MULTIGPU_RANK_FILES` (0/1): Force per-rank log files even in quiet mode.
  - `MULTIGPU_PREEMPT` (0/1): Register preemption handlers (best-effort) when supported. Default 0.
  - `MULTIGPU_SAVE_COMPRESSED` (0/1): Save outputs with compression. Default 0.

- Communication and GPU networking (defaults are safe; advanced tuning optional)
  - `NCCL_DEBUG` (str): `WARN` (default). Use `INFO` for troubleshooting.
  - `NCCL_ASYNC_ERROR_HANDLING=1`: Enabled by default in the launcher to prevent silent hangs.
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`: Stable PCI ordering for consistent GPU selection.
  - Advanced UCX/NCCL knobs (e.g., `UCX_TLS`, `UCX_NET_DEVICES`, `NCCL_P2P_LEVEL`, `NCCL_*`): tune only if you know your fabric; see launcher comments.

- CuPy and diagnostics
  - `CUPY_CACHE_DIR` (path): Cache directory inside container (default `/workspace/.cupy_cache`).
  - `CUPY_ACCELERATORS=cub`: Prefer CUB for reductions when available.
  - `CUPY_COMPILE_WITH_PTX=1`: Embed PTX for forward-compat kernels.
  - `CUPY_DUMP_CUDA_SOURCE_ON_ERROR=1`: Dump source on NVRTC errors (debug).

Notes
- All toggles can be passed via `sbatch --export=ALL,VAR=value,…` or set in the environment before submitting.
- Outputs are written under `data/results_multiGPU/dem_all_<input>.npz`; Nsight traces go to `$LOG_ROOT/nsys/` when `PROFILE=1`.
- In quiet mode (WARNING+), rank 0 still prints start/end markers and the pixels-per-rank distribution so you can track progress succinctly.

**GPU jobs**
Make sure your script contains:

* `#SBATCH --gres=gpu:<N>` to request GPUs
* `singularity exec --nv ...` to expose GPUs inside the container

**Multi-GPU jobs**
Request ≥2 GPUs (e.g., `--gres=gpu:2`) and ensure your code initializes NCCL / distributed mode accordingly.

---

## Configuration

* **Python version**: in `pyproject.toml`.
* **Linters**: Add **ruff** (recommended) or **flake8** to dev-deps.
* **Tests**: Add `pytest` to dev-deps.
* **Slurm**: Adjust partition, time, memory, CPU, and GPU directives in the scripts under `hpc/`.

---

## Troubleshooting

* **Image not found**
  Use an absolute path or set `IMAGE=$PWD/containers/python_poetry.sif`.

* **Entry point not found**
  Ensure you run `python src/<package>/main.py` and that `__init__.py` exists.

* **GPU not visible in container**
  Use `singularity exec --nv ...` and verify that GPUs exist on the node (`nvidia-smi`).

* **MPI issues**
  Host MPI vs. container MPI mismatch. Prefer host MPI (bind libs) or follow site guidance for ABI-compatible builds.

* **Poetry can’t find the package**
  Check `packages = [{ include = "...", from = "src" }]` in `pyproject.toml` and that your packages live under `src/`.

---

## NVTX Profiling

Optional NVTX instrumentation is available for the multi-GPU path. Enable it to obtain rich timeline annotations in **Nsight Systems** / **Nsight Compute**:

```bash
poetry install --with profiling        # install optional nvtx dep
export MULTIGPU_NVTX=1                 # turn on ranges (auto-enabled by launcher when PROFILE=1)
nsys profile -t cuda,nvtx,osrt -o run \
  poetry run python -m src.multiGPU.main --input-dir data/np32 --max-samples 512
```

See `docs/NVTX_PROFILING.md` for detailed guidance (phase list, Slurm usage, disabling, overhead notes). The Slurm launcher auto-enables NVTX when `PROFILE=1` unless you explicitly set `MULTIGPU_NVTX`.

---

## License

TODO
