# HPCP_Assignment

This project combines a real scientific computing application with HPC techniques. It uses CPU vector units, GPUs, MPI, and Dask in a structured way. The focus is on correctness, analysis, and engineering discipline—speedup numbers are secondary.

## Table of contents

- [HPCP_Assignment](#hpcp_assignment)
  - [Project structure](#project-structure)
  - [Quickstart (local)](#quickstart-local)
  - [HPC: Singularity + Slurm](#hpc-singularity--slurm)
  - [Workloads](#workloads)
  - [Benchmarking artifacts](#benchmarking-artifacts)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
  - [NVTX Profiling](#nvtx-profiling)
  - [License](#license)

---

## Project structure

```
HPCP_Assignment/
├─ benchmarking/                    # All benchmark artifacts
│  ├─ baseline/                     # baseline
│  ├─ dask/                         # dask
│  ├─ singlegpu/                    # single-GPU
│  └─ multigpu/                     # multi-GPU
├─ containers/
│  └─ python_poetry.def             # Singularity/Apptainer definition (Python + Poetry + Nsight CLI)
├─ hpc/
│  ├─ slurm_run_baseline.sh         # Baseline CPU (frame 0)
│  ├─ slurm_run_Dask.sh             # Dask (CPU-only, frame 0)
│  ├─ slurm_run_singleGPU.sh        # Single-GPU (CuPy)
│  └─ slurm_run_multiGPU.sh         # Multi-GPU (CuPy + MPI, NVTX-ready)
├─ src/
│  ├─ baseline/                     # CPU baseline
│  ├─ common/                       # shared helpers (nvtx, gpu, profiling, paths)
│  ├─ dask/                         # Dask (CPU-only orchestration)
│  ├─ multiGPU/                     # MPI/CuPy multi-GPU
│  └─ singleGPU/                    # CuPy single-GPU
├─ tests/                           # smoke tests and helpers
├─ Report.md                        # project report
├─ pyproject.toml                   # Poetry config + deps
├─ poetry.lock                      # created by `poetry install`
└─ README.md                        # this file
```

---

## Quickstart (local)

```bash
# Clone and enter the project
git clone <repo-url>
cd HPCP_Assignment

# Install (Poetry creates .venv in project)
poetry install

# Run a workload (choose one)
poetry run python -m src.dask.main          # CPU/Dask
poetry run python -m src.singleGPU.main     # single-GPU (CuPy)
poetry run python -m src.multiGPU.main      # multi-GPU (MPI + CuPy)
```

**Data.** NPZ stacks under `data/np32/` (or your path). See the report’s *Dataset & Preparation* for how they were built and shared.

---

## HPC: Singularity + Slurm

Build the image (example on FHNW):

```bash
srun -p performance --mem=16G   singularity build --fakeroot containers/python_poetry.sif containers/python_poetry.def
```

Submit one of the included scripts (edit `#SBATCH` lines as needed):

```bash
# Baseline (CPU, frame 0)
sbatch hpc/slurm_run_baseline.sh

# Dask (CPU-only, frame 0)
sbatch hpc/slurm_run_Dask.sh

# Single GPU (CuPy)
sbatch hpc/slurm_run_singleGPU.sh

# Multi-GPU (CuPy + MPI)
sbatch hpc/slurm_run_multiGPU.sh
```

GPU jobs require `#SBATCH --gres=gpu:<N>` and `singularity exec --nv ...` inside the script.

---

## Workloads

We evaluate four implementations in this order:

1. **Baseline (CPU / NumPy)** — reference wall-clock and profiling on 4096×4096 frames (frame **0** for timing).  
2. **Dask (CPU-only)** — parallel tile orchestration on CPU (frame **0** for timing).  
3. **Single-GPU (CuPy)** — NumPy→CuPy vectorization of hot paths, device memory pools, optional NVTX.  
4. **Multi-GPU (CuPy + MPI)** — one rank per GPU; adaptive batch sizing; double/triple buffering with NVTX.

See `Report.md` for methodology and results, including normalization notes (Dask ran with 32 CPUs vs. baseline 16).

---

## Benchmarking artifacts

All artifacts are in **`/benchmarking/`**, split per method:

- **baseline/** — wall-times, environment summaries, logs for baseline CPU runs
- **dask/** — wall-times, task-stream summaries, logs for Dask CPU runs
- **singlegpu/** — wall-times, device/env dumps, optional Nsight traces
- **multigpu/** — wall-times, per-rank logs, optional Nsight traces

> Tip: add checksums after runs:
>
> ```bash
> (cd benchmarking && sha256sum baseline/**/* dask/**/* singlegpu/**/* multigpu/**/* > checksums.sha256)
> ```

---

## Configuration

- Python version and dependencies in `pyproject.toml`.
- Linters/tests (ruff/pytest) are included; see `tests/` and CI workflow.
- Slurm partitions/time/mem/CPUs/GPUs: adjust in `hpc/*.sh` scripts.

---

## Troubleshooting

- **GPU not visible in container** → use `singularity exec --nv ...` and verify with `nvidia-smi`.
- **MPI hangs** → prefer host MPI or follow site ABI guidance; set `NCCL_DEBUG=INFO` for comm issues.
- **Poetry import errors** → ensure packages live under `src/` and `pyproject.toml` has `packages = [{ include = "...", from = "src" }]`.
- **Slow first run** → warm-up effects (imports, caches). See Report.md *Benchmark Protocol* for how we measure.

---

## NVTX Profiling

Enable NVTX ranges to get rich Nsight timelines (multi-GPU path):

```bash
poetry install --with profiling
export MULTIGPU_NVTX=1
nsys profile -t cuda,nvtx,osrt -o run   poetry run python -m src.multiGPU.main --input-dir data/np32 --max-samples 512
```

The Slurm launcher auto-enables NVTX when `PROFILE=1`.

---

## License

MIT (or fill in your preferred license).
