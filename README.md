# HPCP\_Assignment

This project combines a real scientific computing application with HPC techniques. It uses CPU vector units, GPUs, MPI, and Dask in a structured way. The focus is on correctness, analysis, and engineering discipline—speedup numbers are secondary.

## Table of contents

* [Project structure](#project-structure)
* [Prerequisites](#prerequisites)
* [Local setup (Poetry)](#local-setup-poetry)
* [Linting & tests](#linting--tests)
* [Continuous Integration (GitHub Actions)](#continuous-integration-github-actions)
* [HPC: Singularity + Slurm](#hpc-singularity--slurm)

  * [Build the Singularity image](#build-the-singularity-image)
  * [Run on Slurm](#run-on-slurm)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Project structure

```
HPCP_Assignment/
├─ .github/
│  └─ workflows/
│     └─ ci.yml                     # CI: lint + tests via Poetry (push + PRs to main)
├─ containers/
│  ├─ python_poetry.def             # Singularity definition (Python 3.11 + Poetry)
│  └─ python_poetry.sif             # (built image; ignored until you build it)
├─ hpc/
│  ├─ slurm_run_singleGPU.sh        # Run src/singleGPU/main.py in Singularity
│  ├─ slurm_run_multiGPU.sh         # Run src/multiGPU/main.py (NCCL, MPI, multi-GPU)
│  └─ slurm_run_Dask.sh             # Run src/Dask/main.py (Dask cluster/client)
├─ src/
│  ├─ singleGPU/
│  │  ├─ __init__.py
│  │  └─ main.py
│  ├─ multiGPU/
│  │  ├─ __init__.py
│  │  └─ main.py
│  └─ Dask/
│     ├─ __init__.py
│     └─ main.py
├─ tests/
│  └─ test_main.py                  # example pytest
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
poetry run python src/Dask/main.py
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

# Dask workload
sbatch hpc/slurm_run_Dask.sh
```

Environment variables these scripts respect:

* `IMAGE` — path to the `.sif` (default: `$REPO_DIR/containers/python_poetry.sif`)
* `REPO_DIR` — repo path to bind (default: `$SLURM_SUBMIT_DIR`)

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

## License

TODO
