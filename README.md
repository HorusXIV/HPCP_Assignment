# HPCP\_Assignment

This project combines a real scientific computing application with HPC techniques. It is expected to make use of CPU vector units, GPUs, MPI, and Dask, applying them in a structured way. The focus is on correctness, analysis, and engineering discipline—speedup numbers are secondary.

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
  - [License](#license)

---

## Project structure

```
HPCP_Assignment/
├─ .github/
│  └─ workflows/
│     └─ ci.yml                 # CI: lint + tests via Poetry (runs on push and PRs to main)
├─ src/
│  └─ [packages]/
│     ├─ __init__.py
│     └─ main.py                # example entry point / module
├─ tests/
│  └─ test_main.py              # example pytest
├─ hpc/
│  └─ slurm-runner.sh           # Slurm job script to run lint+tests inside Singularity
├─ containers/
│  └─ python_poetry.def         # Singularity definition (Python 3.11 + Poetry)
├─ pyproject.toml               # Poetry config + deps (see below)
├─ poetry.lock                  # generated after `poetry install`
├─ README.md                    # this file
└─ .gitignore
```

> Rename `your_package_name` to your actual package name in both the folder and `pyproject.toml`.

---

## Prerequisites

* **Python** 3.10+
* **Poetry** 1.7+ (installer: `curl -sSL https://install.python-poetry.org | python3 -`)
* (For HPC) **Singularity** (or your site’s provided module)
* (For GPUs on HPC) NVIDIA drivers + `singularity exec --nv` support
* (For MPI on HPC) Site-provided MPI modules if using host MPI

---

## Local setup (Poetry)

```bash
# Clone and enter the project
git clone (https://github.com/HorusXIV/HPCP_Assignment)
cd HPCP_Assignment

# Install dependencies (creates .venv in project)
poetry install

# Activate the virtualenv (optional; you can also prefix `poetry run ...`)
poetry env activate
```

---

## Linting & tests

```bash
# Lint
poetry run ruff check .       # or: poetry run flake8

# Tests
poetry run pytest -q
```

---

## Continuous Integration (GitHub Actions)

The workflow is at `.github/workflows/ci.yml`. It:

* Triggers on **every push** and **pull requests targeting `main`**
* Uses **Poetry** to install deps
* Runs **ruff**
* Runs **pytest**

The job runs on a self-hosted runner by default.

---

## HPC: Singularity + Slurm

This repo includes:

* `containers/python_poetry.def` — a Singularity definition that provides **Python 3.11 + Poetry**
* `hpc/slurm-runner.sh` — a Slurm script that runs **the programm** *inside* the image

### Build the Singularity image

We have to run the command using srun since we work on the FHNW SLURM.

```bash
srun - p perfromance singularity build --fakeroot containers/python_poetry.sif containers/python_poetry.def
```

### Run on Slurm

Submit the included script (edit the `#SBATCH` lines to match your cluster’s partition, time, etc.):

```bash
sbatch hpc/slurm-runner.sh
```

What it does:

* Binds your repo into `/workspace` inside the container
* Runs `poetry install`, then lints (ruff/flake8) and runs `pytest`

Environment variables the script respects:

* `IMAGE` — path to the `.sif` (default: `containers/python_poetry.sif`)
* `REPO_DIR` — repo path to bind (default: current working directory)

---

## Configuration

* **Python version**: set in `actions/setup-python` and `pyproject.toml`.
* **Linters**: The CI prefers **ruff**; it falls back to **flake8** if ruff isn’t present. Add one of them to dev-deps.
* **Tests**: Requires `pytest` in dev-deps.
* **Runner labels**: Update `runs-on` in `ci.yml` to match your self-hosted runner labels if you use them.
* **Slurm**: Edit `#SBATCH` options in `hpc/slurm-runner.sh` to suit your site.

---

## Troubleshooting

* **Runner is offline in GitHub**
  Ensure you installed the runner as a service:

  ```bash
  cd ~/actions-runner
  sudo ./svc.sh status
  ```
* **Poetry can’t find the package**
  Check `packages = [{ include = "your_package_name", from = "src" }]` in `pyproject.toml` and that your code lives in `src/your_package_name/`.
* **MPI import/runtime errors**
  Mismatch between container MPI and host MPI. Prefer host MPI or follow your site’s guidance for ABI-compatible builds.
* **GPU not visible in container**
  Use `singularity exec --nv ...` and verify drivers on the host.

---

## License

TODO
