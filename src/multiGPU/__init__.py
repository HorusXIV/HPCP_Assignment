"""multiGPU package

This package contains a small, explicit-MPI, numba/cupy-enabled rework of
the baseline vendor algorithms to run on multi-CPU / multi-GPU clusters.

Notes
- The modules provided here are a lightweight, modular scaffold to:
	* detect and map GPUs to MPI ranks
	* provide GPU-accelerated numerical kernels (CuPy / numba.cuda)
	* orchestrate MPI communication (mpi4py)
	* provide a Slurm-friendly launcher script

This is an initial refactor and scaffold. Computational kernels are
implemented with graceful fallbacks to NumPy/SciPy when CUDA/CuPy
are not available so unit tests and local development are possible.
"""

# Avoid importing submodules at package import time. Importing `main`
# here causes `runpy`/`-m` to import the package and then re-import the
# module, which leads to the runtime warning about modules being present
# in `sys.modules` prior to execution. Consumers should import specific
# submodules explicitly (e.g. `from src.multiGPU import main`).

__all__ = [
    "gpu_kernels",
    "mpi_manager",
    "io",
    "utils",
    "main",
]
