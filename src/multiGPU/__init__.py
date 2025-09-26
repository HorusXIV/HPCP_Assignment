"""Multi-GPU DEM solver package.

This package provides a compact, explicit-MPI orchestrator and GPU kernels
for running Differential Emission Measure (DEM) reconstruction across
multiple GPUs. The design favors clarity and operational robustness in HPC
environments managed by Slurm.

Key modules
- ``gpu_kernels``: CuPy-based hot kernels with optional CUDA streams.
- ``mpi_manager``: Rank/GPU mapping and MPI collectives.
- ``main``: Rank orchestrator and I/O wiring executed as ``-m`` entry.
- ``logging``: Rank-aware logging suitable for clustered runs.
- ``preempt``: Best-effort preemption hooks for schedulers.
"""

# Do not import heavy submodules at package import time to avoid ``-m``
# re-import warnings. Import needed modules explicitly from callers.

__all__ = [
    "gpu_kernels",
    "mpi_manager",
    "io",
    "utils",
    "main",
]
