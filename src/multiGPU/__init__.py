"""Multi-GPU DEM solver package.

This package provides a compact, explicit-MPI orchestrator and GPU kernels
for running Differential Emission Measure (DEM) reconstruction across
multiple GPUs. The design favors clarity and operational robustness in HPC
environments managed by Slurm.

Modules:
    kernels: CuPy-based kernels with CUDA stream pipelining.
    mpi_manager: Rank/GPU mapping and MPI collectives.
    main: Rank orchestrator and I/O wiring executed as ``-m`` entry.
    logging: Rank-aware logging suitable for clustered runs.
    preempt: Best-effort preemption hooks for schedulers.
    io: Lightweight I/O helpers for ``.npz`` inputs.
"""

__all__ = [
    "kernels",
    "mpi_manager",
    "io",
    "logging",
    "preempt",
    "main",
]
