# src/common/threads.py
from __future__ import annotations
"""
Thread-capping helpers for numerical libraries.

This module centralizes two complementary mechanisms:

1) `early_env_caps(threads)`: set environment variables *before* importing
   NumPy/SciPy so low-level backends (OpenBLAS, MKL, BLIS, vecLib, OpenMP)
   initialize with a desired thread cap.

2) `runtime_caps(threads)`: a context manager that uses `threadpoolctl`
   (when available) to temporarily cap thread pools during a critical section,
   e.g., while benchmarking.

Use `early_env_caps()` at process start (prior to importing NumPy). Use
`runtime_caps()` around code regions whose CPU-parallel behavior you want to
stabilize or compare across machines.
"""

import os
from contextlib import contextmanager
from typing import Optional

try:
    from threadpoolctl import threadpool_limits
except Exception:  # optional dependency; runtime_caps becomes a no-op
    threadpool_limits = None  # type: ignore

# Environment variables recognized by common BLAS/OMP runtimes.
ENV_VARS = {
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate (vecLib)
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}


def early_env_caps(threads: Optional[int]) -> None:
    """
    Set process-wide thread caps for numerical libraries via environment vars.

    Call this *before* importing NumPy/SciPy so those libraries initialize with
    the intended parallelism limits. If `threads` is None, this function does
    nothing.

    Parameters
    ----------
    threads : int | None
        Desired thread cap (minimum 1). None leaves the environment unchanged.
    """
    if threads is None:
        return
    t = str(max(1, int(threads)))
    for k in ENV_VARS:
        os.environ[k] = t


@contextmanager
def runtime_caps(threads: Optional[int]):
    """
    Temporarily cap thread pools using `threadpoolctl`.

    If `threadpoolctl` is unavailable or `threads` is None, this context
    manager is a no-op.

    Parameters
    ----------
    threads : int | None
        Desired thread cap during the context (minimum 1). None disables capping.

    Examples
    --------
    >>> from src.common.threads import runtime_caps
    >>> with runtime_caps(4):
    ...     # Code here will run with MKL/OpenBLAS/BLIS/OMP limited to 4 threads
    ...     pass
    """
    if threads is None or threadpool_limits is None:
        # No-op context manager
        yield
        return

    # Limit *all* detected pools (MKL, OpenBLAS, BLIS, OpenMP) to `threads`.
    with threadpool_limits(limits=max(1, int(threads))):
        yield
