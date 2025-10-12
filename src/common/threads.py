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
from typing import Iterator

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


def early_env_caps(threads: int | None) -> None:
    """
    Set process-wide thread caps for numerical libraries via environment vars.

    Call this *before* importing NumPy/SciPy so those libraries initialize with
    the intended parallelism limits. If `threads` is None, this function does
    nothing.

    Parameters
    ----------
    threads : int | None
        Desired thread cap (minimum 1). None leaves the environment unchanged.

    Examples
    --------
    >>> import sys
    >>> # Must be called before importing numpy
    >>> early_env_caps(4)
    >>> import numpy as np  # Will use 4 threads max
    """
    if threads is None:
        return
    t = str(max(1, int(threads)))
    for k in ENV_VARS:
        os.environ[k] = t


@contextmanager
def runtime_caps(threads: int | None) -> Iterator[None]:
    """
    Temporarily cap thread pools using `threadpoolctl`.

    If `threadpoolctl` is unavailable or `threads` is None, this context
    manager is a no-op.

    Parameters
    ----------
    threads : int | None
        Desired thread cap during the context (minimum 1). None disables capping.

    Yields
    ------
    None

    Examples
    --------
    >>> from src.common.threads import runtime_caps
    >>> with runtime_caps(4):
    ...     # Code here will run with MKL/OpenBLAS/BLIS/OMP limited to 4 threads
    ...     result = expensive_computation()
    """
    if threads is None or threadpool_limits is None:
        # No-op context manager
        yield
        return

    # Limit *all* detected pools (MKL, OpenBLAS, BLIS, OpenMP) to `threads`.
    with threadpool_limits(limits=max(1, int(threads))):
        yield


def get_current_threads() -> dict[str, str | None]:
    """
    Get current thread environment variable settings.

    Returns
    -------
    dict[str, str | None]
        Dictionary of environment variables and their values.
        None if variable is not set.

    Examples
    --------
    >>> settings = get_current_threads()
    >>> settings['OMP_NUM_THREADS']
    '4'
    """
    return {k: os.environ.get(k) for k in ENV_VARS}


def has_threadpoolctl() -> bool:
    """
    Check if threadpoolctl is available.

    Returns
    -------
    bool
        True if threadpoolctl is installed and can be imported.

    Examples
    --------
    >>> if has_threadpoolctl():
    ...     print("Runtime thread capping available")
    ... else:
    ...     print("Only early_env_caps() will work")
    """
    return threadpool_limits is not None