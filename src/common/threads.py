# src/common/threads.py
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Optional
try:
    from threadpoolctl import threadpool_limits
except Exception:  # optional dependency
    threadpool_limits = None  # type: ignore

# Libraries we care about: OpenBLAS, MKL, BLIS, Apple Accelerate (vecLib), OpenMP backends.
ENV_VARS = {
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",   # Apple Accelerate
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}

def early_env_caps(threads: Optional[int]) -> None:
    """
    Set process env caps for numerical libs *before* they are imported.
    Use only at process start, before importing numpy/scipy.
    If threads is None, do nothing.
    """
    if threads is None:
        return
    t = str(max(1, int(threads)))
    for k in ENV_VARS:
        os.environ[k] = t

@contextmanager
def runtime_caps(threads: Optional[int]):
    """
    Enforce thread caps during a critical section (benchmark) using threadpoolctl.
    Falls back to a no-op if threadpoolctl isn't available.
    """
    if threads is None or threadpool_limits is None:
        # No-op context manager
        yield
        return
    # Limit *all* detected pools (MKL, OpenBLAS, BLIS, OpenMP) to `threads`
    with threadpool_limits(limits=max(1, int(threads))):
        yield
