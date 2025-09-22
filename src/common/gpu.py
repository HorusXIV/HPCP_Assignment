# src/common/gpu.py
from __future__ import annotations
"""
Lightweight CUDA utilities used by optional GPU paths.

This module intentionally keeps a very small surface area and avoids importing
CuPy at module import time. Functions import CuPy lazily so that CPU-only
environments can still import the package without errors.

Provided helpers
----------------
available()
    Return True if CuPy can be imported.

set_device(idx)
    Make CUDA device `idx` current for the calling thread.

sync()
    Synchronize the null/default CUDA stream.

CudaEventTimer
    Context manager for timing GPU sections using CUDA events.

cuda_event_timer()
    Convenience constructor returning `CudaEventTimer()`.

last_cuda_elapsed()
    (Reserved) Retrieve the last recorded CUDA timing if your code stores it.

pinned_empty(shape, dtype)
    Allocate a NumPy array backed by CUDA **pinned (page-locked)** host memory,
    useful for faster H2D/D2H transfers. The returned array holds a reference
    to the pinned allocation to prevent premature free.
"""

from contextlib import ContextDecorator
from typing import Any, Optional, Sequence, Tuple


def available() -> bool:
    """
    Return True if CuPy is importable in the current environment.

    Notes
    -----
    This does not probe for a working CUDA device; it only checks that
    `import cupy` succeeds.
    """
    try:
        import cupy as _  # noqa: F401
        return True
    except Exception:
        return False


def set_device(idx: int) -> None:
    """
    Set the active CUDA device for the current thread.

    Parameters
    ----------
    idx : int
        Zero-based CUDA device index.
    """
    import cupy as cp  # type: ignore

    cp.cuda.Device(int(idx)).use()


def sync() -> None:
    """
    Synchronize the default (null) CUDA stream.

    Useful after launching asynchronous kernels or copies when you need a
    host-side barrier.
    """
    import cupy as cp  # type: ignore

    cp.cuda.Stream.null.synchronize()


class CudaEventTimer(ContextDecorator):
    """
    Measure elapsed time on the GPU using CUDA events.

    Usage
    -----
    >>> with CudaEventTimer() as t:
    ...     # launch kernels / do GPU work
    ...     pass
    >>> print(t.seconds)  # seconds as float

    Notes
    -----
    - Uses CuPy's `cuda.Event` under the hood.
    - The measurement covers the work recorded between context enter/exit on
      the current stream (default stream by default).
    """

    def __enter__(self) -> "CudaEventTimer":
        import cupy as cp  # type: ignore

        self.cp = cp
        self._start = cp.cuda.Event()
        self._end = cp.cuda.Event()
        self._start.record()
        self.seconds: Optional[float] = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._end.record()
        self._end.synchronize()
        ms = self.cp.cuda.get_elapsed_time(self._start, self._end)  # type: ignore[attr-defined]
        self.seconds = float(ms) / 1e3
        # Do not suppress exceptions from the context body
        return False


def cuda_event_timer() -> CudaEventTimer:
    """
    Convenience constructor for `CudaEventTimer`.

    Returns
    -------
    CudaEventTimer
    """
    return CudaEventTimer()


_last_cuda_secs: Optional[float] = None  # reserved hook if callers want to cache timings


def last_cuda_elapsed() -> Optional[float]:
    """
    Return the last cached CUDA elapsed time in seconds, if your code sets it.

    By default this module does not set `_last_cuda_secs`; it is provided as a
    simple shared slot for callers that prefer a global readback.
    """
    return _last_cuda_secs


def pinned_empty(shape: Sequence[int] | Tuple[int, ...], dtype: Any):
    """
    Allocate a NumPy array backed by CUDA **pinned** (page-locked) host memory.

    Parameters
    ----------
    shape : Sequence[int] | tuple[int, ...]
        Array shape.
    dtype : Any
        NumPy dtype for the allocation.

    Returns
    -------
    numpy.ndarray
        A NumPy array backed by pinned host memory. A reference to the pinned
        allocation is attached to the array (`._pinned_mem`) to keep it alive.

    Notes
    -----
    Pinned memory can significantly speed up host↔device transfers at the cost
    of increased pressure on the pageable memory system. Use judiciously.
    """
    import cupy as cp  # type: ignore
    import numpy as np

    n_elems = int(np.prod(shape))
    nbytes = np.dtype(dtype).itemsize * n_elems

    # Allocate a pinned host buffer and wrap it as a NumPy array.
    mem = cp.cuda.PinnedMemory().alloc(nbytes)
    arr = np.frombuffer(mem, dtype=dtype, count=n_elems).reshape(tuple(shape))
    # Attach the allocation handle so it stays alive as long as the array does.
    setattr(arr, "_pinned_mem", mem)
    return arr
