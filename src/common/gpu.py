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

device_count()
    Return the number of available CUDA devices.

set_device(idx)
    Make CUDA device `idx` current for the calling thread.

get_device()
    Get the current CUDA device index.

sync()
    Synchronize the null/default CUDA stream.

memory_info(device)
    Get memory usage info for a device.

CudaEventTimer
    Context manager for timing GPU sections using CUDA events.

cuda_event_timer()
    Convenience constructor returning `CudaEventTimer()`.

pinned_empty(shape, dtype)
    Allocate a NumPy array backed by CUDA **pinned (page-locked)** host memory,
    useful for faster H2D/D2H transfers.
"""

from contextlib import ContextDecorator
from typing import Any
import numpy as np


def available() -> bool:
    """
    Return True if CuPy is importable and CUDA is functional.

    Returns
    -------
    bool
        True if CuPy can be imported and at least one CUDA device exists.

    Notes
    -----
    This checks both CuPy availability and CUDA runtime functionality.
    A return value of False could mean:
    - CuPy is not installed
    - CUDA drivers are not installed
    - No CUDA-capable GPUs are available
    """
    try:
        import cupy as cp  # type: ignore
        # Try to query device count to ensure CUDA runtime works
        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def device_count() -> int:
    """
    Return the number of available CUDA devices.

    Returns
    -------
    int
        Number of CUDA devices, or 0 if CUDA is unavailable.

    Examples
    --------
    >>> n = device_count()
    >>> if n > 0:
    ...     print(f"Found {n} CUDA device(s)")
    """
    try:
        import cupy as cp  # type: ignore
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def set_device(idx: int) -> None:
    """
    Set the active CUDA device for the current thread.

    Parameters
    ----------
    idx : int
        Zero-based CUDA device index.

    Raises
    ------
    RuntimeError
        If the device index is invalid or CUDA is unavailable.

    Examples
    --------
    >>> set_device(0)  # doctest: +SKIP
    >>> # All subsequent CuPy operations use GPU 0
    """
    try:
        import cupy as cp  # type: ignore

        n_devices = cp.cuda.runtime.getDeviceCount()
        if not (0 <= idx < n_devices):
            raise ValueError(
                f"Device index {idx} out of range [0, {n_devices})"
            )

        cp.cuda.Device(int(idx)).use()
    except ImportError as e:
        raise RuntimeError("CuPy is not available") from e
    except Exception as e:
        raise RuntimeError(f"Failed to set device {idx}: {e}") from e


def get_device() -> int:
    """
    Get the current CUDA device index.

    Returns
    -------
    int
        Current device index, or -1 if CUDA is unavailable.

    Examples
    --------
    >>> device = get_device()  # doctest: +SKIP
    >>> print(f"Current device: {device}")
    """
    try:
        import cupy as cp  # type: ignore
        return int(cp.cuda.Device().id)
    except Exception:
        return -1


def sync(device: int | None = None) -> None:
    """
    Synchronize the default (null) CUDA stream.

    Useful after launching asynchronous kernels or copies when you need a
    host-side barrier.

    Parameters
    ----------
    device : int | None, optional
        Device to synchronize. If None, synchronizes the current device.

    Raises
    ------
    RuntimeError
        If CUDA synchronization fails.

    Examples
    --------
    >>> sync()  # doctest: +SKIP
    >>> # All GPU work on current device is now complete
    """
    try:
        import cupy as cp  # type: ignore

        if device is not None:
            with cp.cuda.Device(device):
                cp.cuda.Stream.null.synchronize()
        else:
            cp.cuda.Stream.null.synchronize()
    except Exception as e:
        raise RuntimeError(f"CUDA synchronization failed: {e}") from e


def memory_info(device: int | None = None) -> dict[str, int]:
    """
    Get memory usage information for a CUDA device.

    Parameters
    ----------
    device : int | None, optional
        Device to query. If None, queries the current device.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'free': Free memory in bytes
        - 'total': Total memory in bytes
        - 'used': Used memory in bytes

    Examples
    --------
    >>> info = memory_info(0)  # doctest: +SKIP
    >>> print(f"GPU 0: {info['used'] / 1e9:.2f} GB used of {info['total'] / 1e9:.2f} GB")
    """
    try:
        import cupy as cp  # type: ignore

        if device is not None:
            with cp.cuda.Device(device):
                free, total = cp.cuda.runtime.memGetInfo()
        else:
            free, total = cp.cuda.runtime.memGetInfo()

        return {
            'free': int(free),
            'total': int(total),
            'used': int(total - free),
        }
    except Exception:
        return {'free': 0, 'total': 0, 'used': 0}


class CudaEventTimer(ContextDecorator):
    """
    Measure elapsed time on the GPU using CUDA events.

    Attributes
    ----------
    seconds : float | None
        Elapsed time in seconds, available after context exit.

    Examples
    --------
    >>> with CudaEventTimer() as timer:  # doctest: +SKIP
    ...     # Launch GPU kernels
    ...     result = gpu_computation()
    >>> print(f"GPU time: {timer.seconds:.4f}s")

    Notes
    -----
    - Uses CuPy's `cuda.Event` for precise GPU timing.
    - Measures wall-clock time on the GPU, not CPU time.
    - The timer synchronizes at exit, ensuring all GPU work is complete.
    """

    def __init__(self, device: int | None = None):
        """
        Initialize timer.

        Parameters
        ----------
        device : int | None, optional
            CUDA device to time on. If None, uses current device.
        """
        self.device = device
        self.seconds: float | None = None
        self._start: Any = None
        self._end: Any = None
        self.cp: Any = None

    def __enter__(self) -> CudaEventTimer:
        try:
            import cupy as cp  # type: ignore
            self.cp = cp

            if self.device is not None:
                cp.cuda.Device(self.device).use()

            self._start = cp.cuda.Event()
            self._end = cp.cuda.Event()
            self._start.record()

        except Exception as e:
            raise RuntimeError(f"Failed to start CUDA timer: {e}") from e

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._end.record()
            self._end.synchronize()
            ms = self.cp.cuda.get_elapsed_time(self._start, self._end)
            self.seconds = float(ms) / 1e3
        except Exception as e:
            # Set seconds to None on error but don't suppress original exception
            self.seconds = None
            if exc_type is None:
                # No exception from context body, raise timer error
                raise RuntimeError(f"Failed to stop CUDA timer: {e}") from e

        # Never suppress exceptions from context body
        return False


def cuda_event_timer(device: int | None = None) -> CudaEventTimer:
    """
    Convenience constructor for `CudaEventTimer`.

    Parameters
    ----------
    device : int | None, optional
        CUDA device to time on. If None, uses current device.

    Returns
    -------
    CudaEventTimer
        Timer context manager.

    Examples
    --------
    >>> timer = cuda_event_timer(device=0)  # doctest: +SKIP
    >>> with timer:
    ...     # GPU work
    ...     pass
    >>> print(f"Elapsed: {timer.seconds}s")
    """
    return CudaEventTimer(device=device)


def pinned_empty(shape: tuple[int, ...] | list[int], dtype: Any) -> Any:
    """
    Allocate a NumPy array backed by CUDA **pinned** (page-locked) host memory.

    Parameters
    ----------
    shape : tuple[int, ...] | list[int]
        Array shape.
    dtype : dtype-like
        NumPy dtype for the allocation.

    Returns
    -------
    numpy.ndarray
        A NumPy array backed by pinned host memory. A reference to the pinned
        allocation is attached as `._pinned_mem` to keep it alive.

    Raises
    ------
    ValueError
        If shape or dtype is invalid.
    RuntimeError
        If pinned memory allocation fails (e.g., out of memory).

    Notes
    -----
    Pinned (page-locked) memory enables:
    - Faster host ↔ device transfers via DMA
    - Asynchronous memory copies
    - Direct GPU access in some cases

    However, pinned memory is a limited resource and increases pressure on the
    system's pageable memory. Use judiciously for transfer buffers, not for
    long-term storage.

    Examples
    --------
    >>> arr = pinned_empty((1024, 1024), dtype=np.float32)  # doctest: +SKIP
    >>> # arr can be filled on CPU, then transferred quickly to GPU
    """
    try:
        import cupy as cp  # type: ignore
        import numpy as np

        # Validate shape
        if not shape:
            raise ValueError("Shape cannot be empty")

        shape_tuple = tuple(int(s) for s in shape)
        if any(s <= 0 for s in shape_tuple):
            raise ValueError(f"All dimensions must be positive, got {shape_tuple}")

        # Validate dtype
        try:
            dt = np.dtype(dtype)
        except TypeError as e:
            raise ValueError(f"Invalid dtype: {dtype}") from e

        # Calculate size
        n_elems = int(np.prod(shape_tuple))
        nbytes = dt.itemsize * n_elems

        if nbytes == 0:
            raise ValueError("Cannot allocate zero-sized array")

        # Allocate pinned memory
        try:
            mem = cp.cuda.PinnedMemory().alloc(nbytes)
        except cp.cuda.memory.OutOfMemoryError as e:
            raise RuntimeError(
                f"Failed to allocate {nbytes / 1e9:.2f} GB of pinned memory"
            ) from e

        # Wrap as NumPy array
        arr = np.frombuffer(mem, dtype=dt, count=n_elems).reshape(shape_tuple)

        # Attach allocation handle to prevent premature garbage collection
        arr._pinned_mem = mem  # type: ignore

        return arr

    except ImportError as e:
        raise RuntimeError("CuPy is not available") from e


def pinned_zeros(shape: tuple[int, ...] | list[int], dtype: Any = np.float32) -> Any:
    """
    Allocate a zero-initialized pinned array.

    Parameters
    ----------
    shape : tuple[int, ...] | list[int]
        Array shape.
    dtype : dtype-like, default float32
        NumPy dtype.

    Returns
    -------
    numpy.ndarray
        Zero-initialized pinned array.

    See Also
    --------
    pinned_empty : Allocate uninitialized pinned memory.

    Examples
    --------
    >>> arr = pinned_zeros((100, 100), dtype=np.float64)  # doctest: +SKIP
    >>> arr.sum()  # Should be 0.0
    """
    arr = pinned_empty(shape, dtype)
    arr.fill(0)
    return arr