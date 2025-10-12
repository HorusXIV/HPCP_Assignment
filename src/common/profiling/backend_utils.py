# src/common/profiling/backend_utils.py
from __future__ import annotations
"""
Backend detection and utilities for cross-backend profiling.

This module provides utilities to detect and work with different array
backends (NumPy, CuPy, Dask) in a unified way. It centralizes backend
detection logic that was previously scattered across profiling modules.

Supported Backends
------------------
- numpy: CPU arrays
- cupy: GPU arrays (single or multi-GPU)
- dask: Distributed arrays

Usage
-----
>>> arr = np.array([1, 2, 3])
>>> backend = detect_backend(arr)
>>> print(backend)  # "numpy"
>>> xp = get_array_module(arr)
>>> # xp is now numpy module, use xp.sum(), xp.mean(), etc.
"""

from typing import Any, Optional
from enum import Enum


class Backend(Enum):
    """
    Enumeration of supported array backends.
    """
    NUMPY = "numpy"
    CUPY = "cupy"
    DASK = "dask"
    UNKNOWN = "unknown"


def detect_backend(arr: Any) -> Backend:
    """
    Detect the backend type from an array instance.

    Parameters
    ----------
    arr : array-like
        Array object (NumPy, CuPy, or Dask).

    Returns
    -------
    Backend
        Detected backend enum value.

    Examples
    --------
    >>> import numpy as np
    >>> detect_backend(np.array([1, 2, 3]))
    <Backend.NUMPY: 'numpy'>

    >>> import cupy as cp
    >>> detect_backend(cp.array([1, 2, 3]))
    <Backend.CUPY: 'cupy'>
    """
    module_name = type(arr).__module__.split('.')[0]

    if module_name == 'numpy':
        return Backend.NUMPY
    elif module_name == 'cupy':
        return Backend.CUPY
    elif module_name == 'dask':
        return Backend.DASK
    else:
        return Backend.UNKNOWN


def get_array_module(arr: Any):
    """
    Get the array module (numpy/cupy/dask.array) from an array instance.

    This is the primary function for backend-agnostic array operations.
    The returned module can be used with the "array API" pattern:

        xp = get_array_module(arr)
        result = xp.sum(arr)  # Works for numpy, cupy, or dask

    Parameters
    ----------
    arr : array-like
        Array object (NumPy, CuPy, or Dask).

    Returns
    -------
    module
        The array module (numpy, cupy, or dask.array).

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> xp = get_array_module(arr)
    >>> xp.sum(arr)
    6
    """
    backend = detect_backend(arr)

    if backend == Backend.NUMPY:
        import numpy
        return numpy
    elif backend == Backend.CUPY:
        import cupy
        return cupy
    elif backend == Backend.DASK:
        import dask.array
        return dask.array
    else:
        # Fallback to numpy for unknown types
        import numpy
        return numpy


def is_gpu_array(arr: Any) -> bool:
    """
    Check if an array is on GPU (CuPy).

    Parameters
    ----------
    arr : array-like
        Array to check.

    Returns
    -------
    bool
        True if array is a CuPy array, False otherwise.
    """
    return detect_backend(arr) == Backend.CUPY


def is_distributed_array(arr: Any) -> bool:
    """
    Check if an array is distributed (Dask).

    Parameters
    ----------
    arr : array-like
        Array to check.

    Returns
    -------
    bool
        True if array is a Dask array, False otherwise.
    """
    return detect_backend(arr) == Backend.DASK


def get_device_id(arr: Any) -> Optional[int]:
    """
    Get the GPU device ID for a CuPy array.

    Parameters
    ----------
    arr : array-like
        Array to query.

    Returns
    -------
    int | None
        Device ID if arr is a CuPy array, None otherwise.

    Examples
    --------
    >>> import cupy as cp
    >>> with cp.cuda.Device(2):
    ...     arr = cp.array([1, 2, 3])
    >>> get_device_id(arr)
    2
    """
    if not is_gpu_array(arr):
        return None

    try:
        import cupy as cp
        return int(arr.device.id)
    except Exception:
        return None


def get_backend_info(arr: Any) -> dict:
    """
    Get comprehensive backend information for an array.

    Parameters
    ----------
    arr : array-like
        Array to inspect.

    Returns
    -------
    dict
        Information dictionary with keys:
          - "backend": Backend enum value
          - "backend_name": String name ("numpy", "cupy", "dask")
          - "is_gpu": Boolean
          - "is_distributed": Boolean
          - "device_id": GPU device ID (if applicable)
          - "shape": Array shape
          - "dtype": Array dtype
          - "nbytes": Estimated memory usage (if available)

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3], dtype=np.float32)
    >>> get_backend_info(arr)
    {
        "backend": <Backend.NUMPY: 'numpy'>,
        "backend_name": "numpy",
        "is_gpu": False,
        "is_distributed": False,
        "device_id": None,
        "shape": (3,),
        "dtype": dtype('float32'),
        "nbytes": 12
    }
    """
    backend = detect_backend(arr)

    info = {
        "backend": backend,
        "backend_name": backend.value,
        "is_gpu": is_gpu_array(arr),
        "is_distributed": is_distributed_array(arr),
        "device_id": get_device_id(arr),
    }

    # Try to get array metadata
    try:
        info["shape"] = tuple(arr.shape)
    except Exception:
        info["shape"] = None

    try:
        info["dtype"] = arr.dtype
    except Exception:
        info["dtype"] = None

    try:
        info["nbytes"] = int(arr.nbytes)
    except Exception:
        info["nbytes"] = None

    return info


def create_synchronize_fn(arr: Any) -> Optional[callable]:
    """
    Create an appropriate synchronization function for an array backend.

    This is useful for accurate timing: GPU operations are asynchronous,
    so we need to synchronize before stopping a timer.

    Parameters
    ----------
    arr : array-like
        Array whose backend determines the sync function.

    Returns
    -------
    callable | None
        Synchronization function, or None if not needed (NumPy).

    Examples
    --------
    >>> import cupy as cp
    >>> arr = cp.array([1, 2, 3])
    >>> sync = create_synchronize_fn(arr)
    >>> # Use in timing:
    >>> t0 = time.time()
    >>> result = some_gpu_operation(arr)
    >>> sync()  # Wait for GPU to finish
    >>> elapsed = time.time() - t0
    """
    backend = detect_backend(arr)

    if backend == Backend.CUPY:
        # For CuPy: synchronize the device
        try:
            import cupy as cp
            device_id = get_device_id(arr)
            if device_id is not None:
                def sync():
                    with cp.cuda.Device(device_id):
                        cp.cuda.Stream.null.synchronize()
                return sync
        except Exception:
            pass

    elif backend == Backend.DASK:
        # For Dask: no sync needed (lazy evaluation)
        # User should call .compute() explicitly
        return None

    # NumPy and others: no sync needed
    return None


def ensure_numpy(arr: Any, copy: bool = False):
    """
    Convert any array to NumPy, handling GPU and Dask arrays.

    Parameters
    ----------
    arr : array-like
        Input array (NumPy, CuPy, or Dask).
    copy : bool, default False
        Whether to force a copy even if already NumPy.

    Returns
    -------
    numpy.ndarray
        NumPy array.

    Notes
    -----
    - CuPy arrays are copied from GPU to CPU
    - Dask arrays are computed (may be expensive!)
    - NumPy arrays are returned as-is (or copied if copy=True)
    """
    import numpy as np

    backend = detect_backend(arr)

    if backend == Backend.NUMPY:
        return np.array(arr, copy=copy)

    elif backend == Backend.CUPY:
        # Transfer from GPU to CPU
        try:
            import cupy as cp
            return cp.asnumpy(arr)
        except Exception:
            return np.array(arr)

    elif backend == Backend.DASK:
        # Compute the Dask array
        try:
            return arr.compute()
        except Exception:
            return np.array(arr)

    else:
        # Unknown: try generic conversion
        return np.array(arr)


def format_memory(nbytes: int) -> str:
    """
    Format byte count as human-readable string.

    Parameters
    ----------
    nbytes : int
        Number of bytes.

    Returns
    -------
    str
        Formatted string (e.g., "1.23 GB").

    Examples
    --------
    >>> format_memory(1234567890)
    '1.15 GB'
    >>> format_memory(12345)
    '12.06 KB'
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(nbytes)
    unit_idx = 0

    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1

    return f"{size:.2f} {units[unit_idx]}"


def summarize_backend_info(arrs: dict) -> dict:
    """
    Summarize backend information for multiple arrays.

    Useful for logging/reporting what backends are in use.

    Parameters
    ----------
    arrs : dict
        Mapping of name -> array.

    Returns
    -------
    dict
        Summary with keys:
          - "arrays": dict of per-array info
          - "backends_used": set of backend names
          - "total_gpu_arrays": count
          - "total_distributed_arrays": count
          - "total_memory": total memory across all arrays
          - "gpu_devices": set of GPU device IDs in use

    Examples
    --------
    >>> import numpy as np
    >>> import cupy as cp
    >>> arrs = {
    ...     "input": np.zeros((100, 100)),
    ...     "weights": cp.zeros((10, 10)),
    ... }
    >>> summary = summarize_backend_info(arrs)
    >>> summary["backends_used"]
    {'numpy', 'cupy'}
    """
    arrays_info = {}
    backends_used = set()
    gpu_count = 0
    distributed_count = 0
    total_memory = 0
    gpu_devices = set()

    for name, arr in arrs.items():
        info = get_backend_info(arr)
        arrays_info[name] = info

        backends_used.add(info["backend_name"])

        if info["is_gpu"]:
            gpu_count += 1
            if info["device_id"] is not None:
                gpu_devices.add(info["device_id"])

        if info["is_distributed"]:
            distributed_count += 1

        if info["nbytes"] is not None:
            total_memory += info["nbytes"]

    return {
        "arrays": arrays_info,
        "backends_used": sorted(backends_used),
        "total_gpu_arrays": gpu_count,
        "total_distributed_arrays": distributed_count,
        "total_memory": total_memory,
        "total_memory_formatted": format_memory(total_memory),
        "gpu_devices": sorted(gpu_devices),
    }


# ---------------------------------------------------------------------------
# Backend capability checking
# ---------------------------------------------------------------------------

def check_backend_available(backend: Backend | str) -> bool:
    """
    Check if a specific backend is available (importable).

    Parameters
    ----------
    backend : Backend | str
        Backend to check (enum or string name).

    Returns
    -------
    bool
        True if backend is available, False otherwise.

    Examples
    --------
    >>> check_backend_available("numpy")
    True
    >>> check_backend_available("cupy")  # Depends on system
    False  # or True if CuPy installed
    """
    if isinstance(backend, str):
        backend = Backend(backend)

    if backend == Backend.NUMPY:
        try:
            import numpy
            return True
        except ImportError:
            return False

    elif backend == Backend.CUPY:
        try:
            import cupy
            return True
        except ImportError:
            return False

    elif backend == Backend.DASK:
        try:
            import dask.array
            return True
        except ImportError:
            return False

    return False


def get_available_backends() -> list[Backend]:
    """
    Get list of available backends on this system.

    Returns
    -------
    list[Backend]
        List of available backend enums.

    Examples
    --------
    >>> get_available_backends()
    [<Backend.NUMPY: 'numpy'>, <Backend.CUPY: 'cupy'>]
    """
    available = []

    for backend in [Backend.NUMPY, Backend.CUPY, Backend.DASK]:
        if check_backend_available(backend):
            available.append(backend)

    return available


def get_gpu_count() -> int:
    """
    Get the number of available GPUs.

    Returns
    -------
    int
        Number of GPUs, or 0 if CuPy not available or no GPUs.

    Examples
    --------
    >>> get_gpu_count()
    4  # If 4 GPUs available
    """
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def get_backend_versions() -> dict[str, str | None]:
    """
    Get version strings for all available backends.

    Returns
    -------
    dict[str, str | None]
        Mapping of backend name -> version string (None if unavailable).

    Examples
    --------
    >>> get_backend_versions()
    {
        'numpy': '1.26.0',
        'cupy': '12.3.0',
        'dask': '2024.1.0'
    }
    """
    versions = {}

    # NumPy
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        versions['numpy'] = None

    # CuPy
    try:
        import cupy
        versions['cupy'] = cupy.__version__
    except ImportError:
        versions['cupy'] = None

    # Dask
    try:
        import dask
        versions['dask'] = dask.__version__
    except ImportError:
        versions['dask'] = None

    return versions


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def print_backend_summary(arrs: dict, verbose: bool = True) -> None:
    """
    Print a human-readable summary of backend usage.

    Parameters
    ----------
    arrs : dict
        Mapping of name -> array.
    verbose : bool, default True
        If True, print per-array details. If False, print only summary.

    Examples
    --------
    >>> import numpy as np
    >>> arrs = {"input": np.zeros((1000, 1000))}
    >>> print_backend_summary(arrs)
    Backend Summary
    ===============
    Backends used: numpy
    Total arrays: 1
    GPU arrays: 0
    Distributed arrays: 0
    Total memory: 7.63 MB

    Array Details:
    - input: numpy, shape=(1000, 1000), dtype=float64, 7.63 MB
    """
    summary = summarize_backend_info(arrs)

    print("Backend Summary")
    print("=" * 50)
    print(f"Backends used: {', '.join(summary['backends_used'])}")
    print(f"Total arrays: {len(arrs)}")
    print(f"GPU arrays: {summary['total_gpu_arrays']}")
    print(f"Distributed arrays: {summary['total_distributed_arrays']}")
    print(f"Total memory: {summary['total_memory_formatted']}")

    if summary['gpu_devices']:
        print(f"GPU devices: {', '.join(map(str, summary['gpu_devices']))}")

    if verbose:
        print("\nArray Details:")
        for name, info in summary['arrays'].items():
            shape = info.get('shape', 'unknown')
            dtype = info.get('dtype', 'unknown')
            nbytes = info.get('nbytes', 0)
            mem_str = format_memory(nbytes) if nbytes else 'unknown'

            details = [
                info['backend_name'],
                f"shape={shape}",
                f"dtype={dtype}",
                mem_str
            ]

            if info['device_id'] is not None:
                details.insert(1, f"device={info['device_id']}")

            print(f"  - {name}: {', '.join(details)}")


def validate_backend_compatibility(*arrs) -> tuple[bool, str]:
    """
    Check if arrays are compatible (same backend or compatible backends).

    Parameters
    ----------
    *arrs : array-like
        Arrays to check for compatibility.

    Returns
    -------
    (bool, str)
        Tuple of (is_compatible, message).

    Examples
    --------
    >>> import numpy as np
    >>> import cupy as cp
    >>> a = np.array([1, 2])
    >>> b = np.array([3, 4])
    >>> validate_backend_compatibility(a, b)
    (True, 'All arrays use numpy')

    >>> c = cp.array([5, 6])
    >>> validate_backend_compatibility(a, c)
    (False, 'Mixed backends: numpy, cupy')
    """
    if not arrs:
        return True, "No arrays provided"

    backends = [detect_backend(arr) for arr in arrs]
    unique_backends = set(backends)

    if len(unique_backends) == 1:
        backend_name = unique_backends.pop().value
        return True, f"All arrays use {backend_name}"
    else:
        backend_names = sorted([b.value for b in unique_backends])
        return False, f"Mixed backends: {', '.join(backend_names)}"


def recommend_synchronization(arr: Any) -> str:
    """
    Recommend synchronization strategy for timing operations on this array.

    Parameters
    ----------
    arr : array-like
        Array to analyze.

    Returns
    -------
    str
        Human-readable recommendation.

    Examples
    --------
    >>> import cupy as cp
    >>> arr = cp.array([1, 2, 3])
    >>> print(recommend_synchronization(arr))
    GPU array detected. Use synchronization for accurate timing:
        sync_fn = create_synchronize_fn(arr)
        # ... operation ...
        sync_fn()
    """
    backend = detect_backend(arr)

    if backend == Backend.CUPY:
        return (
            "GPU array detected. Use synchronization for accurate timing:\n"
            "    sync_fn = create_synchronize_fn(arr)\n"
            "    # ... operation ...\n"
            "    sync_fn()"
        )
    elif backend == Backend.DASK:
        return (
            "Dask array detected. Call .compute() to trigger execution:\n"
            "    result = arr.compute()\n"
            "Note: Timing should wrap the .compute() call."
        )
    else:
        return (
            "NumPy array detected. No synchronization needed for timing."
        )


# ---------------------------------------------------------------------------
# Export all utilities
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "Backend",
    # Detection
    "detect_backend",
    "get_array_module",
    "is_gpu_array",
    "is_distributed_array",
    "get_device_id",
    "get_backend_info",
    # Synchronization
    "create_synchronize_fn",
    # Conversion
    "ensure_numpy",
    # Utilities
    "format_memory",
    "summarize_backend_info",
    # Capability checking
    "check_backend_available",
    "get_available_backends",
    "get_gpu_count",
    "get_backend_versions",
    # Diagnostics
    "print_backend_summary",
    "validate_backend_compatibility",
    "recommend_synchronization",
]