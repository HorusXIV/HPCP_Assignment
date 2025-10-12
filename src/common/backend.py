# src/common/backend.py
from __future__ import annotations

"""
Backend helpers to abstract over NumPy (CPU) and CuPy (GPU).

This tiny utility module centralizes:
  - Feature detection (`has_cupy`)
  - Selecting the array module for a given device (`xp_for`)
  - Copying arrays to/from a CUDA device (`to_device`, `to_host`)

Typical usage
-------------
>>> xp = xp_for("cpu")     # -> numpy
>>> xp = xp_for(0)         # -> cupy with device 0 selected
>>> a_gpu = to_device(a_np, device=0)
>>> a_np  = to_host(a_gpu)
"""

from typing import Any, Literal, Optional


def has_cupy() -> bool:
    """
    Return True if CuPy is importable in the current environment.

    This is a lightweight feature probe and does not select a CUDA device.

    Returns
    -------
    bool
        True if CuPy can be imported, False otherwise.
    """
    try:
        import cupy as _  # noqa: F401
        return True
    except Exception:
        return False


def xp_for(device: Literal["cpu"] | int | None) -> Any:
    """
    Return the array module for the requested device.

    Parameters
    ----------
    device : {"cpu"} | int | None
        - "cpu" or None → return the NumPy module
        - int (0, 1, ...) → select that CUDA device and return the CuPy module

    Returns
    -------
    module
        Either `numpy` or `cupy`.

    Raises
    ------
    RuntimeError
        If device is an integer but CuPy is not available or device is invalid.

    Notes
    -----
    When an integer device is supplied, this function performs:
      `cp.cuda.Device(device).use()`
    to make that device current for the calling thread.

    Examples
    --------
    >>> import numpy as np
    >>> xp = xp_for("cpu")
    >>> isinstance(xp.array([1, 2, 3]), np.ndarray)
    True

    >>> xp = xp_for(0)  # doctest: +SKIP
    >>> # returns cupy module with device 0 selected
    """
    if device is None or device == "cpu":
        import numpy as xp  # type: ignore
        return xp

    try:
        import cupy as cp  # type: ignore
        cp.cuda.Device(int(device)).use()
        return cp
    except ImportError as e:
        raise RuntimeError(
            f"CuPy is not available, cannot use device {device}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to select CUDA device {device}: {e}"
        ) from e


def to_device(
        arr: Any,
        device: int | None,
        dtype: Any | None = None,
        copy: bool = False
) -> Any:
    """
    Copy/convert a host (NumPy) array to a CUDA device (CuPy).

    Parameters
    ----------
    arr : array-like
        NumPy array (or already CuPy array).
    device : int | None
        CUDA device index. If None, returns `arr` unchanged.
    dtype : dtype-like, optional
        Target dtype. If None, preserves input dtype.
    copy : bool, optional
        If True, always copy even if already on correct device with correct dtype.
        Default: False.

    Returns
    -------
    array-like
        CuPy array when `device` is an integer; otherwise the original `arr`.

    Raises
    ------
    RuntimeError
        If CuPy is unavailable, device is invalid, or CUDA out of memory.

    Examples
    --------
    >>> import numpy as np
    >>> a_cpu = np.array([1, 2, 3], dtype=np.float32)
    >>> a_gpu = to_device(a_cpu, device=0)  # doctest: +SKIP
    >>> # a_gpu is now a CuPy array on GPU 0

    >>> a_same = to_device(a_cpu, device=None)
    >>> a_same is a_cpu
    True
    """
    if device is None:
        return arr

    try:
        import cupy as cp  # type: ignore

        # Select device first
        cp.cuda.Device(int(device)).use()

        # Check if already on correct device
        if isinstance(arr, cp.ndarray):
            if arr.device.id == device:
                if dtype is None and not copy:
                    return arr  # Already on correct device
                elif dtype == arr.dtype and not copy:
                    return arr  # Correct device and dtype

        # Transfer/convert
        if dtype is None:
            return cp.asarray(arr, order="C")
        else:
            return cp.asarray(arr, dtype=dtype, order="C")

    except ImportError as e:
        raise RuntimeError(
            f"CuPy is not available, cannot transfer to device {device}"
        ) from e
    except cp.cuda.memory.OutOfMemoryError as e:  # type: ignore
        raise RuntimeError(
            f"CUDA out of memory on device {device} while transferring array "
            f"of shape {getattr(arr, 'shape', 'unknown')}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to transfer array to device {device}: {e}"
        ) from e


def to_host(arr: Any, dtype: Any | None = None) -> Any:
    """
    Convert a CuPy array to a NumPy array; return input unchanged for non-CuPy arrays.

    Parameters
    ----------
    arr : array-like
        Possibly a CuPy ndarray.
    dtype : dtype-like, optional
        Target dtype for the host array. If None, preserves input dtype.

    Returns
    -------
    numpy.ndarray | original
        Host (NumPy) array if input was CuPy; otherwise the original object.

    Notes
    -----
    This function is safe to call even if CuPy is not installed. Non-CuPy
    inputs are returned unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> to_host(a) is a
    True

    >>> # With CuPy array (hypothetical)
    >>> # a_gpu = cp.array([1, 2, 3])
    >>> # a_cpu = to_host(a_gpu)
    >>> # isinstance(a_cpu, np.ndarray)
    >>> # True
    """
    try:
        import cupy as cp  # type: ignore

        if isinstance(arr, cp.ndarray):
            result = cp.asnumpy(arr)
            if dtype is not None and result.dtype != dtype:
                import numpy as np
                result = result.astype(dtype)
            return result
    except ImportError:
        # CuPy not available, arr cannot be a CuPy array
        pass
    except Exception:
        # Other error during conversion; fall through to return original
        pass

    # Not a CuPy array or conversion failed
    if dtype is not None and hasattr(arr, 'astype'):
        return arr.astype(dtype)
    return arr


def get_array_device(arr: Any) -> int | None:
    """
    Get the CUDA device ID of an array, or None if on CPU.

    Parameters
    ----------
    arr : array-like
        Array to check (NumPy or CuPy).

    Returns
    -------
    int | None
        Device ID (0, 1, ...) if arr is a CuPy array, None otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> get_array_device(a) is None
    True
    """
    try:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return int(arr.device.id)
    except Exception:
        pass
    return None


def same_device(*arrays: Any) -> bool:
    """
    Check if all arrays are on the same device (CPU or GPU).

    Parameters
    ----------
    *arrays : array-like
        Arrays to check.

    Returns
    -------
    bool
        True if all arrays are on the same device.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> same_device(a, b)
    True
    """
    if not arrays:
        return True

    devices = [get_array_device(arr) for arr in arrays]
    return len(set(devices)) == 1