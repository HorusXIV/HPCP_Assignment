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
    """
    try:
        import cupy as _  # noqa: F401
        return True
    except Exception:
        return False


def xp_for(device: Optional[Literal["cpu"] | int]) -> Any:
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

    Notes
    -----
    When an integer device is supplied, this function performs:
      `cp.cuda.Device(device).use()`
    to make that device current for the calling thread.
    """
    if device is None or device == "cpu":
        import numpy as xp  # type: ignore
        return xp
    import cupy as cp  # type: ignore

    cp.cuda.Device(int(device)).use()
    return cp


def to_device(arr: Any, device: Optional[int]) -> Any:
    """
    Copy/convert a host (NumPy) array to a CUDA device (CuPy) if `device` is not None.

    Parameters
    ----------
    arr : array-like
        NumPy array (or already CuPy).
    device : int | None
        CUDA device index. If None, returns `arr` unchanged.

    Returns
    -------
    array-like
        CuPy array when `device` is an integer; otherwise the original `arr`.
    """
    if device is None:
        return arr
    import cupy as cp  # type: ignore

    return cp.asarray(arr, order="C")


def to_host(arr: Any) -> Any:
    """
    Convert a CuPy array to a NumPy array; return input unchanged for non-CuPy arrays.

    Parameters
    ----------
    arr : array-like
        Possibly a CuPy ndarray.

    Returns
    -------
    numpy.ndarray | original
        Host copy if input was CuPy; otherwise the original object.
    """
    try:
        import cupy as cp  # type: ignore

        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        # CuPy not available or `arr` is not a CuPy array.
        pass
    return arr
