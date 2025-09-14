# src/common/backend.py
from __future__ import annotations
from typing import Literal, Optional

def has_cupy() -> bool:
    try:
        import cupy as _  # noqa: F401
        return True
    except Exception:
        return False

def xp_for(device: Optional[Literal["cpu"]] | int):
    """
    Return numpy for CPU or cupy for GPU device index (0..).
    device=None or "cpu" -> numpy; device=int -> cupy.
    """
    if device is None or device == "cpu":
        import numpy as xp
        return xp
    import cupy as cp
    cp.cuda.Device(int(device)).use()
    return cp

def to_device(arr, device: Optional[int]):
    """Copy/convert host numpy -> device (cupy) if device is not None."""
    if device is None:
        return arr
    import cupy as cp
    return cp.asarray(arr, order="C")

def to_host(arr):
    """cupy -> numpy (no-op for numpy)."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return arr
