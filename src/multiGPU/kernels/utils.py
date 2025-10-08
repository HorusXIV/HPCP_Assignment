"""
Utility helpers for multiGPU kernels.

- nvtx_range: Optional NVTX annotation context manager
- verbose_enabled: Env-driven verbose logging toggle
- _pinned_empty: Allocate NumPy arrays backed by CUDA pinned memory
"""

from __future__ import annotations

import os
import numpy as np
from contextlib import contextmanager


@contextmanager
def nvtx_range(msg: str, color: int | None = None):
    """Lightweight NVTX context manager controlled by MULTIGPU_NVTX.

    When the environment variable ``MULTIGPU_NVTX`` is set to "1", emits
    NVTX ranges; otherwise acts as a no-op.

    Args:
        msg (str): Annotation label.
        color (int | None, optional): Optional RGB integer color.
    """
    if os.environ.get("MULTIGPU_NVTX", "0") != "1":
        yield
        return
    cm = None
    try:
        import nvtx as _nvtx  # type: ignore

        kwargs = {"message": msg}
        if color is not None:
            kwargs["color"] = int(color)
        cm = _nvtx.annotate(**kwargs)
    except Exception:
        cm = None  # degrade silently
    if cm is None:
        yield
    else:
        with cm:
            yield


def verbose_enabled() -> bool:
    """Return True if verbose multiGPU logging is enabled via env.

    Controlled by the environment variable ``MULTIGPU_VERBOSE``.

    Returns:
        bool: True when verbose mode is active.
    """
    try:
        return int(os.environ.get("MULTIGPU_VERBOSE", "0")) > 0
    except Exception:
        return False


def _pinned_empty(shape, dtype):
    """Allocate a NumPy array backed by CUDA pinned (page-locked) memory.

    Uses ``cp.cuda.alloc_pinned_memory`` when available; falls back to
    ``cp.cuda.PinnedMemory`` on some CuPy versions.

    Args:
        shape: Array shape.
        dtype: NumPy dtype for the array.

    Returns:
        np.ndarray: Host array backed by pinned memory.

    Raises:
        RuntimeError: If pinned memory allocation fails.
    """
    import cupy as cp  # local import to avoid hard dependency when unused

    n_elems = int(np.prod(shape))
    nbytes = np.dtype(dtype).itemsize * n_elems
    mem = None
    try:
        mem = cp.cuda.alloc_pinned_memory(int(nbytes))
    except Exception:
        try:
            mem = cp.cuda.PinnedMemory(int(nbytes))  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"Failed to allocate pinned host memory: {exc}")
    arr = np.frombuffer(mem, dtype=dtype, count=n_elems).reshape(tuple(shape))
    return arr


__all__ = ["nvtx_range", "verbose_enabled", "_pinned_empty"]
