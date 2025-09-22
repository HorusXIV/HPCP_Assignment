# src/common/nvtx.py
from __future__ import annotations
"""
Tiny NVTX helper.

Provides a single context manager, `nvtx_range(msg)`, that annotates a code
region with an NVTX range *if* the optional `nvtx` package is available.
If NVTX is not installed (or any error occurs while importing/using it), the
context manager degrades to a no-op so callers can unconditionally wrap blocks.

Typical usage
-------------
>>> from src.common.nvtx import nvtx_range
>>> with nvtx_range("baseline-solve"):
...     # work to profile in Nsight Systems / Nsight Compute
...     pass
"""

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def nvtx_range(msg: str) -> Iterator[None]:
    """
    Enter an NVTX range labeled `msg`, or no-op if `nvtx` is unavailable.

    Parameters
    ----------
    msg : str
        The label shown in NVIDIA profiling tools (Nsight Systems/Compute).

    Notes
    -----
    - Requires the optional `nvtx` Python package for functionality.
    - Safe to use even when NVTX is not installed.
    """
    try:
        import nvtx  # type: ignore

        with nvtx.annotate(msg):  # type: ignore[attr-defined]
            yield
    except Exception:
        # Graceful no-op if nvtx is not present or any error occurs.
        yield
