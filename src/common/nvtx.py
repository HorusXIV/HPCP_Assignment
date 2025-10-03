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

import os
from contextlib import contextmanager
from typing import Iterator, Optional, Any, Callable

__all__ = ["nvtx_range", "nvtx_available", "annotate_if_enabled"]

# Lazy cached state to avoid import attempts when disabled
_NVTX_IMPORTED = False
_NVTX_MOD: Any = None
_NVTX_DISABLED = False  # permanently disabled after first failure


def _want_nvtx() -> bool:
    """Return True if NVTX instrumentation is requested via env var.

    Environment variables (evaluated once per process):
      MULTIGPU_NVTX=1      -> enable (default off)
      MULTIGPU_NVTX=0      -> force disable
    """
    return os.environ.get("MULTIGPU_NVTX", "0") == "1"


def _import_nvtx() -> Optional[Any]:
    global _NVTX_IMPORTED, _NVTX_MOD, _NVTX_DISABLED
    if _NVTX_DISABLED:
        return None
    if _NVTX_IMPORTED:
        return _NVTX_MOD
    if not _want_nvtx():
        _NVTX_DISABLED = True  # honor explicit disable for future calls
        return None
    try:  # pragma: no cover - import success path trivial
        import nvtx as _n  # type: ignore

        _NVTX_IMPORTED = True
        _NVTX_MOD = _n
        return _n
    except Exception:  # pragma: no cover - failure becomes silent no-op
        _NVTX_DISABLED = True
        return None


def nvtx_available() -> bool:
    """Return whether NVTX is both requested and import succeeded."""
    return _import_nvtx() is not None


@contextmanager
def nvtx_range(msg: str, color: Optional[int] = None) -> Iterator[None]:
    """Enter an NVTX range labeled `msg` (no-op if disabled/unavailable).

    Parameters
    ----------
    msg : str
        Range label (keep < 64 chars for compact timelines).
    color : int, optional
        0xRRGGBB style integer or any int accepted by nvtx (if provided)
        for consistent phase coloring across ranks. Ignored if NVTX off.
    """
    mod = _import_nvtx()
    if mod is None:  # fast no-op path
        yield
        return
    cm = None
    try:  # pragma: no cover (import + annotate simple)
        kwargs = {"message": msg}
        if color is not None:
            kwargs["color"] = int(color)
        cm = mod.annotate(**kwargs)  # type: ignore[arg-type]
    except Exception:
        cm = None
    if cm is None:
        # degrade to no-op if annotate construction failed
        yield
    else:
        with cm:
            yield


def annotate_if_enabled(
    label: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: wrap a function body in an NVTX range if enabled."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def _inner(*a, **k):
            with nvtx_range(label):
                return fn(*a, **k)

        _inner.__name__ = fn.__name__
        _inner.__doc__ = fn.__doc__
        return _inner

    return _wrap
