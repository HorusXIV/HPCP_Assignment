# src/common/nvtx.py
from __future__ import annotations

"""
Tiny NVTX helper for profiling with NVIDIA Nsight Systems/Compute.

Provides a context manager `nvtx_range(msg)` that annotates code regions
with NVTX ranges *if* the optional `nvtx` package is available and enabled.
If NVTX is not installed or disabled, the context manager degrades to a no-op,
so callers can unconditionally wrap blocks without runtime overhead.

Environment Variables
---------------------
HPCP_NVTX : {"0", "1"}
    Enable (1) or disable (0) NVTX instrumentation. Default: 0 (disabled).
    Set to "1" to enable profiling annotations.

Typical Usage
-------------
>>> from src.common.nvtx import nvtx_range
>>> with nvtx_range("baseline-solve"):
...     # work to profile in Nsight Systems / Nsight Compute
...     pass

>>> # Or use as decorator
>>> @annotate_if_enabled("heavy_computation")
... def my_func():
...     pass
"""

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

__all__ = [
    "nvtx_range",
    "nvtx_available",
    "annotate_if_enabled",
    "Colors",
]

# Lazy cached state to avoid import attempts when disabled
_NVTX_IMPORTED = False
_NVTX_MOD: Any = None
_NVTX_DISABLED = False  # permanently disabled after first failure

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


class Colors:
    """
    Common NVTX color constants in 0xRRGGBB format.

    Use these for consistent coloring across profiling sessions.

    Examples
    --------
    >>> with nvtx_range("compute", color=Colors.GREEN):
    ...     pass  # doctest: +SKIP
    """
    # Primary colors
    RED = 0xFF0000
    GREEN = 0x00FF00
    BLUE = 0x0000FF

    # Secondary colors
    YELLOW = 0xFFFF00
    CYAN = 0x00FFFF
    MAGENTA = 0xFF00FF

    # Grayscale
    WHITE = 0xFFFFFF
    LIGHT_GRAY = 0xCCCCCC
    GRAY = 0x888888
    DARK_GRAY = 0x444444
    BLACK = 0x000000

    # Additional useful colors
    ORANGE = 0xFF8800
    PURPLE = 0x8800FF
    PINK = 0xFF88FF
    LIME = 0x88FF00
    TEAL = 0x008888
    NAVY = 0x000088


def _want_nvtx() -> bool:
    """
    Return True if NVTX instrumentation is requested via environment variable.

    Returns
    -------
    bool
        True if HPCP_NVTX=1, False otherwise.

    Notes
    -----
    Environment variables (evaluated once per process):
      HPCP_NVTX=1      -> enable (default off)
      HPCP_NVTX=0      -> force disable

    Alternative names are checked for backward compatibility:
      - MULTIGPU_NVTX (legacy)
      - NVTX_ENABLED
    """
    # Check primary variable
    if os.environ.get("HPCP_NVTX", "0") == "1":
        return True

    # Backward compatibility with old name
    if os.environ.get("MULTIGPU_NVTX", "0") == "1":
        return True

    # Alternative name
    if os.environ.get("NVTX_ENABLED", "0") == "1":
        return True

    return False


def _import_nvtx() -> Any | None:
    """
    Lazily import and cache the nvtx module.

    Returns
    -------
    module | None
        The nvtx module if available and enabled, None otherwise.

    Notes
    -----
    This function maintains global state to avoid repeated import attempts.
    Once NVTX is determined to be unavailable or disabled, subsequent calls
    return None immediately without trying to import.
    """
    global _NVTX_IMPORTED, _NVTX_MOD, _NVTX_DISABLED

    if _NVTX_DISABLED:
        return None

    if _NVTX_IMPORTED:
        return _NVTX_MOD

    if not _want_nvtx():
        _NVTX_DISABLED = True  # honor explicit disable for future calls
        return None

    try:
        import nvtx as _n  # type: ignore
        _NVTX_IMPORTED = True
        _NVTX_MOD = _n
        return _n
    except Exception:
        # Import failed - disable permanently to avoid repeated attempts
        _NVTX_DISABLED = True
        return None


def nvtx_available() -> bool:
    """
    Check whether NVTX is both requested and successfully imported.

    Returns
    -------
    bool
        True if NVTX is enabled and available, False otherwise.

    Examples
    --------
    >>> if nvtx_available():  # doctest: +SKIP
    ...     print("NVTX profiling is active")
    ... else:
    ...     print("NVTX profiling is disabled")
    """
    return _import_nvtx() is not None


@contextmanager
def nvtx_range(msg: str, color: int | None = None) -> Iterator[None]:
    """
    Enter an NVTX range labeled `msg` (no-op if disabled/unavailable).

    This context manager creates a profiling annotation that appears in
    NVIDIA Nsight Systems and Nsight Compute timelines. If NVTX is not
    available or not enabled, this becomes a no-op with minimal overhead.

    Parameters
    ----------
    msg : str
        Range label. Keep under 64 characters for compact timelines.
        Use descriptive names like "forward_pass" or "data_loading".
    color : int | None, optional
        Color in 0xRRGGBB format (e.g., 0xFF0000 for red).
        Use the `Colors` class for predefined constants.
        If None, NVTX assigns a color automatically.

    Yields
    ------
    None

    Examples
    --------
    >>> with nvtx_range("data_loading", color=Colors.BLUE):  # doctest: +SKIP
    ...     data = load_data()

    >>> with nvtx_range("gpu_compute", color=0xFF8800):  # doctest: +SKIP
    ...     result = model(input_gpu)

    Notes
    -----
    - NVTX ranges can be nested to show hierarchical relationships.
    - The overhead when disabled is just a module-level boolean check.
    - When enabled, NVTX adds ~1-2 microseconds per range.
    """
    mod = _import_nvtx()

    if mod is None:
        # Fast no-op path when NVTX is disabled
        yield
        return

    # NVTX is available - create range
    cm = None
    try:
        kwargs = {"message": msg}
        if color is not None:
            kwargs["color"] = int(color)
        cm = mod.annotate(**kwargs)  # type: ignore
    except Exception:
        # If annotation creation fails, degrade to no-op
        cm = None

    if cm is None:
        yield
    else:
        with cm:
            yield


def annotate_if_enabled(
        label: str,
        color: int | None = None,
) -> Callable[[F], F]:
    """
    Decorator: wrap a function body in an NVTX range if enabled.

    Parameters
    ----------
    label : str
        NVTX range label for the function.
    color : int | None, optional
        Color in 0xRRGGBB format. If None, use automatic coloring.

    Returns
    -------
    Callable
        Decorated function with NVTX instrumentation.

    Examples
    --------
    >>> @annotate_if_enabled("heavy_computation", color=Colors.RED)
    ... def compute(x):
    ...     return x ** 2
    >>> result = compute(10)  # doctest: +SKIP
    >>> # In Nsight, this appears as "heavy_computation" range

    Notes
    -----
    - Function name, docstring, and annotations are preserved via functools.wraps.
    - When NVTX is disabled, the decorator adds zero overhead.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with nvtx_range(label, color=color):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def push_range(msg: str, color: int | None = None) -> None:
    """
    Push an NVTX range onto the stack (no-op if disabled).

    Must be paired with a corresponding `pop_range()` call.
    Consider using `nvtx_range()` context manager instead for automatic cleanup.

    Parameters
    ----------
    msg : str
        Range label.
    color : int | None, optional
        Color in 0xRRGGBB format.

    See Also
    --------
    pop_range : Pop range from stack.
    nvtx_range : Recommended context manager interface.

    Examples
    --------
    >>> push_range("manual_range", color=Colors.GREEN)  # doctest: +SKIP
    >>> # ... work ...
    >>> pop_range()  # doctest: +SKIP
    """
    mod = _import_nvtx()
    if mod is None:
        return

    try:
        kwargs = {"message": msg}
        if color is not None:
            kwargs["color"] = int(color)
        mod.push_range(**kwargs)  # type: ignore
    except Exception:
        pass  # Silently fail - profiling shouldn't break code


def pop_range() -> None:
    """
    Pop the top NVTX range from the stack (no-op if disabled).

    Must be paired with a preceding `push_range()` call.

    See Also
    --------
    push_range : Push range onto stack.
    nvtx_range : Recommended context manager interface.

    Examples
    --------
    >>> push_range("manual_range")  # doctest: +SKIP
    >>> # ... work ...
    >>> pop_range()  # doctest: +SKIP
    """
    mod = _import_nvtx()
    if mod is None:
        return

    try:
        mod.pop_range()  # type: ignore
    except Exception:
        pass  # Silently fail