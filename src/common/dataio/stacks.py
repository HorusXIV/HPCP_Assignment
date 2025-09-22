# src/common/dataio/stacks.py
from __future__ import annotations
"""
NumPy-based loaders for NPZ "stack" files.

This module provides helpers to:
  • Select frames by flexible index specifications (int/slice/sequence)
  • Load one or more NPZ files containing a 'bands' array of shape (6, H, W)
  • Return data either channel-first (N, 6, H, W) or channel-last (N, H, W, 6)
  • Extract a single frame formatted for the solver as (H, W, 6)

Conventions
-----------
- Each NPZ must contain a key 'bands' with shape (6, H, W).
- Negative indices are supported in selectors; additionally, `idx == -1`
  means "all frames" for convenience (project-wide convention).
"""

from pathlib import Path
from typing import Optional, Sequence, Union, Tuple

import numpy as np

IndexLike = Union[int, slice, Sequence[int], None]


def _ensure_6hw(a: np.ndarray, *, src: Union[str, Path]) -> np.ndarray:
    """
    Validate that an array has shape (6, H, W).

    Parameters
    ----------
    a : np.ndarray
        Input array to validate.
    src : str | Path
        Source identifier used in error messages.

    Returns
    -------
    np.ndarray
        The input array `a` if validation passes.

    Raises
    ------
    ValueError
        If `a` does not have rank 3 or `a.shape[0] != 6`.
    """
    if a.ndim != 3 or a.shape[0] != 6:
        raise ValueError(f"{src} expected (6,H,W), got {a.shape}")
    return a


def _cast(a: np.ndarray, *, dtype: Optional[np.dtype], contiguous: bool) -> np.ndarray:
    """
    Optionally cast dtype and enforce C-contiguity.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    dtype : np.dtype | None
        Desired dtype; if None, the dtype is preserved.
    contiguous : bool
        If True, ensure the result is C-contiguous.

    Returns
    -------
    np.ndarray
        Possibly cast and contiguously laid out array.
    """
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    if contiguous:
        a = np.ascontiguousarray(a)
    return a


def _select_indices(n: int, idx: IndexLike) -> list[int]:
    """
    Resolve a flexible `idx` selector into concrete indices.

    Semantics
    ---------
    - None          → all indices [0, 1, ..., n-1]
    - -1            → all indices (project-wide convenience)
    - int (≠ -1)    → [idx] (supports negative indexing in Python style)
    - slice         → list(range(n))[idx]
    - Sequence[int] → validated elementwise (negative indices supported)

    Parameters
    ----------
    n : int
        Total number of items.
    idx : IndexLike
        Selector as described above.

    Returns
    -------
    list[int]
        Selected indices.

    Raises
    ------
    IndexError
        If any resolved index falls outside [0, n).
    """
    if idx is None or idx == -1:
        return list(range(n))
    if isinstance(idx, int):
        j = idx + n if idx < 0 else idx
        if not (0 <= j < n):
            raise IndexError(f"Index {idx} out of range for n={n}")
        return [j]
    if isinstance(idx, slice):
        return list(range(n))[idx]
    out = []
    for i in idx:
        j = i + n if i < 0 else i
        if not (0 <= j < n):
            raise IndexError(f"Index {i} out of range for n={n}")
        out.append(j)
    return out


def load_np_stack(
    file_list: Sequence[Union[str, Path]],
    idx: IndexLike = -1,
    *,
    channels_last: bool = False,  # True -> return (N, H, W, 6)
    dtype: Optional[np.dtype] = None,
    contiguous: bool = True,
    return_paths: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list[Path]]]:
    """
    Load one or more NPZ stacks and return a stacked NumPy array.

    Each NPZ file must contain 'bands' with shape (6, H, W). Shapes across
    selected files must match exactly.

    Parameters
    ----------
    file_list : Sequence[str | Path]
        List of NPZ files to consider (ordering preserved after sorting).
    idx : IndexLike, default -1
        Frame selector:
          None/-1 → all files,
          int     → single file (negative ok),
          slice   → range selection,
          Sequence[int] → explicit indices.
    channels_last : bool, default False
        If True, return (N, H, W, 6). Otherwise (N, 6, H, W).
    dtype : np.dtype | None, default None
        Optional dtype cast of the stacked result.
    contiguous : bool, default True
        Ensure the returned array is C-contiguous.
    return_paths : bool, default False
        If True, also return the list of `Path`s in the order stacked.

    Returns
    -------
    np.ndarray
        Stacked array shaped (N, 6, H, W) or (N, H, W, 6) if `channels_last=True`.
    (np.ndarray, list[pathlib.Path])
        If `return_paths=True`, the tuple of (array, paths).

    Raises
    ------
    FileNotFoundError
        If `file_list` is empty.
    KeyError
        If an NPZ is missing the 'bands' array.
    ValueError
        If a 'bands' array has an unexpected shape or shapes mismatch.
    """
    files = sorted(Path(f) for f in file_list)
    if not files:
        raise FileNotFoundError("No .npz files provided.")

    select = _select_indices(len(files), idx)
    arrays = []
    ref_shape = None

    for j in select:
        p = files[j]
        with np.load(p, allow_pickle=False) as z:
            if "bands" not in z:
                raise KeyError(f"{p.name} missing 'bands' array")
            a = _ensure_6hw(z["bands"], src=p)  # (6,H,W)
        if ref_shape is None:
            ref_shape = a.shape
        elif a.shape != ref_shape:
            raise ValueError(f"Shape mismatch: {p.name} {a.shape} != {ref_shape}")
        arrays.append(a)

    arr = np.stack(arrays, axis=0)  # (N,6,H,W)
    if channels_last:
        arr = np.moveaxis(arr, 1, -1)  # (N,H,W,6)
    arr = _cast(arr, dtype=dtype, contiguous=contiguous)

    return (arr, [files[j] for j in select]) if return_paths else arr


def frame_for_solver(stack_any: np.ndarray, i: int = 0) -> np.ndarray:
    """
    Extract a single frame as (H, W, 6) suitable for the solver.

    Accepts either channel-first (N, 6, H, W) or channel-last (N, H, W, 6)
    stacks and returns the i-th frame with channels last.

    Parameters
    ----------
    stack_any : np.ndarray
        Input stack array of rank 4.
    i : int, default 0
        Frame index to extract.

    Returns
    -------
    np.ndarray
        Frame shaped (H, W, 6), C-contiguous.

    Raises
    ------
    ValueError
        If the selected frame is not rank-3 or the channel dimension is not 6.
    """
    f = stack_any[i]
    if f.ndim != 3:
        raise ValueError(f"Expected 3D frame, got {f.shape}")
    if f.shape[0] == 6:  # (6,H,W) -> (H,W,6)
        f = np.moveaxis(f, 0, -1)
    elif f.shape[-1] != 6:
        raise ValueError(f"Cannot infer channels axis for shape {f.shape}")
    return np.ascontiguousarray(f)
