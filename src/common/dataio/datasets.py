# src/common/dataio/datasets.py
from __future__ import annotations
"""
Dask-backed dataset builders for NPZ stacks.

This module provides utilities to:
  • Select frame indices from a list of files (supporting int/slice/sequence)
  • Compute crop bounds
  • Load cropped blocks from NPZ files containing 'bands' arrays
  • Construct a lazy Dask array stack with tile-aligned chunking

Conventions
-----------
Input NPZ files must contain an array named 'bands' with shape (6, H, W).
Output arrays are shaped (F, Hc, Wc, 6), where Hc/Wc reflect the crop.
Chunks are (1, Th, Tw, 6) so each Dask chunk corresponds to a single tile.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import dask.array as da
from dask import delayed

IndexLike = Union[int, slice, Sequence[int], None]


def _select_indices(n: int, idx: IndexLike) -> list[int]:
    """
    Resolve an index selector into a concrete list of indices.

    Parameters
    ----------
    n : int
        Total number of available items (e.g., number of files).
    idx : int | slice | Sequence[int] | None
        - None            → all indices [0, 1, ..., n-1]
        - int             → [idx] (supports negative indexing like -1 for last)
        - slice           → list(range(n))[idx]
        - Sequence[int]   → list(idx) as provided

    Returns
    -------
    list[int]
        The selected indices.

    Raises
    ------
    ValueError
        If `idx` is an unsupported type.
    """
    if idx is None:
        return list(range(n))
    if isinstance(idx, int):
        return [idx]
    if isinstance(idx, slice):
        return list(range(n))[idx]
    return list(idx)


def _crop_bounds(
    H: int, W: int, crop_hw: Tuple[int, int] | None
) -> tuple[int, int, int, int]:
    """
    Compute crop bounds (y0, y1, x0, x1) within a source (H, W).

    Parameters
    ----------
    H, W : int
        Source height and width.
    crop_hw : (int, int) | None
        Desired crop (h, w). If None, returns the full image.

    Returns
    -------
    (int, int, int, int)
        Tuple (y0, y1, x0, x1) suitable for array slicing.

    Raises
    ------
    ValueError
        If the requested crop exceeds the source shape.
    """
    if crop_hw is None:
        return 0, H, 0, W
    h, w = crop_hw
    if h > H or w > W:
        raise ValueError(f"Crop {crop_hw} exceeds source shape {(H, W)}")
    return 0, h, 0, w


def _load_block(
    path: Union[str, Path], y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """
    Load one NPZ file, crop to (6, h, w), and return (h, w, 6) float32 contiguous.

    The file must contain a key 'bands' with shape (6, H, W).

    Parameters
    ----------
    path : str | Path
        Path to the NPZ file.
    y0, y1, x0, x1 : int
        Crop bounds.

    Returns
    -------
    np.ndarray
        Cropped array of shape (h, w, 6), dtype float32, C-contiguous.

    Raises
    ------
    KeyError
        If 'bands' is missing in the NPZ.
    ValueError
        If 'bands' has unexpected rank or channels != 6.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if "bands" not in z:
            raise KeyError(f"{path.name} missing 'bands' array")
        a = z["bands"]  # (6, H, W)
        if a.ndim != 3 or a.shape[0] != 6:
            raise ValueError(f"{path.name}: expected (6,H,W), got {a.shape}")
        a = a[:, y0:y1, x0:x1]  # (6, h, w)
        a = np.moveaxis(a, 0, -1)  # (h, w, 6)
        a = a.astype(np.float32, copy=False)
        return np.ascontiguousarray(a)


def build_lazy_npz_stack(
    file_list: Sequence[Union[str, Path]],
    idx: IndexLike,
    *,
    crop_hw: Tuple[int, int] | None,
    tile_hw: Tuple[int, int],
) -> da.Array:
    """
    Construct a lazy Dask array (F, Hc, Wc, 6) from NPZ files with 'bands'.

    Each selected file contributes one frame. The result is chunked so that each
    chunk corresponds to a single tile: (1, Th, Tw, 6).

    Parameters
    ----------
    file_list : Sequence[str | Path]
        Ordered list of NPZ files containing 'bands' arrays of shape (6, H, W).
    idx : int | slice | Sequence[int] | None
        Frame selection:
          - None            → all files
          - int (incl. -1)  → single file at that index (Python indexing applies)
          - slice           → range-based selection
          - Sequence[int]   → explicit indices
    crop_hw : (int, int) | None
        Crop (Hc, Wc). If None, uses the full (H, W) from the first file.
    tile_hw : (int, int)
        Tile (Th, Tw) used to define Dask chunks: (1, Th, Tw, 6).

    Returns
    -------
    dask.array.Array
        Lazy stack with shape (F, Hc, Wc, 6) and chunks (1, Th, Tw, 6).

    Raises
    ------
    FileNotFoundError
        If `file_list` is empty.
    KeyError
        If the first file lacks a 'bands' array.
    ValueError
        If the first file's 'bands' array has unexpected shape or if selection
        produces no frames.
    """
    files = [Path(f) for f in file_list]
    if not files:
        raise FileNotFoundError("No .npz files provided.")

    # Probe first file to derive crop bounds
    with np.load(files[0], allow_pickle=False) as z0:
        if "bands" not in z0:
            raise KeyError(f"{files[0].name} missing 'bands' array")
        _, H, W = z0["bands"].shape

    y0, y1, x0, x1 = _crop_bounds(H, W, crop_hw)
    Hc, Wc = y1 - y0, x1 - x0
    Th, Tw = tile_hw

    select = _select_indices(len(files), idx)
    frames: list[da.Array] = []
    for j in select:
        d = delayed(_load_block)(files[j], y0, y1, x0, x1)  # → (Hc, Wc, 6)
        fr = da.from_delayed(d, shape=(Hc, Wc, 6), dtype=np.float32)[None, ...]  # (1,Hc,Wc,6)
        frames.append(fr)

    if not frames:
        raise ValueError("Selection produced no frames.")

    darr = da.concatenate(frames, axis=0)
    return darr.rechunk((1, Th, Tw, 6))
