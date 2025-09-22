# src/common/thermo_maps.py
from __future__ import annotations
"""
Utilities to derive temperature summary maps from a DEM cube.

Currently provides:
- `dem_to_temp_maps(demmap, logT_bins)` → (mean_log10T, peak_log10T)

The inputs follow the convention used throughout the project:
  - `demmap` has shape (H, W, NT), non-negative emission measures per bin.
  - `logT_bins` is a 1D array of base-10 log-temperature *bin centers* of length NT.
"""

import numpy as np
from typing import Tuple


def dem_to_temp_maps(demmap: np.ndarray, logT_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two temperature maps from a DEM cube:
      1) DEM-weighted **mean log10(T)** per pixel
      2) **Peak log10(T)** (bin center of the maximal DEM per pixel)

    Parameters
    ----------
    demmap : np.ndarray
        DEM cube with shape (H, W, NT). Values are expected to be non-negative.
    logT_bins : np.ndarray
        1D array of length NT with base-10 log-temperature bin centers.

    Returns
    -------
    mean_log10T : np.ndarray, shape (H, W), dtype float32
        DEM-weighted mean of log10(T) per pixel. For pixels with zero total EM,
        the value is NaN.
    peak_log10T : np.ndarray, shape (H, W), dtype float32
        Bin-center log10(T) at which the DEM attains its maximum for that pixel.
        For pixels with zero total EM, the value is NaN.

    Notes
    -----
    - The DEM-weighted mean is computed in **linear T** (K), then converted to
      log10(T):  mean_log10T = log10( sum(DEM * T_center) / sum(DEM) ).
    - Pixels with zero total emission measure are marked as NaN in both outputs.
    - The function is vectorized and avoids intermediate allocations where possible.
    """
    if demmap.ndim != 3:
        raise ValueError(f"`demmap` must be (H, W, NT); got shape {demmap.shape}")
    H, W, NT = demmap.shape

    logT_bins = np.asarray(logT_bins).reshape(-1)
    if logT_bins.size != NT:
        raise ValueError(
            f"len(logT_bins) must equal NT={NT}; got {logT_bins.size}"
        )

    # Ensure non-negative, contiguous float32 without copying unnecessarily.
    dem = np.clip(demmap, 0.0, None).astype(np.float32, copy=False)  # (H, W, NT)

    # Total emission measure and validity mask.
    EM = dem.sum(axis=2)  # (H, W)
    valid = EM > 0

    # Peak log10(T): argmax over bins -> index into centers.
    imax = dem.argmax(axis=2).astype(np.intp)
    peak_logT = np.full((H, W), np.nan, dtype=np.float32)
    # Safe indexing only where valid emission exists.
    np.put_along_axis(
        peak_logT, imax[..., None], logT_bins.astype(np.float32)[imax][..., None], axis=1
    )  # temporary fill; will overwrite below for valid mask

    # Simpler and faster: use take with where.
    peak_logT = np.where(valid, np.take(logT_bins.astype(np.float32), imax), np.nan)

    # DEM-weighted mean temperature in linear space, then log10.
    T_centers = (10.0 ** logT_bins).astype(np.float32)  # (NT,)
    # num = sum_k DEM_ijk * T_k
    num = np.einsum("ijk,k->ij", dem, T_centers, optimize=True)  # (H, W)
    mean_logT = np.full_like(EM, np.nan, dtype=np.float32)
    # mean_T = num / EM where valid
    np.divide(num, EM, out=mean_logT, where=valid)
    # Convert to log10(T) only for valid pixels.
    mean_logT = np.log10(mean_logT, out=mean_logT, where=valid)

    return mean_logT, peak_logT
