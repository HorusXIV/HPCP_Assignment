# src/common/dn2dem_wrapper.py
from __future__ import annotations
"""
Thin, well-documented wrapper around the vendor `dn2dem_pos` solver.

This wrapper:
  - Accepts a single frame shaped **(6, H, W)** (channels-first).
  - Transposes it to **(H, W, 6)** and sanitizes values (NaNs/±inf → 0, clamp to ≥ 0).
  - Builds a simple Poisson-like uncertainty model: `edn = sqrt(f) + 1e-6`.
  - Calls the vendor solver and returns its outputs unchanged.

Use this helper when you want a minimal, explicit bridge from our data layout to
the vendor function without pulling in the broader pipeline.
"""

from typing import Tuple

import numpy as np
from src.baseline.vendor.dn2dem_pos import dn2dem_pos as _dn2dem_pos


def dn2dem(
    frame_6hw: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    nmu: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the vendor DEM solver on a single frame.

    Parameters
    ----------
    frame_6hw : np.ndarray
        Input image with **shape (6, H, W)** (channels-first). Values are sanitized:
        NaNs/±inf → 0, then clamped to be non-negative.
    T_RESP : np.ndarray
        Temperature response matrix, shaped (N_tresp, N_filters).
    T_RESP_LOGT : np.ndarray
        1D vector of log(T) samples for `T_RESP`, length N_tresp.
    TEMPS : np.ndarray
        1D vector of DEM bin edges in T-space, length NT + 1 (strictly increasing).
    nmu : int, default 42
        Regularization knob passed through to the vendor implementation.

    Returns
    -------
    demmap : np.ndarray
        DEM estimate, shape (H, W, NT).
    edemmap : np.ndarray
        Estimated uncertainty for DEM, shape (H, W, NT).
    logT_bins : np.ndarray
        1D vector of logT bin centers used by the solver, length NT.
    chisq : np.ndarray
        Per-pixel chi-square, shape (H, W).
    dn_reg : np.ndarray
        Regularized data term returned by the vendor (shape (H, W, N_filters)).

    Notes
    -----
    - Uncertainty model: `edn = sqrt(f) + 1e-6`, with `f` the sanitized input
      after converting to (H, W, 6). This mirrors the lightweight model used
      elsewhere in the codebase for baseline runs.
    - This function does not perform shape validation on the temperature
      response inputs; it forwards them to the vendor solver as-is.

    Raises
    ------
    ValueError
        If `frame_6hw` does not have 3 dimensions or its leading dimension is not 6.
    """
    if frame_6hw.ndim != 3 or frame_6hw.shape[0] != 6:
        raise ValueError(f"Expected frame shaped (6, H, W); got {frame_6hw.shape}")

    # (6, H, W) -> (H, W, 6), sanitize, clamp to non-negative, float32 & contiguous
    f = np.moveaxis(frame_6hw, 0, -1).astype(np.float32, copy=False)  # (H, W, 6)
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    f = np.ascontiguousarray(f)

    # Simple Poisson-ish uncertainties
    edn = np.sqrt(f, dtype=np.float32) + 1e-6

    # Delegate to vendor
    return _dn2dem_pos(f, edn, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)
