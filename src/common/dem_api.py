# src/common/dem_api.py
from __future__ import annotations

"""
Thin, well-documented wrapper around the vendor `dn2dem_pos` solver.

This wrapper:
  - Accepts a single frame shaped **(6, H, W)** (channels-first).
  - Transposes it to **(H, W, 6)** (channels-last) as expected by vendor.
  - Sanitizes values (NaNs/±inf → 0, clamp to ≥ 0).
  - Builds a simple Poisson-like uncertainty model: `edn = sqrt(a*f + b)`.
  - Calls the vendor solver and returns its outputs.

Use this helper when you want a minimal, explicit bridge from our data layout to
the vendor function without pulling in the broader pipeline.

Important
---------
**Vendor dn2dem_pos expects channels-last format (H, W, 6)!**
This function handles the transpose automatically from (6, H, W) to (H, W, 6).
"""

import numpy as np

from src.baseline.vendor.dn2dem_pos import dn2dem_pos as _dn2dem_pos


def dn2dem(
        frame_6hw: np.ndarray,
        T_RESP: np.ndarray,
        T_RESP_LOGT: np.ndarray,
        TEMPS: np.ndarray,
        nmu: int = 42,
        err_a: float = 1.0,
        err_b: float = 1e-6,
        **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the vendor DEM solver on a single frame.

    Parameters
    ----------
    frame_6hw : np.ndarray, shape (6, H, W)
        Input image with **channels-first** layout. Values are sanitized:
        NaNs/±inf → 0, then clamped to be non-negative.
    T_RESP : np.ndarray, shape (n_tresp, 6)
        Temperature response matrix. Each column is the response for one filter
        evaluated at n_tresp temperature samples.
    T_RESP_LOGT : np.ndarray, shape (n_tresp,)
        Log10(T) sample points corresponding to rows of T_RESP.
        Must be monotonically increasing.
    TEMPS : np.ndarray, shape (nt + 1,)
        DEM bin edges in linear temperature (Kelvin), length nt + 1.
        Must be monotonically increasing and positive.
    nmu : int, default 42
        Regularization parameter (number of samples for λ search).
    err_a : float, default 1.0
        Error model parameter: edn = sqrt(a*counts + b)
    err_b : float, default 1e-6
        Error model parameter: minimum error floor
    **kwargs
        Additional keyword arguments passed to vendor dn2dem_pos
        (e.g., reg_tweak, max_iter, rgt_fact)

    Returns
    -------
    demmap : np.ndarray, shape (H, W, nt)
        DEM estimate [cm⁻⁵ K⁻¹] or [cm⁻⁵] depending on vendor settings.
    edemmap : np.ndarray, shape (H, W, nt)
        Estimated uncertainty for DEM (same units as demmap).
    logT_bins : np.ndarray, shape (nt,)
        Log10(T) bin centers used by the solver.
    chisq : np.ndarray, shape (H, W)
        Per-pixel reduced chi-squared.
    dn_reg : np.ndarray, shape (H, W, 6)
        Reconstructed DN counts from the DEM solution (for validation).

    Raises
    ------
    ValueError
        If frame_6hw has wrong shape, or temperature arrays are invalid.

    Notes
    -----
    - **Data layout**: Input is (6, H, W) channels-first, automatically
      transposed to (H, W, 6) for the vendor which expects channels-last.
    - **Error model**: `edn = sqrt(a * counts + b)` provides a simple
      Poisson-like model. Default (a=1.0, b=1e-6) works well for typical
      AIA DN values (~1-10000 DN/pixel/s).
    - **Temperature arrays**: T_RESP provides high-resolution response curves
      which the vendor interpolates onto the DEM temperature grid defined by TEMPS.

    Examples
    --------
    >>> import numpy as np
    >>> frame = np.random.rand(6, 100, 100).astype(np.float32) * 1000
    >>> T_RESP = np.random.rand(200, 6).astype(np.float32)
    >>> T_RESP_LOGT = np.linspace(5.5, 7.5, 200).astype(np.float32)
    >>> TEMPS = np.logspace(5.5, 7.5, 25).astype(np.float32)
    >>> dem, edem, logt, chisq, dn_reg = dn2dem(
    ...     frame, T_RESP, T_RESP_LOGT, TEMPS, nmu=42
    ... )
    >>> dem.shape
    (100, 100, 24)
    """
    # ===== Validate inputs =====

    if frame_6hw.ndim != 3:
        raise ValueError(
            f"frame_6hw must be 3D (6, H, W), got {frame_6hw.ndim}D: {frame_6hw.shape}"
        )

    if frame_6hw.shape[0] != 6:
        raise ValueError(
            f"frame_6hw must have 6 channels, got {frame_6hw.shape[0]}: "
            f"shape is {frame_6hw.shape}"
        )

    C, H, W = frame_6hw.shape

    if H == 0 or W == 0:
        raise ValueError(f"Empty spatial dimensions: H={H}, W={W}")

    # Validate temperature response matrix
    if T_RESP.ndim != 2:
        raise ValueError(
            f"T_RESP must be 2D (n_tresp, 6), got {T_RESP.ndim}D: {T_RESP.shape}"
        )

    if T_RESP.shape[1] != 6:
        raise ValueError(
            f"T_RESP must have 6 filters, got {T_RESP.shape[1]}"
        )

    n_tresp = T_RESP.shape[0]

    # Validate T_RESP_LOGT
    if T_RESP_LOGT.ndim != 1:
        raise ValueError(
            f"T_RESP_LOGT must be 1D, got {T_RESP_LOGT.ndim}D: {T_RESP_LOGT.shape}"
        )

    if len(T_RESP_LOGT) != n_tresp:
        raise ValueError(
            f"T_RESP_LOGT length ({len(T_RESP_LOGT)}) must match "
            f"T_RESP rows ({n_tresp})"
        )

    if not np.all(np.diff(T_RESP_LOGT) > 0):
        raise ValueError("T_RESP_LOGT must be strictly monotonically increasing")

    # Validate TEMPS
    if TEMPS.ndim != 1:
        raise ValueError(
            f"TEMPS must be 1D, got {TEMPS.ndim}D: {TEMPS.shape}"
        )

    if len(TEMPS) < 2:
        raise ValueError(
            f"TEMPS must have at least 2 edges (1 bin), got {len(TEMPS)}"
        )

    if not np.all(TEMPS > 0):
        raise ValueError("TEMPS must be all positive (linear temperature in Kelvin)")

    if not np.all(np.diff(TEMPS) > 0):
        raise ValueError("TEMPS must be strictly monotonically increasing")

    # ===== Transpose and sanitize input =====

    # (6, H, W) -> (H, W, 6) - vendor expects channels-last
    f = np.moveaxis(frame_6hw, 0, -1).astype(np.float32, copy=False)

    # Sanitize: NaN/Inf -> 0, then clamp to non-negative
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.clip(f, 0.0, None, out=f)

    # Ensure C-contiguous for vendor
    f = np.ascontiguousarray(f)

    # ===== Build error model =====

    # Poisson-like: edn = sqrt(a * counts + b)
    # This gives ~sqrt(N) scaling for Poisson noise with a minimum floor
    edn = np.sqrt(err_a * f + err_b, dtype=np.float32)
    edn = np.ascontiguousarray(edn)

    # ===== Call vendor solver =====

    try:
        dem, edem, logt_bins, chisq, dn_reg = _dn2dem_pos(
            f, edn, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu, **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Vendor dn2dem_pos failed for frame shape {f.shape}: {e}"
        ) from e

    # ===== Validate outputs =====

    nt = len(TEMPS) - 1

    if dem.shape != (H, W, nt):
        raise RuntimeError(
            f"Vendor returned unexpected DEM shape: got {dem.shape}, "
            f"expected ({H}, {W}, {nt})"
        )

    if edem.shape != (H, W, nt):
        raise RuntimeError(
            f"Vendor returned unexpected edem shape: got {edem.shape}, "
            f"expected ({H}, {W}, {nt})"
        )

    if chisq.shape != (H, W):
        raise RuntimeError(
            f"Vendor returned unexpected chisq shape: got {chisq.shape}, "
            f"expected ({H}, {W})"
        )

    if dn_reg.shape != (H, W, 6):
        raise RuntimeError(
            f"Vendor returned unexpected dn_reg shape: got {dn_reg.shape}, "
            f"expected ({H}, {W}, 6)"
        )

    if logt_bins.shape != (nt,):
        raise RuntimeError(
            f"Vendor returned unexpected logt_bins shape: got {logt_bins.shape}, "
            f"expected ({nt},)"
        )

    # ===== Return results =====

    return dem, edem, logt_bins, chisq, dn_reg


def dn2dem_simple(
        frame_6hw: np.ndarray,
        T_RESP: np.ndarray,
        T_RESP_LOGT: np.ndarray,
        TEMPS: np.ndarray,
        nmu: int = 42,
) -> np.ndarray:
    """
    Convenience wrapper that returns only the DEM cube.

    Parameters
    ----------
    frame_6hw : np.ndarray, shape (6, H, W)
        Input frame (channels-first).
    T_RESP : np.ndarray, shape (n_tresp, 6)
        Temperature response matrix.
    T_RESP_LOGT : np.ndarray, shape (n_tresp,)
        Log10(T) samples for T_RESP.
    TEMPS : np.ndarray, shape (nt + 1,)
        DEM bin edges in Kelvin.
    nmu : int, default 42
        Regularization parameter.

    Returns
    -------
    dem : np.ndarray, shape (H, W, nt)
        DEM estimate only.

    See Also
    --------
    dn2dem : Full interface with all outputs.

    Examples
    --------
    >>> dem = dn2dem_simple(frame, T_RESP, T_RESP_LOGT, TEMPS)
    >>> dem.shape
    (100, 100, 24)
    """
    dem, _edem, _logt, _chisq, _dn_reg = dn2dem(
        frame_6hw, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu
    )
    return dem