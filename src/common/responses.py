# src/common/responses.py
from __future__ import annotations

"""
Temperature response functions for DEM inversions.

This module provides:
1. Synthetic responses for testing (deterministic, well-conditioned)
2. Utilities for building response matrices on DEM temperature grids
3. Proper Jacobian factors for DEM vs EMD calculations

Typical usage
-------------
>>> T_RESP, logT, TEMPS = prepare_synthetic_responses(nt=24, nf=6)
>>> T_RESP.shape   # (n_tresp, nf)
(200, 6)
>>> logT.shape     # (n_tresp,)
(200,)
>>> TEMPS.shape    # (nt + 1,) - bin edges
(25,)

>>> # Or get responses already binned for DEM calculation:
>>> rmatrix, logt, dlogt = build_binned_responses(nt=24, nf=6)
>>> rmatrix.shape  # (nt, nf) - ready for dn2dem_pos
(24, 6)
"""

import numpy as np


def prepare_synthetic_responses(
        logT_min: float = 5.5,
        logT_max: float = 7.5,
        n_tresp: int = 200,
        nt: int = 24,
        nf: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build synthetic instrument temperature responses.

    Parameters
    ----------
    logT_min, logT_max : float
        Range of log10(T) covered by the responses and DEM bin edges.
        Typical solar values: 5.5 (10^5.5 K ≈ 316,000 K) to 7.5 (10^7.5 K ≈ 31.6 MK)
    n_tresp : int, default 200
        Number of logT samples for the response curves (high resolution).
    nt : int, default 24
        Number of DEM bins (TEMPS will have length nt + 1 as edges).
    nf : int, default 6
        Number of response channels/filters (6 for AIA).

    Returns
    -------
    T_RESP : np.ndarray, shape (n_tresp, nf), dtype float32
        Synthetic response matrix; each column is a broad Gaussian in log10(T),
        shifted so channels are separated and reasonably conditioned.
        Units: arbitrary (DN / cm^5 or similar)
    logT : np.ndarray, shape (n_tresp,), dtype float32
        Sample points (centers) for T_RESP along log10(T).
    TEMPS : np.ndarray, shape (nt + 1,), dtype float32
        DEM bin edges in **linear** temperature (Kelvin), monotonically increasing.

    Raises
    ------
    ValueError
        If parameters are invalid (e.g., logT_max <= logT_min, nt < 1, etc.)

    Notes
    -----
    - A small floor (1e-30) is added to T_RESP to avoid exact zeros.
    - The Gaussian width (0.15 in log10(T)) is chosen to:
      1. Give reasonable overlap between adjacent filters
      2. Ensure stable matrix inversions in DEM calculations
      3. Mimic real AIA response functions qualitatively
    - These are NOT calibrated AIA responses; use for testing only.

    Examples
    --------
    >>> T_RESP, logT, TEMPS = prepare_synthetic_responses(nt=24, nf=6)
    >>> T_RESP.min() > 0  # All positive due to floor
    True
    >>> len(TEMPS) == 25  # nt bins → nt+1 edges
    True
    """
    # Validate inputs
    if logT_max <= logT_min:
        raise ValueError(
            f"logT_max ({logT_max}) must be > logT_min ({logT_min})"
        )
    if n_tresp < 10:
        raise ValueError(f"n_tresp must be >= 10, got {n_tresp}")
    if nt < 1:
        raise ValueError(f"nt must be >= 1, got {nt}")
    if nf < 1:
        raise ValueError(f"nf must be >= 1, got {nf}")

    # Sample points along log10(T) for the response curves
    logT = np.linspace(logT_min, logT_max, int(n_tresp), dtype=np.float32)

    # Place nf Gaussian centers evenly inside the range (with margin)
    # Margin prevents responses from peaking right at boundaries
    margin = 0.2
    if nf == 1:
        centers = np.array([(logT_min + logT_max) / 2], dtype=np.float32)
    else:
        centers = np.linspace(
            logT_min + margin,
            logT_max - margin,
            int(nf),
            dtype=np.float32
        )

    # Gaussian width (standard deviation in log10(T))
    # 0.15 gives ~0.3 FWHM, reasonable overlap
    width = np.float32(0.15)

    # Build responses: (n_tresp, nf)
    T_RESP = np.exp(-0.5 * ((logT[:, None] - centers[None, :]) / width) ** 2)

    # Add small floor to prevent exact zeros (helps with numerical stability)
    T_RESP = T_RESP.astype(np.float32) + np.float32(1e-30)

    # DEM bin edges in linear T (Kelvin): length = nt + 1
    TEMPS = np.logspace(logT_min, logT_max, int(nt) + 1, dtype=np.float32)

    return T_RESP, logT, TEMPS


def build_binned_responses(
        nt: int,
        nf: int,
        *,
        n_tresp: int = 200,
        logT_min: float = 5.5,
        logT_max: float = 7.5,
        include_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a response matrix on DEM bins by bin-averaging synthetic responses.

    This function produces a response matrix **ready for DEM inversions**,
    including the temperature Jacobian factor if requested.

    Parameters
    ----------
    nt : int
        Number of DEM temperature bins.
    nf : int
        Number of filters/channels.
    n_tresp : int, default 200
        Resolution of high-res response curves before binning.
    logT_min, logT_max : float
        Temperature range in log10(K).
    include_jacobian : bool, default True
        If True, multiply response by dT = T * ln(10) * dlog10(T)
        to convert from EMD to DEM response.
        Set to False if working in EMD space.

    Returns
    -------
    rmatrix : np.ndarray, shape (nt, nf), dtype float64
        Response per DEM bin and filter.
        If include_jacobian=True: units are [DN / (cm^-5 K^-1)]
        If include_jacobian=False: units are [DN / cm^-5]
    logt : np.ndarray, shape (nt,), dtype float64
        Bin-center log10(T) values.
    dlogt : np.ndarray, shape (nt,), dtype float64
        Bin widths in log10(T).

    Raises
    ------
    ValueError
        If nt or nf < 1.

    Notes
    -----
    The Jacobian factor arises from:
        DEM(T) = EM(T) / dT
    where dT = T * ln(10) * d(log10 T)

    For proper DEM inversions, the forward model is:
        DN = ∫ DEM(T) * R(T) * dT
           = ∫ DEM(T) * R(T) * T * ln(10) * d(log10 T)

    So the discretized response matrix should include the T * ln(10) factor.

    Examples
    --------
    >>> rmatrix, logt, dlogt = build_binned_responses(nt=24, nf=6)
    >>> rmatrix.shape
    (24, 6)
    >>> (rmatrix > 0).all()  # All positive
    True
    >>> np.allclose(dlogt, dlogt[0])  # Uniform spacing in log
    True
    """
    # Validate
    if nt < 1:
        raise ValueError(f"nt must be >= 1, got {nt}")
    if nf < 1:
        raise ValueError(f"nf must be >= 1, got {nf}")

    # Get high-resolution responses
    T_RESP, logT_samples, TEMPS = prepare_synthetic_responses(
        logT_min=logT_min, logT_max=logT_max,
        n_tresp=n_tresp, nt=nt, nf=nf
    )

    # Edges in log10(T)
    log_edges = np.log10(TEMPS.astype(np.float64))

    # Bin centers and widths
    logt = 0.5 * (log_edges[:-1] + log_edges[1:])
    dlogt = log_edges[1:] - log_edges[:-1]

    # Bin-average T_RESP over logT_samples into DEM bins
    rmatrix = np.empty((nt, nf), dtype=np.float64)

    for i in range(nt):
        lo, hi = log_edges[i], log_edges[i + 1]

        # Select samples in this bin
        if i < nt - 1:
            sel = (logT_samples >= lo) & (logT_samples < hi)
        else:
            # Last bin: include right edge
            sel = (logT_samples >= lo) & (logT_samples <= hi)

        if np.any(sel):
            # Average over selected samples
            rmatrix[i, :] = T_RESP[sel, :].mean(axis=0)
        else:
            # Fallback: interpolate at bin center
            lc = logt[i]
            for k in range(nf):
                rmatrix[i, k] = np.interp(lc, logT_samples, T_RESP[:, k])

    # Apply Jacobian factor if requested (for DEM, not EMD)
    if include_jacobian:
        # dT = T * ln(10) * dlog10(T)
        T_centers = 10.0 ** logt  # Linear temperature at bin centers
        jacobian = T_centers * np.log(10.0 ** dlogt)  # dT in Kelvin
        rmatrix = rmatrix * jacobian[:, np.newaxis]

    # Ensure strictly positive floor to avoid divide-by-zero downstream
    rmatrix = np.maximum(rmatrix, 1e-30)

    return rmatrix, logt, dlogt


def build_response_matrix_for_solver(
        nt: int = 24,
        nf: int = 6,
        scale_factor: float = 1e15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a response matrix scaled appropriately for the vendor solver.

    This is a convenience function that:
    1. Builds binned responses with Jacobian
    2. Scales by a factor to avoid numerical underflow
    3. Returns in the format expected by dn2dem_pos

    Parameters
    ----------
    nt : int, default 24
        Number of DEM temperature bins.
    nf : int, default 6
        Number of filters (must be 6 for vendor code).
    scale_factor : float, default 1e15
        Scaling factor to avoid tiny numbers.

    Returns
    -------
    rmatrix : np.ndarray, shape (nt, nf), dtype float64
        Scaled response matrix for solver.
    logt : np.ndarray, shape (nt,), dtype float64
        Bin-center log10(T) values.
    dlogt : np.ndarray, shape (nt,), dtype float64
        Bin widths in log10(T).

    Notes
    -----
    The vendor solver (dn2dem_pos) expects:
    - rmatrix: (nt, nf) response on DEM bins
    - logt: (nt,) bin centers
    - dlogt: (nt,) bin widths

    The scale_factor is applied to rmatrix and should be removed from
    the final DEM output: DEM_true = DEM_output * scale_factor

    Examples
    --------
    >>> rmatrix, logt, dlogt = build_response_matrix_for_solver()
    >>> rmatrix.shape
    (24, 6)
    >>> rmatrix.max() < 1e20  # Scaled but not too large
    True
    """
    rmatrix, logt, dlogt = build_binned_responses(
        nt=nt, nf=nf, include_jacobian=True
    )

    # Apply scaling
    rmatrix = rmatrix * scale_factor

    return rmatrix, logt, dlogt


def validate_response_matrix(
        rmatrix: np.ndarray,
        logt: np.ndarray,
        dlogt: np.ndarray,
        nf_expected: int = 6,
) -> None:
    """
    Validate a response matrix for use with DEM solver.

    Parameters
    ----------
    rmatrix : np.ndarray
        Response matrix.
    logt : np.ndarray
        Bin-center log10(T) values.
    dlogt : np.ndarray
        Bin widths.
    nf_expected : int, default 6
        Expected number of filters.

    Raises
    ------
    ValueError
        If validation fails.

    Examples
    --------
    >>> rmatrix, logt, dlogt = build_binned_responses(nt=24, nf=6)
    >>> validate_response_matrix(rmatrix, logt, dlogt, nf_expected=6)
    >>> # No error raised - validation passed
    """
    if rmatrix.ndim != 2:
        raise ValueError(
            f"rmatrix must be 2D (nt, nf), got {rmatrix.ndim}D"
        )

    nt, nf = rmatrix.shape

    if nf != nf_expected:
        raise ValueError(
            f"Expected {nf_expected} filters, got {nf}"
        )

    if logt.ndim != 1 or len(logt) != nt:
        raise ValueError(
            f"logt must be 1D with length {nt}, got shape {logt.shape}"
        )

    if dlogt.ndim != 1 or len(dlogt) != nt:
        raise ValueError(
            f"dlogt must be 1D with length {nt}, got shape {dlogt.shape}"
        )

    if not np.all(rmatrix > 0):
        raise ValueError("rmatrix must have all positive values")

    if not np.all(np.diff(logt) > 0):
        raise ValueError("logt must be strictly increasing")

    if not np.all(dlogt > 0):
        raise ValueError("dlogt must have all positive values")