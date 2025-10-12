# src/common/post.py
from __future__ import annotations

"""
Utilities to derive temperature summary maps from a DEM cube.

Provides functions to extract temperature information from
Differential Emission Measure (DEM) distributions:

- `dem_to_temp_maps(demmap, logT_bins)` → (mean_log10T, peak_log10T)
- `dem_to_all_temp_maps(demmap, logT_bins)` → (mean, peak, median, std)

Input Convention
----------------
- `demmap` has shape (H, W, NT), non-negative emission measures per bin.
- `logT_bins` is a 1D array of base-10 log-temperature *bin centers* of length NT.

Output Convention
-----------------
All returned temperature maps are in log10(T) [K] with NaN for invalid pixels.
"""

import numpy as np


def dem_to_temp_maps(
        demmap: np.ndarray,
        logT_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute two temperature maps from a DEM cube:
      1) DEM-weighted **mean log10(T)** per pixel
      2) **Peak log10(T)** (bin center of the maximal DEM per pixel)

    Parameters
    ----------
    demmap : np.ndarray, shape (H, W, NT)
        DEM cube. Values should be non-negative emission measures.
    logT_bins : np.ndarray, shape (NT,)
        Base-10 log-temperature bin centers.

    Returns
    -------
    mean_log10T : np.ndarray, shape (H, W), dtype float32
        DEM-weighted mean of log10(T) per pixel.
        NaN for pixels with zero total emission measure.
    peak_log10T : np.ndarray, shape (H, W), dtype float32
        Bin-center log10(T) at which the DEM attains its maximum.
        NaN for pixels with zero total emission measure.

    Raises
    ------
    ValueError
        If demmap is not 3D or logT_bins length doesn't match NT.

    Notes
    -----
    - The DEM-weighted mean is computed in **linear T** (K), then converted to
      log10(T):  mean_log10T = log10( sum(DEM * T_center) / sum(DEM) ).
    - Pixels with zero total emission measure are marked as NaN in both outputs.
    - The function is vectorized for performance.

    Examples
    --------
    >>> dem = np.random.rand(100, 100, 24).astype(np.float32)
    >>> logT = np.linspace(5.5, 7.5, 24).astype(np.float32)
    >>> mean_T, peak_T = dem_to_temp_maps(dem, logT)
    >>> mean_T.shape
    (100, 100)
    >>> np.isfinite(mean_T).all() or (mean_T > 0).any()
    True
    """
    # Validate inputs
    if demmap.ndim != 3:
        raise ValueError(
            f"`demmap` must be 3D (H, W, NT); got {demmap.ndim}D: {demmap.shape}"
        )

    H, W, NT = demmap.shape

    logT_bins = np.asarray(logT_bins, dtype=np.float32).reshape(-1)
    if logT_bins.size != NT:
        raise ValueError(
            f"len(logT_bins) must equal NT={NT}; got {logT_bins.size}"
        )

    # Ensure non-negative, contiguous float32
    dem = np.clip(demmap, 0.0, None).astype(np.float32, copy=False)  # (H, W, NT)

    # Total emission measure and validity mask
    EM = dem.sum(axis=2)  # (H, W)
    valid = EM > 0

    # Peak log10(T): argmax over bins
    imax = dem.argmax(axis=2)  # (H, W)
    peak_logT = np.where(valid, logT_bins[imax], np.nan)

    # DEM-weighted mean temperature in linear space, then log10
    T_centers = 10.0 ** logT_bins  # (NT,)

    # Weighted sum: sum_k DEM_ijk * T_k
    num = np.einsum("ijk,k->ij", dem, T_centers, optimize=True)  # (H, W)

    # Compute mean temperature in linear space
    mean_T_linear = np.full_like(EM, np.nan, dtype=np.float32)
    np.divide(num, EM, out=mean_T_linear, where=valid)

    # Convert to log10(T)
    mean_logT = np.full_like(EM, np.nan, dtype=np.float32)
    np.log10(mean_T_linear, out=mean_logT, where=valid)

    return mean_logT, peak_logT


def dem_to_median_temp(
        demmap: np.ndarray,
        logT_bins: np.ndarray
) -> np.ndarray:
    """
    Compute DEM-weighted median log10(T) for each pixel.

    Parameters
    ----------
    demmap : np.ndarray, shape (H, W, NT)
        DEM cube with non-negative emission measures.
    logT_bins : np.ndarray, shape (NT,)
        Base-10 log-temperature bin centers.

    Returns
    -------
    median_log10T : np.ndarray, shape (H, W), dtype float32
        Median log10(T) weighted by DEM.
        NaN for pixels with zero total emission measure.

    Notes
    -----
    The median is computed by finding the temperature bin where the
    cumulative DEM reaches 50% of the total.

    Examples
    --------
    >>> dem = np.random.rand(50, 50, 24).astype(np.float32)
    >>> logT = np.linspace(5.5, 7.5, 24).astype(np.float32)
    >>> median_T = dem_to_median_temp(dem, logT)
    >>> median_T.shape
    (50, 50)
    """
    if demmap.ndim != 3:
        raise ValueError(f"demmap must be 3D, got {demmap.ndim}D")

    H, W, NT = demmap.shape
    logT_bins = np.asarray(logT_bins, dtype=np.float32).reshape(-1)

    if logT_bins.size != NT:
        raise ValueError(f"logT_bins length {logT_bins.size} != NT {NT}")

    dem = np.clip(demmap, 0.0, None).astype(np.float32, copy=False)

    # Cumulative sum along temperature axis
    cumsum = np.cumsum(dem, axis=2)  # (H, W, NT)
    total = cumsum[:, :, -1]  # (H, W)
    valid = total > 0

    # Find first bin where cumsum >= 0.5 * total
    threshold = 0.5 * total[:, :, np.newaxis]  # (H, W, 1)
    median_idx = np.argmax(cumsum >= threshold, axis=2)  # (H, W)

    median_logT = np.where(valid, logT_bins[median_idx], np.nan)

    return median_logT


def dem_to_std_temp(
        demmap: np.ndarray,
        logT_bins: np.ndarray
) -> np.ndarray:
    """
    Compute DEM-weighted standard deviation of log10(T) for each pixel.

    Parameters
    ----------
    demmap : np.ndarray, shape (H, W, NT)
        DEM cube with non-negative emission measures.
    logT_bins : np.ndarray, shape (NT,)
        Base-10 log-temperature bin centers.

    Returns
    -------
    std_log10T : np.ndarray, shape (H, W), dtype float32
        Standard deviation of log10(T) weighted by DEM.
        NaN for pixels with zero total emission measure.

    Notes
    -----
    Computed as: sqrt(sum(DEM * (logT - mean_logT)^2) / sum(DEM))

    Examples
    --------
    >>> dem = np.random.rand(50, 50, 24).astype(np.float32)
    >>> logT = np.linspace(5.5, 7.5, 24).astype(np.float32)
    >>> std_T = dem_to_std_temp(dem, logT)
    >>> std_T.shape
    (50, 50)
    """
    if demmap.ndim != 3:
        raise ValueError(f"demmap must be 3D, got {demmap.ndim}D")

    H, W, NT = demmap.shape
    logT_bins = np.asarray(logT_bins, dtype=np.float32).reshape(-1)

    if logT_bins.size != NT:
        raise ValueError(f"logT_bins length {logT_bins.size} != NT {NT}")

    dem = np.clip(demmap, 0.0, None).astype(np.float32, copy=False)

    EM = dem.sum(axis=2)  # (H, W)
    valid = EM > 0

    # Compute mean in log space (simpler than converting to linear)
    weighted_sum = np.einsum("ijk,k->ij", dem, logT_bins, optimize=True)
    mean_logT = np.full_like(EM, np.nan, dtype=np.float32)
    np.divide(weighted_sum, EM, out=mean_logT, where=valid)

    # Compute variance
    diff_sq = (logT_bins[np.newaxis, np.newaxis, :] - mean_logT[:, :, np.newaxis]) ** 2
    weighted_var = np.einsum("ijk,ijk->ij", dem, diff_sq, optimize=True)

    variance = np.full_like(EM, np.nan, dtype=np.float32)
    np.divide(weighted_var, EM, out=variance, where=valid)

    std_logT = np.sqrt(variance, where=valid, out=variance)

    return std_logT


def dem_to_all_temp_maps(
        demmap: np.ndarray,
        logT_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all temperature summary statistics from a DEM cube.

    Parameters
    ----------
    demmap : np.ndarray, shape (H, W, NT)
        DEM cube with non-negative emission measures.
    logT_bins : np.ndarray, shape (NT,)
        Base-10 log-temperature bin centers.

    Returns
    -------
    mean_log10T : np.ndarray, shape (H, W), dtype float32
        DEM-weighted mean log10(T).
    peak_log10T : np.ndarray, shape (H, W), dtype float32
        Peak (mode) log10(T).
    median_log10T : np.ndarray, shape (H, W), dtype float32
        Median log10(T).
    std_log10T : np.ndarray, shape (H, W), dtype float32
        Standard deviation of log10(T).

    Notes
    -----
    This function computes all statistics in a single pass for efficiency.
    All outputs are NaN for pixels with zero total emission measure.

    Examples
    --------
    >>> dem = np.random.rand(100, 100, 24).astype(np.float32)
    >>> logT = np.linspace(5.5, 7.5, 24).astype(np.float32)
    >>> mean, peak, median, std = dem_to_all_temp_maps(dem, logT)
    >>> all(x.shape == (100, 100) for x in [mean, peak, median, std])
    True
    """
    mean_logT, peak_logT = dem_to_temp_maps(demmap, logT_bins)
    median_logT = dem_to_median_temp(demmap, logT_bins)
    std_logT = dem_to_std_temp(demmap, logT_bins)

    return mean_logT, peak_logT, median_logT, std_logT


def dem_emission_weighted_median(
        demmap: np.ndarray,
        logT_bins: np.ndarray
) -> np.ndarray:
    """
    Compute emission-weighted median temperature (most commonly reported metric).

    This is the temperature at which 50% of the total emission measure
    is below and 50% is above, weighted by the DEM values.

    Parameters
    ----------
    demmap : np.ndarray, shape (H, W, NT)
        DEM cube.
    logT_bins : np.ndarray, shape (NT,)
        Log10 temperature bin centers.

    Returns
    -------
    ewm_log10T : np.ndarray, shape (H, W), dtype float32
        Emission-weighted median log10(T).

    Notes
    -----
    This is often the most robust single-value temperature estimate
    as it's less sensitive to outliers than the mean.

    See Also
    --------
    dem_to_median_temp : Alias for this function.
    """
    return dem_to_median_temp(demmap, logT_bins)