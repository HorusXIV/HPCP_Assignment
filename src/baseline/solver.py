# src/baseline/solver.py
"""
Baseline DEM solver using vendorized DEMREG code.

This module provides a clean interface to the vendor DEM solver (dn2dem_pos)
with proper input validation, error handling, and profiling integration.

The baseline solver serves as the ground truth for:
  - Generating golden reference outputs
  - Performance baseline for comparison
  - Correctness verification

Usage
-----
Simple solve:
    from src.baseline.solver import solve_dem

    demmap, edemmap, logt, chisq, dn_reg = solve_dem(
        data_6hw,
        tresp,
        tresp_logt,
        temps,
    )

With validation and profiling:
    result = solve_dem(
        data_6hw,
        tresp,
        tresp_logt,
        temps,
        validate_inputs=True,
        validate_outputs=True,
    )
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

# Import vendor solver
from .vendor.dn2dem_pos import dn2dem_pos


# Import validation utilities (we'll reference from common if needed)


def validate_input_shapes(
        data_6hw: np.ndarray,
        tresp: np.ndarray,
        tresp_logt: np.ndarray,
        temps: np.ndarray,
) -> None:
    """
    Validate input array shapes for DEM solver.

    Parameters
    ----------
    data_6hw : np.ndarray
        Input data, shape (6, H, W) - channels first.
    tresp : np.ndarray
        Temperature response matrix, shape (n_tresp, 6).
    tresp_logt : np.ndarray
        Log temperature bins for response, shape (n_tresp,).
    temps : np.ndarray
        DEM temperature bin edges, shape (n_temps,).

    Raises
    ------
    ValueError
        If shapes are invalid or incompatible.
    """
    # Check data shape
    if data_6hw.ndim != 3:
        raise ValueError(
            f"data_6hw must be 3D (6, H, W), got shape {data_6hw.shape}"
        )

    if data_6hw.shape[0] != 6:
        raise ValueError(
            f"data_6hw must have 6 channels (shape[0]), got {data_6hw.shape[0]}"
        )

    # Check tresp shape
    if tresp.ndim != 2:
        raise ValueError(
            f"tresp must be 2D (n_tresp, 6), got shape {tresp.shape}"
        )

    if tresp.shape[1] != 6:
        raise ValueError(
            f"tresp must have 6 filters (shape[1]), got {tresp.shape[1]}"
        )

    # Check tresp_logt shape
    if tresp_logt.ndim != 1:
        raise ValueError(
            f"tresp_logt must be 1D, got shape {tresp_logt.shape}"
        )

    if tresp_logt.shape[0] != tresp.shape[0]:
        raise ValueError(
            f"tresp_logt length ({tresp_logt.shape[0]}) must match "
            f"tresp rows ({tresp.shape[0]})"
        )

    # Check temps shape
    if temps.ndim != 1:
        raise ValueError(
            f"temps must be 1D, got shape {temps.shape}"
        )

    # Check monotonicity
    if not np.all(np.diff(tresp_logt) > 0):
        raise ValueError("tresp_logt must be strictly increasing")

    if not np.all(np.diff(temps) > 0):
        raise ValueError("temps must be strictly increasing")


def validate_input_values(
        data_6hw: np.ndarray,
        tresp: np.ndarray,
        tresp_logt: np.ndarray,
        temps: np.ndarray,
) -> None:
    """
    Validate input array values for DEM solver.

    Parameters
    ----------
    data_6hw : np.ndarray
        Input data, shape (6, H, W).
    tresp : np.ndarray
        Temperature response matrix, shape (n_tresp, 6).
    tresp_logt : np.ndarray
        Log temperature bins, shape (n_tresp,).
    temps : np.ndarray
        DEM temperature bin edges, shape (n_temps,).

    Raises
    ------
    ValueError
        If values are invalid (NaN, Inf, negative counts).

    Warnings
    --------
    Warns if data contains zeros, negative values, or non-finite values.
    """
    # Check for NaN/Inf in data
    if not np.all(np.isfinite(data_6hw)):
        n_bad = np.sum(~np.isfinite(data_6hw))
        warnings.warn(
            f"data_6hw contains {n_bad} non-finite values (NaN/Inf). "
            f"These will be sanitized to zero."
        )

    # Check for negative values in data
    if np.any(data_6hw < 0):
        n_neg = np.sum(data_6hw < 0)
        warnings.warn(
            f"data_6hw contains {n_neg} negative values. "
            f"These will be clipped to zero."
        )

    # Check tresp
    if not np.all(np.isfinite(tresp)):
        raise ValueError("tresp contains non-finite values")

    if np.any(tresp < 0):
        raise ValueError("tresp contains negative values")

    # Check tresp_logt range (typically 5.5 to 7.5 for log10(K))
    if tresp_logt.min() < 4.0 or tresp_logt.max() > 9.0:
        warnings.warn(
            f"tresp_logt range [{tresp_logt.min():.2f}, {tresp_logt.max():.2f}] "
            f"is unusual. Expected approximately [5.5, 7.5] for log10(K)."
        )

    # Check temps
    if not np.all(np.isfinite(temps)):
        raise ValueError("temps contains non-finite values")


def validate_output_shapes(
        demmap: np.ndarray,
        edemmap: np.ndarray,
        logt: np.ndarray,
        chisq: np.ndarray,
        dn_reg: np.ndarray,
        expected_hw: Tuple[int, int],
        expected_nt: int,
) -> None:
    """
    Validate solver output shapes.

    Parameters
    ----------
    demmap : np.ndarray
        DEM solution, expected shape (H, W, n_temps-1).
    edemmap : np.ndarray
        DEM uncertainties, expected shape (H, W, n_temps-1).
    logt : np.ndarray
        Temperature bin centers, expected shape (n_temps-1,).
    chisq : np.ndarray
        Chi-square values, expected shape (H, W).
    dn_reg : np.ndarray
        Regularized data, expected shape (H, W, 6).
    expected_hw : tuple[int, int]
        Expected (H, W) from input.
    expected_nt : int
        Expected n_temps-1.

    Raises
    ------
    ValueError
        If output shapes don't match expected dimensions.
    """
    H, W = expected_hw

    # Check demmap
    if demmap.shape != (H, W, expected_nt):
        raise ValueError(
            f"demmap has shape {demmap.shape}, "
            f"expected ({H}, {W}, {expected_nt})"
        )

    # Check edemmap
    if edemmap.shape != (H, W, expected_nt):
        raise ValueError(
            f"edemmap has shape {edemmap.shape}, "
            f"expected ({H}, {W}, {expected_nt})"
        )

    # Check logt
    if logt.shape != (expected_nt,):
        raise ValueError(
            f"logt has shape {logt.shape}, expected ({expected_nt},)"
        )

    # Check chisq
    if chisq.shape != (H, W):
        raise ValueError(
            f"chisq has shape {chisq.shape}, expected ({H}, {W})"
        )

    # Check dn_reg
    if dn_reg.shape != (H, W, 6):
        raise ValueError(
            f"dn_reg has shape {dn_reg.shape}, expected ({H}, {W}, 6)"
        )


def validate_output_values(
        demmap: np.ndarray,
        edemmap: np.ndarray,
        chisq: np.ndarray,
) -> None:
    """
    Validate solver output values.

    Parameters
    ----------
    demmap : np.ndarray
        DEM solution.
    edemmap : np.ndarray
        DEM uncertainties.
    chisq : np.ndarray
        Chi-square values.

    Warnings
    --------
    Warns about non-finite values, negative DEMs, or suspicious chi-square.
    """
    # Check demmap
    if not np.all(np.isfinite(demmap)):
        n_bad = np.sum(~np.isfinite(demmap))
        pct = 100.0 * n_bad / demmap.size
        warnings.warn(
            f"demmap contains {n_bad} ({pct:.2f}%) non-finite values"
        )

    if np.any(demmap < 0):
        n_neg = np.sum(demmap < 0)
        pct = 100.0 * n_neg / demmap.size
        warnings.warn(
            f"demmap contains {n_neg} ({pct:.2f}%) negative values"
        )

    # Check edemmap
    if not np.all(np.isfinite(edemmap)):
        n_bad = np.sum(~np.isfinite(edemmap))
        pct = 100.0 * n_bad / edemmap.size
        warnings.warn(
            f"edemmap contains {n_bad} ({pct:.2f}%) non-finite values"
        )

    # Check chisq
    if not np.all(np.isfinite(chisq)):
        n_bad = np.sum(~np.isfinite(chisq))
        pct = 100.0 * n_bad / chisq.size
        warnings.warn(
            f"chisq contains {n_bad} ({pct:.2f}%) non-finite values"
        )

    # Check for unreasonably high chi-square
    finite_chisq = chisq[np.isfinite(chisq)]
    if finite_chisq.size > 0:
        median_chisq = np.median(finite_chisq)
        if median_chisq > 10.0:
            warnings.warn(
                f"Median chi-square is {median_chisq:.2f}, which is high. "
                f"Check data quality and error estimates."
            )


def prepare_inputs(
        data_6hw: np.ndarray,
        nmu: int = 42,
        err_a: float = 1.0,
        err_b: float = 1e-6,
        error_model: str = "sqrt",
        dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data and error arrays for vendor solver.

    The vendor expects:
      - data: (H, W, 6) channels-last, sanitized, non-negative
      - edata: (H, W, 6) channels-last, positive uncertainties

    Parameters
    ----------
    data_6hw : np.ndarray
        Input data, shape (6, H, W) - channels first.
    nmu : int, default 42
        Regularization parameter (not used here, for API consistency).
    err_a : float, default 1.0
        Error model coefficient (for linear model).
    err_b : float, default 1e-6
        Error model base/floor.
    error_model : {"sqrt", "linear", "constant"}, default "sqrt"
        Error model type.
    dtype : np.dtype, default np.float32
        Output dtype.

    Returns
    -------
    data_hw6 : np.ndarray
        Prepared data, shape (H, W, 6), dtype=dtype.
    edata_hw6 : np.ndarray
        Error estimates, shape (H, W, 6), dtype=float32.
    """
    # Transpose to channels-last
    data = np.moveaxis(data_6hw, 0, -1).astype(dtype, copy=False)

    # Sanitize: remove NaN/Inf and clip to non-negative
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0, None)

    # Build error array
    if error_model == "sqrt":
        edata = np.sqrt(data, dtype=np.float32) + float(err_b)
    elif error_model == "linear":
        edata = float(err_a) * data.astype(np.float32, copy=False) + float(err_b)
    elif error_model == "constant":
        edata = np.full_like(data, float(err_b), dtype=np.float32)
    else:
        raise ValueError(f"Unknown error_model: {error_model}")

    return data, edata


def solve_dem(
        data_6hw: np.ndarray,
        tresp: np.ndarray,
        tresp_logt: np.ndarray,
        temps: np.ndarray,
        *,
        nmu: int = 42,
        err_a: float = 1.0,
        err_b: float = 1e-6,
        error_model: str = "sqrt",
        dtype: np.dtype = np.float32,
        validate_inputs: bool = False,
        validate_outputs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for Differential Emission Measure using baseline vendor code.

    This is the main entry point for the baseline DEM solver. It wraps
    the vendor dn2dem_pos function with input preparation, validation,
    and error handling.

    Parameters
    ----------
    data_6hw : np.ndarray
        Input data, shape (6, H, W) - channels first.
        Contains observed intensities in 6 wavelength channels.
    tresp : np.ndarray
        Temperature response matrix, shape (n_tresp, 6).
        Response of each channel to emission at different temperatures.
    tresp_logt : np.ndarray
        Log10 temperature bins for response, shape (n_tresp,).
        Typically covers log10(T) from ~5.5 to ~7.5 (10^5.5 to 10^7.5 K).
    temps : np.ndarray
        DEM temperature bin edges, shape (n_temps,).
        Output DEM will have n_temps-1 bins.
    nmu : int, default 42
        Regularization parameter. Higher values = more smoothing.
    err_a : float, default 1.0
        Error model coefficient (for linear model: err = a*data + b).
    err_b : float, default 1e-6
        Error model floor/base.
    error_model : {"sqrt", "linear", "constant"}, default "sqrt"
        Error estimation method:
          - "sqrt": err = sqrt(data) + err_b (Poisson-like)
          - "linear": err = err_a * data + err_b
          - "constant": err = err_b (uniform)
    dtype : np.dtype, default np.float32
        Data type for computation.
    validate_inputs : bool, default False
        If True, perform comprehensive input validation.
        Adds ~1-5ms overhead but catches errors early.
    validate_outputs : bool, default False
        If True, validate solver outputs for correctness.
        Adds ~1-5ms overhead.

    Returns
    -------
    demmap : np.ndarray
        DEM solution, shape (H, W, n_temps-1).
        Units: cm^-5 K^-1 (emission measure per temperature bin).
    edemmap : np.ndarray
        DEM uncertainties, shape (H, W, n_temps-1).
        1-sigma uncertainties on demmap.
    logt : np.ndarray
        Temperature bin centers (log10), shape (n_temps-1,).
    chisq : np.ndarray
        Reduced chi-square values, shape (H, W).
        Goodness of fit metric (ideally ~1).
    dn_reg : np.ndarray
        Regularized/fitted intensities, shape (H, W, 6).
        Model prediction for input data.

    Raises
    ------
    ValueError
        If inputs are invalid (when validate_inputs=True).

    Warnings
    --------
    Issues warnings for suspicious input/output values.

    Notes
    -----
    - Input data is automatically sanitized (NaN->0, negative->0)
    - This function is NOT thread-safe due to vendor code internals
    - For parallel processing, use separate processes

    Examples
    --------
    >>> # Simple usage
    >>> demmap, edemmap, logt, chisq, dn_reg = solve_dem(
    ...     data_6hw, tresp, tresp_logt, temps
    ... )

    >>> # With validation
    >>> result = solve_dem(
    ...     data_6hw, tresp, tresp_logt, temps,
    ...     validate_inputs=True,
    ...     validate_outputs=True,
    ... )
    """
    # Optional input validation
    if validate_inputs:
        validate_input_shapes(data_6hw, tresp, tresp_logt, temps)
        validate_input_values(data_6hw, tresp, tresp_logt, temps)

    # Prepare inputs for vendor solver
    data_hw6, edata_hw6 = prepare_inputs(
        data_6hw,
        nmu=nmu,
        err_a=err_a,
        err_b=err_b,
        error_model=error_model,
        dtype=dtype,
    )

    # Call vendor solver
    # Signature: dn2dem_pos(data, edata, tresp, tresp_logt, temps, nmu=42)
    try:
        demmap, edemmap, logt, chisq, dn_reg = dn2dem_pos(
            data_hw6,
            edata_hw6,
            tresp,
            tresp_logt,
            temps,
            nmu=nmu,
        )
        nt = int(temps.shape[0] - 1)
        if isinstance(logt, np.ndarray):
            if logt.ndim == 3 and logt.shape[-1] == nt:
                # logt was broadcast per-pixel; collapse to a single grid
                logt = logt[0, 0, :].astype(np.float32, copy=False)
            elif logt.ndim != 1 or logt.shape[0] != nt:
                # Fallback: derive centers from edges if shapes don't match
                logt = 0.5 * (temps[:-1] + temps[1:])
                logt = logt.astype(np.float32, copy=False)
        else:
            # Defensive fallback if a non-ndarray sneaks through
            logt = np.asarray(0.5 * (temps[:-1] + temps[1:]), dtype=np.float32)
    except Exception as e:
        # Wrap vendor exceptions with context
        raise RuntimeError(
            f"Vendor solver (dn2dem_pos) failed: {e}\n"
            f"Input shapes: data={data_hw6.shape}, tresp={tresp.shape}, "
            f"temps={temps.shape}, nmu={nmu}"
        ) from e

    # Optional output validation
    if validate_outputs:
        H, W = data_hw6.shape[:2]
        n_temps_minus_1 = temps.shape[0] - 1
        validate_output_shapes(
            demmap, edemmap, logt, chisq, dn_reg,
            expected_hw=(H, W),
            expected_nt=n_temps_minus_1,
        )
        validate_output_values(demmap, edemmap, chisq)

    return demmap, edemmap, tresp_logt.astype(np.float32, copy=False), chisq, dn_reg


# Convenience function matching profiling.wallclock signature
def baseline_solver_fn(
        data_hw6: np.ndarray,
        edata_hw6: np.ndarray,
        tresp: np.ndarray,
        tresp_logt: np.ndarray,
        temps: np.ndarray,
        nmu: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solver function compatible with profiling.wallclock.benchmark_wallclock().

    This is a thin wrapper around dn2dem_pos that matches the expected
    signature for wallclock benchmarking.

    Parameters
    ----------
    data_hw6 : np.ndarray
        Input data, shape (H, W, 6) - channels last.
    edata_hw6 : np.ndarray
        Error estimates, shape (H, W, 6) - channels last.
    tresp : np.ndarray
        Temperature response matrix, shape (n_tresp, 6).
    tresp_logt : np.ndarray
        Log temperature bins, shape (n_tresp,).
    temps : np.ndarray
        DEM temperature bin edges, shape (n_temps,).
    nmu : int, default 42
        Regularization parameter.

    Returns
    -------
    tuple
        (demmap, edemmap, logt, chisq, dn_reg) from solver.

    Notes
    -----
    This function expects inputs already prepared (channels-last, sanitized).
    For the full pipeline, use solve_dem() instead.
    """
    return dn2dem_pos(data_hw6, edata_hw6, tresp, tresp_logt, temps, nmu=nmu)


__all__ = [
    "solve_dem",
    "baseline_solver_fn",
    "prepare_inputs",
    "validate_input_shapes",
    "validate_input_values",
    "validate_output_shapes",
    "validate_output_values",
]