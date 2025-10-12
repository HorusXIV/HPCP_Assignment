# src/dask/solver.py
from __future__ import annotations

"""
Dask DEM solver - wraps baseline vendor code for distributed execution.
"""

import os
import numpy as np
from typing import Tuple

# ========== CRITICAL: Disable vendor's nested parallelism ==========
# Set BEFORE importing vendor code
os.environ['HPCP_INNER_PROCS'] = '1'

# ========== Monkey-patch demmap_pos BEFORE importing dn2dem_pos ==========
import sys
from unittest.mock import patch


# We need to patch the ProcessPoolExecutor in the vendor code
# to prevent it from trying to spawn processes when already in a daemon process
def _mock_process_pool_executor(*args, **kwargs):
    """
    Mock ProcessPoolExecutor that runs serially instead of spawning processes.
    This prevents the 'daemonic processes cannot have children' error.
    """
    from concurrent.futures import Executor

    class SerialExecutor(Executor):
        """Executor that runs tasks serially in the current process."""

        def submit(self, fn, *args, **kwargs):
            """Execute function immediately and return a completed future."""
            from concurrent.futures import Future
            future = Future()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future

        def shutdown(self, wait=True):
            pass

    return SerialExecutor()


# Patch ProcessPoolExecutor before importing vendor code
import concurrent.futures

_original_pool_executor = concurrent.futures.ProcessPoolExecutor
concurrent.futures.ProcessPoolExecutor = _mock_process_pool_executor

# Now import vendor code (it will use our mocked executor)
from src.baseline.vendor.dn2dem_pos import dn2dem_pos

# Restore original (optional, but good practice)
concurrent.futures.ProcessPoolExecutor = _original_pool_executor

from src.common.solver_utils import get_bins, synthesize_tresp

print("[solver.py] Successfully patched ProcessPoolExecutor to prevent nested process spawning")

# ========== Global calibration cache (computed once) ==========

_CALIB_CACHE = {}


def _get_calibration(nt: int = 50, nf: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get or compute calibration data (cached globally to avoid recomputation).

    Returns
    -------
    (tresp, tresp_logt, temps) : tuple
        - tresp: (nt, nf) temperature response matrix
        - tresp_logt: (nt,) log10(T) bin centers
        - temps: (nt+1,) temperature bin edges in Kelvin
    """
    key = (nt, nf)
    if key not in _CALIB_CACHE:
        logt_centers, temps_edges = get_bins(nt=nt)
        tresp = synthesize_tresp(
            logt_centers,
            nf=nf,
            include_jacobian=True,
            normalize="l1",
            width=0.20,
        )
        _CALIB_CACHE[key] = (tresp, logt_centers, temps_edges)

    return _CALIB_CACHE[key]


# ========== Error model (matches baseline) ==========

def _build_error_model(
        tile_hw6: np.ndarray,
        err_a: float = 1.0,
        err_b: float = 1e-6,
) -> np.ndarray:
    """
    Build Poisson-like error model: e = sqrt(a*f + b).

    Parameters
    ----------
    tile_hw6 : np.ndarray, shape (H, W, 6)
        Input tile (channels-last).
    err_a, err_b : float
        Error model parameters.

    Returns
    -------
    edn : np.ndarray, shape (H, W, 6)
        Error estimates for the tile.
    """
    # Sanitize input
    f = np.nan_to_num(tile_hw6, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.clip(f, 0.0, None)

    # Poisson-like error
    edn = np.sqrt(err_a * f + err_b).astype(np.float32)

    return edn


# ========== Core solver function (called by Dask workers) ==========

def solve_tile(
        tile_hw6: np.ndarray,
        *,
        nmu: int = 42,
        nt: int = 50,
        err_a: float = 1.0,
        err_b: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve DEM for a single tile using baseline vendor code.

    This function is mapped over Dask chunks by `da.map_blocks`.

    Parameters
    ----------
    tile_hw6 : np.ndarray, shape (H, W, 6)
        Input tile in channels-last format (vendor expects this).
    nmu : int, default 42
        Regularization parameter.
    nt : int, default 50
        Number of DEM temperature bins.
    err_a, err_b : float
        Error model parameters.

    Returns
    -------
    dem : np.ndarray, shape (H, W, nt)
        DEM solution.
    edem : np.ndarray, shape (H, W, nt)
        DEM uncertainties.
    chisq : np.ndarray, shape (H, W)
        Reduced chi-squared per pixel.
    logt : np.ndarray, shape (nt,)
        Temperature bin centers (log10 K).

    Notes
    -----
    - Input must be channels-last: (H, W, 6)
    - Output is also channels-last: (H, W, nt)
    - This matches vendor dn2dem_pos interface
    """
    # Get calibration (cached)
    tresp, tresp_logt, temps = _get_calibration(nt=nt, nf=6)

    # Build error model
    edn = _build_error_model(tile_hw6, err_a=err_a, err_b=err_b)

    # Call vendor solver (expects channels-last)
    # Returns: (dem, edem, logt, chisq, dn_reg)
    dem, edem, logt, chisq, _dn_reg = dn2dem_pos(
        tile_hw6,  # (H, W, 6) - channels last
        edn,  # (H, W, 6) - errors
        tresp,  # (nt, 6) - response matrix
        tresp_logt,  # (nt,) - log10(T) centers
        temps,  # (nt+1,) - T edges in Kelvin
        nmu=nmu,
    )

    return dem, edem, chisq, logt


# ========== Dask-friendly wrapper (for map_blocks) ==========

def solve_tile_dask_wrapper(
        block: np.ndarray,
        *args,
        **kwargs
) -> np.ndarray:
    """
    Wrapper for da.map_blocks that handles block metadata.

    Dask passes: (block, *block_id_tuple, *extra_args)
    We extract nmu and nt from args after block_id.
    """
    # Extract nmu and nt from the end of args
    if len(args) >= 2:
        nmu = args[-2]
        nt = args[-1]
    else:
        nmu = kwargs.get('nmu', 42)
        nt = kwargs.get('nt', 50)

    # Handle batch dimension if present
    if block.ndim == 4 and block.shape[0] == 1:
        tile = block[0]  # (H, W, 6)
        dem, _edem, _chisq, _logt = solve_tile(tile, nmu=nmu, nt=nt)
        return dem[None, ...]  # (1, H, W, nt)
    else:
        dem, _edem, _chisq, _logt = solve_tile(block, nmu=nmu, nt=nt)
        return dem


def solve_edem_dask_wrapper(
        block: np.ndarray,
        *args,
        **kwargs
) -> np.ndarray:
    """Wrapper for edem output."""
    if len(args) >= 2:
        nmu = args[-2]
        nt = args[-1]
    else:
        nmu = kwargs.get('nmu', 42)
        nt = kwargs.get('nt', 50)

    if block.ndim == 4 and block.shape[0] == 1:
        tile = block[0]
        _dem, edem, _chisq, _logt = solve_tile(tile, nmu=nmu, nt=nt)
        return edem[None, ...]
    else:
        _dem, edem, _chisq, _logt = solve_tile(block, nmu=nmu, nt=nt)
        return edem


def solve_chisq_dask_wrapper(
        block: np.ndarray,
        *args,
        **kwargs
) -> np.ndarray:
    """Wrapper for chisq output."""
    if len(args) >= 2:
        nmu = args[-2]
        nt = args[-1]
    else:
        nmu = kwargs.get('nmu', 42)
        nt = kwargs.get('nt', 50)

    if block.ndim == 4 and block.shape[0] == 1:
        tile = block[0]
        _dem, _edem, chisq, _logt = solve_tile(tile, nmu=nmu, nt=nt)
        return chisq[None, ...]  # (1, H, W)
    else:
        _dem, _edem, chisq, _logt = solve_tile(block, nmu=nmu, nt=nt)
        return chisq