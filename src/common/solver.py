# src/common/solver.py
from __future__ import annotations
"""
LEGACY CODE (used as a fallback in the single-gpu approach)

Unified CPU-side solver utilities.

This module provides a small, stable surface that the rest of the project can
call to obtain DEM outputs from a 3D tile `(H, W, 6)`:

- `solve_tile_all(...)` → `(dem, edem, chisq, logT_centers)`
- `solve_tile(...)`     → `dem` (convenience wrapper)
- `get_logt_bins_once(...)` → `(NT, logT_centers)`

Implementation notes
--------------------
- The implementation prefers the vendor function `dn2dem_pos` when available
  and falls back to `demmap_pos` otherwise. Both are lazily imported to avoid
  circular imports via `src.baseline.__init__`.
- When using the fallback path, synthetic temperature responses are generated so
  the solver remains runnable without external assets.
- The error model used for per-pixel/channel uncertainties is a simple
  `sqrt(a*DN + b)` by default and can be extended as needed.
"""

from typing import Optional, Tuple
import logging
import os
from pathlib import Path
import importlib.util
import importlib.machinery

import numpy as np

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    """
    Read a boolean-like environment variable.

    Accepts: "1", "true", "yes", "on" (case-insensitive).
    """
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


SOLVER_DEBUG = _env_flag("HPCP_SOLVER_DEBUG", False)
REQUIRE_VENDOR = _env_flag("HPCP_REQUIRE_VENDOR", False)


def _load_module_from_file(mod_name: str, file_path: Path):
    """
    Load a Python module directly from a file, bypassing package `__init__`.

    This avoids circular imports caused by `src.baseline.__init__`.

    Parameters
    ----------
    mod_name : str
        Synthetic module name.
    file_path : Path
        Path to the source file.

    Returns
    -------
    module
    """
    loader = importlib.machinery.SourceFileLoader(mod_name, str(file_path))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_vendor_file(rel_path: str) -> Optional[Path]:
    """
    Locate a vendor file relative to this file:
    `solver.py` → `../../baseline/vendor/<rel_path>`.
    """
    here = Path(__file__).resolve()
    root = here.parents[2]  # .../src
    candidate = root / "baseline" / "vendor" / rel_path
    return candidate if candidate.exists() else None


def _load_vendor():
    """
    Lazily load vendor symbols.

    Tries filesystem-based import first (to bypass `src.baseline.__init__`),
    then falls back to a normal import if files aren’t where we expect.

    Returns
    -------
    tuple[callable | None, callable | None]
        `(dn2dem_pos, demmap_pos)` — any may be `None` if not importable.
    """
    dn2 = dmm = None

    # --- try direct file loading (preferred to avoid circular import) ---
    dn2_file = _find_vendor_file("dn2dem_pos.py")
    dmm_file = _find_vendor_file("demmap_pos.py")

    try:
        if dn2_file:
            mod = _load_module_from_file("_vendor_dn2dem_pos", dn2_file)
            dn2 = getattr(mod, "dn2dem_pos", None)
            if SOLVER_DEBUG:
                log.info("[solver] loaded vendor dn2dem_pos from %s", dn2_file)
    except Exception as e:
        if SOLVER_DEBUG:
            log.warning("[solver] failed loading dn2dem_pos from file: %s", e)

    try:
        if dmm_file:
            mod = _load_module_from_file("_vendor_demmap_pos", dmm_file)
            dmm = getattr(mod, "demmap_pos", None)
            if SOLVER_DEBUG:
                log.info("[solver] loaded vendor demmap_pos from %s", dmm_file)
    except Exception as e:
        if SOLVER_DEBUG:
            log.warning("[solver] failed loading demmap_pos from file: %s", e)

    # --- fallback: standard import path (may hit package __init__) ---
    if dn2 is None or dmm is None:
        try:
            import importlib

            if dn2 is None:
                dn2 = importlib.import_module(
                    "src.baseline.vendor.dn2dem_pos"
                ).dn2dem_pos  # type: ignore[attr-defined]
                if SOLVER_DEBUG:
                    log.info("[solver] loaded vendor dn2dem_pos via importlib")
            if dmm is None:
                dmm = importlib.import_module(
                    "src.baseline.vendor.demmap_pos"
                ).demmap_pos  # type: ignore[attr-defined]
                if SOLVER_DEBUG:
                    log.info("[solver] loaded vendor demmap_pos via importlib")
        except Exception as e:
            if SOLVER_DEBUG:
                log.warning("[solver] standard vendor import failed: %s", e)

    return dn2, dmm


# -----------------------------------------------------------------------------
# Temperature binning & synthetic responses
# -----------------------------------------------------------------------------


def get_logt_bins_once(
    nmu: Optional[int] = None, nt: Optional[int] = None
) -> Tuple[int, np.ndarray]:
    """
    Compute `(NT, logT_centers)` used by the solver/vendor code.

    If `nt` is provided, it takes precedence. Otherwise, derive `NT` from `nmu`
    with a simple, stable mapping.

    Parameters
    ----------
    nmu : int | None
        Regularization parameter controlling the effective number of bins.
    nt : int | None
        Explicit number of temperature bins (overrides `nmu` heuristic).

    Returns
    -------
    NT : int
        Number of temperature bins.
    logT_centers : np.ndarray, shape (NT,), dtype float32
        Evenly spaced centers over `[5.5, 7.5]` in log10(T).
    """
    if nt is None:
        nt = 24 if not nmu else int(max(8, min(64, round(0.6 * nmu))))
    logT = np.linspace(5.5, 7.5, int(nt), dtype=np.float32)
    return int(nt), logT


def _synth_responses(nt: int, nf: int = 6):
    """
    Build deterministic synthetic temperature responses.

    Returns
    -------
    T_RESP : (n_tresp, nf) float32
    T_RESP_LOGT : (n_tresp,) float32
    TEMPS : (nt + 1,) float32  (bin edges in linear Kelvin)
    """
    logT = np.linspace(5.5, 7.5, 200, dtype=np.float32)  # response support
    temps = np.logspace(5.5, 7.5, nt + 1, dtype=np.float32)  # bin edges
    centers = np.linspace(5.7, 7.3, nf, dtype=np.float32)
    T_RESP = np.exp(-0.5 * ((logT[:, None] - centers[None, :]) / 0.20) ** 2) + 1e-30
    return T_RESP.astype(np.float32), logT.astype(np.float32), temps.astype(np.float32)


def _err_from_counts(
    counts6: np.ndarray, a: float = 1.0, b: float = 1e-6
) -> np.ndarray:
    """
    Simple per-pixel/channel error model: `sqrt(a * DN + b)`.

    Parameters
    ----------
    counts6 : (H, W, 6) float-like
        Input counts (already sanitized to be finite and non-negative).
    a, b : float
        Error model parameters.

    Returns
    -------
    (H, W, 6) float32
        Per-pixel/channel uncertainties.
    """
    return np.sqrt(a * np.clip(counts6, 0, None) + b).astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Per-tile solver
# -----------------------------------------------------------------------------


def solve_tile_all(
    counts6: np.ndarray,
    *,
    nmu: Optional[int] = 42,
    nt: Optional[int] = None,
    error_model: str = "sqrt",
    err_a: float = 1.0,
    err_b: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve a single tile `(H, W, 6)` and return `(dem, edem, chisq, logT_centers)`.

    Parameters
    ----------
    counts6 : np.ndarray
        Input tile with shape `(H, W, 6)` (channels-last).
    nmu : int | None, default 42
        Regularization parameter (passed to vendor; also used to derive `NT` when `nt` is None).
    nt : int | None, default None
        Explicit number of temperature bins. If set, overrides the `nmu` mapping.
    error_model : {"sqrt"}, default "sqrt"
        Error model used to produce per-pixel/channel uncertainties.
    err_a, err_b : float
        Parameters for the error model.

    Returns
    -------
    dem : np.ndarray, shape (H, W, NT), dtype float32
    edem : np.ndarray, shape (H, W, NT), dtype float32
    chisq : np.ndarray, shape (H, W), dtype float32
    logT_centers : np.ndarray, shape (NT,), dtype float32
    """
    assert counts6.ndim == 3 and counts6.shape[-1] == 6, "expects (H,W,6) tile"
    H, W, nf = counts6.shape
    NT, logT_centers = get_logt_bins_once(nmu=nmu, nt=nt)

    # Sanitize inputs
    f = np.clip(
        np.nan_to_num(counts6, nan=0.0, posinf=0.0, neginf=0.0), 0, None
    ).astype(np.float32, copy=False)

    # Build per-pixel/channel uncertainties (keep variable name 'e' for vendor call)
    if error_model == "sqrt":
        e = _err_from_counts(f, a=err_a, b=err_b)
    else:
        raise ValueError(f"Unsupported error_model: {error_model!r}")

    # Build synthetic responses
    T_RESP, T_RESP_LOGT, TEMPS = _synth_responses(nt=NT, nf=nf)

    # Load vendor functions lazily (and cycle-safe)
    dn2dem_pos, demmap_pos = _load_vendor()

    # ---- vendor path (preferred) ----
    if dn2dem_pos is not None:
        if SOLVER_DEBUG:
            log.info(
                "[solver] trying vendor dn2dem_pos (H=%d, W=%d, nf=%d, NT=%d, nmu=%s)",
                H,
                W,
                nf,
                NT,
                str(nmu),
            )
        try:
            dem, edem, logT_bins_out, chisq, _dn_reg = dn2dem_pos(
                f, e, T_RESP, T_RESP_LOGT, TEMPS, nmu=int(nmu or 42)
            )
            if (
                isinstance(logT_bins_out, np.ndarray)
                and logT_bins_out.shape == logT_centers.shape
            ):
                logT_centers = logT_bins_out.astype(np.float32, copy=False)
            if SOLVER_DEBUG:
                log.info("[solver] vendor dn2dem_pos SUCCESS")
            return (
                np.asarray(dem, dtype=np.float32, order="C"),
                np.asarray(edem, dtype=np.float32, order="C"),
                np.asarray(chisq, dtype=np.float32, order="C"),
                np.asarray(logT_centers, dtype=np.float32, order="C"),
            )
        except Exception as e_exc:
            if SOLVER_DEBUG:
                log.warning("[solver] vendor dn2dem_pos FAILED → fallback: %s", e_exc)

    # If required, refuse to run fallback
    if REQUIRE_VENDOR and dn2dem_pos is None:
        raise RuntimeError(
            "HPCP_REQUIRE_VENDOR=1 set, but vendor dn2dem_pos is unavailable"
        )

    # ---- fallback via demmap_pos ----
    if demmap_pos is None:
        # As a last resort, refuse with a clear error
        raise RuntimeError(
            "Vendor import failed and fallback demmap_pos is unavailable"
        )

    if SOLVER_DEBUG:
        log.info("[solver] using fallback demmap_pos")

    nx, ny, nt_bins = int(H), int(W), int(NT)
    dn1d = f.reshape(nx * ny, nf)
    edn1d = e.reshape(nx * ny, nf)
    rmatrix = T_RESP  # (n_tresp, nf)
    logt = T_RESP_LOGT  # (n_tresp,)

    # Some vendor code expects `dlogt` to be indexable (per-logT sample).
    dlogt = np.full(logt.shape, np.median(np.diff(logt)), dtype=np.float32)

    # IMPORTANT: 'glc' is a per-channel mask/vector — length must be nf (not NT)
    glc = np.ones((nf,), dtype=np.float32)

    # Provide a properly shaped initial DEM guess for vendor fallback
    # demmap_pos slices dem_norm0[i*n_par:(i+1)*n_par, :], so it must be 2D
    dem_norm0 = np.zeros((nx * ny, nt_bins), dtype=np.float32)

    # Mirror typical vendor defaults
    reg_tweak = 1.0
    max_iter = 10
    rgt_fact = 1.5
    warn = False
    l_emd = False
    rscl = False

    dem1d, edem1d, elogt1d, chisq1d, _dn_reg1d = demmap_pos(
        dn1d,
        edn1d,
        rmatrix,
        logt,
        dlogt,
        glc,
        reg_tweak=reg_tweak,
        max_iter=max_iter,
        rgt_fact=rgt_fact,
        dem_norm0=dem_norm0,  # <- array, not an int
        nmu=int(nmu or 42),
        warn=warn,
        l_emd=l_emd,
        rscl=rscl,
    )

    dem = dem1d.reshape(nx, ny, nt_bins).astype(np.float32, copy=False)
    edem = edem1d.reshape(nx, ny, nt_bins).astype(np.float32, copy=False)
    chisq = chisq1d.reshape(nx, ny).astype(np.float32, copy=False)
    _ = elogt1d  # reserved for future error propagation

    return dem, edem, chisq, logT_centers.astype(np.float32, copy=False)


def solve_tile(
    counts6: np.ndarray, *, nmu: Optional[int] = 42, nt: Optional[int] = None
) -> np.ndarray:
    """
    Convenience wrapper around `solve_tile_all(...)` that returns only the DEM cube.

    Parameters
    ----------
    counts6 : (H, W, 6) float-like
    nmu, nt : see `solve_tile_all`.

    Returns
    -------
    dem : np.ndarray, shape (H, W, NT), dtype float32
    """
    dem, _edem, _chisq, _logt = solve_tile_all(counts6, nmu=nmu, nt=nt)
    return dem
