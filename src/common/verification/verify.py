# src/common/verify.py
from __future__ import annotations
"""
Lightweight helpers to save and compare DEM results against a "golden".

This module provides:
- `save_golden`: persist arrays to a compressed NPZ and write a JSON sidecar
  containing a SHA-256 of the NPZ (useful for provenance).
- `compare_to_golden`: load a reference NPZ and compare arrays with tolerances.
  The `chisq` comparison supports modes:
    * "exact" — strict elementwise tolerance check
    * "auto"  — if strict check fails, attempt to match up to an additive or
                multiplicative constant (useful when chisq differs by a constant)
    * "skip"  — do not compare chisq

Return shapes:
- `save_golden` returns the SHA-256 hex digest of the written NPZ.
- `compare_to_golden` returns a tuple `(ok, reports)` where `ok` is a boolean and
  `reports` is a list of per-array comparison dicts.
"""

from pathlib import Path
import hashlib
import json
from typing import Any, Dict, Tuple

import numpy as np

DEFAULT_TOLS: Dict[str, float] = dict(rtol=1e-4, atol=1e-6)


def save_golden(
    path: str | Path,
    *,
    demmap: np.ndarray,
    edemmap: np.ndarray,
    chisq: np.ndarray,
    logT_bins: np.ndarray | None = None,
    meta: Dict[str, Any],
) -> str:
    """
    Write a golden NPZ (demmap, edemmap, chisq, logT_bins) and a JSON sidecar.

    The JSON sidecar contains the provided `meta` dict plus a `sha256` field
    computed from the exact NPZ bytes (helps detect accidental edits).

    Parameters
    ----------
    path : str | pathlib.Path
        Destination for the NPZ file (e.g., ".../baseline.npz").
    demmap, edemmap, chisq : np.ndarray
        Arrays to persist under standardized keys.
    logT_bins : np.ndarray | None, default None
        Optional 1D temperature bin centers.
    meta : dict
        Arbitrary metadata to include in the sidecar; `sha256` is added.

    Returns
    -------
    str
        Hex-encoded SHA-256 of the NPZ content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, demmap=demmap, edemmap=edemmap, chisq=chisq, logT_bins=logT_bins
    )
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    meta_out = dict(meta, sha256=sha)
    (path.with_suffix(".json")).write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    return sha


def _array_report(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """
    Compare two arrays with `np.allclose` and basic error summaries.

    Returns
    -------
    dict
        {
          "name": str,
          "equal": bool,
          "max_abs": float,
          "mean_abs": float,
          "frac_within_tol": float
        }
    """
    same = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    diff = np.abs(a - b)
    frac_ok = float(np.mean(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)))
    return {
        "name": name,
        "equal": bool(same),
        "max_abs": float(np.nanmax(diff)),
        "mean_abs": float(np.nanmean(diff)),
        "frac_within_tol": frac_ok,
    }


def _chisq_report(
    a: np.ndarray,
    b: np.ndarray,
    mode: str,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """
    Compare `chisq` arrays with optional conventions.

    Modes
    -----
    exact :
        Strict `allclose`.
    auto :
        If strict check fails, attempt to match by either an additive constant
        (b ≈ a + c) or a multiplicative factor (b ≈ s * a) determined from
        medians (robust to outliers). If either succeeds, mark as equal and
        annotate the note.
    skip :
        Do not compare; return an always-equal report.

    Returns
    -------
    dict
        Same structure as `_array_report`, possibly with a `"note"`.
    """
    if mode == "skip":
        return {"name": "chisq", "equal": True, "note": "skipped"}

    # direct check first
    rep = _array_report("chisq", a, b, rtol, atol)
    if rep["equal"] or mode == "exact":
        return rep

    # Try additive constant: b ≈ a + c
    c = np.nanmedian(b - a)
    if np.allclose(a + c, b, rtol=rtol, atol=atol, equal_nan=True):
        r = _array_report("chisq (additive-adjusted)", a + c, b, rtol, atol)
        r["equal"] = True
        r["note"] = f"matched with additive constant c≈{float(c):.6g}"
        return r

    # Try multiplicative constant: b ≈ s * a
    mask = np.isfinite(a) & (np.abs(a) > 1e-12)
    if mask.any():
        s = np.nanmedian(b[mask] / a[mask])
        if np.allclose(a * s, b, rtol=rtol, atol=atol, equal_nan=True):
            r = _array_report("chisq (scale-adjusted)", a * s, b, rtol, atol)
            r["equal"] = True
            r["note"] = f"matched with scale s≈{float(s):.6g}"
            return r

    rep["note"] = "mismatch; not fixed by additive or multiplicative constant"
    return rep


def compare_to_golden(
    golden_npz: str | Path,
    *,
    demmap: np.ndarray,
    edemmap: np.ndarray,
    chisq: np.ndarray | None,
    logT_bins: np.ndarray | None = None,
    rtol: float = DEFAULT_TOLS["rtol"],
    atol: float = DEFAULT_TOLS["atol"],
    chisq_mode: str = "auto",
) -> Tuple[bool, list[Dict[str, Any]]]:
    """
    Compare computed arrays to those stored in a golden NPZ.

    Parameters
    ----------
    golden_npz : str | pathlib.Path
        Path to reference NPZ with keys: demmap, edemmap, chisq, logT_bins.
    demmap, edemmap : np.ndarray
        Computed arrays to compare.
    chisq : np.ndarray | None
        Computed chi^2 map (optional; comparison can be skipped/relaxed via `chisq_mode`).
    logT_bins : np.ndarray | None, default None
        Computed temperature bin centers.
    rtol, atol : float
        Tolerances for `np.allclose`.
    chisq_mode : {"auto", "exact", "skip"}
        Comparison policy for chisq.

    Returns
    -------
    (ok, reports) : (bool, list[dict])
        - ok: True if all requested comparisons passed
        - reports: per-array comparison details
    """
    g = np.load(golden_npz)
    reports: list[Dict[str, Any]] = []

    reports.append(_array_report("demmap", demmap, g["demmap"], rtol, atol))
    reports.append(_array_report("edemmap", edemmap, g["edemmap"], rtol, atol))

    if "chisq" in g and chisq is not None:
        reports.append(_chisq_report(chisq, g["chisq"], chisq_mode, rtol, atol))

    if "logT_bins" in g and logT_bins is not None:
        reports.append(_array_report("logT_bins", logT_bins, g["logT_bins"], rtol, atol))

    ok = all(r.get("equal", True) for r in reports)
    return ok, reports
