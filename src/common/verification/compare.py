# src/common/verification/compare.py
from __future__ import annotations

"""
Solver-agnostic comparison utilities for DEM results.

This module compares computed DEM outputs against golden reference NPZ files.
It does NOT know how to compute DEMs - that's the job of each backend's solver.

Typical usage
-------------
>>> from src.common.verification import compare_to_golden
>>> 
>>> # After computing DEM with any backend:
>>> result = compare_to_golden(
...     golden_npz=Path("data/golden/1024x1024/baseline.npz"),
...     demmap=dem_computed,
...     edemmap=edem_computed,
...     chisq=chisq_computed,
...     logT_bins=logt_computed,
... )
>>> 
>>> if result['passed']:
...     print("✓ Verification passed!")
>>> else:
...     print(f"✗ Failed: {result['summary']}")
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ComparisonResult:
    """Results from comparing a single array."""
    name: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    correlation: float
    shape_match: bool
    notes: list[str]


def load_golden(golden_npz: Path | str) -> dict[str, np.ndarray]:
    """
    Load golden reference arrays from NPZ.

    Parameters
    ----------
    golden_npz : Path | str
        Path to golden NPZ file with keys:
        demmap, edemmap, chisq, logT_bins

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of arrays from the golden file.

    Raises
    ------
    FileNotFoundError
        If golden file doesn't exist.
    KeyError
        If required keys are missing.

    Examples
    --------
    >>> golden = load_golden("data/golden/1024x1024/baseline.npz")
    >>> golden.keys()
    dict_keys(['demmap', 'edemmap', 'chisq', 'logT_bins'])
    """
    golden_path = Path(golden_npz)

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden file not found: {golden_path}")

    data = np.load(golden_path)

    # Validate required keys
    required = ['demmap', 'edemmap', 'chisq', 'logT_bins']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Golden file missing required keys: {missing}. "
            f"Found: {list(data.keys())}"
        )

    return {k: data[k] for k in required}


def save_golden(
        output_npz: Path | str,
        *,
        demmap: np.ndarray,
        edemmap: np.ndarray,
        chisq: np.ndarray,
        logT_bins: np.ndarray,
        metadata: dict | None = None,
) -> None:
    """
    Save arrays as a golden reference NPZ.

    Parameters
    ----------
    output_npz : Path | str
        Output path for golden NPZ file.
    demmap : np.ndarray
        DEM array.
    edemmap : np.ndarray
        DEM error array.
    chisq : np.ndarray
        Chi-squared array.
    logT_bins : np.ndarray
        Temperature bin centers.
    metadata : dict | None
        Optional metadata to save as sidecar JSON.

    Examples
    --------
    >>> save_golden(
    ...     "data/golden/1024x1024/baseline.npz",
    ...     demmap=dem,
    ...     edemmap=edem,
    ...     chisq=chisq,
    ...     logT_bins=logt,
    ...     metadata={'nmu': 42, 'size': '1024x1024'}
    ... )
    """
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.savez_compressed(
        output_path,
        demmap=demmap,
        edemmap=edemmap,
        chisq=chisq,
        logT_bins=logT_bins,
    )

    # Save metadata sidecar if provided
    if metadata is not None:
        json_path = output_path.with_suffix('.json')

        # Add array shapes to metadata
        metadata['shapes'] = {
            'demmap': list(demmap.shape),
            'edemmap': list(edemmap.shape),
            'chisq': list(chisq.shape),
            'logT_bins': list(logT_bins.shape),
        }

        json_path.write_text(
            json.dumps(metadata, indent=2),
            encoding='utf-8'
        )


def _compare_array(
        name: str,
        computed: np.ndarray,
        golden: np.ndarray,
        rtol: float,
        atol: float,
) -> ComparisonResult:
    """
    Compare two arrays with detailed diagnostics.

    Parameters
    ----------
    name : str
        Name of the array being compared.
    computed : np.ndarray
        Computed array.
    golden : np.ndarray
        Golden reference array.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Returns
    -------
    ComparisonResult
        Detailed comparison result.
    """
    notes = []

    # Check shapes
    shape_match = computed.shape == golden.shape
    if not shape_match:
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs_error=np.inf,
            max_rel_error=np.inf,
            mean_abs_error=np.inf,
            correlation=0.0,
            shape_match=False,
            notes=[f"Shape mismatch: {computed.shape} vs {golden.shape}"],
        )

    # Compute errors
    abs_diff = np.abs(computed - golden)

    # Relative error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / (np.abs(golden) + 1e-10)

    # Correlation coefficient
    flat_c = computed.flatten()
    flat_g = golden.flatten()
    valid = np.isfinite(flat_c) & np.isfinite(flat_g)

    if np.sum(valid) > 1:
        try:
            correlation = np.corrcoef(flat_c[valid], flat_g[valid])[0, 1]
        except Exception:
            correlation = 0.0
    else:
        correlation = 0.0
        notes.append("Insufficient valid values for correlation")

    # Check if arrays match within tolerance
    passed = np.allclose(computed, golden, rtol=rtol, atol=atol, equal_nan=True)

    # Additional diagnostics if failed
    if not passed:
        frac_within = np.mean(
            np.isclose(computed, golden, rtol=rtol, atol=atol, equal_nan=True)
        )
        notes.append(f"{frac_within:.1%} of values within tolerance")

        # Find worst pixel
        if abs_diff.size > 0:
            worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            worst_val = abs_diff[worst_idx]
            notes.append(f"Worst pixel at {worst_idx}: diff={worst_val:.3e}")

    return ComparisonResult(
        name=name,
        passed=passed,
        max_abs_error=float(np.nanmax(abs_diff)),
        max_rel_error=float(np.nanmax(rel_diff)),
        mean_abs_error=float(np.nanmean(abs_diff)),
        correlation=float(correlation),
        shape_match=True,
        notes=notes,
    )


def compare_to_golden(
        golden_npz: Path | str,
        *,
        demmap: np.ndarray,
        edemmap: np.ndarray,
        chisq: np.ndarray,
        logT_bins: np.ndarray | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        check_logt: bool = True,
) -> dict:
    """
    Compare computed DEM results to golden reference.

    Parameters
    ----------
    golden_npz : Path | str
        Path to golden NPZ file.
    demmap : np.ndarray
        Computed DEM array.
    edemmap : np.ndarray
        Computed DEM error array.
    chisq : np.ndarray
        Computed chi-squared array.
    logT_bins : np.ndarray | None
        Computed temperature bin centers (optional).
    rtol : float, default 1e-4
        Relative tolerance for np.allclose.
    atol : float, default 1e-6
        Absolute tolerance for np.allclose.
    check_logt : bool, default True
        Whether to compare logT_bins (disable if not provided).

    Returns
    -------
    dict
        Verification results:
        {
            'passed': bool,              # Overall pass/fail
            'results': list[ComparisonResult],  # Per-array details
            'summary': str,              # Human-readable summary
            'golden_path': str,          # Path to golden used
        }

    Examples
    --------
    >>> result = compare_to_golden(
    ...     golden_npz="data/golden/1024x1024/baseline.npz",
    ...     demmap=dem,
    ...     edemmap=edem,
    ...     chisq=chisq,
    ...     logT_bins=logt,
    ... )
    >>> if result['passed']:
    ...     print("✓ Verification passed")
    >>> else:
    ...     for r in result['results']:
    ...         if not r.passed:
    ...             print(f"  {r.name}: FAILED (max_err={r.max_abs_error:.3e})")
    """
    # Load golden reference
    golden = load_golden(golden_npz)

    # Compare each array
    results = []

    results.append(_compare_array('demmap', demmap, golden['demmap'], rtol, atol))
    results.append(_compare_array('edemmap', edemmap, golden['edemmap'], rtol, atol))
    results.append(_compare_array('chisq', chisq, golden['chisq'], rtol, atol))

    if check_logt and logT_bins is not None:
        results.append(_compare_array(
            'logT_bins', logT_bins, golden['logT_bins'], rtol, atol
        ))

    # Overall pass/fail
    passed = all(r.passed for r in results)

    # Generate summary
    if passed:
        summary = "All arrays match within tolerance"
    else:
        failed = [r.name for r in results if not r.passed]
        summary = f"Mismatch in: {', '.join(failed)}"

    return {
        'passed': passed,
        'results': results,
        'summary': summary,
        'golden_path': str(golden_npz),
        'tolerances': {'rtol': rtol, 'atol': atol},
    }


def print_comparison_report(result: dict) -> None:
    """
    Print human-readable comparison report.

    Parameters
    ----------
    result : dict
        Output from compare_to_golden.

    Examples
    --------
    >>> result = compare_to_golden(...)
    >>> print_comparison_report(result)
    ✓ Verification PASSED
      demmap  : ✓ (max_err=1.23e-07, corr=1.000000)
      edemmap : ✓ (max_err=4.56e-08, corr=0.999998)
      chisq   : ✓ (max_err=2.34e-06, corr=0.999995)
    """
    passed = result['passed']
    status = "✓ PASSED" if passed else "✗ FAILED"

    print(f"\nVerification: {status}")
    print(f"Golden: {result['golden_path']}")
    print(f"Tolerances: rtol={result['tolerances']['rtol']}, atol={result['tolerances']['atol']}")
    print("\nPer-array results:")

    for r in result['results']:
        if isinstance(r, ComparisonResult):
            status_icon = "✓" if r.passed else "✗"
            print(
                f"  {r.name:10s}: {status_icon} "
                f"(max_err={r.max_abs_error:.3e}, corr={r.correlation:.6f})"
            )
            for note in r.notes:
                print(f"             {note}")
        else:
            # Handle dict format for backwards compatibility
            status_icon = "✓" if r.get('passed', False) else "✗"
            print(f"  {r.get('name', 'unknown'):10s}: {status_icon}")

    print(f"\nSummary: {result['summary']}\n")


def save_comparison_report(result: dict, output_json: Path | str) -> None:
    """
    Save comparison report as JSON.

    Parameters
    ----------
    result : dict
        Output from compare_to_golden.
    output_json : Path | str
        Output path for JSON report.

    Examples
    --------
    >>> result = compare_to_golden(...)
    >>> save_comparison_report(result, "verification_report.json")
    """
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert ComparisonResult dataclasses to dicts
    serializable = dict(result)
    serializable['results'] = [
        asdict(r) if isinstance(r, ComparisonResult) else r
        for r in result['results']
    ]

    output_path.write_text(
        json.dumps(serializable, indent=2),
        encoding='utf-8'
    )