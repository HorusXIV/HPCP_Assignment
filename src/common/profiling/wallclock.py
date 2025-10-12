# src/common/profiling/wallclock.py
from __future__ import annotations

"""
Backend-agnostic wall-clock benchmarking utilities for DEM solvers.

This module provides:
  • A backend-agnostic wrapper (`run_dn2dem`) that prepares inputs and calls
    any compatible solver (NumPy, CuPy, or Dask-based)
  • A timing helper (`time_one`) that measures a single run and derives DEM/s
  • A streaming benchmark (`benchmark_wallclock`) that sweeps a set of square
    crop sizes with repeats, writing progress to CSV/JSONL/Markdown as it goes

Design notes
------------
- Outputs are written incrementally so partial results are preserved if the job
  is preempted or interrupted.
- File writes use simple durability measures (flush + fsync; atomic rename for
  the summary Markdown).
- The solver is backend-agnostic and passed as a parameter.
- Error-modeling is intentionally minimal.

Backend Support
---------------
Automatically detects and supports:
  • NumPy arrays (CPU baseline)
  • CuPy arrays (single/multi GPU)
  • Dask arrays (distributed)
"""

import csv
import json
import os
import time
import statistics as stats
from pathlib import Path
from typing import Iterable, List, Tuple, Callable, Optional, Any

import numpy as np

from .backend_utils import get_array_module as _get_array_module


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    Write text to a file atomically: create `<path>.tmp`, fsync, then rename.

    Parameters
    ----------
    path : pathlib.Path
        Destination path for the final file.
    text : str
        File contents to write.
    encoding : str, default "utf-8"
        Text encoding.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)  # atomic rename on POSIX


def _append_row_csv(csv_path: Path, row: dict) -> None:
    """
    Append one row to a CSV file, creating the header if needed, then fsync.

    Parameters
    ----------
    csv_path : pathlib.Path
        Destination CSV path.
    row : dict
        Mapping of column -> value. Keys define header order on first write.
    """
    csv_path = Path(csv_path)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists or csv_path.stat().st_size == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _rewrite_summary_md(md_path: Path, rows: List[dict]) -> None:
    """
    Rewrite a compact Markdown summary table from accumulated rows, atomically.

    Parameters
    ----------
    md_path : pathlib.Path
        Destination Markdown file path.
    rows : list[dict]
        Accumulated benchmark rows as produced by `benchmark_wallclock`.
    """
    lines = [
        "| Size | repeats | time_median [s] | DEMs/s (median) | dtype | nmu |",
        "|------|---------|-----------------|------------------|-------|-----|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['repeats']} | {r['time_median']:.3f} | "
            f"{r['dems_per_s_median']:.0f} | {r['dtype']} | {r['nmu']} |"
        )
    _atomic_write_text(md_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Core runner (backend-agnostic solver wrapper and single-run timer)
# ---------------------------------------------------------------------------

def run_dn2dem(
        frame_6hw,  # Can be np.ndarray, cp.ndarray, or da.Array
        T_RESP,
        T_RESP_LOGT,
        TEMPS,
        *,
        nmu: int = 42,
        dtype=None,
        solver_fn: Callable,
        error_model: str = "sqrt",
        err_a: float = 1.0,
        err_b: float = 1e-6,
):
    """
    Prepare inputs and call the provided solver (backend-agnostic).

    Parameters
    ----------
    frame_6hw : array-like
        Channel-first frame shaped (6, H, W). Can be NumPy, CuPy, or Dask array.
        It is converted to (H, W, 6) for the solver, sanitized to be
        finite/non-negative, and cast to `dtype`.
    T_RESP : array-like
        Temperature-response matrix (n_tresp, nf).
    T_RESP_LOGT : array-like
        1-D array of log(T) sample positions (length n_tresp).
    TEMPS : array-like
        1-D array of DEM bin edges.
    nmu : int, default 42
        Regularization parameter forwarded to the solver.
    dtype : dtype-like, default None
        Dtype for the finite/clipped input intensities. If None, uses
        float32 from the detected backend.
    solver_fn : Callable
        Solver function with signature compatible to `dn2dem_pos`.
        **REQUIRED** - no default solver to maintain backend independence.
    error_model : {"sqrt", "linear", "constant"}, default "sqrt"
        How to build the per-pixel/channel uncertainties `e` from `f`.
    err_a : float, default 1.0
        Coefficient for "linear" error model (e = a*f + b).
    err_b : float, default 1e-6
        Base term for error models.

    Returns
    -------
    tuple
        (demmap, edemmap, logT_bins, chisq, dn_reg) as provided by the solver.

    Notes
    -----
    - Backend (numpy/cupy/dask) is auto-detected from `frame_6hw` type.
    - All array operations use the detected backend's functions.
    """
    # Detect backend
    xp = _get_array_module(frame_6hw)

    # Set default dtype if not provided
    if dtype is None:
        dtype = xp.float32

    # Move channels: (6, H, W) -> (H, W, 6)
    f = xp.moveaxis(frame_6hw, 0, -1)

    # Cast to target dtype
    if hasattr(f, 'astype'):
        f = f.astype(dtype, copy=False)

    # Sanitize: remove NaN/Inf and clip to non-negative
    if hasattr(xp, 'nan_to_num'):
        f = xp.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    elif hasattr(xp, 'isfinite'):  # Dask fallback
        # For Dask, replace non-finite with zero
        f = xp.where(xp.isfinite(f), f, 0.0)

    # Clip to non-negative
    if hasattr(xp, 'clip'):
        f = xp.clip(f, 0, None)
    elif hasattr(xp, 'maximum'):  # Alternative for some backends
        f = xp.maximum(f, 0)

    # Build error array based on error model
    if error_model == "sqrt":
        e = xp.sqrt(f)
        if hasattr(e, 'astype'):
            e = e.astype(xp.float32, copy=False)
        e = e + float(err_b)
    elif error_model == "linear":
        if hasattr(f, 'astype'):
            f_float = f.astype(xp.float32, copy=False)
        else:
            f_float = f
        e = float(err_a) * f_float + float(err_b)
    elif error_model == "constant":
        if hasattr(xp, 'full_like'):
            e = xp.full_like(f, float(err_b), dtype=xp.float32)
        else:
            # Fallback for backends without full_like
            e = xp.ones_like(f) * float(err_b)
            if hasattr(e, 'astype'):
                e = e.astype(xp.float32, copy=False)
    else:
        raise ValueError(f"Unknown error_model: {error_model}")

    return solver_fn(f, e, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)


def time_one(
        frame_6hw,
        T_RESP,
        T_RESP_LOGT,
        TEMPS,
        *,
        nmu: int = 42,
        dtype=None,
        solver_fn: Callable,
        error_model: str = "sqrt",
        err_a: float = 1.0,
        err_b: float = 1e-6,
        synchronize_fn: Optional[Callable] = None,
) -> Tuple[float, float]:
    """
    Run the solver once and return (elapsed_seconds, DEMs_per_second).

    Parameters
    ----------
    frame_6hw : array-like
        Input frame shaped (6, H, W). Can be NumPy, CuPy, or Dask array.
    T_RESP, T_RESP_LOGT, TEMPS :
        See `run_dn2dem` for details.
    nmu, dtype, solver_fn, error_model, err_a, err_b :
        Forwarded to `run_dn2dem`.
    synchronize_fn : Callable | None, optional
        Optional synchronization function to call before stopping the timer.
        For GPU: pass `cupy.cuda.Device().synchronize`
        For Dask: pass `result.compute` or similar
        For NumPy: None (no synchronization needed)

    Returns
    -------
    (float, float)
        Tuple of (elapsed_seconds, pixels_per_second) where the latter is
        computed as (H * W) / elapsed_seconds.

    Notes
    -----
    - For GPU benchmarks, always provide `synchronize_fn` to ensure accurate timing.
    - For Dask benchmarks, synchronization may trigger computation.
    """
    t0 = time.perf_counter()

    result = run_dn2dem(
        frame_6hw,
        T_RESP,
        T_RESP_LOGT,
        TEMPS,
        nmu=nmu,
        dtype=dtype,
        solver_fn=solver_fn,
        error_model=error_model,
        err_a=err_a,
        err_b=err_b,
    )

    # Synchronize if needed (GPU or Dask)
    if synchronize_fn is not None:
        synchronize_fn()

    dt = time.perf_counter() - t0

    # Get spatial dimensions (backend-agnostic)
    shape = frame_6hw.shape
    H, W = shape[1], shape[2]  # Assumes (6, H, W)

    return dt, (H * W) / dt


# ---------------------------------------------------------------------------
# Wall-clock benchmark (streaming outputs)
# ---------------------------------------------------------------------------

def benchmark_wallclock(
        STACK,  # Can be np.ndarray, cp.ndarray, or da.Array
        T_RESP,
        T_RESP_LOGT,
        TEMPS,
        *,
        sizes: Iterable[int] = (14, 64, 256, 1024),
        repeats: int = 5,
        nmu: int = 42,
        dtype=None,
        outdir: Path | str = "benchmark_out",
        solver_fn: Callable,
        error_model: str = "sqrt",
        err_a: float = 1.0,
        err_b: float = 1e-6,
        synchronize_fn: Optional[Callable] = None,
        warmup_runs: int = 3,
) -> List[dict]:
    """
    Run repeated wall-clock timings over several square crops, streaming results.

    Streaming behavior
    ------------------
    After completing each size:
      1) Append a row to CSV (`wallclock.csv`)
      2) Append the same row as JSON to a JSONL file (`progress.jsonl`)
      3) Rewrite a compact Markdown summary table (`summary.md`) atomically

    Parameters
    ----------
    STACK : array-like
        Input stack with at least one frame. Only frame 0 is used here.
        Expected shape: (F, 6, H, W) - channels-first.
        Can be NumPy, CuPy, or Dask array.
    T_RESP, T_RESP_LOGT, TEMPS :
        See `run_dn2dem`.
    sizes : Iterable[int], default (14, 64, 256, 1024)
        Square crop sizes to benchmark (sz → sz×sz).
    repeats : int, default 5
        Number of timed repetitions per size (after warm-up).
    nmu, dtype, solver_fn, error_model, err_a, err_b :
        Forwarded to `time_one` / `run_dn2dem`.
    outdir : str | pathlib.Path, default "benchmark_out"
        Output directory for CSV/JSONL/Markdown. Created if missing.
    synchronize_fn : Callable | None, optional
        Synchronization function for accurate timing (see `time_one`).
    warmup_runs : int, default 3
        Number of warm-up runs before timing (for JIT, cache warming, etc.).

    Returns
    -------
    list[dict]
        Accumulated per-size rows containing means/medians, throughput, etc.

    Notes
    -----
    - Only the first frame (STACK[0]) is used.
    - Backend is auto-detected from STACK type.
    - For GPU: provide `synchronize_fn` for accurate timing.
    - For Dask: may need special handling (compute triggers).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Detect backend
    xp = _get_array_module(STACK)

    # Set default dtype
    if dtype is None:
        dtype = xp.float32

    # Warn if multiple frames present (only the first is used)
    try:
        nframes = int(STACK.shape[0])
        if nframes > 1:
            (outdir / "WARNING_multiple_frames.txt").write_text(
                f"NOTE: STACK contains {nframes} frames; benchmark uses only the first (index 0).\n",
                encoding="utf-8",
            )
    except Exception:
        # If shape probing fails, proceed silently.
        pass

    csv_path = outdir / "wallclock.csv"
    md_path = outdir / "summary.md"
    jl_path = outdir / "progress.jsonl"  # one JSON object per line, per size

    rows: List[dict] = []

    for sz in sizes:
        # Take a square crop from the first frame: (6, sz, sz)
        frame = STACK[0, :, :sz, :sz]

        # Warm-up runs to stabilize JIT/BLAS/caches
        for _ in range(warmup_runs):
            _ = time_one(
                frame,
                T_RESP,
                T_RESP_LOGT,
                TEMPS,
                nmu=nmu,
                dtype=dtype,
                solver_fn=solver_fn,
                error_model=error_model,
                err_a=err_a,
                err_b=err_b,
                synchronize_fn=synchronize_fn,
            )

        # Timed repeats
        dts, thr = [], []
        for _ in range(repeats):
            dt, dems = time_one(
                frame,
                T_RESP,
                T_RESP_LOGT,
                TEMPS,
                nmu=nmu,
                dtype=dtype,
                solver_fn=solver_fn,
                error_model=error_model,
                err_a=err_a,
                err_b=err_b,
                synchronize_fn=synchronize_fn,
            )
            dts.append(dt)
            thr.append(dems)

        row = dict(
            size=f"{sz}x{sz}",
            H=sz,
            W=sz,
            nf=int(frame.shape[0]),
            repeats=repeats,
            time_mean=float(np.mean(dts)),
            time_std=float(np.std(dts)),
            time_median=float(stats.median(dts)),
            dems_per_s_mean=float(np.mean(thr)),
            dems_per_s_median=float(stats.median(thr)),
            nmu=int(nmu),
            dtype=str(dtype),
            backend=xp.__name__,  # Record backend used
        )
        rows.append(row)

        # --- STREAMING OUTPUTS ---
        # 1) CSV append
        _append_row_csv(csv_path, row)

        # 2) JSONL append (progress log)
        with jl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(row) + "\n")
            jf.flush()
            os.fsync(jf.fileno())

        # 3) Rewrite MD summary atomically from all rows so far
        _rewrite_summary_md(md_path, rows)

    return rows