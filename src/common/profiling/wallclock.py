# src/common/profiling/wallclock.py
from __future__ import annotations
"""
Simple wall-clock benchmarking utilities for the vendor DEM solver.

This module provides:
  • A thin wrapper (`run_dn2dem`) that prepares inputs and calls the vendor
    solver `dn2dem_pos`
  • A timing helper (`time_one`) that measures a single run and derives DEM/s
  • A streaming benchmark (`benchmark_wallclock`) that sweeps a set of square
    crop sizes with repeats, writing progress to CSV/JSONL/Markdown as it goes

Design notes
------------
- Outputs are written incrementally so partial results are preserved if the job
  is preempted or interrupted.
- File writes use simple durability measures (flush + fsync; atomic rename for
  the summary Markdown).
- The solver itself is treated as a black box and is imported from
  `src.baseline.vendor.dn2dem_pos`. Error-modeling is intentionally minimal.
"""

import csv
import json
import os
import time
import statistics as stats
from pathlib import Path
from typing import Iterable, List, Tuple, Callable, Optional

import numpy as np

# vendor solver entry point (treated as a black box here)
from src.baseline.vendor.dn2dem_pos import dn2dem_pos


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
# Core runner (vendor solver wrapper and single-run timer)
# ---------------------------------------------------------------------------

def run_dn2dem(
    frame_6hw: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    solver_fn: Optional[Callable] = None,
    error_model: str = "sqrt",
    err_a: float = 1.0,
    err_b: float = 1e-6,
):
    """
    Prepare inputs and call the provided solver (defaults to vendor `dn2dem_pos`).

    Parameters
    ----------
    frame_6hw : np.ndarray
        Channel-first frame shaped (6, H, W). It is converted to (H, W, 6)
        for the solver, sanitized to be finite/non-negative, and cast to `dtype`.
    T_RESP : np.ndarray
        Temperature-response matrix (n_tresp, nf).
    T_RESP_LOGT : np.ndarray
        1-D array of log(T) sample positions (length n_tresp).
    TEMPS : np.ndarray
        1-D array of DEM bin edges.
    nmu : int, default 42
        Regularization parameter forwarded to the solver.
    dtype : np.dtype, default np.float32
        Dtype for the finite/clipped input intensities.
    solver_fn : Callable | None, default None
        Solver function with signature compatible to `dn2dem_pos`. If None,
        `dn2dem_pos` is used.
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
    """
    if solver_fn is None:
        solver_fn = dn2dem_pos

    f = np.moveaxis(frame_6hw, 0, -1).astype(dtype, copy=False)  # (H, W, 6)
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0, None)

    if error_model == "sqrt":
        e = np.sqrt(f, dtype=np.float32) + float(err_b)
    elif error_model == "linear":
        e = float(err_a) * f.astype(np.float32, copy=False) + float(err_b)
    elif error_model == "constant":
        e = np.full_like(f, float(err_b), dtype=np.float32)
    else:
        raise ValueError(f"Unknown error_model: {error_model}")

    return solver_fn(f, e, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)


def time_one(
    frame_6hw: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    solver_fn: Optional[Callable] = None,
    error_model: str = "sqrt",
    err_a: float = 1.0,
    err_b: float = 1e-6,
) -> Tuple[float, float]:
    """
    Run the solver once and return (elapsed_seconds, DEMs_per_second).

    Parameters
    ----------
    frame_6hw : np.ndarray
        Input frame shaped (6, H, W).
    T_RESP, T_RESP_LOGT, TEMPS :
        See `run_dn2dem` for details.
    nmu, dtype, solver_fn, error_model, err_a, err_b :
        Forwarded to `run_dn2dem`.

    Returns
    -------
    (float, float)
        Tuple of (elapsed_seconds, pixels_per_second) where the latter is
        computed as (H * W) / elapsed_seconds.
    """
    t0 = time.perf_counter()
    _ = run_dn2dem(
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
    dt = time.perf_counter() - t0
    H, W = frame_6hw.shape[1], frame_6hw.shape[2]
    return dt, (H * W) / dt


# ---------------------------------------------------------------------------
# Wall-clock benchmark (streaming outputs)
# ---------------------------------------------------------------------------

def benchmark_wallclock(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    sizes: Iterable[int] = (14, 64, 256, 1024),
    repeats: int = 5,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    outdir: Path | str = "benchmark_out",
    solver_fn: Optional[Callable] = None,
    error_model: str = "sqrt",
    err_a: float = 1.0,
    err_b: float = 1e-6,
) -> List[dict]:
    """
    Run repeated wall-clock timings over several square crops, streaming results.

    Streaming behavior
    ------------------
    After completing each size:
      1) Append a row to CSV (`baseline_wallclock.csv`)
      2) Append the same row as JSON to a JSONL file (`progress.jsonl`)
      3) Rewrite a compact Markdown summary table (`summary.md`) atomically

    Parameters
    ----------
    STACK : np.ndarray
        Input stack with at least one frame. Only frame 0 is used here.
        Expected shapes are (F, H, W, 6) or (F, 6, H, W) depending on caller.
        (This function indexes as `STACK[0, :, :sz, :sz]`, so the caller should
        pass a stack where slicing this way yields a (6, sz, sz) plane-first view.)
    T_RESP, T_RESP_LOGT, TEMPS :
        See `run_dn2dem`.
    sizes : Iterable[int], default (14, 64, 256, 1024)
        Square crop sizes to benchmark (sz → sz×sz).
    repeats : int, default 5
        Number of timed repetitions per size (after a fixed warm-up).
    nmu, dtype, solver_fn, error_model, err_a, err_b :
        Forwarded to `time_one` / `run_dn2dem`.
    outdir : str | pathlib.Path, default "benchmark_out"
        Output directory for CSV/JSONL/Markdown. Created if missing.

    Returns
    -------
    list[dict]
        Accumulated per-size rows containing means/medians, throughput, etc.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    csv_path = outdir / "baseline_wallclock.csv"
    md_path = outdir / "summary.md"
    jl_path = outdir / "progress.jsonl"  # one JSON object per line, per size

    rows: List[dict] = []

    for sz in sizes:
        # Take a square crop from the first frame; calling code ensures layout.
        frame = STACK[0, :, :sz, :sz]

        # Warm-up runs to stabilize JIT/BLAS/caches
        for __ in range(3):
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
