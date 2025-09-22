# src/common/profiling/io_helpers.py
from __future__ import annotations
"""
CSV I/O helpers for benchmark bookkeeping.

This module centralizes how benchmark rows are appended to a well-known CSV
(`bench.csv`) inside a configurable output directory. The destination can be:

  1) Set at import time via the BENCH_OUTDIR environment variable, or
  2) Overridden at runtime by calling `set_bench_outdir(path)`.

Behavior
--------
- Appends rows with a header written on first write.
- Flushes and attempts to `fsync` for durability (best-effort).
- Field order is taken from the keys of the first appended row.
"""

import csv
import os
from pathlib import Path
from typing import Dict, Any

# Default sink (can be overridden with set_bench_outdir or BENCH_OUTDIR)
_BENCH_OUTDIR = Path(os.environ.get("BENCH_OUTDIR", "benchmark_out"))


def set_bench_outdir(path: str | Path) -> None:
    """
    Set the output directory used for benchmark CSVs.

    The directory will contain (at minimum):
      - bench.csv

    Parameters
    ----------
    path : str | Path
        Destination directory. Created if missing.
    """
    global _BENCH_OUTDIR
    _BENCH_OUTDIR = Path(path)
    _BENCH_OUTDIR.mkdir(parents=True, exist_ok=True)


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    """
    Append a single dictionary row to a CSV file, creating a header if needed.

    Parameters
    ----------
    path : Path
        Target CSV path.
    row : dict
        Mapping of column -> value. The first row written defines the header
        (column order). Subsequent rows are written with the same fieldnames.

    Notes
    -----
    - This function flushes and attempts to fsync for durability.
    - If fsync is not available on the filesystem, the exception is ignored.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # Best-effort durability; ignore on unsupported filesystems.
            pass


def bench_row(**kw) -> None:
    """
    Append a row to `bench.csv` in the configured benchmark output directory.

    Parameters
    ----------
    **kw :
        Arbitrary key/value pairs composing one CSV record.
    """
    _BENCH_OUTDIR.mkdir(parents=True, exist_ok=True)
    row = dict(kw)
    _append_row(_BENCH_OUTDIR / "bench.csv", row)


def flush_bench_csv() -> None:
    """
    No-op placeholder for API parity.

    Rows are written immediately; this function exists to match interfaces
    where explicit flushing might be required.
    """
    return
