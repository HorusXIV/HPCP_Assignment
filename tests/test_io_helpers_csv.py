"""Smoke test for bench.csv writing helpers.

Verifies that:
- `set_bench_outdir` redirects CSV output to a temporary directory, and
- `bench_row` appends a row with expected fields to `bench.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.common.profiling.io_helpers import bench_row, set_bench_outdir


def test_bench_csv_append(tmp_path: Path) -> None:
    """`bench_row` should create/append to bench.csv in the configured outdir."""
    set_bench_outdir(tmp_path)

    # minimal row — types intentionally simple to keep IO path exercised
    bench_row(stamp="x", mode="baseline-cpu", frames=1, H=8, W=8)  # type: ignore[arg-type]

    p = tmp_path / "bench.csv"
    assert p.exists()

    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows, "bench.csv should contain at least one row"
    assert rows[0]["mode"] == "baseline-cpu"
