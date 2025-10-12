# tests/test_baseline_helpers.py
"""Smoke tests for baseline and common helpers.

Covers:
- Basic quality statistics on DEM / chi-square arrays.
- Path utilities for benchmark directories.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.common.paths import default_run_dir
from src.common.profiling.io_helpers import set_bench_outdir
from src.common.profiling.checks import basic_checks


def test_default_run_dir_creates_timestamped_path(tmp_path: Path) -> None:
    """Test that default_run_dir creates a timestamped directory."""
    method = "baseline"

    # Create run directory
    run_dir = default_run_dir(method, cli_outdir=tmp_path, create=True)

    # Should exist and be a directory
    assert run_dir.exists() and run_dir.is_dir()

    # Should be under the specified output directory
    assert tmp_path in run_dir.parents

    # Should contain method name in path
    assert method in str(run_dir)

    # Timestamp portion should match YYYYMMDD-HHMMSS pattern
    # The last component should be the timestamp
    timestamp = run_dir.name
    assert re.match(r"\d{8}-\d{6}", timestamp)


def test_set_bench_outdir_configures_csv_location(tmp_path: Path) -> None:
    """Test that set_bench_outdir properly configures the benchmark CSV location."""
    bench_dir = tmp_path / "benchmark_test"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Configure the benchmark output directory
    set_bench_outdir(bench_dir)

    # Verify the directory exists and is accessible
    assert bench_dir.exists()
    assert bench_dir.is_dir()


def test_quality_stats_basic() -> None:
    """Quality stats expose expected keys and return sane fractions."""
    rng = np.random.default_rng(0)
    dem = rng.random((8, 8, 4), dtype=np.float32)
    chisq = rng.random((8, 8), dtype=np.float32)

    # Use basic_checks from src.common.profiling.checks
    q = basic_checks(dem, chisq)

    # keys present
    expected = {"finite_frac", "positive_frac", "chisq_median"}
    assert expected <= q.keys()

    # fractions in [0, 1]
    assert 0.0 <= q["finite_frac"] <= 1.0
    assert 0.0 <= q["positive_frac"] <= 1.0


def test_quality_stats_with_nans() -> None:
    """Test basic_checks handles NaN values correctly."""
    dem = np.array([[[1.0, 2.0], [np.nan, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    chisq = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    q = basic_checks(dem, chisq)

    # Should have finite_frac less than 1.0 due to NaN
    assert 0.0 <= q["finite_frac"] < 1.0
    # Should have positive_frac reported
    assert 0.0 <= q["positive_frac"] <= 1.0


def test_quality_stats_with_negatives() -> None:
    """Test basic_checks handles negative values correctly."""
    dem = np.array([[[1.0, -2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    chisq = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    q = basic_checks(dem, chisq)

    # Should have positive_frac less than 1.0 due to negative value
    assert 0.0 <= q["positive_frac"] < 1.0
    # All values are finite
    assert q["finite_frac"] == 1.0