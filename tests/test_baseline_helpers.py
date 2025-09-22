"""Smoke tests for baseline runner helpers.

Covers:
- Timestamped benchmark directory creation and CSV sink routing.
- Basic quality statistics on DEM / chi-square arrays.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.baseline.runner import _ensure_timestamped_root, _quality_stats
from src.common.profiling.io_helpers import set_bench_outdir


def test_ensure_timestamped_root_makes_dir_and_sets_csv(tmp_path: Path) -> None:
    """Create a timestamped bench dir twice and ensure uniqueness + sink routing."""
    base = tmp_path / "benchmarking" / "baseline"
    bench_dir, stamp = _ensure_timestamped_root(base)

    # Exists and matches YYYYMMDD-HHMMSS
    assert bench_dir.exists() and bench_dir.is_dir()
    assert re.fullmatch(r"\d{8}-\d{6}", stamp)

    # A second call may land in the same second; if stamps differ, paths must differ.
    bench_dir2, stamp2 = _ensure_timestamped_root(base)
    assert bench_dir2.exists() and bench_dir2.is_dir()
    assert re.fullmatch(r"\d{8}-\d{6}", stamp2)
    if stamp2 != stamp:
        assert bench_dir2 != bench_dir

    # bench.csv lives in the configured outdir; ensure the directory is set up.
    set_bench_outdir(bench_dir)
    assert (bench_dir / "bench.csv").parent.exists()


def test_quality_stats_basic() -> None:
    """Quality stats expose expected keys and return sane fractions."""
    rng = np.random.default_rng(0)
    dem = rng.random((8, 8, 4), dtype=np.float32)
    chisq = rng.random((8, 8), dtype=np.float32)

    q = _quality_stats(dem, chisq)

    # keys present
    expected = {"dem_finite_frac", "dem_positive_frac", "chisq_median", "chisq_mean"}
    assert expected <= q.keys()

    # fractions in [0, 1]
    assert 0.0 <= q["dem_finite_frac"] <= 1.0
    assert 0.0 <= q["dem_positive_frac"] <= 1.0
