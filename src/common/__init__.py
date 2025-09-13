# src/common/__init__.py
"""
Common utilities shared across baseline, Dask, and GPU variants.

This package re-exports the most frequently used helpers so you can write:
    from src.common import load_np_stack, prepare_synthetic_responses, dn2dem, dem_to_temp_maps
    from src.common import run_baseline_suite  # full profiling/benchmark suite

Modules:
- io.py         : Data loading helpers (NPZ stacks)
- responses.py  : Synthetic (and future real) temperature-response builders
- dem_api.py    : Thin wrapper around the provided DEM solver
- post.py       : Post-processing (DEM → temperature maps)
- profiling.py  : Baseline benchmarking/profiling utilities
"""

# IO
from .io import load_np_stack

# Temperature responses
from .responses import prepare_synthetic_responses

# Solver wrapper
from .dem_api import dn2dem

# Post-processing
from .post import dem_to_temp_maps

# Profiling / benchmarking
from .profiling import (
    set_single_thread_caps,
    run_dn2dem,
    time_one,
    benchmark_wallclock,
    run_cprofile,
    run_line_profiler,
    write_env_snapshot,
    run_baseline_suite,
)

__all__ = (
    # io
    "load_np_stack",
    # responses
    "prepare_synthetic_responses",
    # solver
    "dn2dem",
    # post
    "dem_to_temp_maps",
    # profiling
    "set_single_thread_caps",
    "run_dn2dem",
    "time_one",
    "benchmark_wallclock",
    "run_cprofile",
    "run_line_profiler",
    "write_env_snapshot",
    "run_baseline_suite",
)
