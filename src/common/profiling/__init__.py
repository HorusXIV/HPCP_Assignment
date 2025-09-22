# src/common/profiling/__init__.py
"""
Public API for profiling & reporting across CPU, Dask, and (future) GPU runs.

Stable, existing surface:
    from src.common.profiling import (
        Profiler, SystemSampler, NVMLSampler,
        write_bench_row, write_run_card_md, write_json,
        aggregate_task_stream,
    )
"""

# Modern API re-exports (only from files that exist in this repo)
from .profiler import Profiler
from .samplers import SystemSampler, NVMLSampler
from .io_helpers import bench_row as write_bench_row
from .reporting import write_run_card_md, write_json
from .task_agg import aggregate_task_stream

__all__ = [
    "Profiler",
    "SystemSampler",
    "NVMLSampler",
    "write_bench_row",
    "write_run_card_md",
    "write_json",
    "aggregate_task_stream",
]
