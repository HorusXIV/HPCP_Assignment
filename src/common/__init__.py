# src/common/__init__.py
"""
Common package initializer. Keep minimal to avoid import cycles.
Expose only stable, shared entry points.
"""

# Canonical data I/O (tiny, stable re-exports)
from .dataio import (
    default_files,
    build_lazy_npz_stack,
    load_np_stack,
    frame_for_solver,
    write_manifest_and_hash,
)

# Canonical profiling surface
from .profiling import (
    Profiler,
    SystemSampler,
    NVMLSampler,
    write_bench_row,
    write_run_card_md,
    write_json,
    aggregate_task_stream,
)

__all__ = [
    # dataio
    "default_files",
    "build_lazy_npz_stack",
    "load_np_stack",
    "frame_for_solver",
    "write_manifest_and_hash",
    # profiling
    "Profiler",
    "SystemSampler",
    "NVMLSampler",
    "write_bench_row",
    "write_run_card_md",
    "write_json",
    "aggregate_task_stream",
]
