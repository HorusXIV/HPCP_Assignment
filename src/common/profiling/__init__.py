# src/common/profiling/__init__.py
"""
Public API for profiling & reporting across CPU, Dask, and GPU runs.

This module provides comprehensive profiling utilities for benchmarking
DEMREG solvers across different backends (NumPy, CuPy, Dask).

Main Components
---------------
Profiler : Cross-backend profiling orchestrator
    Main context manager for collecting timings, system metrics, and Dask
    task streams. Use this for all profiling scenarios.

SystemSampler, NVMLSampler : Background metric samplers
    Continuous sampling of system (CPU/memory) and GPU (NVML) metrics.

CUDAProfiler, MultiGPUProfiler : GPU event timing
    High-precision timing using CUDA events for single and multi-GPU scenarios.

Backend utilities : Backend detection and compatibility
    Auto-detect array backends (NumPy/CuPy/Dask) and create appropriate
    synchronization functions for accurate timing.

Wallclock benchmarking : Standardized timing harness
    Backend-agnostic benchmarking with streaming CSV/JSON/Markdown output.

I/O and reporting : Data persistence
    CSV appending, JSON writing, Markdown report generation.

Usage Examples
--------------
Baseline (NumPy):
    from src.common.profiling import Profiler
    with Profiler(None, benchdir, stamp) as prof:
        prof.section("compute", start=True)
        result = solver(...)
        prof.section("compute", start=False)

Single GPU (CuPy):
    from src.common.profiling import Profiler, CUDAProfiler
    with Profiler(None, benchdir, stamp, enable_gpu_sampler=True) as prof:
        gpu_prof = CUDAProfiler(device_id=0)
        gpu_prof.section("compute", start=True)
        result = gpu_solver(...)
        gpu_prof.section("compute", start=False)
        prof.register_gpu_timings(gpu_prof.get_timings())

Multi-GPU:
    from src.common.profiling import Profiler, MultiGPUProfiler
    with Profiler(None, benchdir, stamp, enable_gpu_sampler=True) as prof:
        mgpu_prof = MultiGPUProfiler([0, 1, 2, 3])
        mgpu_prof.section_all("compute", start=True)
        result = multi_gpu_solver(...)
        mgpu_prof.section_all("compute", start=False)
        prof.register_gpu_timings(mgpu_prof.get_summary())

Dask (Multi-Node):
    from src.common.profiling import Profiler
    from dask.distributed import Client
    client = Client()
    with Profiler(client, benchdir, stamp) as prof:
        prof.snapshot_workers("before")
        with prof.compute_context():
            result = client.compute(lazy_result)
            result = result.result()
        prof.snapshot_workers("after")

Backend Detection:
    from src.common.profiling import detect_backend, create_synchronize_fn
    backend = detect_backend(arr)  # Returns Backend.NUMPY, CUPY, or DASK
    sync_fn = create_synchronize_fn(arr)  # Auto-create sync function
"""

# Core profiling
from .profiler import Profiler, simple_profile

# Background samplers
from .samplers import SystemSampler, NVMLSampler

# GPU profiling (CUDA events)
from .gpu_profiling import (
    CUDAProfiler,
    MultiGPUProfiler,
    MemoryTracker,
    MemorySnapshot,
    get_memory_info,
    get_all_memory_info,
)

# Backend utilities
from .backend_utils import (
    Backend,
    detect_backend,
    get_array_module,
    is_gpu_array,
    is_distributed_array,
    get_device_id,
    get_backend_info,
    create_synchronize_fn,
    ensure_numpy,
    format_memory,
    summarize_backend_info,
    check_backend_available,
    get_available_backends,
    get_gpu_count,
    get_backend_versions,
    print_backend_summary,
    validate_backend_compatibility,
    recommend_synchronization,
)

# Wallclock benchmarking
from .wallclock import (
    run_dn2dem,
    time_one,
    benchmark_wallclock,
)

# I/O helpers
from .io_helpers import (
    bench_row as write_bench_row,
    set_bench_outdir,
    flush_bench_csv,
)

# Reporting
from .reporting import (
    write_json,
    write_run_card_md,
)

# Task stream aggregation
from .task_agg import (
    aggregate_task_stream,
    write_task_csv,
    write_agg_csv,
)

# Quality checks
from .checks import basic_checks

# Environment snapshots
from .env import write_env_snapshot

# Public API
__all__ = [
    # Core profiling
    "Profiler",
    "simple_profile",
    # Samplers
    "SystemSampler",
    "NVMLSampler",
    # GPU profiling
    "CUDAProfiler",
    "MultiGPUProfiler",
    "MemoryTracker",
    "MemorySnapshot",
    "get_memory_info",
    "get_all_memory_info",
    # Backend utilities
    "Backend",
    "detect_backend",
    "get_array_module",
    "is_gpu_array",
    "is_distributed_array",
    "get_device_id",
    "get_backend_info",
    "create_synchronize_fn",
    "ensure_numpy",
    "format_memory",
    "summarize_backend_info",
    "check_backend_available",
    "get_available_backends",
    "get_gpu_count",
    "get_backend_versions",
    "print_backend_summary",
    "validate_backend_compatibility",
    "recommend_synchronization",
    # Wallclock benchmarking
    "run_dn2dem",
    "time_one",
    "benchmark_wallclock",
    # I/O
    "write_bench_row",
    "set_bench_outdir",
    "flush_bench_csv",
    # Reporting
    "write_json",
    "write_run_card_md",
    # Task aggregation
    "aggregate_task_stream",
    "write_task_csv",
    "write_agg_csv",
    # Checks
    "basic_checks",
    # Environment
    "write_env_snapshot",
]

# Version info
__version__ = "2.0.0"  # Major refactor: backend-agnostic, fixed Dask, GPU support