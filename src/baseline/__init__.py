# src/baseline/__init__.py
"""
Baseline DEM solver using vendorized DEMREG code.

This module provides the ground truth implementation for DEM solving,
serving as the reference for correctness verification and performance
comparison of optimized implementations.

Main Components
---------------
solve_dem : Primary solver interface
    High-level interface with validation and error handling.

baseline_solver_fn : Wallclock-compatible solver
    Thin wrapper for use with profiling.wallclock.benchmark_wallclock().

prepare_inputs : Input preparation
    Sanitize and prepare inputs for vendor solver.

Validation functions : Input/output checking
    Comprehensive validation for debugging and correctness.

Usage
-----
Simple solve:
    from src.baseline import solve_dem
    
    demmap, edemmap, logt, chisq, dn_reg = solve_dem(
        data_6hw,      # (6, H, W) channels-first
        tresp,         # (n_tresp, 6)
        tresp_logt,    # (n_tresp,)
        temps,         # (n_temps,)
    )

With validation:
    result = solve_dem(
        data_6hw, tresp, tresp_logt, temps,
        validate_inputs=True,
        validate_outputs=True,
    )

For benchmarking:
    from src.baseline import baseline_solver_fn
    from src.common.profiling import benchmark_wallclock
    
    results = benchmark_wallclock(
        STACK, T_RESP, T_RESP_LOGT, TEMPS,
        solver_fn=baseline_solver_fn,
        sizes=[64, 256, 1024],
    )

Running the baseline:
    # Command line interface
    python -m src.baseline.run --benchmark --sizes 64,256,1024
    
    # Or in code
    from src.baseline.run import run_benchmark
    results = run_benchmark(STACK, T_RESP, T_RESP_LOGT, TEMPS, ...)

Notes
-----
- The baseline uses vendorized DEMREG code (src.baseline.vendor)
- All inputs are automatically sanitized (NaN->0, negative->0)
- The solver expects data in channels-first format (6, H, W)
- The vendor code is NOT thread-safe; use separate processes for parallelism

Performance
-----------
Typical performance on modern CPU (single core):
  - 64x64:   ~0.1 seconds
  - 256x256: ~1-2 seconds
  - 1024x1024: ~15-30 seconds
  - 4096x4096: ~5-10 minutes

The baseline is intentionally unoptimized to serve as a reference.
Optimized implementations (GPU, vectorized, Dask) should provide
significant speedups while maintaining correctness.
"""

from .solver import (
    solve_dem,
    baseline_solver_fn,
    prepare_inputs,
    validate_input_shapes,
    validate_input_values,
    validate_output_shapes,
    validate_output_values,
)

__all__ = [
    "solve_dem",
    "baseline_solver_fn",
    "prepare_inputs",
    "validate_input_shapes",
    "validate_input_values",
    "validate_output_shapes",
    "validate_output_values",
]

__version__ = "1.0.0"