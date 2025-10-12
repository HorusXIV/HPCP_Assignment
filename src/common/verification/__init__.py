# src/common/verification/__init__.py
"""
Verification utilities for comparing DEM results against golden references.

This module provides solver-agnostic comparison functions. Golden generation
is handled by each backend (baseline, GPU, Dask) individually.

Public API
----------
compare_to_golden : Compare computed arrays to a golden NPZ
ComparisonResult : Dataclass holding comparison details
load_golden : Load golden reference from NPZ
"""

from .compare import (
    compare_to_golden,
    ComparisonResult,
    load_golden,
    save_golden,
)

__all__ = [
    "compare_to_golden",
    "ComparisonResult",
    "load_golden",
    "save_golden",
]