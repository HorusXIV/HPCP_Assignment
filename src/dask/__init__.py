# src/dask/__init__.py
"""
Dask runner shims / legacy entrypoints.
"""

from .runner import run_dask_suite

__all__ = ("run_dask_suite",)
