"""Lightweight package init for src.dask.

Avoid importing heavy modules (like the cluster runner that pulls in
`dask.distributed`) at import time so that tests importing
`src.dask.tiles` or `src.dask.main` don't require the full dask stack.
"""

# Intentionally do not import .runner here.
__all__ = ()
