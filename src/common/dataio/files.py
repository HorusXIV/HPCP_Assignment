# src/common/dataio/files.py
from __future__ import annotations
"""
File discovery helpers for NPZ-based datasets.

This module provides a single convenience function, `default_files`, which
collects input files from a directory using one or more glob patterns and
returns a unique, sorted list of `Path` objects.

Notes
-----
- Matching is **non-recursive** (uses `Path.glob`, not `rglob`).
- Results are deduplicated across patterns and sorted lexicographically.
"""

from pathlib import Path
from typing import List, Sequence, Union


def default_files(
    directory: Union[str, Path],
    ext: Union[str, Sequence[str]] = "*.npz",
) -> List[Path]:
    """
    Discover input files in a directory.

    Parameters
    ----------
    directory : str | Path
        Folder to search.
    ext : str | Sequence[str], default "*.npz"
        Glob pattern(s). Examples: "*.npz" or ["*.npz", "*.npz.gz"].
        Matching is non-recursive.

    Returns
    -------
    list[pathlib.Path]
        Unique, lexicographically sorted list of matching files.
    """
    root = Path(directory) if directory is not None else Path(".")
    patterns = [ext] if isinstance(ext, str) else list(ext)
    out: List[Path] = []
    for pat in patterns:
        out.extend(p for p in root.glob(pat) if p.is_file())
    # unique + sorted
    return sorted(set(out))
