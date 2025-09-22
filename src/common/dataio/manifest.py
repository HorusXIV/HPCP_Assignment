# src/common/dataio/manifest.py
from __future__ import annotations
"""
Manifest helpers for benchmarking artifacts.

This module provides utilities to:
  • Compute a stable SHA-256 hash over a sequence of strings
  • Write a newline-separated manifest file for a benchmark run and return
    both the file path and the corresponding hash

Conventions
-----------
- Manifests are written as UTF-8 text with one basename per line and a
  trailing newline.
- The hash is computed over the exact sequence of lines (each terminated by
  '\n') to match the file content semantics.
"""

import hashlib
from pathlib import Path
from typing import Sequence, Tuple


def sha256_of_strings(items: Sequence[str]) -> str:
    """
    Compute a SHA-256 hex digest over a sequence of strings.

    Each string is encoded as UTF-8 and a newline ('\\n') is appended between
    items, mirroring the on-disk manifest format.

    Parameters
    ----------
    items : Sequence[str]
        The strings to hash, in order.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_manifest_and_hash(
    basenames: Sequence[str], out_dir: Path, stamp: str
) -> Tuple[Path, str]:
    """
    Write a manifest of basenames and return its path and SHA-256 hash.

    The manifest is written to:
        <out_dir>/manifest_<stamp>.txt

    Each line contains a single basename, and the file ends with a trailing
    newline. The returned hash matches the content produced (same newline
    convention) via `sha256_of_strings`.

    Parameters
    ----------
    basenames : Sequence[str]
        Ordered list of basenames to record in the manifest.
    out_dir : pathlib.Path
        Output directory; created if it does not exist.
    stamp : str
        Timestamp or unique run identifier included in the filename.

    Returns
    -------
    (pathlib.Path, str)
        Tuple of (manifest_path, sha256_hex_digest).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"manifest_{stamp}.txt"
    txt.write_text("\n".join(basenames) + "\n", encoding="utf-8")
    return txt, sha256_of_strings(basenames)
