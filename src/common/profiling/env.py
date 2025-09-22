# src/common/profiling/env.py
from __future__ import annotations
"""
Environment and input-shape snapshotting for reproducibility.

This module records a concise JSON "fingerprint" of the runtime environment
(Python, NumPy, platform, thread env vars) together with the shapes of the
key arrays used in a run. The goal is to make benchmark artifacts easier to
reproduce and diagnose without being verbose.
"""

import os
import sys
import json
import platform
from pathlib import Path
from typing import Optional

import numpy as np


def write_env_snapshot(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    outdir: str | Path = "benchmark_out",
    extra: Optional[dict] = None,
) -> Path:
    """
    Write a JSON file capturing environment info and core array shapes.

    The JSON includes:
      - Python version, interpreter path, and platform string
      - NumPy version
      - Threading-related environment variables (BLAS/OpenMP hints)
      - Shapes/lengths of key arrays: STACK, T_RESP, T_RESP_LOGT, TEMPS
      - Any user-provided `extra` fields

    Parameters
    ----------
    STACK : np.ndarray
        Input stack, typically shaped (F, H, W, 6) or (N, 6, H, W).
        Only the shape is recorded.
    T_RESP : np.ndarray
        Temperature response matrix, e.g. (n_tresp, n_filters).
    T_RESP_LOGT : np.ndarray
        1-D array of log(T) sample positions, length n_tresp.
    TEMPS : np.ndarray
        1-D array of DEM bin edges or centers (implementation-specific).
    outdir : str | Path, keyword-only, default "benchmark_out"
        Directory where `env.json` will be written; created if missing.
    extra : dict | None, keyword-only
        Optional key/value pairs to merge into the JSON snapshot.

    Returns
    -------
    pathlib.Path
        Path to the written `env.json`.

    Notes
    -----
    - Only metadata is stored; no array contents are serialized.
    - Thread caps are read from common environment variables and may be None
      if not explicitly set.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "thread_caps": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "stack_shape": tuple(map(int, STACK.shape)),
        "tresp_shape": tuple(map(int, T_RESP.shape)),
        "tresp_logt_len": int(T_RESP_LOGT.shape[0]),
        "temps_len": int(TEMPS.shape[0]),
    }
    if extra:
        info.update(extra)

    p = outdir / "env.json"
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return p
