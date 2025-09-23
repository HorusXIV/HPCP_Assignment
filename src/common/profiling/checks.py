# src/common/profiling/checks.py
from __future__ import annotations
"""
Lightweight correctness/quality checks for benchmark outputs.

These checks are intentionally simple and fast—they provide quick signals to
spot gross regressions without imposing heavy validation cost.
"""

from typing import Dict

import numpy as np


def basic_checks(demmap: np.ndarray, chisq: np.ndarray) -> Dict[str, float]:
    """
    Compute minimal, robust quality indicators for a DEM solve.

    Parameters
    ----------
    demmap : np.ndarray
        DEM array of any shape (e.g., (H, W, NT) or tiled views).
    chisq : np.ndarray
        Chi-square array (e.g., (H, W)) aligned with `demmap`'s spatial axes.

    Returns
    -------
    dict
        Dictionary with:
          - "finite_frac":   fraction of finite values in `demmap`
          - "positive_frac": fraction of strictly positive values in `demmap`
          - "chisq_median":  median of `chisq` (NaNs ignored)

    Notes
    -----
    These metrics are coarse sanity checks; they are not a substitute for
    golden-file verification.
    """
    finite = np.isfinite(demmap)
    positive = demmap > 0
    return {
        "finite_frac": float(finite.mean()) if demmap.size else float("nan"),
        "positive_frac": float(positive.mean()) if demmap.size else float("nan"),
        "chisq_median": float(np.nanmedian(chisq)) if chisq.size else float("nan"),
    }
