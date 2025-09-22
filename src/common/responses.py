# src/common/synthetic.py
from __future__ import annotations
"""
Deterministic synthetic temperature responses for testing and demos.

This utility generates a bank of broad, well-behaved Gaussian response curves
over log10(T), along with the corresponding logT sample vector and DEM bin
edges in linear T (Kelvin).

Typical usage
-------------
>>> T_RESP, logT, TEMPS = prepare_synthetic_responses(nt=24, nf=6)
>>> T_RESP.shape   # (n_tresp, nf)
(200, 6)
>>> logT.shape     # (n_tresp,)
(200,)
>>> TEMPS.shape    # (nt + 1,)
(25,)
"""

from typing import Tuple
import numpy as np


def prepare_synthetic_responses(
    logT_min: float = 5.5,
    logT_max: float = 7.5,
    n_tresp: int = 200,
    nt: int = 24,
    nf: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build synthetic instrument temperature responses.

    Parameters
    ----------
    logT_min, logT_max : float
        Range of log10(T) covered by the responses and DEM bin edges.
    n_tresp : int
        Number of logT samples for the response curves (resolution of T_RESP / logT).
    nt : int
        Number of DEM bins (TEMPS will have length nt + 1 as edges).
    nf : int
        Number of response channels/filters.

    Returns
    -------
    T_RESP : np.ndarray, shape (n_tresp, nf), dtype float32
        Synthetic response matrix; each column is a broad Gaussian in log10(T),
        shifted so channels are separated and reasonably conditioned.
    logT : np.ndarray, shape (n_tresp,), dtype float32
        Sample points (centers) for T_RESP along log10(T).
    TEMPS : np.ndarray, shape (nt + 1,), dtype float32
        DEM bin edges in **linear** temperature (Kelvin), monotonically increasing.

    Notes
    -----
    - A small floor (1e-30) is added to T_RESP to avoid exact zeros.
    - The Gaussian width is chosen to encourage stable inversions in tests while
      remaining simple and deterministic.
    """
    # Sample points along log10(T) for the response curves
    logT = np.linspace(logT_min, logT_max, int(n_tresp), dtype=np.float32)

    # Place nf Gaussian centers evenly inside the range (with a small margin)
    centers = np.linspace(logT_min + 0.2, logT_max - 0.2, int(nf), dtype=np.float32)
    width = np.float32(0.15)

    # Build responses: (n_tresp, nf)
    T_RESP = np.exp(-0.5 * ((logT[:, None] - centers[None, :]) / width) ** 2, dtype=np.float32)
    T_RESP += np.float32(1e-30)  # numerical floor

    # DEM bin edges in linear T (Kelvin): length = nt + 1
    TEMPS = np.logspace(logT_min, logT_max, int(nt) + 1, dtype=np.float32)

    return T_RESP.astype(np.float32, copy=False), logT, TEMPS
