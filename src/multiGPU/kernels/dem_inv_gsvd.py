"""
GSVD-equivalent factorization used by DEM solver.

Implements vendor-compatible outputs using SVD of A @ pinv(B).
"""

from __future__ import annotations

import numpy as np

from .linalg import safe_pinv, safe_svd


def dem_inv_gsvd(A: np.ndarray, B: np.ndarray):
    """Compute GSVD-equivalent factors using SVD of ``A @ pinv(B)``.

    This mirrors the vendor pipeline to produce equivalent outputs to a
    GSVD, but implemented with a standard SVD on the product ``A @ pinv(B)``.
    Conceptually, this balances two quadratic forms (data-fit and smoothness)
    without relying on a dedicated GSVD implementation.

    Args:
        A (np.ndarray): Left matrix in the generalized system.
        B (np.ndarray): Right matrix; must be full-rank (or near) for pinv.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            alpha, beta, U_T, V_T, W matching the vendor convention.

    Raises:
        RuntimeError: If SVD or pseudo-inverse fails on the provided inputs.
    """
    AB1 = A @ safe_pinv(B)
    sze = AB1.shape
    C = np.zeros([max(sze), max(sze)], dtype=np.float64)
    C[: sze[0], : sze[1]] = AB1
    u, s, v = safe_svd(C, full_matrices=True)
    beta = 1.0 / np.sqrt(1.0 + s**2)
    alpha = s * beta
    SB = np.diag(beta)
    SB_inv = safe_pinv(SB)
    W = safe_pinv(SB_inv @ v @ B)
    return alpha, beta, u.T[:, : sze[0]], v.T, W


__all__ = ["dem_inv_gsvd"]
