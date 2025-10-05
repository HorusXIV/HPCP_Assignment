"""
Regularization parameter selection via the discrepancy principle.
"""

from __future__ import annotations

import numpy as np


def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    """Select a regularization parameter via the discrepancy principle.

    Args:
        sigmaa (array-like): Singular values from the left term (size ≥ nf).
        sigmab (array-like): Singular values from the right term (size ≥ nf).
        U (array-like): Left singular vectors; rows indexed by kk for nf rows.
        W (array-like): Not used directly here; included for parity.
        data (array-like): Observation vector (nf,).
        err (array-like): 1-sigma errors (nf,) used in residual target.
        reg_tweak (float): Multiplier for the discrepancy target.
        nmu (int, optional): Number of mu values to evaluate. Defaults 500.

    Returns:
        float: Selected mu (lambda) minimizing absolute discrepancy.

    Raises:
        ValueError: If U does not have at least nf rows after orientation fix.
    """
    nf = data.shape[0]

    eps = np.finfo(float).tiny
    sigs = np.asarray(sigmaa[:nf]) / np.maximum(np.asarray(sigmab[:nf]), eps)
    sigs = sigs[np.isfinite(sigs) & (sigs > 0)]
    if sigs.size == 0:
        minx, maxx = 1e-8, 1e2
    else:
        maxx = float(np.max(sigs))
        minx = float((np.min(sigs) ** 2) * 1e-4)
        minx = max(minx, 1e-300)
        if not (maxx > minx):
            maxx = minx * 10.0

    nmu_eff = int(max(nmu, 2))
    mu = np.geomspace(minx, maxx, num=nmu_eff, dtype=float)

    # Ensure U indexing matches vendor expectations (row-major access per kk)
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must be 2D")
    if U.shape[0] < nf and U.shape[1] >= nf:
        U = U.T  # make rows addressable by kk
    if U.shape[0] < nf:
        raise ValueError("U must have at least nf rows for dem_reg_map")

    arg = np.zeros((nf, nmu_eff), dtype=np.float64)
    for kk in range(nf):
        coef = float(np.dot(data, U[kk, :]))
        num = mu * (sigmab[kk] ** 2) * coef
        den = (sigmaa[kk] ** 2) + mu * (sigmab[kk] ** 2)
        arg[kk, :] = (num / den) ** 2
    discr = np.sum(arg, axis=0) - np.sum(err**2) * float(reg_tweak)
    opt = float(mu[int(np.argmin(np.abs(discr)))])
    return opt


__all__ = ["dem_reg_map"]
