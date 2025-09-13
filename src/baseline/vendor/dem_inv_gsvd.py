import numpy as np
from numpy.linalg import inv,pinv,svd

# DEMREG/python/dem_inv_gsvd.py

import numpy as np
import scipy.linalg as sla

def safe_pinv(A, rcond=1e-12):
    """Robust pseudo-inverse using safe_svd instead of np.linalg.svd."""
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)

    U, s, Vh = safe_svd(A, full_matrices=False)
    tol = np.max(s) * rcond
    s_inv = np.array([1/x if x > tol else 0 for x in s])
    return (Vh.T * s_inv) @ U.T

def safe_svd(A, full_matrices=True, compute_uv=True, **kwargs):
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)
    try:
        return sla.svd(A,
                       full_matrices=full_matrices,
                       compute_uv=compute_uv,
                       lapack_driver="gesvd",
                       check_finite=False)
    except Exception:
        try:
            return sla.svd(A,
                           full_matrices=full_matrices,
                           compute_uv=compute_uv,
                           lapack_driver="gesdd",
                           check_finite=False)
        except Exception:
            import numpy.linalg as npl
            return npl.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)

def dem_inv_gsvd(A, B):
    """
    Generalized SVD used in DEMREG.
    A = U*SA*W^-1, B = V*SB*W^-1
    """
    AB1 = A @ np.linalg.inv(B)
    sze = AB1.shape
    C = np.zeros([max(sze), max(sze)])
    C[:sze[0], :sze[1]] = AB1

    u, s, v = safe_svd(C, full_matrices=True, compute_uv=True)

    beta  = 1.0 / np.sqrt(1.0 + s**2)
    alpha = s * beta

    SB = np.diag(beta)
    W  = safe_pinv(np.linalg.inv(SB) @ v @ B)

    return alpha, beta, u.T[:, :sze[0]], v.T, W
