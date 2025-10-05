"""
Numerically safe linear algebra helpers built on CuPy for GPU SVDs.

- safe_svd: Performs SVD on GPU with input sanitization and returns NumPy arrays
- safe_pinv: Pseudo-inverse via SVD with small-value truncation
"""

from __future__ import annotations

import numpy as np


def safe_svd(A: np.ndarray, full_matrices: bool = True):
    """Compute a numerically safe SVD on GPU and return NumPy arrays.

    Args:
        A (np.ndarray): Input matrix.
        full_matrices (bool, optional): See numpy.linalg.svd. Defaults True.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: U, s, Vh as NumPy arrays.

    Raises:
        RuntimeError: If the underlying CuPy SVD fails.
    """
    import cupy as cp

    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)
    A_gpu = cp.asarray(A)
    try:
        u, s, vh = cp.linalg.svd(A_gpu, full_matrices=full_matrices)
        return cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)
    except Exception as exc:
        raise RuntimeError(f"CuPy SVD failed: {exc}")


def safe_pinv(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Compute a robust pseudo-inverse via SVD with small-value truncation.

    Args:
        A (np.ndarray): Input matrix.
        rcond (float, optional): Relative cut-off for small singular values.
            Defaults to 1e-12.

    Returns:
        np.ndarray: Pseudo-inverse of A.
    """
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)
    U, s, Vh = safe_svd(A, full_matrices=False)
    tol = np.max(s) * rcond if s.size else rcond
    s_inv = np.array([1.0 / x if x > tol else 0.0 for x in s], dtype=np.float64)
    return (Vh.T * s_inv) @ U.T


__all__ = ["safe_svd", "safe_pinv"]
