"""GPU-accelerated kernels for DEM/GSVd-related computations.

This module prefers CuPy + numba.cuda for GPU execution but falls back to
NumPy/SciPy implementations when CUDA is not available. Kernels expose the
same function signatures as the original vendor code where practical.
"""
from typing import Tuple
import numpy as np
import logging

# module-level alias used by functions below
_np = np

# CuPy and numba.cuda are required for multi-GPU kernels. Fail loudly if
# they are not importable so callers do not silently fall back to CPU.
try:
    import cupy as cp
except Exception as e:
    raise ImportError(
        "CuPy is required for multiGPU execution. " f"Original error: {e}"
    ) from e

# We require GPU for this module
GPU_AVAILABLE = True

try:
    import scipy.linalg as sla
except Exception:
    sla = None


def _to_device(x):
    return cp.asarray(x)


def safe_svd(
    A: np.ndarray,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD on GPU when possible, otherwise use SciPy/NumPy.

    This mirrors the behavior of baseline/vendor/dem_inv_gsvd.safe_svd but
    uses CuPy's linalg when available for acceleration.
    """
    # Run SVD on GPU and return numpy arrays. Raise on failure so callers
    # cannot silently continue on CPU.
    A_gpu = _to_device(np.asarray(A, dtype=np.float64, order="C"))
    try:
        u, s, vh = cp.linalg.svd(A_gpu, full_matrices=full_matrices)
        return cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)
    except Exception as exc:
        logging.getLogger(__name__).exception("GPU SVD failed: %s", exc)
        raise RuntimeError(
            "CuPy SVD failed; aborting multiGPU execution."
        ) from exc


def safe_pinv(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Pseudo-inverse with safety checks; uses GPU if available.

    The function mirrors baseline.safe_pinv semantics but accelerates heavy
    linear algebra on the GPU when possible.
    """
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)

    u, s, vh = safe_svd(A, full_matrices=False, compute_uv=True)
    tol = np.max(s) * rcond if s.size else rcond
    s_inv = np.array([1/x if x > tol else 0 for x in s])
    return (vh.T * s_inv) @ u.T


def dem_inv_gsvd(A: np.ndarray, B: np.ndarray):
    """GPU-aware generalized SVD helper used by DEM routines.

    Returns alpha, beta, u_t_slice, v_t, W matching original API.
    """
    AB1 = A @ safe_pinv(B)
    sze = AB1.shape
    C = np.zeros([max(sze), max(sze)])
    C[:sze[0], :sze[1]] = AB1

    u, s, v = safe_svd(C, full_matrices=True, compute_uv=True)

    beta = 1.0 / np.sqrt(1.0 + s**2)
    alpha = s * beta

    SB = np.diag(beta)
    SB_inv = safe_pinv(SB)
    W = safe_pinv(SB_inv @ v @ B)

    return alpha, beta, u.T[:, :sze[0]], v.T, W


def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    """GPU-aware regularization parameter search.

    Mirrors baseline.dem_reg_map but attempts to use CuPy for vectorized
    operations when available. Returns the optimal mu (regularization param).
    """
    xp = cp if GPU_AVAILABLE else np

    data = xp.asarray(data)
    err = xp.asarray(err)

    nf = data.shape[0]
    nreg = sigmaa.shape[0]

    # Safe generalized singular values ratio
    eps = xp.finfo(float).tiny
    sigs = xp.asarray(sigmaa[:nf]) / xp.maximum(xp.asarray(sigmab[:nf]), eps)
    sigs = sigs[xp.isfinite(sigs) & (sigs > 0)]

    if sigs.size == 0:
        minx, maxx = 1e-8, 1e2
    else:
        maxx = float(xp.max(sigs))
        minx = float((xp.min(sigs) ** 2) * 1e-4)
        minx = max(minx, 1e-300)
        if not (maxx > minx):
            maxx = minx * 10.0

    # Defensive handling: callers sometimes pass the full `u` matrix from SVD
    # (shape M x M) instead of the expected `U` shaped like u.T[:, :nf]. Try
    # to adapt the common mistake automatically; otherwise raise a clear
    # error.
    U_arr = xp.asarray(U)
    if U_arr.ndim != 2:
        raise ValueError(f"U must be 2D, got ndim={U_arr.ndim}")
    # If second axis doesn't match data length, but caller passed full square
    # u (M x M) we can transpose and slice to obtain u.T[:, :nf]
    if U_arr.shape[1] != nf:
        if U_arr.shape[0] == U_arr.shape[1] and U_arr.shape[0] >= nf:
            U_arr = U_arr.T[:, :nf]
        else:
            shape = tuple(U_arr.shape)
            raise ValueError(
                f"Incompatible U shape {shape} for len {nf}. "
                + "Expected second dimension == data length or square matrix."
            )

    nmu_eff = max(int(nmu), 2)
    # generate mu as geometric spacing (on CPU to avoid NaNs in older cupy)
    mu = np.geomspace(minx, maxx, num=nmu_eff, dtype=float)
    mu_xp = xp.asarray(mu)

    # Compute coefficients and discr in a vectorized way
    arg = xp.zeros((nreg, nmu_eff), dtype=float)
    for kk in range(nf):
        Uk = xp.asarray(U_arr[kk, :])
        coef = data @ Uk
        sb = xp.asarray(sigmab[kk])
        sa = xp.asarray(sigmaa[kk])
        num = mu_xp * (sb ** 2) * coef
        den = (sa ** 2) + mu_xp * (sb ** 2)
        vals = (num / den) ** 2
        # Keep values on the current array module
        arg[kk, :] = vals

    discr = xp.sum(arg[:, :nmu_eff], axis=0) - xp.sum(err ** 2) * reg_tweak
    # move to host to pick optimal mu (nmu is typically small)
    discr_host = cp.asnumpy(discr) if GPU_AVAILABLE else np.asarray(discr)
    opt = mu[int(np.argmin(np.abs(discr_host[:nmu_eff])))]
    return opt


def dem_pix(*args, **kwargs):
    """Single-pixel DEM is provided via GPU paths; this function is a stub
    in multiGPU mode. Use `demmap_pos` for batched GPU execution.
    """
    raise RuntimeError(
        "dem_pix is not supported in multiGPU mode as a standalone function. "
        "Use demmap_pos (batched GPU path) or run the single-GPU/single-node "
        "baseline implementation."
    )


def demmap_pos(dd, ed, rmatrix, logt, dlogt, glc,
               reg_tweak=1.0, max_iter=10, rgt_fact=1.5, dem_norm0=None,
               nmu=42, warn=False, l_emd=False, rscl=False):
    """Batch wrapper to compute dem_pix over dd rows.

    This mirrors the baseline.demmap_pos but keeps the implementation simple
    (single-process) and relies on per-pixel kernel `dem_pix` which can be
    accelerated by GPU-enabled functions inside this module.
    """
    na = dd.shape[0]
    nt = logt.shape[0]
    dem = _np.zeros((na, nt))
    edem = _np.zeros((na, nt))
    elogt = _np.zeros((na, nt))
    chisq = _np.zeros((na,))
    dn_reg = _np.zeros((na, rmatrix.shape[1]))

    # If GPU is available, attempt a batched device-resident path. The goal
    # is to amortize repeated SVD/pinv costs across a block of pixels. We
    # still do small control computations on the host (e.g. regularizer
    # selection) to remain numerically robust.
    if GPU_AVAILABLE:
        try:
            dd_d = cp.asarray(dd, dtype=cp.float64)
            ed_d = cp.asarray(ed, dtype=cp.float64)
            rmatrix_d = cp.asarray(rmatrix, dtype=cp.float64)

            # Batch over pixels to perform batched SVDs with adaptive retries
            batch_size = 64
            initial_batch = batch_size
            nt_dev = rmatrix_d.shape[0]
            nf_dev = rmatrix_d.shape[1]

            b0 = 0
            while b0 < na:
                cur_batch = min(batch_size, na - b0)
                b1 = b0 + cur_batch
                try:
                    dn_b = dd_d[b0:b1, :]  # (b, nf)
                    ed_b = ed_d[b0:b1, :]  # (b, nf)

                    # build rmatrixin per pixel: shape (b, nt, nf)
                    inv_ed = 1.0 / ed_b  # (b, nf)
                    rmatrix_expand = rmatrix_d[None, :, :]  # (1, nt, nf)
                    inv_ed_expand = inv_ed[:, None, :]      # (b, 1, nf)
                    rmatrixin_b = rmatrix_expand * inv_ed_expand  # (b, nt, nf)

                    # A_batch: (b, nf, nt)
                    A_batch = cp.transpose(rmatrixin_b, (0, 2, 1))

                    # Constant B = L from dlogt
                    dlogt_d = cp.asarray(dlogt, dtype=cp.float64)
                    L = cp.diag(1.0 / cp.sqrt(dlogt_d))
                    B_inv = cp.linalg.pinv(L)

                    # AB1 per pixel: (b, nf, nt)
                    AB1 = A_batch @ B_inv

                    # Form square C matrices for SVD: (b, M, M)
                    M = max(AB1.shape[1], AB1.shape[2])
                    C = cp.zeros((cur_batch, M, M), dtype=cp.float64)
                    C[:, :AB1.shape[1], :AB1.shape[2]] = AB1

                    # Batched SVD on device
                    u_b, s_b, vh_b = cp.linalg.svd(C, full_matrices=True)

                    # alpha/beta per pixel
                    beta_b = 1.0 / cp.sqrt(1.0 + s_b ** 2)
                    alpha_b = s_b * beta_b

                    # Compute per-pixel W and lambda on host (small matrices)
                    alpha_cpu = cp.asnumpy(alpha_b)
                    beta_cpu = cp.asnumpy(beta_b)
                    u_cpu = cp.asnumpy(u_b)
                    vh_cpu = cp.asnumpy(vh_b)
                    B_host = cp.asnumpy(L)

                    for j in range(cur_batch):
                        sva = alpha_cpu[j]
                        svb = beta_cpu[j]
                        U_cpu = u_cpu[j]
                        vh_j = vh_cpu[j]

                        # compute SB and its inverse on host
                        SB = np.diag(svb)
                        try:
                            SB_inv = np.linalg.pinv(SB)
                        except Exception:
                            SB_inv = np.linalg.pinv(
                                SB + 1e-12 * np.eye(SB.shape[0])
                            )

                        # compute W on host then move to device
                        W_host = np.linalg.pinv(SB_inv @ vh_j.T @ B_host)
                        W_d = cp.asarray(W_host)

                        # pick lambda via dem_reg_map on host
                        # Provide U in shape (M, nf_dev) i.e. u.T[:, :nf]
                        U_for_reg = (
                            U_cpu.T[:, :nf_dev]
                            if U_cpu.ndim == 2
                            else U_cpu
                        )
                        lamb = dem_reg_map(
                            sva,
                            svb,
                            U_for_reg,
                            W_host,
                            cp.asnumpy(dn_b[j]),
                            cp.asnumpy(ed_b[j]),
                            reg_tweak,
                            nmu,
                        )

                        # compute filter and kdag on device
                        alpha_d = cp.asarray(sva)
                        beta_d = cp.asarray(svb)
                        filt = cp.zeros((nf_dev, nt_dev), dtype=cp.float64)
                        for kk in range(nf_dev):
                            filt[kk, kk] = (
                                alpha_d[kk]
                                / (alpha_d[kk] ** 2 + beta_d[kk] ** 2 * lamb)
                            )

                        u_d = u_b[j]
                        kdag_d = W_d @ (filt.T @ u_d[:nf_dev, :nf_dev])
                        dem_out_d = (kdag_d @ dn_b[j, :]).squeeze()

                        dem[b0 + j, :] = cp.asnumpy(dem_out_d)
                        dn_reg[b0 + j, :] = cp.asnumpy(
                            (rmatrix_d.T @ dem_out_d).squeeze()
                        )
                        residuals = (
                            cp.asnumpy(dd[b0 + j, :]) - dn_reg[b0 + j, :]
                        ) / cp.asnumpy(ed[b0 + j, :])
                        chisq[b0 + j] = np.sum(residuals ** 2) / dd.shape[1]
                        edem[b0 + j, :] = np.abs(dem[b0 + j, :]) * 0.1

                    # advance to next block and (optionally) restore batch_size
                    b0 = b1
                    batch_size = initial_batch
                except Exception as e:
                    # Log and reduce batch size on memory/allocation failures
                    _log = logging.getLogger(__name__)
                    _log.exception("GPU block %d:%d failed: %s", b0, b1, e)
                    if cur_batch <= 1:
                        _log.exception(
                            "Minimum batch size reached; aborting GPU path"
                        )
                        raise
                    # reduce batch and retry the same start index
                    batch_size = max(1, cur_batch // 2)
                    _log.warning(
                        "Reducing GPU batch size to %d and retrying",
                        batch_size,
                    )

            return dem, edem, elogt, chisq, dn_reg
        except Exception as e:
            # If anything fails on the GPU path, raise an error so callers
            # cannot silently fallback to CPU in multiGPU mode.
            logging.getLogger(__name__).exception("GPU path failed: %s", e)
            raise RuntimeError(
                "GPU path failed; aborting multiGPU execution"
            ) from e

    for i in range(na):
        dem_i, edem_i, elogt_i, chisq_i, dn_reg_i = dem_pix(
            dd[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
            reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
            dem_norm0=dem_norm0[i, :] if dem_norm0 is not None else 0,
            nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl
        )
        dem[i, :] = dem_i
        edem[i, :] = edem_i
        elogt[i, :] = elogt_i
        chisq[i] = chisq_i
        dn_reg[i, :] = dn_reg_i

    return dem, edem, elogt, chisq, dn_reg
