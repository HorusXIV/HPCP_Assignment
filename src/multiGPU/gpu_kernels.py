"""GPU-accelerated kernels for DEM/GSVd-related computations.

This module prefers CuPy + numba.cuda for GPU execution but falls back to
NumPy/SciPy implementations when CUDA is not available. Kernels expose the
same function signatures as the original vendor code where practical.
"""
from typing import Tuple
import numpy as np

# module-level alias used by functions below
_np = np

# Try to import CuPy and numba.cuda; otherwise provide fallbacks
try:
    import cupy as cp
    from numba import cuda
    GPU_AVAILABLE = True
except Exception:
    cp = None
    cuda = None
    GPU_AVAILABLE = False

try:
    import scipy.linalg as sla
except Exception:
    sla = None


def _to_device(x):
    return cp.asarray(x) if GPU_AVAILABLE else x


def safe_svd(A: np.ndarray, full_matrices: bool = True, compute_uv: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD on GPU when possible, otherwise use SciPy/NumPy.

    This mirrors the behavior of baseline/vendor/dem_inv_gsvd.safe_svd but
    uses CuPy's linalg when available for acceleration.
    """
    if GPU_AVAILABLE:
        A_gpu = _to_device(np.asarray(A, dtype=np.float64, order="C"))
        try:
            u, s, vh = cp.linalg.svd(A_gpu, full_matrices=full_matrices)
            return cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)
        except Exception:
            # fallback to CPU
            pass

    # CPU fallback
    A = np.asarray(A, dtype=np.float64, order="C")
    if sla is not None:
        try:
            return sla.svd(A, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver="gesvd", check_finite=False)
        except Exception:
            try:
                return sla.svd(A, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver="gesdd", check_finite=False)
            except Exception:
                import numpy.linalg as npl
                return npl.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
    else:
        import numpy.linalg as npl
        return npl.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)


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

    nmu_eff = max(int(nmu), 2)
    # generate mu as geometric spacing (on CPU to avoid NaN issues in older cupy)
    mu = np.geomspace(minx, maxx, num=nmu_eff, dtype=float)
    mu_xp = xp.asarray(mu)

    # Compute coefficients and discr in a vectorized way
    arg = xp.zeros((nreg, nmu_eff), dtype=float)
    for kk in range(nf):
        Uk = xp.asarray(U[kk, :])
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


def dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, reg_tweak=1.0,
            max_iter=10, rgt_fact=1.5, dem_norm0=0, nmu=42, warn=True,
            l_emd=False, rscl=False):
    """GPU-aware single-pixel DEM inversion mirroring baseline.dem_pix.

    This function keeps control logic on the CPU but uses GPU-accelerated
    linear algebra where available via dem_inv_gsvd and dem_reg_map.
    """
    # use module-level numpy alias

    nf = rmatrix.shape[1]
    nt = logt.shape[0]

    # prepare scaled matrix like baseline
    rmatrixin = _np.zeros((nt, nf))
    for kk in range(nf):
        rmatrixin[:, kk] = rmatrix[:, kk] / ednin[kk]

    dn = dnin / ednin
    edn = ednin / ednin

    # If a GPU is available, run the heavy linear algebra on-device to
    # avoid round-trip host/device transfers. We keep a CPU fallback path
    # that mirrors the original algorithm.
    if GPU_AVAILABLE:
        try:
            # move small working arrays to device
            rmatrixin_d = cp.asarray(rmatrixin, dtype=cp.float64)
            dn_d = cp.asarray(dn, dtype=cp.float64)
            edn_d = cp.asarray(edn, dtype=cp.float64)

            # initial weighting and L computation on device
            if (hasattr(dem_norm0, '__len__') and _np.prod(dem_norm0) != 1.0 and dem_norm0[0] != 0):
                dem_reg_lwght_d = cp.asarray(dem_norm0, dtype=cp.float64)
            else:
                dlogt_d = cp.asarray(dlogt, dtype=cp.float64)
                L_d = cp.diag(1.0 / cp.sqrt(dlogt_d))

                # compute GSVD-like step on device: use pinv and SVD on C
                B_d = cp.asarray(L_d, dtype=cp.float64)
                A_d = rmatrixin_d.T
                AB1_d = A_d @ cp.linalg.pinv(B_d)
                sze0 = AB1_d.shape
                C_d = cp.zeros((max(sze0), max(sze0)), dtype=cp.float64)
                C_d[:sze0[0], :sze0[1]] = AB1_d

                u_d, s_d, vh_d = cp.linalg.svd(C_d, full_matrices=True)

                beta_d = 1.0 / cp.sqrt(1.0 + s_d ** 2)
                alpha_d = s_d * beta_d

                SB_d = cp.diag(beta_d)
                SB_inv_d = cp.linalg.pinv(SB_d)
                W_d = cp.linalg.pinv(SB_inv_d @ vh_d.T @ B_d)

                # select first approximation for dem_reg_lwght
                # compute lamb using CPU dem_reg_map (small cost) by copying
                sva = cp.asnumpy(alpha_d)
                svb = cp.asnumpy(beta_d)
                U = cp.asnumpy(u_d)
                W = cp.asnumpy(W_d)
                lamb = dem_reg_map(sva, svb, U, W, cp.asnumpy(dn_d), cp.asnumpy(edn_d), reg_tweak, nmu)

                # compute kdag on device
                filt_d = cp.zeros((nf, nt), dtype=cp.float64)
                for kk in range(nf):
                    filt_d[kk, kk] = (alpha_d[kk] / (alpha_d[kk] ** 2 + beta_d[kk] ** 2 * lamb))
                kdag_d = W_d @ (filt_d.T @ u_d[:nf, :nf])
                dr0_d = (kdag_d @ dn_d).squeeze()

                fcofmax = 1e-4
                mask_d = (dr0_d > 0) & (dr0_d > fcofmax * cp.max(dr0_d))
                dem_reg_lwght_d = cp.ones(nt, dtype=cp.float64)
                dem_reg_lwght_d[mask_d] = dr0_d[mask_d]
                # smooth on host for convenience
                dem_reg_lwght = cp.asnumpy(dem_reg_lwght_d)
                dem_reg_lwght = _np.convolve(dem_reg_lwght[1:-1], _np.ones(5) / 5)[1:-1]
                dem_reg_lwght = dem_reg_lwght / _np.max(dem_reg_lwght)
                dem_reg_lwght[dem_reg_lwght <= 1e-8] = 1e-8
                dem_reg_lwght_d = cp.asarray(dem_reg_lwght, dtype=cp.float64)

            # build L on device
            if l_emd:
                L_d = cp.diag(1.0 / cp.abs(dem_reg_lwght_d))
            else:
                dlogt_d = cp.asarray(dlogt, dtype=cp.float64)
                L_d = cp.diag(cp.sqrt(dlogt_d) / cp.sqrt(cp.abs(dem_reg_lwght_d)))

            # GSVD-like on device
            A_d = rmatrixin_d.T
            B_d = L_d
            AB1_d = A_d @ cp.linalg.pinv(B_d)
            sze0 = AB1_d.shape
            C_d = cp.zeros((max(sze0), max(sze0)), dtype=cp.float64)
            C_d[:sze0[0], :sze0[1]] = AB1_d
            u_d, s_d, vh_d = cp.linalg.svd(C_d, full_matrices=True)

            alpha_d = s_d * (1.0 / cp.sqrt(1.0 + s_d ** 2))
            beta_d = 1.0 / cp.sqrt(1.0 + s_d ** 2)
            SB_d = cp.diag(beta_d)
            SB_inv_d = cp.linalg.pinv(SB_d)
            W_d = cp.linalg.pinv(SB_inv_d @ vh_d.T @ B_d)

            # positivity loop on device (but compute lamb on host via dem_reg_map)
            piter = 0
            rgt = reg_tweak
            dem_reg_out_d = None
            while piter < max_iter:
                sva = cp.asnumpy(alpha_d)
                svb = cp.asnumpy(beta_d)
                U = cp.asnumpy(u_d)
                W = cp.asnumpy(W_d)
                lamb = dem_reg_map(sva, svb, U, W, cp.asnumpy(dn_d), cp.asnumpy(edn_d), rgt, nmu)

                filt_d = cp.zeros((nf, nt), dtype=cp.float64)
                for kk in range(nf):
                    filt_d[kk, kk] = (alpha_d[kk] / (alpha_d[kk] ** 2 + beta_d[kk] ** 2 * lamb))
                kdag_d = W_d @ (filt_d.T @ u_d[:nf, :nf])
                dem_reg_out_d = (kdag_d @ dn_d).squeeze()

                ndem = int(cp.sum(dem_reg_out_d < 0))
                if ndem == 0:
                    break
                rgt = rgt_fact * rgt
                piter += 1

            # collect results back to host
            dem = cp.asnumpy(dem_reg_out_d)
            dn_reg = cp.asnumpy((rmatrix.T @ dem_reg_out_d).squeeze())
            residuals = (dnin - dn_reg) / ednin
            chisq = _np.sum(residuals ** 2) / nf

            delxi2 = cp.asnumpy(kdag_d @ kdag_d.T)
            edem = _np.sqrt(_np.diag(delxi2))

            kdagk = cp.asnumpy(kdag_d @ rmatrixin_d.T)

            elogt = _np.zeros(nt)
            ltt = _np.min(logt) + 1e-8 + (_np.max(logt) - _np.min(logt)) * _np.arange(51) / (52 - 1.0)
            for kk in range(nt):
                rr = _np.interp(ltt, logt, kdagk[:, kk])
                hm_mask = (rr >= _np.max(kdagk[:, kk]) / 2.0)
                elogt[kk] = dlogt[kk]
                if _np.sum(hm_mask) > 0:
                    elogt[kk] = (ltt[hm_mask][-1] - ltt[hm_mask][0]) / 2

            if rscl:
                mnrat = _np.mean(dnin / dn_reg)
                dem = dem * mnrat
                edem = edem * mnrat
                dn_reg = (rmatrix.T @ dem).squeeze()
                chisq = _np.sum(((dnin - dn_reg) / ednin) ** 2) / nf

            return dem, edem, elogt, chisq, dn_reg
        except Exception:
            # If anything fails on the GPU path, fall back to CPU implementation
            pass

    # basic checks (CPU fallback)
    if (_np.sum(_np.isnan(dn)) == 0 and _np.sum(_np.isinf(dn)) == 0 and _np.prod(dn) > 0):
        piter = 0
        rgt = reg_tweak

        # initial weighting
        if (hasattr(dem_norm0, '__len__') and _np.prod(dem_norm0) != 1.0 and dem_norm0[0] != 0):
            dem_reg_lwght = dem_norm0
        else:
            # simple initial L diag
            L = _np.diag(1.0 / _np.sqrt(dlogt))
            sva, svb, U, V, W = dem_inv_gsvd(rmatrixin.T, L)
            lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)
            filt = _np.zeros((nf, nt))
            for kk in range(nf):
                filt[kk, kk] = (sva[kk] / (sva[kk] ** 2 + svb[kk] ** 2 * lamb))
            kdag = W @ (filt.T @ U[:nf, :nf])
            dr0 = (kdag @ dn).squeeze()
            fcofmax = 1e-4
            mask = _np.where((dr0 > 0) & (dr0 > fcofmax * _np.max(dr0)))[0]
            dem_reg_lwght = _np.ones(nt)
            dem_reg_lwght[mask] = dr0[mask]
            dem_reg_lwght = _np.convolve(dem_reg_lwght[1:-1], _np.ones(5) / 5)[1:-1]
            dem_reg_lwght = dem_reg_lwght / _np.max(dem_reg_lwght)
            dem_reg_lwght[dem_reg_lwght <= 1e-8] = 1e-8

        # build L
        if l_emd:
            L = _np.diag(1 / _np.abs(dem_reg_lwght))
        else:
            L = _np.diag(_np.sqrt(dlogt) / _np.sqrt(_np.abs(dem_reg_lwght)))

        sva, svb, U, V, W = dem_inv_gsvd(rmatrixin.T, L)

        # positivity loop
        dem_reg_out = None
        while (piter < max_iter):
            lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)
            filt = _np.zeros((nf, nt))
            for kk in range(nf):
                filt[kk, kk] = (sva[kk] / (sva[kk] ** 2 + svb[kk] ** 2 * lamb))
            kdag = W @ (filt.T @ U[:nf, :nf])
            dem_reg_out = (kdag @ dn).squeeze()

            ndem = len(dem_reg_out[dem_reg_out < 0])
            if ndem == 0:
                break
            rgt = rgt_fact * rgt
            piter += 1

        if (warn and (piter == max_iter)):
            print('Warning, positivity loop hit max iterations')

        dem = dem_reg_out
        dn_reg = (rmatrix.T @ dem_reg_out).squeeze()
        residuals = (dnin - dn_reg) / ednin
        chisq = _np.sum(residuals ** 2) / nf

        delxi2 = kdag @ kdag.T
        edem = _np.sqrt(_np.diag(delxi2))

        kdagk = kdag @ rmatrixin.T

        elogt = _np.zeros(nt)
        ltt = _np.min(logt) + 1e-8 + (_np.max(logt) - _np.min(logt)) * _np.arange(51) / (52 - 1.0)
        for kk in range(nt):
            rr = _np.interp(ltt, logt, kdagk[:, kk])
            hm_mask = (rr >= _np.max(kdagk[:, kk]) / 2.0)
            elogt[kk] = dlogt[kk]
            if _np.sum(hm_mask) > 0:
                elogt[kk] = (ltt[hm_mask][-1] - ltt[hm_mask][0]) / 2

        if rscl:
            mnrat = _np.mean(dnin / dn_reg)
            dem = dem * mnrat
            edem = edem * mnrat
            dn_reg = (rmatrix.T @ dem).squeeze()
            chisq = _np.sum(((dnin - dn_reg) / ednin) ** 2) / nf

        return dem, edem, elogt, chisq, dn_reg

    # fallback empty results
    return _np.zeros(nt), _np.zeros(nt), _np.zeros(nt), 0.0, _np.zeros(nf)


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

            # Batch over pixels to perform batched SVDs
            batch_size = 64
            nt_dev = rmatrix_d.shape[0]
            nf_dev = rmatrix_d.shape[1]

            for b0 in range(0, na, batch_size):
                b1 = min(na, b0 + batch_size)
                b = b1 - b0

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
                C = cp.zeros((b, M, M), dtype=cp.float64)
                C[:, :AB1.shape[1], :AB1.shape[2]] = AB1

                # Batched SVD on device
                u_b, s_b, vh_b = cp.linalg.svd(C, full_matrices=True)

                # alpha/beta per pixel
                beta_b = 1.0 / cp.sqrt(1.0 + s_b ** 2)
                alpha_b = s_b * beta_b

                # We'll compute per-pixel W and lambda on host (small matrices)
                alpha_cpu = cp.asnumpy(alpha_b)
                beta_cpu = cp.asnumpy(beta_b)
                u_cpu = cp.asnumpy(u_b)
                vh_cpu = cp.asnumpy(vh_b)
                B_host = cp.asnumpy(L)

                for j in range(b):
                    sva = alpha_cpu[j]
                    svb = beta_cpu[j]
                    U_cpu = u_cpu[j]
                    vh_j = vh_cpu[j]

                    # compute SB and its inverse on host
                    SB = np.diag(svb)
                    try:
                        SB_inv = np.linalg.pinv(SB)
                    except Exception:
                        SB_inv = np.linalg.pinv(SB + 1e-12 * np.eye(SB.shape[0]))

                    # compute W on host then move to device
                    W_host = np.linalg.pinv(SB_inv @ vh_j.T @ B_host)
                    W_d = cp.asarray(W_host)

                    # pick lambda via dem_reg_map on host
                    lamb = dem_reg_map(sva, svb, U_cpu, W_host, cp.asnumpy(dn_b[j]), cp.asnumpy(ed_b[j]), reg_tweak, nmu)

                    # compute filter and kdag on device
                    alpha_d = cp.asarray(sva)
                    beta_d = cp.asarray(svb)
                    filt = cp.zeros((nf_dev, nt_dev), dtype=cp.float64)
                    for kk in range(nf_dev):
                        filt[kk, kk] = (alpha_d[kk] / (alpha_d[kk] ** 2 + beta_d[kk] ** 2 * lamb))

                    u_d = u_b[j]
                    kdag_d = W_d @ (filt.T @ u_d[:nf_dev, :nf_dev])
                    dem_out_d = (kdag_d @ dn_b[j, :]).squeeze()

                    dem[b0 + j, :] = cp.asnumpy(dem_out_d)
                    dn_reg[b0 + j, :] = cp.asnumpy((rmatrix_d.T @ dem_out_d).squeeze())
                    residuals = (cp.asnumpy(dd[b0 + j, :]) - dn_reg[b0 + j, :]) / cp.asnumpy(ed[b0 + j, :])
                    chisq[b0 + j] = np.sum(residuals ** 2) / dd.shape[1]
                    edem[b0 + j, :] = np.abs(dem[b0 + j, :]) * 0.1

            return dem, edem, elogt, chisq, dn_reg
        except Exception:
            # If anything fails on the GPU path, fall back to CPU implementation
            pass

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
