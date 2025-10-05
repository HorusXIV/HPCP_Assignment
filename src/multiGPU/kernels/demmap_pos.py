"""
CuPy-accelerated batched DEM reconstruction kernel (multiGPU).

This module contains the core implementation previously housed in
src/multiGPU/gpu_kernels.py, refactored for modularity.
"""

from __future__ import annotations

import logging
import math
import os
import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception as e:  # pragma: no cover - import checked at runtime path
    raise ImportError("CuPy is required for multiGPU execution: %s" % e) from e

from .memory import _adaptive_batch_size, _bytes_per_sample_estimate
from .utils import nvtx_range, verbose_enabled, _pinned_empty


def demmap_pos(
    dd,
    ed,
    rmatrix,
    logt,
    dlogt,
    glc,
    reg_tweak=1.0,
    max_iter=10,
    rgt_fact=1.5,
    dem_norm0=None,
    nmu=42,
    warn=False,
    l_emd=False,
    rscl=False,
):
    """Reconstruct Differential Emission Measure (DEM) for many samples on GPU.

    Batched CuPy implementation that adaptively chooses batch size from
    available device memory, performs two-pass SVD-based inversion with
    vendor-compatible behavior, and asynchronously transfers results back
    to pinned host memory.

    Args:
        dd (array-like): Data matrix of shape (na, nf); counts per filter.
        ed (array-like): 1-sigma errors of shape (na, nf) or (1, nf).
        rmatrix (array-like): Response matrix of shape (nt, nf).
        logt (array-like): Temperature grid centers of shape (nt,).
        dlogt (array-like): Bin widths in log(T) of shape (nt,).
        glc (array-like): Global constraints mask per filter (nf,) where
            positive entries enable L0-style pass; if all non-positive,
            an internal L0 pass seeds the main SVD iteration.
        reg_tweak (float, optional): Discrepancy principle multiplier for
            target residual. Defaults to 1.0.
        max_iter (int, optional): Max iterations of non-negativity relaxation
            during lambda selection. Defaults to 10.
        rgt_fact (float, optional): Multiplicative factor to increase
            regularization when negative DEM entries are detected. Defaults to 1.5.
        dem_norm0 (array-like | None, optional): Initial DEM normalization
            of shape (na, nt) or broadcastable (nt,). Currently used for
            parity with vendor; can be None. Defaults to None.
        nmu (int, optional): Number of candidate regularization parameters
            sampled on a log grid per batch element. Defaults to 42.
        warn (bool, optional): Unused placeholder for vendor parity.
        l_emd (bool, optional): If True, use L1-like weighting; otherwise
            use sqrt(weight)/sqrt(dlogt) per vendor behavior. Defaults False.
        rscl (bool, optional): If True, rescale DEM to match mean ratio of
            observed/predicted data. Defaults False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - dem: (na, nt) reconstructed DEM
            - edem: (na, nt) error estimates per temperature bin
            - elogt: (na, nt) effective log(T) half-width per bin
            - chisq: (na,) reduced chi-square per sample
            - dn_reg: (na, nf) predicted data under the chosen regularization

    Raises:
        ImportError: If CuPy is not installed/available at runtime.
        RuntimeError: If no CUDA device is visible or if GPU OOM persists
            after downshifting batch size.

    Environment:
        - MULTIGPU_BATCH_SIZE: Force a fixed batch size when > 0; otherwise
          an adaptive size is computed.
        - MULTIGPU_BATCH_MEM_FRAC: Fraction of free GPU memory targeted by
          the adaptive planner (default 0.7, clamped to [0.1, 0.9]).
        - MULTIGPU_VERBOSE: When set, logs plan/OOM retries as [metrics].
        - MULTIGPU_NVTX: When "1", enables NVTX ranges (requires nvtx pkg).
        - MULTIGPU_*POOL_LIMIT_*: Optional CuPy pool soft-limits.
    """
    # Convert inputs
    dd = np.asarray(dd, dtype=np.float64)
    ed = np.asarray(ed, dtype=np.float64)
    rmatrix = np.asarray(rmatrix, dtype=np.float64)
    logt = np.asarray(logt, dtype=np.float64)
    dlogt = np.asarray(dlogt, dtype=np.float64)
    glc = np.asarray(glc)

    na, nf = dd.shape
    nt = logt.shape[0]

    if dem_norm0 is None:
        dem_norm0 = np.ones((na, nt), dtype=np.float64)
    else:
        dem_norm0 = np.asarray(dem_norm0, dtype=np.float64)
        if dem_norm0.ndim == 1:
            dem_norm0 = np.broadcast_to(dem_norm0[None, :], (na, nt)).copy()

    if na == 0:
        return (
            np.zeros((0, nt), dtype=np.float64),
            np.zeros((0, nt), dtype=np.float64),
            np.zeros((0, nt), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, nf), dtype=np.float64),
        )

    dem = np.zeros((na, nt), dtype=np.float64)
    edem = np.zeros((na, nt), dtype=np.float64)
    elogt = np.zeros((na, nt), dtype=np.float64)
    chisq = np.zeros((na,), dtype=np.float64)
    dn_reg = np.zeros((na, nf), dtype=np.float64)

    with nvtx_range("DEM_SOLVE_INIT", color=0x455A64):
        ltt = (
            float(np.min(logt))
            + 1e-8
            + (float(np.max(logt)) - float(np.min(logt)))
            * (np.arange(51, dtype=np.float64) / (52.0 - 1.0))
        )

    def _have_cuda_devices() -> bool:
        """Return True if at least one CUDA device is available."""
        try:
            rt = getattr(cp, "cuda").runtime  # type: ignore[attr-defined]
            get_dc = getattr(rt, "getDeviceCount", None)
            return bool(callable(get_dc) and int(get_dc()) > 0)
        except Exception:
            return False

    if not _have_cuda_devices():
        raise RuntimeError(
            "No CUDA device available. multiGPU.demmap_pos requires a GPU."
        )

    if _have_cuda_devices():
        with nvtx_range("MEMPOOL_CONFIG", color=0x00695C):
            try:
                pool = cp.get_default_memory_pool()
                pin_pool = cp.get_default_pinned_memory_pool()
                total_b = None
                try:
                    _free_b0, _total_b0 = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
                    total_b = int(_total_b0)
                except Exception:
                    total_b = None
                limit_bytes = None
                frac_s = os.environ.get("MULTIGPU_POOL_LIMIT_FRACTION", None)
                if frac_s is not None and total_b is not None:
                    try:
                        frac = float(frac_s)
                        if 0.1 <= frac <= 0.95:
                            limit_bytes = int(total_b * frac)
                    except Exception:
                        limit_bytes = None
                if limit_bytes is None:
                    by_s = os.environ.get("MULTIGPU_POOL_LIMIT_BYTES", None)
                    if by_s is not None:
                        try:
                            limit_bytes = int(by_s)
                        except Exception:
                            limit_bytes = None
                if limit_bytes is not None:
                    try:
                        pool.set_limit(limit_bytes)
                    except Exception:
                        pass
                try:
                    pin_limit_env = os.environ.get(
                        "MULTIGPU_PINNED_POOL_LIMIT_BYTES", None
                    )
                    if pin_limit_env is not None:
                        pin_pool.set_limit(int(pin_limit_env))
                    else:
                        pin_pool.set_limit(int(1 * 1024**3))
                except Exception:
                    pass
                try:
                    pool.free_all_blocks()
                    pin_pool.free_all_blocks()
                except Exception:
                    pass
            except Exception:
                pass

        dem = _pinned_empty((na, nt), np.float64)
        edem = _pinned_empty((na, nt), np.float64)
        elogt = _pinned_empty((na, nt), np.float64)
        chisq = _pinned_empty((na,), np.float64)
        dn_reg = _pinned_empty((na, nf), np.float64)

        # Streams and device constants (allocate first, then size the batch)
        with nvtx_range("DEM_DEVICE_CONSTS", color=0x00796B):
            compute_stream = cp.cuda.Stream(non_blocking=True)
            copy_stream = cp.cuda.Stream(non_blocking=True)
            h2d_stream = cp.cuda.Stream(non_blocking=True)

            rmatrix_d = cp.asarray(rmatrix)
            dlogt_d = cp.asarray(dlogt)
            b0_inv_vec = cp.sqrt(dlogt_d)
            # Interpolation helpers on device
            ltt_d = cp.asarray(ltt)
            seg_idx = np.clip(np.searchsorted(logt, ltt, side="right") - 1, 0, nt - 2)
            seg_idx_d = cp.asarray(seg_idx)
            seg_left = logt[seg_idx]
            seg_right = logt[seg_idx + 1]
            t_frac_d = cp.asarray((ltt - seg_left) / (seg_right - seg_left + 1e-300))

        try:
            env_bs = int(os.environ.get("MULTIGPU_BATCH_SIZE", "0"))
        except Exception:
            env_bs = 0
        batch_size = env_bs if env_bs > 0 else _adaptive_batch_size(na, nf, nt, nmu)

        if verbose_enabled():
            log = logging.getLogger(__name__)
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
            except Exception:
                free_b, total_b = None, None
            bps = _bytes_per_sample_estimate(nf, nt, nmu)
            nb = int(math.ceil(na / max(1, batch_size)))
            log.info(
                "[metrics] plan: na=%d nf=%d nt=%d nmu=%d batch_size=%d num_batches=%d bytes_per_sample=%d est_batch_bytes=%d free_bytes=%s total_bytes=%s",
                na,
                nf,
                nt,
                int(nmu),
                int(batch_size),
                nb,
                int(bps),
                int(bps * batch_size),
                str(int(free_b)) if free_b is not None else "<n/a>",
                str(int(total_b)) if total_b is not None else "<n/a>",
            )

        with nvtx_range("DEM_DEVICE_IO", color=0x00796B):
            while True:
                try:
                    dn_dev = [
                        cp.empty((int(batch_size), int(nf)), dtype=cp.float64)
                        for _ in range(2)
                    ]
                    ed_dev = [
                        cp.empty((int(batch_size), int(nf)), dtype=cp.float64)
                        for _ in range(2)
                    ]
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                    try:
                        free_b3, _ = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
                        bps3 = _bytes_per_sample_estimate(nf, nt, nmu)
                        try:
                            frac_env = float(
                                os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7")
                            )
                            mem_frac = float(min(max(frac_env, 0.1), 0.9))
                        except Exception:
                            mem_frac = 0.55
                        est3 = int((free_b3 * mem_frac) // max(1, bps3))
                        new_bs = max(1, min(est3, na))
                    except Exception:
                        new_bs = max(1, batch_size // 2)
                    if new_bs >= batch_size and batch_size > 1:
                        new_bs = max(1, batch_size // 2)
                    if verbose_enabled():
                        logging.getLogger(__name__).warning(
                            "[metrics] OOM during IO alloc: batch_size %d -> %d",
                            int(batch_size),
                            int(new_bs),
                        )
                    if new_bs == batch_size == 1:
                        raise
                    batch_size = new_bs
            h2d_ready = [cp.cuda.Event() for _ in range(2)]

        # Common mu linspace used for both self-normalized and main passes
        nmu_eff = int(max(int(nmu), 2))
        tlin = cp.linspace(0.0, 1.0, nmu_eff, dtype=cp.float64)

        def _smooth_and_clamp(wraw_b: "cp.ndarray") -> "cp.ndarray":
            """Apply vendor-equivalent smoothing and clamping to weights."""
            x = wraw_b[:, 1:-1]
            z = cp.pad(x, ((0, 0), (4, 4)))
            cs = cp.cumsum(z, axis=1)
            cs = cp.concatenate([cp.zeros((z.shape[0], 1), dtype=z.dtype), cs], axis=1)
            y_full = cs[:, 5:] - cs[:, :-5]
            y = y_full[:, 1:-1] / 5.0
            maxv = cp.maximum(cp.max(wraw_b, axis=1, keepdims=True), 1e-12)
            out = y / maxv
            return cp.maximum(out, 1e-8)

        idx = 0
        cur_batch = batch_size
        completed_gpu = True
        oom_retries = 0

        RING = 3
        ring = [{"in_use": False, "done_evt": None, "keep": None} for _ in range(RING)]

        def _sync_and_clear_slot(i: int):
            slot = ring[i]
            if slot["in_use"] and slot["done_evt"] is not None:
                try:
                    slot["done_evt"].synchronize()
                except Exception:
                    pass
            ring[i] = {"in_use": False, "done_evt": None, "keep": None}

        def _stage_h2d(b0: int, cur: int, slot: int):
            with nvtx_range("H2D_STAGE", color=0x0077CC):
                dn_dst = dn_dev[slot][:cur, :]
                ed_dst = ed_dev[slot][:cur, :]
                try:
                    kind_h2d = cp.cuda.runtime.memcpyHostToDevice  # type: ignore[attr-defined]
                except AttributeError:  # pragma: no cover
                    kind_h2d = cp.cuda.runtime.cudaMemcpyHostToDevice  # type: ignore[attr-defined]
                elsize = np.dtype(np.float64).itemsize
                with h2d_stream:
                    h_dn = dd[b0 : b0 + cur, :]
                    h_ed = ed[b0 : b0 + cur, :]
                    cp.cuda.runtime.memcpyAsync(
                        dn_dst.data.ptr,
                        h_dn.ctypes.data,
                        int(dn_dst.size) * elsize,
                        kind_h2d,
                        h2d_stream.ptr,
                    )
                    cp.cuda.runtime.memcpyAsync(
                        ed_dst.data.ptr,
                        h_ed.ctypes.data,
                        int(ed_dst.size) * elsize,
                        kind_h2d,
                        h2d_stream.ptr,
                    )
                    h2d_ready[slot].record(h2d_stream)

        # Pipeline state for ping-pong inputs
        slot = 0
        pre_staged = False

        while idx < na:
            attempt = min(cur_batch, na - idx)
            try:
                b0 = idx
                b1 = b0 + attempt
                cur = attempt
                with nvtx_range(f"BATCH[{b0}:{b1})", color=0xFF6F00):
                    # Stage current batch if not pre-staged
                    if not pre_staged:
                        _stage_h2d(b0, cur, slot)
                    with nvtx_range("BATCH_PREP", color=0x8E24AA):
                        # Ensure compute waits until inputs are ready
                        compute_stream.wait_event(h2d_ready[slot])
                        with compute_stream:
                            dn_b = dn_dev[slot][:cur, :]
                            ed_b = ed_dev[slot][:cur, :]
                            # Preserve original errors for residuals; clamp a normalized copy for stability
                            ed_b_orig = ed_b
                            ed_b = cp.maximum(ed_b, cp.array(1e-12, dtype=cp.float64))

                            rmatrixin_b = (
                                rmatrix_d[None, :, :] * (1.0 / ed_b)[:, None, :]
                            )
                            A_b = cp.transpose(rmatrixin_b, (0, 2, 1))
                            dprime_b = dn_b / ed_b

                            # Guard invalid rows (NaN/Inf or non-positive product after normalization)
                            valid_rows = cp.logical_and(
                                cp.all(cp.isfinite(dprime_b), axis=1),
                                cp.prod(dprime_b, axis=1) > 0.0,
                            )
                            # Zero-out invalid rows in inputs to keep SVD stable; they'll be set to zeros in outputs later
                            inv_mask = ~valid_rows
                            if bool(cp.any(inv_mask)):
                                A_b = A_b.copy()
                                dprime_b = dprime_b.copy()
                                A_b[inv_mask, :, :] = 0.0
                                dprime_b[inv_mask, :] = 0.0

                    use_gloci = bool(np.sum(glc > 0) > 0)
                    if use_gloci:
                        with nvtx_range("GLOCI_WEIGHTS", color=0x6A1B9A):
                            gd = cp.asarray(np.nonzero(glc > 0)[0])
                            dn_sel = dn_b[:, gd]
                            tr_sel = rmatrix_d[:, gd]
                            em = dn_sel[:, None, :] / cp.maximum(
                                tr_sel[None, :, :], 1e-300
                            )
                            wraw_b = cp.min(em, axis=2)
                    else:
                        with nvtx_range("L0_PASS", color=0x5E35B1):
                            with compute_stream:
                                AB10 = A_b * b0_inv_vec[None, None, :]
                                U0, s0, Vh0 = cp.linalg.svd(AB10, full_matrices=False)
                                coef0 = cp.matmul(
                                    cp.transpose(U0, (0, 2, 1)), dprime_b[:, :, None]
                                ).squeeze(-1)
                                eps = cp.finfo(cp.float64).tiny
                                s_safe0 = cp.maximum(s0, eps)
                                minx0 = cp.maximum(
                                    (cp.min(s_safe0, axis=1) ** 2) * 1e-4,
                                    cp.array(1e-300),
                                )
                                maxx0 = cp.max(s_safe0, axis=1)
                                maxx0 = cp.where(maxx0 > minx0, maxx0, minx0 * 10.0)
                                mu0 = cp.exp(
                                    cp.log(minx0)[:, None]
                                    + (cp.log(maxx0) - cp.log(minx0))[:, None]
                                    * tlin[None, :]
                                )
                                s2 = s0**2
                                vals0 = (
                                    mu0[:, None, :] / (s2[:, :, None] + mu0[:, None, :])
                                ) * coef0[:, :, None]
                                arg_sum0 = cp.sum(vals0**2, axis=1)
                                # Vendor parity: after normalization by ed, discrepancy target uses sum(err^2)=nf
                                err_sq0 = cp.full((cur,), float(nf), dtype=cp.float64)
                                discr0 = (
                                    arg_sum0 - (err_sq0 * float(reg_tweak))[:, None]
                                )
                                idx0 = cp.argmin(cp.abs(discr0), axis=1)
                                lamb0 = mu0[cp.arange(cur), idx0]
                                V0 = cp.transpose(Vh0, (0, 2, 1))
                                filt0 = s0 / (s0**2 + lamb0[:, None])
                                xprime0 = cp.matmul(
                                    V0, (filt0 * coef0)[:, :, None]
                                ).squeeze(-1)
                                dem0 = b0_inv_vec[None, :] * xprime0
                                fcofmax = 1e-4
                                mx = cp.max(dem0, axis=1, keepdims=True)
                                mask = (dem0 > 0) & (dem0 > (fcofmax * mx))
                                wraw_b = cp.where(mask, dem0, cp.ones_like(dem0))

                    with nvtx_range("SMOOTH_WEIGHTS", color=0x303F9F):
                        with compute_stream:
                            weights_b = _smooth_and_clamp(wraw_b)
                            if l_emd:
                                bvec = cp.abs(weights_b)
                            else:
                                bvec = (
                                    cp.sqrt(cp.abs(weights_b))
                                    / cp.sqrt(dlogt_d)[None, :]
                                )

                    with nvtx_range("SVD_MAIN", color=0x1E88E5):
                        with compute_stream:
                            AB1 = A_b * bvec[:, None, :]
                            U, s, Vh = cp.linalg.svd(AB1, full_matrices=False)
                            V = cp.transpose(Vh, (0, 2, 1))
                            s_safe = cp.maximum(s, cp.finfo(cp.float64).tiny)
                            minx = cp.maximum(
                                (cp.min(s_safe, axis=1) ** 2) * 1e-4, cp.array(1e-300)
                            )
                            maxx = cp.max(s_safe, axis=1)
                            maxx = cp.where(maxx > minx, maxx, minx * 10.0)
                            mu_b = cp.exp(
                                cp.log(minx)[:, None]
                                + (cp.log(maxx) - cp.log(minx))[:, None] * tlin[None, :]
                            )
                            coef = cp.matmul(
                                cp.transpose(U, (0, 2, 1)), dprime_b[:, :, None]
                            ).squeeze(-1)
                            # Vendor parity: normalized system -> sum(err^2)=nf
                            err_sq = cp.full((cur,), float(nf), dtype=cp.float64)

                    next_b0 = b1
                    if next_b0 < na:
                        next_cur = min(cur_batch, na - next_b0)
                        _stage_h2d(next_b0, next_cur, 1 - slot)
                        pre_staged = True
                    else:
                        pre_staged = False

                    with nvtx_range("LAMBDA_SELECT", color=0x3949AB):
                        with compute_stream:
                            reg_vec = cp.full(
                                (cur,), float(reg_tweak), dtype=cp.float64
                            )
                            for _it in range(int(max_iter)):
                                vals = (
                                    mu_b[:, None, :]
                                    / (s_safe[:, :, None] ** 2 + mu_b[:, None, :])
                                ) * coef[:, :, None]
                                arg_sum = cp.sum(vals**2, axis=1)
                                discr = arg_sum - (err_sq * reg_vec)[:, None]
                                idx_mu = cp.argmin(cp.abs(discr), axis=1)
                                lamb = mu_b[cp.arange(cur), idx_mu]
                                filt = s / (s**2 + lamb[:, None])
                                xprime = cp.matmul(
                                    V, (filt * coef)[:, :, None]
                                ).squeeze(-1)
                                dem_out = bvec * xprime
                                dn_pred = cp.matmul(dem_out, rmatrix_d)
                                # Use original (unclamped) errors for residuals to match vendor behavior
                                resid = (dn_b - dn_pred) / ed_b_orig
                                chisq_b = cp.sum(resid**2, axis=1) / nf
                                neg_mask = cp.any(dem_out < 0, axis=1)
                                if not bool(cp.any(neg_mask)):
                                    break
                                reg_vec = cp.where(
                                    neg_mask, reg_vec * float(rgt_fact), reg_vec
                                )

                    with nvtx_range("EDEM_ELOGT", color=0x6D4C41):
                        with compute_stream:
                            kVT = V * filt[:, None, :]
                            kdag = cp.matmul(kVT, cp.transpose(U, (0, 2, 1)))
                            kdag = bvec[:, :, None] * kdag
                            edem_b = cp.sqrt(cp.sum(kdag**2, axis=2))

                            kdagk = cp.matmul(
                                kdag, cp.transpose(rmatrixin_b, (0, 2, 1))
                            )
                            j = seg_idx_d
                            left = kdagk[:, :, j]
                            right = kdagk[:, :, j + 1]
                            rr = left + (right - left) * t_frac_d
                            thr = cp.max(kdagk, axis=1) / 2.0
                            hm = rr >= thr[:, :, None]
                            hm_int = hm.astype(cp.int8)
                            first = cp.argmax(hm_int, axis=2)
                            last = (hm.shape[2] - 1) - cp.argmax(
                                hm_int[:, :, ::-1], axis=2
                            )
                            has_any = cp.any(hm, axis=2)
                            width = (ltt_d[last] - ltt_d[first]) / 2.0
                            elogt_b = cp.where(has_any, width, dlogt_d[None, :])

                    if rscl:
                        with nvtx_range("RSCL_ADJUST", color=0x8D6E63):
                            with compute_stream:
                                mnrat = cp.mean(
                                    dn_b / cp.maximum(dn_pred, 1e-300), axis=1
                                )
                                dem_out = dem_out * mnrat[:, None]
                                edem_b = edem_b * mnrat[:, None]
                                dn_pred = cp.matmul(dem_out, rmatrix_d)
                                resid = (dn_b - dn_pred) / ed_b_orig
                                chisq_b = cp.sum(resid**2, axis=1) / nf

                    # Zero outputs for invalid rows (match vendor skip of bad pixels)
                    with nvtx_range("INVALID_ROW_ZERO", color=0x9E9E9E):
                        if "valid_rows" in locals() and bool(cp.any(~valid_rows)):
                            vr = valid_rows
                            dem_out = dem_out.copy()
                            edem_b = edem_b.copy()
                            elogt_b = elogt_b.copy()
                            dn_pred = dn_pred.copy()
                            chisq_b = chisq_b.copy()
                            dem_out[~vr, :] = 0.0
                            edem_b[~vr, :] = 0.0
                            elogt_b[~vr, :] = dlogt_d[None, :]
                            dn_pred[~vr, :] = 0.0
                            chisq_b[~vr] = 0.0
                    with nvtx_range("DEVICE_TO_HOST_ASYNC", color=0x795548):
                        slot_id = (b0 // max(1, cur_batch)) % RING
                        _sync_and_clear_slot(slot_id)

                        # Record compute completion for this batch
                        ready_evt = cp.cuda.Event()
                        ready_evt.record(compute_stream)

                        dem_dst = dem[b0:b1, :]
                        edem_dst = edem[b0:b1, :]
                        elogt_dst = elogt[b0:b1, :]
                        chisq_dst = chisq[b0:b1]
                        dn_reg_dst = dn_reg[b0:b1, :]

                        copy_stream.wait_event(ready_evt)
                        with copy_stream:
                            dem_src = cp.ascontiguousarray(dem_out)
                            edem_src = cp.ascontiguousarray(edem_b)
                            elogt_src = cp.ascontiguousarray(elogt_b)
                            chisq_src = cp.ascontiguousarray(chisq_b)
                            dn_pred_src = cp.ascontiguousarray(dn_pred)
                        try:
                            kind = cp.cuda.runtime.memcpyDeviceToHost  # type: ignore[attr-defined]
                        except AttributeError:  # pragma: no cover - older CuPy
                            kind = cp.cuda.runtime.cudaMemcpyDeviceToHost  # type: ignore[attr-defined]
                        elsize = np.dtype(np.float64).itemsize

                        cp.cuda.runtime.memcpyAsync(
                            dem_dst.ctypes.data,
                            dem_src.data.ptr,
                            int(dem_src.size) * elsize,
                            kind,
                            copy_stream.ptr,
                        )
                        cp.cuda.runtime.memcpyAsync(
                            edem_dst.ctypes.data,
                            edem_src.data.ptr,
                            int(edem_src.size) * elsize,
                            kind,
                            copy_stream.ptr,
                        )
                        cp.cuda.runtime.memcpyAsync(
                            elogt_dst.ctypes.data,
                            elogt_src.data.ptr,
                            int(elogt_src.size) * elsize,
                            kind,
                            copy_stream.ptr,
                        )
                        cp.cuda.runtime.memcpyAsync(
                            chisq_dst.ctypes.data,
                            chisq_src.data.ptr,
                            int(chisq_src.size) * elsize,
                            kind,
                            copy_stream.ptr,
                        )
                        cp.cuda.runtime.memcpyAsync(
                            dn_reg_dst.ctypes.data,
                            dn_pred_src.data.ptr,
                            int(dn_pred_src.size) * elsize,
                            kind,
                            copy_stream.ptr,
                        )

                        done_evt = cp.cuda.Event()
                        done_evt.record(copy_stream)
                        ring[slot_id] = {
                            "in_use": True,
                            "done_evt": done_evt,
                            "keep": (dem_out, edem_b, elogt_b, chisq_b, dn_pred),
                        }

                idx = b1
                cur_batch = attempt
                slot = 1 - slot
                cp.get_default_memory_pool().free_all_blocks()
            except cp.cuda.memory.OutOfMemoryError:
                with nvtx_range("OOM_RETRY", color=0xD32F2F):
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                    attempt2 = attempt
                    try:
                        free_b2, _ = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
                        bps2 = _bytes_per_sample_estimate(nf, nt, nmu)
                        # Use the same configurable fraction for consistency
                        try:
                            frac_env = float(
                                os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7")
                            )
                            mem_frac = float(min(max(frac_env, 0.1), 0.9))
                        except Exception:
                            mem_frac = 0.55
                        est2 = int((free_b2 * mem_frac) // max(1, bps2))
                        attempt2 = max(1, min(est2, na - idx))
                    except Exception:
                        attempt2 = max(1, attempt // 2)
                    if attempt2 >= attempt:
                        attempt2 = max(1, attempt // 2)
                    if attempt2 == attempt:
                        completed_gpu = False
                        break
                    if verbose_enabled():
                        logging.getLogger(__name__).warning(
                            "[metrics] OOM: reducing batch %d -> %d",
                            int(attempt),
                            int(attempt2),
                        )
                    oom_retries += 1
                    cur_batch = attempt2
                    pre_staged = False
                    continue
        for i in range(RING):
            _sync_and_clear_slot(i)

        if completed_gpu:
            if verbose_enabled():
                logging.getLogger(__name__).info(
                    "[metrics] completed on GPU: batches=%d oom_retries=%d",
                    int(math.ceil(na / max(1, batch_size))),
                    int(oom_retries),
                )
            return dem, edem, elogt, chisq, dn_reg
        # If we reach here, we could not complete on GPU even after retries
        raise RuntimeError(
            "GPU OOM: exhausted retries; try reducing MULTIGPU_BATCH_SIZE or input size"
        )


__all__ = ["demmap_pos"]
