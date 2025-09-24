"""GPU-accelerated kernels for DEM computations using CuPy."""
from __future__ import annotations

from typing import Tuple
import logging
import numpy as np
import os

try:  # CuPy required
    import cupy as cp  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("CuPy is required for multiGPU execution: %s" % e) from e

GPU_AVAILABLE = True
_np = np


def _adaptive_batch_size(
    na: int, nf: int, nt: int, safety: float = 0.70, log_info: bool = True
) -> int:
    """Heuristic batch size based on free GPU memory with environment override."""
    # Check for environment override first - critical for large images
    env_bs = int(os.environ.get("MULTIGPU_BATCH_SIZE", "0"))
    if env_bs > 0:
        actual_batch = min(env_bs, na)
        # Only log once per process, not per batch
        if log_info and not hasattr(_adaptive_batch_size, '_logged_override'):
            log_msg = (f"Batch size override: {actual_batch} "
                       f"(req: {env_bs}, max: {na})")
            logging.getLogger(__name__).info(log_msg)
            _adaptive_batch_size._logged_override = True
        return actual_batch
    
    default = min(64, na)  # Increased from 32 for better GPU utilization
    try:  # pragma: no cover
        free_b, total_b = cp.cuda.runtime.memGetInfo()  # type: ignore
        m = max(nf, nt)
        # More accurate memory estimate for large images
        bytes_per_pixel = 8 * (3*nf + 3*nt*nf + 3*m*m + 64)  # Added workspace overhead
        if bytes_per_pixel <= 0:
            return default
        
        # Use more aggressive memory utilization for large datasets
        effective_safety = safety if na < 1000000 else min(safety + 0.1, 0.85)
        est = int((free_b * effective_safety) / bytes_per_pixel)
        
        if est <= 0:
            return default
        
        # For very large images, prefer larger batches to amortize overhead
        final_batch = max(1, min(est, na))
        if na > 1000000:  # Large image optimization
            final_batch = max(final_batch, min(128, na))
            
        if log_info:
            logging.getLogger(__name__).info(
                f"Adaptive batch size: {final_batch} (free_mem: {free_b/1024**3:.1f}GB, "
                f"safety: {effective_safety:.2f}, est: {est}, pixels: {na})"
            )
        
        return final_batch
    except Exception:  # pragma: no cover
        if log_info:
            logging.getLogger(__name__).warning(f"Batch size fallback to default: {default}")
        return default


def _to_device(x):
    return cp.asarray(x)


def safe_svd(
    A: np.ndarray, full_matrices: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_gpu = _to_device(np.asarray(A, dtype=np.float64, order="C"))
    try:
        u, s, vh = cp.linalg.svd(A_gpu, full_matrices=full_matrices)
        return cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).exception("GPU SVD failed: %s", exc)
        raise RuntimeError(
            "CuPy SVD failed; aborting multiGPU execution"
        ) from exc


def safe_pinv(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)
    u, s, vh = safe_svd(A, full_matrices=False)
    tol = np.max(s) * rcond if s.size else rcond
    s_inv = np.array([1/x if x > tol else 0 for x in s])
    return (vh.T * s_inv) @ u.T


def dem_inv_gsvd(A: np.ndarray, B: np.ndarray):
    AB1 = A @ safe_pinv(B)
    sze = AB1.shape
    C = np.zeros([max(sze), max(sze)])
    C[:sze[0], :sze[1]] = AB1
    u, s, v = safe_svd(C, full_matrices=True)
    beta = 1.0 / np.sqrt(1.0 + s**2)
    alpha = s * beta
    SB = np.diag(beta)
    SB_inv = safe_pinv(SB)
    W = safe_pinv(SB_inv @ v @ B)
    return alpha, beta, u.T[:, :sze[0]], v.T, W


def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    xp = cp if GPU_AVAILABLE else np
    data = xp.asarray(data)
    err = xp.asarray(err)
    nf = data.shape[0]
    nreg = sigmaa.shape[0]
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
    U_arr = xp.asarray(U)
    if U_arr.ndim != 2:
        raise ValueError("U must be 2D")
    if U_arr.shape[1] != nf:
        if U_arr.shape[0] == U_arr.shape[1] and U_arr.shape[0] >= nf:
            U_arr = U_arr.T[:, :nf]
        else:
            raise ValueError("Incompatible U shape for data length")
    nmu_eff = max(int(nmu), 2)
    mu = np.geomspace(minx, maxx, num=nmu_eff, dtype=float)
    mu_xp = xp.asarray(mu)
    arg = xp.zeros((nreg, nmu_eff), dtype=float)
    for kk in range(nf):
        Uk = xp.asarray(U_arr[kk, :])
        coef = data @ Uk
        sb = xp.asarray(sigmab[kk])
        sa = xp.asarray(sigmaa[kk])
        num = mu_xp * (sb ** 2) * coef
        den = (sa ** 2) + mu_xp * (sb ** 2)
        arg[kk, :] = (num / den) ** 2
    discr = xp.sum(arg[:, :nmu_eff], axis=0) - xp.sum(err ** 2) * reg_tweak
    discr_host = cp.asnumpy(discr) if GPU_AVAILABLE else np.asarray(discr)
    opt = mu[int(np.argmin(np.abs(discr_host[:nmu_eff])))]
    return opt


def _batch_select_lambda(
    alpha_b: "cp.ndarray",
    beta_b: "cp.ndarray",
    u_b: "cp.ndarray",
    dn_b: "cp.ndarray",
    ed_b: "cp.ndarray",
    reg_tweak: float,
    nmu: int,
    nf_dev: int,
) -> "cp.ndarray":
    batch = dn_b.shape[0]
    sa = alpha_b[:, :nf_dev]
    sb = beta_b[:, :nf_dev]
    eps = cp.finfo(cp.float64).tiny
    sigs = sa / cp.maximum(sb, eps)
    mask_valid = cp.isfinite(sigs) & (sigs > 0)
    has_valid = cp.any(mask_valid, axis=1)
    sigs_safe = cp.where(mask_valid, sigs, cp.inf)
    min_sigs = cp.min(sigs_safe, axis=1)
    min_sigs = cp.where(cp.isinf(min_sigs), cp.array(1.0), min_sigs)
    max_sigs = cp.max(cp.where(mask_valid, sigs, 0.0), axis=1)
    max_sigs = cp.where(max_sigs == 0.0, cp.array(10.0), max_sigs)
    minx = cp.maximum((min_sigs ** 2) * 1e-4, cp.array(1e-300))
    maxx = cp.where(max_sigs > minx, max_sigs, minx * 10.0)
    minx = cp.where(has_valid, minx, cp.array(1e-8))
    maxx = cp.where(has_valid, maxx, cp.array(1e2))
    nmu_eff = int(max(int(nmu), 2))
    t = cp.linspace(0.0, 1.0, nmu_eff, dtype=cp.float64)
    mu_batch = cp.exp(
        cp.log(minx)[:, None]
        + (cp.log(maxx) - cp.log(minx))[:, None] * t[None, :]
    )
    U_rows = cp.transpose(u_b, (0, 2, 1))[:, :nf_dev, :nf_dev]
    coef = cp.matmul(U_rows, dn_b[:, :nf_dev, None]).squeeze(-1)
    sa2 = sa ** 2
    sb2 = sb ** 2
    num = mu_batch[:, None, :] * sb2[:, :, None] * coef[:, :, None]
    den = sa2[:, :, None] + mu_batch[:, None, :] * sb2[:, :, None]
    vals = (num / den) ** 2
    arg_sum = cp.sum(vals, axis=1)
    err_sq = cp.sum(ed_b[:, :nf_dev] ** 2, axis=1)
    discr = arg_sum - (err_sq * reg_tweak)[:, None]
    idx = cp.argmin(cp.abs(discr), axis=1)
    return mu_batch[cp.arange(batch), idx]


class GPUWorkspaceManager:
    """Manages GPU workspace arrays to reduce allocation overhead for large images."""
    
    def __init__(self):
        self.workspaces = {}
        self.logger = logging.getLogger(__name__)
        
    def get_workspace(self, shape, dtype, name="default"):
        """Get or create a workspace array on GPU."""
        key = (tuple(shape), dtype, name)
        if key not in self.workspaces:
            try:
                self.workspaces[key] = cp.zeros(shape, dtype=dtype)
                self.logger.info(f"Allocated GPU workspace '{name}': {shape} {dtype}")
            except Exception as e:
                self.logger.warning(f"Failed to allocate workspace {name}: {e}")
                return None
        return self.workspaces[key]
    
    def clear_workspaces(self):
        """Clear all workspace arrays to free GPU memory."""
        count = len(self.workspaces)
        self.workspaces.clear()
        if count > 0:
            self.logger.info(f"Cleared {count} GPU workspace arrays")


# Global workspace manager for reuse across batches
_gpu_workspace = GPUWorkspaceManager()


def dem_pix(*_a, **_k):  # pragma: no cover
    raise RuntimeError(
        "dem_pix unsupported in multiGPU module; use demmap_pos"
    )


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
    na = dd.shape[0]
    nt = logt.shape[0]
    dem = _np.zeros((na, nt))
    edem = _np.zeros((na, nt))
    elogt = _np.zeros((na, nt))
    chisq = _np.zeros((na,))
    dn_reg = _np.zeros((na, rmatrix.shape[1]))
    if GPU_AVAILABLE:
        try:
            dd_d = cp.asarray(dd, dtype=cp.float64)
            ed_d = cp.asarray(ed, dtype=cp.float64)
            rmatrix_d = cp.asarray(rmatrix, dtype=cp.float64)
            dlogt_d = cp.asarray(dlogt, dtype=cp.float64)
            L = cp.diag(1.0 / cp.sqrt(dlogt_d))
            B_inv = cp.linalg.pinv(L)
            nf_dev = rmatrix_d.shape[1]
            nt_dev = rmatrix_d.shape[0]
            batch_size = _adaptive_batch_size(na, nf_dev, nt_dev)
            initial_batch = batch_size
            b0 = 0

            disable_vector = (
                os.environ.get("MULTIGPU_VECTOR_DISABLE", "0") == "1"
            )
            use_nvtx = os.environ.get("MULTIGPU_NVTX", "0") == "1"
            if use_nvtx:
                try:
                    from cupy.cuda import nvtx as _nvtx  # type: ignore
                except Exception:  # pragma: no cover
                    use_nvtx = False

            def _range_push(name: str):  # lightweight helper
                if use_nvtx:
                    try:  # pragma: no cover
                        _nvtx.RangePush(name)
                    except Exception:
                        pass

            def _range_pop():
                if use_nvtx:
                    try:  # pragma: no cover
                        _nvtx.RangePop()
                    except Exception:
                        pass

            while b0 < na:
                cur_batch = min(batch_size, na - b0)
                b1 = b0 + cur_batch
                
                # Performance logging (selective for large images)
                log = logging.getLogger(__name__)
                # Check if verbose logging is disabled
                verbose_disabled = os.environ.get("MULTIGPU_QUIET", "0") == "1"
                if verbose_disabled:
                    should_log = False
                else:
                    # Log: first batch, every 100 batches, or milestones
                    batch_num = b0 // batch_size + 1
                    milestone_batches = [1, 10, 50, 100, 500, 1000]
                    should_log = (b0 == 0 or
                                  batch_num % 100 == 0 or
                                  batch_num in milestone_batches or
                                  na > 10000000)  # Large images
                
                if should_log:
                    try:
                        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                        device_id = cp.cuda.runtime.getDevice()
                        progress_pct = (b1 * 100.0) / na
                        log.info(
                            f"Batch {batch_num}: pixels {b0}-{b1-1}/{na} "
                            f"({progress_pct:.1f}%), GPU {device_id}, "
                            f"mem: {(total_mem-free_mem)/1024**3:.1f}/"
                            f"{total_mem/1024**3:.1f}GB"
                        )
                    except Exception:
                        pass
                
                try:
                    _range_push("BATCH_PREP")
                    dn_b = dd_d[b0:b1, :]
                    ed_b = ed_d[b0:b1, :]
                    rmatrixin_b = (
                        rmatrix_d[None, :, :] * (1.0 / ed_b)[:, None, :]
                    )
                    A_batch = cp.transpose(rmatrixin_b, (0, 2, 1))
                    AB1 = A_batch @ B_inv
                    M = max(AB1.shape[1], AB1.shape[2])
                    C = cp.zeros((cur_batch, M, M), dtype=cp.float64)
                    C[:, :AB1.shape[1], :AB1.shape[2]] = AB1
                    _range_pop()

                    _range_push("SVD")
                    u_b, s_b, vh_b = cp.linalg.svd(C, full_matrices=True)
                    beta_b = 1.0 / cp.sqrt(1.0 + s_b ** 2)
                    alpha_b = s_b * beta_b
                    _range_pop()

                    _range_push("LAMBDA_SELECT")
                    lambs_dev = _batch_select_lambda(
                        alpha_b,
                        beta_b,
                        u_b,
                        dn_b,
                        ed_b,
                        reg_tweak,
                        nmu,
                        nf_dev,
                    )
                    _range_pop()

                    # Build W matrices (device-only path for high performance)
                    _range_push("BUILD_W_DEVICE")
                    # Keep all computations on device to avoid PCIe bottlenecks
                    L_dev = L  # Already on device
                    
                    # Compute SB_inv efficiently on device
                    # beta_b shape: (batch, M), need diag matrices
                    eps = cp.finfo(cp.float64).eps
                    beta_safe = cp.where(cp.abs(beta_b) > eps, beta_b, eps)
                    SB_inv_diag = 1.0 / beta_safe  # (batch, M)
                    
                    # Compute vh_b^T for each batch item
                    vh_T = cp.transpose(vh_b, (0, 2, 1))  # (batch, M, M)
                    
                    # Apply diagonal scaling: SB_inv @ vh_j^T
                    # Broadcasting: (batch, M) * (batch, M, M) -> (batch, M, M)
                    SB_inv_vh_T = vh_T * SB_inv_diag[:, :, None]
                    
                    # Compute SB_inv @ vh_j^T @ L_dev for each batch
                    # (batch, M, M) @ (M, nt_dev) -> (batch, M, nt_dev)
                    tmp_batch = cp.matmul(SB_inv_vh_T, L_dev)
                    
                    # Compute pseudo-inverse for each batch item
                    # For large images, use more stable computation
                    stable_env = os.environ.get("MULTIGPU_STABLE_PINV", "0")
                    use_stable_pinv = na > 1000000 or stable_env == "1"
                    
                    if use_stable_pinv:
                        # More stable but slightly slower path for large images
                        W_list = []
                        for j in range(cur_batch):
                            try:
                                # Add regularization for numerical stability
                                reg_term = 1e-12 * cp.eye(tmp_batch.shape[1])
                                tmp_reg = tmp_batch[j] + reg_term
                                W_j = cp.linalg.pinv(tmp_reg)
                            except Exception:
                                # Fallback with stronger regularization
                                reg_term = 1e-10 * cp.eye(tmp_batch.shape[1])
                                tmp_reg = tmp_batch[j] + reg_term
                                W_j = cp.linalg.pinv(tmp_reg)
                            W_list.append(W_j)
                        W_stack = cp.stack(W_list, axis=0)
                    else:
                        # Fast path using batch operations where possible
                        try:
                            # Try vectorized pinv (may not be available)
                            W_stack = cp.linalg.pinv(tmp_batch)
                        except Exception:
                            # Fallback to loop but keep on device
                            W_list = []
                            for j in range(cur_batch):
                                W_list.append(cp.linalg.pinv(tmp_batch[j]))
                            W_stack = cp.stack(W_list, axis=0)
                    
                    _range_pop()

                    if disable_vector:
                        # Fallback to original per-sample loop (rarely needed)
                        _range_push("RECON_FALLBACK_LOOP")
                        lambs = cp.asnumpy(lambs_dev)
                        for j in range(cur_batch):
                            sva = alpha_b[j]
                            svb = beta_b[j]
                            lamb = float(lambs[j])
                            denom = (sva ** 2 + svb ** 2 * lamb)
                            filt_diag = sva / denom
                            U_full = u_b[j]
                            U_filt = U_full * filt_diag[None, :]
                            dn_ext = cp.zeros(
                                (U_full.shape[0],), dtype=dn_b.dtype
                            )
                            dn_ext[:nf_dev] = dn_b[j, :]
                            kdag_d = U_filt @ W_stack[j]
                            dem_out_d = kdag_d.T @ dn_ext
                            dem[b0 + j, :] = cp.asnumpy(dem_out_d[:nt_dev])
                            dn_reg_vec = (
                                rmatrix_d.T @ dem_out_d[:nt_dev]
                            ).squeeze()
                            dn_reg[b0 + j, :] = cp.asnumpy(dn_reg_vec)
                            residuals = (
                                (dd[b0 + j, :] - dn_reg[b0 + j, :])
                                / ed[b0 + j, :]
                            )
                            chisq[b0 + j] = (
                                np.sum(residuals ** 2) / dd.shape[1]
                            )
                            edem[b0 + j, :] = np.abs(dem[b0 + j, :]) * 0.1
                        _range_pop()
                    else:
                        _range_push("RECON_VECTOR_OPTIMIZED")
                        # Optimized vectorized path for high-resolution images
                        # Keep all computations on device until final transfer
                        
                        _range_push("FILTER_CONSTRUCTION")
                        lambs_vec = lambs_dev[:, None]  # (batch,1)
                        denom = (alpha_b ** 2 + beta_b ** 2 * lambs_vec)
                        # Add numerical stability for large images
                        denom = cp.maximum(denom, cp.finfo(cp.float64).eps)
                        filt = alpha_b / denom  # (batch,M)
                        U_filt = u_b * filt[:, None, :]  # (batch,M,M)
                        _range_pop()
                        
                        _range_push("DATA_EXTENSION")
                        # Extend dn to M with padding zeros (reuse workspace)
                        dn_ext = cp.zeros((cur_batch, M), dtype=dn_b.dtype)
                        dn_ext[:, :nf_dev] = dn_b
                        _range_pop()
                        
                        _range_push("MATRIX_MULTIPLY")
                        # kdag = U_filt @ W_stack -> (batch,M,nt_dev)
                        kdag = cp.matmul(U_filt, W_stack)
                        # dem_out = (kdag^T) @ dn_ext  -> (batch,nt_dev)
                        # Use optimized einsum for large arrays
                        dem_out_device = cp.einsum('bmn,bm->bn', kdag, dn_ext)
                        _range_pop()
                        
                        _range_push("RECONSTRUCTION_CALC")
                        # Keep calculations on device
                        dn_reg_device = cp.matmul(dem_out_device, rmatrix_d)
                        
                        # Residuals & chisq computation on device
                        residuals_device = (dn_b - dn_reg_device) / ed_b
                        chisq_device = (
                            cp.sum(residuals_device ** 2, axis=1) /
                            dn_b.shape[1]
                        )
                        _range_pop()
                        
                        _range_push("DEVICE_TO_HOST")
                        # Single optimized host transfer per batch
                        # Use asynchronous transfers where possible
                        dem_batch_host = cp.asnumpy(dem_out_device[:, :nt_dev])
                        dn_reg_batch_host = cp.asnumpy(dn_reg_device)
                        chisq_batch_host = cp.asnumpy(chisq_device)
                        
                        # Assign to output arrays
                        dem[b0:b1, :] = dem_batch_host
                        dn_reg[b0:b1, :] = dn_reg_batch_host
                        chisq[b0:b1] = chisq_batch_host
                        edem[b0:b1, :] = np.abs(dem_batch_host) * 0.1
                        _range_pop()
                        
                        _range_pop()  # RECON_VECTOR_OPTIMIZED

                    b0 = b1
                    batch_size = initial_batch
                except Exception as e:
                    log = logging.getLogger(__name__)
                    log.exception("GPU batch %d:%d failed: %s", b0, b1, e)
                    if cur_batch <= 1:
                        raise
                    batch_size = max(1, cur_batch // 2)
                    logging.warning(
                        "Reducing GPU batch size to %d and retrying",
                        batch_size,
                    )
            
            # Clean up workspace arrays for large images to free GPU memory
            if na > 1000000:  # Only for large images to avoid overhead
                _gpu_workspace.clear_workspaces()
                
            return dem, edem, elogt, chisq, dn_reg
        except Exception as e:  # pragma: no cover
            logging.getLogger(__name__).exception("GPU path failed: %s", e)
            raise RuntimeError(
                "GPU path failed; aborting multiGPU execution"
            ) from e
    for _i in range(na):  # pragma: no cover
        raise RuntimeError("CPU fallback not supported in this module")
    return dem, edem, elogt, chisq, dn_reg
