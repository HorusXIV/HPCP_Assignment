"""GPU-accelerated kernels used by the multi-GPU DEM solver.

This module provides the hot-path numerical routines implemented with CuPy
and optional CUDA streams. It also contains small CPU fallbacks for sanity
checks and unit tests, but production use requires a CUDA-capable GPU and
CuPy.

Terminology
- DEM: Differential Emission Measure reconstruction per spatial sample.
- Batch: A contiguous slice of samples processed together on a GPU.

Environment flags
- MULTIGPU_BATCH_SIZE: int, override the auto batch size (0 = auto).
- MULTIGPU_NVTX: "1" to enable NVTX ranges inside kernels.
- MULTIGPU_STREAMS: "1" to enable async D2H overlap via CUDA streams.
- MULTIGPU_STREAMS_DEPTH: ring-buffer depth for pinned host staging.
- MULTIGPU_NO_FUSE: "1" to disable cp.fuse() for debug/determinism.
- MULTIGPU_VERBOSE: "1" to enable extra informational logging.
"""

from __future__ import annotations

from typing import Tuple
from collections import deque
import logging
import numpy as np
import os

try:  # CuPy required
    import cupy as cp  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("CuPy is required for multiGPU execution: %s" % e) from e

_np = np


def _adaptive_batch_size(
    na: int,
    nf: int,
    nt: int,
    nmu: int,
    safety: float = 0.70,
    log_info: bool = True,
) -> int:
    """Choose a batch size based on current free GPU memory.

    The calculation includes dominant per-pixel tensor terms for SVD and
    lambda selection to avoid oversizing batches on large images.

    Args:
        na: Total number of samples (pixels) to process.
        nf: Number of bands/filters per sample.
        nt: Number of temperature bins (response rows).
        nmu: Count of candidate regularization strengths.
        safety: Fraction of free memory to target (0 < safety <= 1).
        log_info: Whether to log informational details once per process.

    Returns:
        An integer batch size in [1, na]. Environment variable
        ``MULTIGPU_BATCH_SIZE`` overrides this heuristic when set > 0.
    """
    # Check for environment override first - critical for large images
    env_bs = int(os.environ.get("MULTIGPU_BATCH_SIZE", "0"))
    if env_bs > 0:
        actual_batch = min(env_bs, na)
        # Only log once per process, not per batch
        if log_info and not hasattr(_adaptive_batch_size, "_logged_override"):
            log_msg = f"Batch size override: {actual_batch} (req: {env_bs}, max: {na})"
            logging.getLogger(__name__).info(log_msg)
            _adaptive_batch_size._logged_override = True
        return actual_batch

    default = min(64, na)  # Increased from 32 for better GPU utilization
    try:  # pragma: no cover
        free_b, total_b = cp.cuda.runtime.memGetInfo()  # type: ignore
        # Per-pixel memory estimate of dominant tensors present during
        # SVD + lambda selection. Use k = min(nf, nt).
        k = min(nf, nt)
        # SVD-related: AB1 (nf*nt), U (nf*k), s (k), Vh (k*nt)
        svd_terms = (nf * nt) + (nf * k) + k + (k * nt)
        # Lambda-selection dominant 3D tensors: f and vals ~ 2 * (k * nmu)
        # plus per-pixel vectors (arg/discr/mu) ~ O(nmu).
        lambda_terms = (2 * k * max(nmu, 2)) + (2 * max(nmu, 2))
        # Modest constant overhead per pixel
        bytes_per_pixel = 8 * (svd_terms + lambda_terms + 64)
        if bytes_per_pixel <= 0:
            return default

        # Adjust safety conservatively for large images or large nmu to
        # accommodate allocator fragmentation and concurrent buffers.
        effective_safety = safety
        if na >= 1_000_000 or nmu >= 32:
            effective_safety = max(safety - 0.10, 0.50)
        est = int((free_b * effective_safety) / bytes_per_pixel)

        if est <= 0:
            return default

        # For very large images, prefer larger batches to amortize overhead
        final_batch = max(1, min(est, na))
        if na > 1000000:  # Large image optimization
            final_batch = max(final_batch, min(128, na))

        verbose = os.environ.get("MULTIGPU_VERBOSE", "0") == "1"
        if log_info and verbose:
            free_gb = free_b / 1024**3
            logging.getLogger(__name__).info(
                "Adaptive batch size: %d (free_mem: %.1fGB, safety: %.2f, est: %d, pixels: %d, nf: %d, nt: %d, nmu: %d, k: %d)",
                final_batch,
                free_gb,
                effective_safety,
                est,
                na,
                nf,
                nt,
                nmu,
                k,
            )

        return final_batch
    except Exception:  # pragma: no cover
        verbose = os.environ.get("MULTIGPU_VERBOSE", "0") == "1"
        if log_info and verbose:
            logging.getLogger(__name__).warning(
                "Batch size fallback to default: %d", default
            )
        return default


def _to_device(x):
    """Transfer a NumPy array-like to the current GPU device.

    Args:
        x: Array-like object convertible to ``cp.ndarray``.

    Returns:
        cupy.ndarray on the active device.
    """
    return cp.asarray(x)


def safe_svd(
    A: np.ndarray, full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD on GPU with CPU NumPy arrays returned.

    Args:
        A: Input matrix (will be copied to device as float64, C-order).
        full_matrices: Whether to compute full-sized U/VH.

    Returns:
        Tuple of (U, S, VH) as NumPy arrays.

    Raises:
        RuntimeError: If the GPU SVD fails for any reason.
    """
    A_gpu = _to_device(np.asarray(A, dtype=np.float64, order="C"))
    try:
        u, s, vh = cp.linalg.svd(A_gpu, full_matrices=full_matrices)
        return cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vh)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).exception("GPU SVD failed: %s", exc)
        raise RuntimeError("CuPy SVD failed; aborting multiGPU execution") from exc


def safe_pinv(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Numerically robust Moore–Penrose pseudoinverse on GPU.

    Applies clipping and NaN/inf sanitization before SVD to improve
    stability with ill-conditioned inputs.

    Args:
        A: Input matrix (NumPy) to invert.
        rcond: Relative threshold for singular value truncation.

    Returns:
        NumPy array containing the pseudoinverse of ``A``.
    """
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)
    u, s, vh = safe_svd(A, full_matrices=False)
    tol = np.max(s) * rcond if s.size else rcond
    s_inv = np.array([1 / x if x > tol else 0 for x in s])
    return (vh.T * s_inv) @ u.T


def dem_inv_gsvd(A: np.ndarray, B: np.ndarray):
    """Compute GSVD-like factors used in the DEM reconstruction.

    Args:
        A: Response matrix A (nf x nt or compatible).
        B: Diagonal scaling matrix B.

    Returns:
        Tuple ``(alpha, beta, U_T, V_T, W)`` consistent with the DEM
        formulation used by the solver.
    """
    AB1 = A @ safe_pinv(B)
    sze = AB1.shape
    C = np.zeros([max(sze), max(sze)])
    C[: sze[0], : sze[1]] = AB1
    u, s, v = safe_svd(C, full_matrices=False)
    beta = 1.0 / np.sqrt(1.0 + s**2)
    alpha = s * beta
    SB = np.diag(beta)
    SB_inv = safe_pinv(SB)
    W = safe_pinv(SB_inv @ v @ B)
    return alpha, beta, u.T[:, : sze[0]], v.T, W


def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    """Select Tikhonov regularization strength for a single pixel.

    Args:
        sigmaa: Singular values for A.
        sigmab: Singular values for B.
        U: Left singular vectors (2D array) compatible with ``data``.
        W: Transformation matrix used in DEM reconstruction.
        data: Observed data vector ``d`` (nf,).
        err: Uncertainty vector ``ed`` (nf,).
        reg_tweak: Scalar factor multiplying the chi-square target.
        nmu: Number of candidates in the geometric grid over ``mu``.

    Returns:
        Selected regularization strength (float).
    """
    data = cp.asarray(data)
    err = cp.asarray(err)
    nf = data.shape[0]
    eps = cp.finfo(float).tiny
    sigs = cp.asarray(sigmaa[:nf]) / cp.maximum(cp.asarray(sigmab[:nf]), eps)
    sigs = sigs[cp.isfinite(sigs) & (sigs > 0)]
    if sigs.size == 0:
        minx, maxx = 1e-8, 1e2
    else:
        maxx = float(cp.max(sigs))
        minx = float((cp.min(sigs) ** 2) * 1e-4)
        minx = max(minx, 1e-300)
        if not (maxx > minx):
            maxx = minx * 10.0
    U_arr = cp.asarray(U)
    if U_arr.ndim != 2:
        raise ValueError("U must be 2D")
    if U_arr.shape[1] != nf:
        if U_arr.shape[0] == U_arr.shape[1] and U_arr.shape[0] >= nf:
            U_arr = U_arr.T[:, :nf]
        else:
            raise ValueError("Incompatible U shape for data length")
    # Ensure we can index rows 0..nf-1
    if U_arr.shape[0] < nf:
        raise ValueError("U must have at least nf rows after adjustment")
    nmu_eff = max(int(nmu), 2)
    mu = np.geomspace(minx, maxx, num=nmu_eff, dtype=float)
    mu_xp = cp.asarray(mu)
    # Use nf rows to safely index arg[kk] for kk in range(nf)
    arg = cp.zeros((nf, nmu_eff), dtype=float)
    for kk in range(nf):
        Uk = cp.asarray(U_arr[kk, :])
        coef = data @ Uk
        sb = cp.asarray(sigmab[kk])
        sa = cp.asarray(sigmaa[kk])
        num = mu_xp * (sb**2) * coef
        den = (sa**2) + mu_xp * (sb**2)
        arg[kk, :] = (num / den) ** 2
    discr = cp.sum(arg[:, :nmu_eff], axis=0) - cp.sum(err**2) * reg_tweak
    discr_host = cp.asnumpy(discr)
    opt = mu[int(np.argmin(np.abs(discr_host[:nmu_eff])))]
    return float(opt)


def _batch_select_lambda(
    u_b: "cp.ndarray",
    s_b: "cp.ndarray",
    dn_b: "cp.ndarray",
    ed_b: "cp.ndarray",
    reg_tweak: float,
    nmu: int,
) -> "cp.ndarray":
    """Vectorized lambda selection for a batch of samples on device.

    Args:
        u_b: Batched U matrices (batch, nf, k).
        s_b: Batched singular values (batch, k).
        dn_b: Batched data (batch, nf).
        ed_b: Batched uncertainties (batch, nf).
        reg_tweak: Target chi-square scaling factor.
        nmu: Number of candidate ``mu`` values to test.

    Returns:
        Vector of selected ``mu`` of shape ``(batch,)`` on device.
    """
    batch = dn_b.shape[0]
    # d' = d / ed
    dprime = dn_b / ed_b
    # c = U^T d'
    coef = cp.matmul(cp.transpose(u_b, (0, 2, 1)), dprime[:, :, None]).squeeze(
        -1
    )  # (batch, k)
    eps = cp.finfo(cp.float64).tiny
    s_safe = cp.maximum(s_b, eps)
    minx = cp.maximum((cp.min(s_safe, axis=1) ** 2) * 1e-4, cp.array(1e-300))
    maxx = cp.max(s_safe, axis=1)
    maxx = cp.where(maxx > minx, maxx, minx * 10.0)
    nmu_eff = int(max(int(nmu), 2))
    t = cp.linspace(0.0, 1.0, nmu_eff, dtype=cp.float64)
    mu_batch = cp.exp(
        cp.log(minx)[:, None] + (cp.log(maxx) - cp.log(minx))[:, None] * t[None, :]
    )
    s2 = s_b**2
    f = mu_batch[:, None, :] / (s2[:, :, None] + mu_batch[:, None, :])
    vals = (f * coef[:, :, None]) ** 2
    arg_sum = cp.sum(vals, axis=1)
    err_sq = cp.sum(ed_b**2, axis=1)
    discr = arg_sum - (err_sq * reg_tweak)[:, None]
    idx = cp.argmin(cp.abs(discr), axis=1)
    return mu_batch[cp.arange(batch), idx]


class GPUWorkspaceManager:
    """Manage reusable GPU workspaces/pools for large images.

    The manager caches arrays keyed by shape/dtype/name, reducing allocator
    overhead when processing many batches. When CuPy's default memory pool is
    available, it is used to back allocations for improved performance.
    """

    def __init__(self):
        self.workspaces = {}
        self.logger = logging.getLogger(__name__)
        # Guard for reduced CuPy shims in tests
        _get_pool = getattr(cp, "get_default_memory_pool", None)
        _get_pinned = getattr(cp, "get_default_pinned_memory_pool", None)
        self.memory_pool = _get_pool() if callable(_get_pool) else None
        self.pinned_pool = _get_pinned() if callable(_get_pinned) else None

    def get_workspace(self, shape, dtype, name="default"):
        """Get or create a workspace array on the GPU.

        Args:
            shape: Array shape.
            dtype: NumPy/CuPy dtype.
            name: Logical name used as part of the cache key.

        Returns:
            cupy.ndarray or ``None`` if allocation fails.
        """
        key = (tuple(shape), dtype, name)
        if key not in self.workspaces:
            try:
                # Use memory pool for faster allocation/deallocation
                self.workspaces[key] = cp.zeros(shape, dtype=dtype)
                self.logger.info(
                    "Allocated GPU workspace '%s': %s %s",
                    name,
                    shape,
                    dtype,
                )
            except Exception as e:
                self.logger.warning("Failed to allocate workspace %s: %s", name, e)
                return None
        return self.workspaces[key]

    def get_workspace_with_pool(self, shape, dtype, name="default"):
        """Get a pooled workspace array when a CuPy memory pool is present.

        Args:
            shape: Array shape.
            dtype: NumPy/CuPy dtype.
            name: Logical name used as part of the cache key.

        Returns:
            cupy.ndarray or ``None`` if allocation fails.
        """
        key = (tuple(shape), dtype, name)
        if key not in self.workspaces:
            try:
                # Use the memory pool via CuPy's allocator context manager
                # MemoryPool itself is not a context manager; use
                # cp.cuda.using_allocator with pool.malloc
                # If memory_pool is unavailable (e.g., test shim), fall back
                if self.memory_pool is not None:
                    with cp.cuda.using_allocator(self.memory_pool.malloc):
                        self.workspaces[key] = cp.zeros(shape, dtype=dtype)
                else:
                    self.workspaces[key] = cp.zeros(shape, dtype=dtype)
                self.logger.info(
                    "Allocated GPU workspace (pooled) '%s': %s %s",
                    name,
                    shape,
                    dtype,
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to allocate pooled workspace %s: %s", name, e
                )
                return None
        return self.workspaces[key]

    def clear_workspaces(self):
        """Clear cached arrays and free CuPy pool blocks if available."""
        count = len(self.workspaces)
        self.workspaces.clear()
        # Force memory pool cleanup for large datasets
        if count > 0 and self.memory_pool is not None:
            try:
                self.memory_pool.free_all_blocks()
            except Exception:
                pass
            self.logger.info(
                f"Cleared {count} GPU workspace arrays and freed memory pool"
            )

    def get_memory_usage(self):
        """Return memory pool usage statistics.

        Returns:
            Dict with keys ``used_mb``, ``total_mb``, ``utilization``.
        """
        try:
            if self.memory_pool is None:
                return {"used_mb": 0, "total_mb": 0, "utilization": 0}
            used_bytes = self.memory_pool.used_bytes()
            total_bytes = self.memory_pool.total_bytes()
            return {
                "used_mb": used_bytes / (1024**2),
                "total_mb": total_bytes / (1024**2),
                "utilization": used_bytes / max(total_bytes, 1),
            }
        except Exception:
            return {"used_mb": 0, "total_mb": 0, "utilization": 0}


# Global workspace manager for reuse across batches
_gpu_workspace = GPUWorkspaceManager()


"""Initialize cp.fuse helpers once per process; provide safe fallbacks."""
_FUSE_OK = True
try:

    @cp.fuse()
    def _fused_residuals_chisq(dn_b, dn_reg_device, ed_b, nf):
        residuals_device = (dn_b - dn_reg_device) / ed_b
        chisq_device = cp.sum(residuals_device**2, axis=1) / nf
        return residuals_device, chisq_device
except Exception:
    _FUSE_OK = False

    def _fused_residuals_chisq(dn_b, dn_reg_device, ed_b, nf):  # type: ignore
        residuals_device = (dn_b - dn_reg_device) / ed_b
        chisq_device = cp.sum(residuals_device**2, axis=1) / nf
        return residuals_device, chisq_device


def _residuals_and_chisq(dn_b, dn_reg_device, ed_b, nf):
    """Compute residuals and per-sample chi-square on device.

    Uses a fused kernel when available unless disabled via
    ``MULTIGPU_NO_FUSE=1``.

    Args:
        dn_b: Batched observed data (batch, nf).
        dn_reg_device: Batched model predictions (batch, nf) on device.
        ed_b: Batched uncertainties (batch, nf) on device.
        nf: Number of filters (normalization factor).

    Returns:
        Tuple ``(residuals_device, chisq_device)`` on device.
    """
    # Allow disabling the fused kernel for determinism/debugging
    if os.environ.get("MULTIGPU_NO_FUSE", "0") == "1":
        residuals_device = (dn_b - dn_reg_device) / ed_b
        chisq_device = cp.sum(residuals_device**2, axis=1) / nf
        return residuals_device, chisq_device
    return _fused_residuals_chisq(dn_b, dn_reg_device, ed_b, nf)


def dem_pix(*_a, **_k):  # pragma: no cover
    """Legacy single-pixel entry point.

    Raises:
        RuntimeError: Always; use :func:`demmap_pos` instead.
    """
    raise RuntimeError("dem_pix unsupported in multiGPU module; use demmap_pos")


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
    nmu=24,
    warn=False,
    l_emd=False,
    rscl=False,
):
    """Reconstruct DEM for many samples on GPU with batched processing.

    This is the multi-sample GPU implementation used by the orchestrator.
    It performs a rectangular SVD per batch and selects a Tikhonov
    regularization parameter per sample, optionally overlapping compute and
    transfers using CUDA streams and pinned host memory.

    Args:
        dd: Observed data matrix with shape ``(na, nf)``.
        ed: Uncertainty matrix with shape ``(na, nf)``.
        rmatrix: Temperature response matrix of shape ``(nt, nf)``.
        logt: Temperature grid ``log10(T)`` of shape ``(nt,)``.
        dlogt: Bin widths for ``logt`` of shape ``(nt,)``.
        glc: Gain/linear coefficients per filter (unused placeholder).
        reg_tweak: Chi-square target scaling factor.
        max_iter: Unused (kept for API compatibility).
        rgt_fact: Unused (kept for API compatibility).
        dem_norm0: Unused (kept for API compatibility).
        nmu: Number of candidate ``mu`` values for regularization.
        warn: Unused (kept for API compatibility).
        l_emd: Unused (kept for API compatibility).
        rscl: Unused (kept for API compatibility).

    Returns:
        Tuple of NumPy arrays ``(dem, edem, elogt, chisq, dn_reg)`` where
        ``dem`` has shape ``(na, nt)`` and ``dn_reg`` has shape ``(na, nf)``.

    Raises:
        RuntimeError: If the GPU path fails during processing.
    """
    na = dd.shape[0]
    nt = logt.shape[0]
    dem = _np.zeros((na, nt))
    edem = _np.zeros((na, nt))
    elogt = _np.zeros((na, nt))
    chisq = _np.zeros((na,))
    dn_reg = _np.zeros((na, rmatrix.shape[1]))
    try:
        dd_d = cp.asarray(dd, dtype=cp.float64)
        ed_d = cp.asarray(ed, dtype=cp.float64)
        rmatrix_d = cp.asarray(rmatrix, dtype=cp.float64)
        dlogt_d = cp.asarray(dlogt, dtype=cp.float64)
        L = cp.diag(1.0 / cp.sqrt(dlogt_d))
        B_inv = cp.linalg.pinv(L)
        nf_dev = rmatrix_d.shape[1]
        nt_dev = rmatrix_d.shape[0]
        batch_size = _adaptive_batch_size(na, nf_dev, nt_dev, nmu)
        initial_batch = batch_size
        b0 = 0

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

        # Optional CUDA Streams + pinned-memory async D2H to overlap
        # compute and transfers. Enable with MULTIGPU_STREAMS=1 (default).
        use_streams = os.environ.get("MULTIGPU_STREAMS", "1") == "1"
        have_cuda = hasattr(cp, "cuda") and hasattr(cp.cuda, "Stream")
        have_memcpy_async = hasattr(cp, "cuda") and hasattr(cp.cuda, "runtime")
        have_pinned = hasattr(cp, "cuda") and hasattr(cp.cuda, "alloc_pinned_memory")

        streams_enabled = (
            use_streams and have_cuda and have_memcpy_async and have_pinned
        )

        # Double/triple buffering depth for host-pinned staging
        buf_depth = (
            max(2, int(os.environ.get("MULTIGPU_STREAMS_DEPTH", "2")))
            if streams_enabled
            else 0
        )

        # Pre-create streams
        if streams_enabled:
            compute_stream = cp.cuda.Stream(non_blocking=True)
            copy_stream = cp.cuda.Stream(non_blocking=True)
        else:
            compute_stream = None  # type: ignore
            copy_stream = None  # type: ignore

        # Helper: allocate pinned NumPy array
        def _alloc_pinned_array(shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            mem = cp.cuda.alloc_pinned_memory(nbytes)  # type: ignore
            # Create a NumPy array that uses the pinned buffer
            arr = np.ndarray(  # type: ignore[arg-type]
                shape=shape, dtype=dtype, buffer=mem
            )
            return arr, mem  # return mem to keep reference alive

        # Pinned buffer pool (round-robin)
        if streams_enabled:
            pinned_pool = []
            for _k in range(buf_depth):
                host_dem, mem_dem = _alloc_pinned_array(
                    (min(initial_batch, na), nt_dev), np.float64
                )
                host_dnreg, mem_dnreg = _alloc_pinned_array(
                    (min(initial_batch, na), nf_dev), np.float64
                )
                host_chi, mem_chi = _alloc_pinned_array(
                    (min(initial_batch, na),), np.float64
                )
                pinned_pool.append(
                    {
                        "dem": host_dem,
                        "dem_mem": mem_dem,
                        "dnreg": host_dnreg,
                        "dnreg_mem": mem_dnreg,
                        "chi": host_chi,
                        "chi_mem": mem_chi,
                    }
                )

            # Pending copy operations queue
            # each item: dict with 'slice', 'buffers', 'done_event'
            pending = deque()

        while b0 < na:
            cur_batch = min(batch_size, na - b0)
            b1 = b0 + cur_batch

            # Quiet by default: suppress per-batch logs

            try:
                _range_push("BATCH_PREP")
                dn_b = dd_d[b0:b1, :]
                ed_b = ed_d[b0:b1, :]
                rmatrixin_b = rmatrix_d[None, :, :] * (1.0 / ed_b)[:, None, :]
                A_batch = cp.transpose(rmatrixin_b, (0, 2, 1))
                AB1 = A_batch @ B_inv  # (batch, nf, nt)
                _range_pop()

                _range_push("SVD")
                # Rectangular SVD directly on AB1 (no square padding)
                if streams_enabled:
                    with compute_stream:  # type: ignore[union-attr]
                        u_b, s_b, vh_b = cp.linalg.svd(AB1, full_matrices=False)
                else:
                    u_b, s_b, vh_b = cp.linalg.svd(AB1, full_matrices=False)
                _range_pop()

                _range_push("LAMBDA_SELECT")
                if streams_enabled:
                    with compute_stream:  # type: ignore[union-attr]
                        lambs_dev = _batch_select_lambda(
                            u_b,
                            s_b,
                            dn_b,
                            ed_b,
                            reg_tweak,
                            nmu,
                        )
                else:
                    lambs_dev = _batch_select_lambda(
                        u_b,
                        s_b,
                        dn_b,
                        ed_b,
                        reg_tweak,
                        nmu,
                    )
                _range_pop()
                # Reconstruction using rectangular SVD
                _range_push("RECONSTRUCTION_CALC")
                if streams_enabled:
                    # Ensure reconstruction ops are enqueued on the compute stream
                    with compute_stream:  # type: ignore[union-attr]
                        # d' = d / ed
                        dprime = dn_b / ed_b  # (batch, nf_dev)
                        # c = U^T d'
                        coef = cp.matmul(
                            cp.transpose(u_b, (0, 2, 1)), dprime[:, :, None]
                        ).squeeze(-1)  # (batch, k)
                        # f = s / (s^2 + lambda)
                        lambs_vec = lambs_dev[:, None]
                        filt = s_b / (s_b**2 + lambs_vec)
                        # x' = V * (f * c)
                        x_prime = cp.matmul(
                            cp.transpose(vh_b, (0, 2, 1)), (filt * coef)[:, :, None]
                        ).squeeze(-1)  # (batch, nt_dev)
                        # x = B^{-1} x'
                        dem_out_device = cp.matmul(x_prime, B_inv)
                        # predicted data
                        dn_reg_device = cp.matmul(dem_out_device, rmatrix_d)
                        residuals_device, chisq_device = _residuals_and_chisq(
                            dn_b, dn_reg_device, ed_b, nf_dev
                        )
                else:
                    # Synchronous path on default stream
                    # d' = d / ed
                    dprime = dn_b / ed_b  # (batch, nf_dev)
                    # c = U^T d'
                    coef = cp.matmul(
                        cp.transpose(u_b, (0, 2, 1)), dprime[:, :, None]
                    ).squeeze(-1)  # (batch, k)
                    # f = s / (s^2 + lambda)
                    lambs_vec = lambs_dev[:, None]
                    filt = s_b / (s_b**2 + lambs_vec)
                    # x' = V * (f * c)
                    x_prime = cp.matmul(
                        cp.transpose(vh_b, (0, 2, 1)), (filt * coef)[:, :, None]
                    ).squeeze(-1)  # (batch, nt_dev)
                    # x = B^{-1} x'
                    dem_out_device = cp.matmul(x_prime, B_inv)
                    # predicted data
                    dn_reg_device = cp.matmul(dem_out_device, rmatrix_d)
                    residuals_device, chisq_device = _residuals_and_chisq(
                        dn_b, dn_reg_device, ed_b, nf_dev
                    )
                _range_pop()

                _range_push("DEVICE_TO_HOST")
                if not streams_enabled:
                    # Synchronous path (original behavior)
                    dem_batch_host = cp.asnumpy(dem_out_device[:, :nt_dev])
                    dn_reg_batch_host = cp.asnumpy(dn_reg_device)
                    chisq_batch_host = cp.asnumpy(chisq_device)
                    dem[b0:b1, :] = dem_batch_host
                    dn_reg[b0:b1, :] = dn_reg_batch_host
                    chisq[b0:b1] = chisq_batch_host
                    edem[b0:b1, :] = np.abs(dem_batch_host) * 0.1
                else:
                    # Asynchronous D2H into pinned buffers using a
                    # separate stream
                    # Drain oldest pending if pool is full
                    if len(pending) >= buf_depth:
                        old = pending.popleft()
                        # Wait for D2H completion then copy into
                        # final outputs
                        old["done_event"].synchronize()
                        s0, s1 = old["slice"]
                        hb = old["buffers"]
                        # Copy pinned -> pageable outputs on CPU
                        # (overlaps with next GPU compute)
                        dem[s0:s1, :] = hb["dem"][: (s1 - s0), :]
                        dn_reg[s0:s1, :] = hb["dnreg"][: (s1 - s0), :]
                        chisq[s0:s1] = hb["chi"][: (s1 - s0)]
                        edem[s0:s1, :] = np.abs(hb["dem"][: (s1 - s0), :]) * 0.1

                    buf = pinned_pool[(b0 // max(1, cur_batch)) % buf_depth]
                    # Ensure pinned views match current batch length
                    dem_view = buf["dem"][:cur_batch, :nt_dev]
                    dnreg_view = buf["dnreg"][:cur_batch, :nf_dev]
                    chi_view = buf["chi"][:cur_batch]

                    # Schedule async copies on copy_stream after
                    # compute_stream completes
                    evt = cp.cuda.Event(disable_timing=True)
                    # type: ignore[arg-type]
                    evt.record(compute_stream)
                    with copy_stream:  # type: ignore[union-attr]
                        # type: ignore[arg-type]
                        copy_stream.wait_event(evt)
                        # Device -> pinned host memcpyAsync
                        # type: ignore[attr-defined]
                        cp.cuda.runtime.memcpyAsync(
                            dem_view.__array_interface__["data"][0],
                            int(dem_out_device[:, :nt_dev].data.ptr),
                            dem_view.nbytes,
                            2,  # cudaMemcpyDeviceToHost
                            copy_stream.ptr,
                        )
                        # type: ignore[attr-defined]
                        cp.cuda.runtime.memcpyAsync(
                            dnreg_view.__array_interface__["data"][0],
                            int(dn_reg_device.data.ptr),
                            dnreg_view.nbytes,
                            2,
                            copy_stream.ptr,
                        )
                        # type: ignore[attr-defined]
                        cp.cuda.runtime.memcpyAsync(
                            chi_view.__array_interface__["data"][0],
                            int(chisq_device.data.ptr),
                            chi_view.nbytes,
                            2,
                            copy_stream.ptr,
                        )
                        # Record an event to signal copy completion
                        done_evt = cp.cuda.Event(disable_timing=True)
                        done_evt.record(copy_stream)
                    pending.append(
                        {
                            "slice": (b0, b1),
                            "buffers": buf,
                            "done_event": done_evt,
                            # Hold device refs alive until copy completes
                            "dev_dem": dem_out_device[:, :nt_dev],
                            "dev_dnreg": dn_reg_device,
                            "dev_chi": chisq_device,
                        }
                    )
                _range_pop()

                b0 = b1
                batch_size = initial_batch
            except Exception as e:
                log = logging.getLogger(__name__)
                log.exception("GPU batch %d:%d failed: %s", b0, b1, e)
                if cur_batch <= 1:
                    raise
                batch_size = max(1, cur_batch // 2)
                logging.warning(
                    "Reducing GPU batch size to %d and retrying", batch_size
                )

        # Finalize any pending async copies
        if streams_enabled:
            while pending:
                old = pending.popleft()
                old["done_event"].synchronize()
                s0, s1 = old["slice"]
                hb = old["buffers"]
                dem[s0:s1, :] = hb["dem"][: (s1 - s0), :]
                dn_reg[s0:s1, :] = hb["dnreg"][: (s1 - s0), :]
                chisq[s0:s1] = hb["chi"][: (s1 - s0)]
                edem[s0:s1, :] = np.abs(hb["dem"][: (s1 - s0), :]) * 0.1

        # Clean up workspace arrays for large images to free GPU memory
        if na > 1000000:  # Only for large images to avoid overhead
            _gpu_workspace.clear_workspaces()

        # Log memory usage statistics for monitoring
        mem_stats = _gpu_workspace.get_memory_usage()
        _verbose = os.environ.get("MULTIGPU_VERBOSE", "0") == "1"
        if mem_stats["total_mb"] > 0 and _verbose:
            logging.getLogger(__name__).info(
                "GPU memory pool: %.1f/%.1f MB (%.1f%% utilization)",
                mem_stats["used_mb"],
                mem_stats["total_mb"],
                mem_stats["utilization"] * 100,
            )

        return dem, edem, elogt, chisq, dn_reg
    except Exception as e:  # pragma: no cover
        logging.getLogger(__name__).exception("GPU path failed: %s", e)
        raise RuntimeError("GPU path failed; aborting multiGPU execution") from e
    # Unreachable: GPU path always returns or raises above
