# flake8: noqa: E501
"""
GPU-accelerated kernels for DEM inversion, mirroring vendor behavior with
OOM-aware batching on the GPU path and vendor-parity per-sample fallback.

This module provides:
- A batched, CuPy-accelerated DEM reconstruction that is algorithmically
  equivalent to the vendor baseline (SVD-of-AB1 formulation for GSVD parity).
- Automatic, OOM-aware downscaling of batch.
"""

from __future__ import annotations

import os
import math
import logging
from typing import Tuple, Dict, Any
import numpy as np
from contextlib import contextmanager


@contextmanager
def nvtx_range(msg: str, color: int | None = None):
    """Lightweight NVTX context manager controlled by MULTIGPU_NVTX.

    When the environment variable ``MULTIGPU_NVTX`` is set to "1", emits
    NVTX ranges; otherwise acts as a no-op.
    """
    if os.environ.get("MULTIGPU_NVTX", "0") != "1":
        yield
        return
    cm = None
    try:
        import nvtx as _nvtx  # type: ignore

        kwargs = {"message": msg}
        if color is not None:
            kwargs["color"] = int(color)
        cm = _nvtx.annotate(**kwargs)
    except Exception:
        cm = None  # degrade silently
    if cm is None:
        yield
    else:
        with cm:
            yield


try:
    import cupy as cp  # type: ignore
except Exception as e:
    raise ImportError("CuPy is required for multiGPU execution: %s" % e) from e


def _bytes_per_sample_estimate(nf: int, nt: int, nmu: int) -> int:
    """Estimate peak bytes per sample used by the batched solver (Float64).

    This is an intentionally conservative estimate designed to reduce
    OOM-induced batch resizes. It accounts for the largest simultaneously
    live arrays in both the self-normalized L pass and the main SVD pass,
    as well as dominant intermediates used in uncertainty and elogt paths.

    Args:
        nf: Number of filters (channels).
        nt: Number of temperature bins.
        nmu: Number of lambda grid points in the discrepancy search.

    Returns:
        Estimated peak bytes per sample as an integer.
    """
    k = min(nf, nt)
    nmu_eff = max(int(nmu), 2)

    core_terms = (
        (nf * nt)  # A_b
        + 2 * (nf * k + k + k * nt)  # two SVDs worth of (U, s, Vh)
        + 2 * (k * nmu_eff)  # discrepancy vals for two passes (approx)
        + (nt * nf)  # kdagk dominant slice used for elogt
    )
    io_terms = nf + nf + nt + nt + nt  # dn, ed, dem, edem, elogt
    safety = 1.35  # cover allocator/workspace and transient temporaries
    bytes_f64 = 8.0
    return int(bytes_f64 * safety * (core_terms + io_terms + 64))


def _adaptive_batch_size(na: int, nf: int, nt: int, nmu: int) -> int:
    """Pick a batch size based on free GPU memory and problem size.

    The heuristic estimates per-sample memory footprint from matrix sizes
    and singular vector intermediates, then targets ~70% of currently free
    device memory. Falls back to `min(64, na)` if device memory is not
    available (e.g., running under a CPU-only shim).

    Args:
      na: Number of samples (pixels) to process.
      nf: Number of filters (data channels).
      nt: Number of temperature bins.
      nmu: Number of lambda grid points used in the discrepancy search.

    Returns:
      An integer batch size in the range [1, na].
    """
    default = min(64, na)
    try:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        free_b, _ = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
        bytes_per = _bytes_per_sample_estimate(nf, nt, nmu)
        if bytes_per <= 0:
            return default
        try:
            frac_env = float(os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7"))
            mem_frac = float(min(max(frac_env, 0.1), 0.9))
        except Exception:
            mem_frac = 0.55
        est = int((free_b * mem_frac) // bytes_per)
        return max(1, min(est, na))
    except Exception:
        return default


def verbose_enabled() -> bool:
    """Return True if verbose multiGPU logging is enabled via env.

    Controlled by the environment variable ``MULTIGPU_VERBOSE``.
    """
    try:
        return int(os.environ.get("MULTIGPU_VERBOSE", "0")) > 0
    except Exception:
        return False


def _pinned_empty(shape, dtype):
    """Allocate a NumPy array backed by CUDA pinned (page-locked) memory.

    Uses ``cp.cuda.alloc_pinned_memory`` when available; falls back to
    ``cp.cuda.PinnedMemory`` on some CuPy versions.
    """
    n_elems = int(np.prod(shape))
    nbytes = np.dtype(dtype).itemsize * n_elems
    mem = None
    try:
        mem = cp.cuda.alloc_pinned_memory(int(nbytes))
    except Exception:
        try:
            mem = cp.cuda.PinnedMemory(int(nbytes))  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"Failed to allocate pinned host memory: {exc}")
    arr = np.frombuffer(mem, dtype=dtype, count=n_elems).reshape(tuple(shape))
    return arr


def estimate_batch_plan(na: int, nf: int, nt: int, nmu: int) -> Dict[str, Any]:
    """Provide a batch sizing and memory usage estimate.

    This mirrors the internal heuristic and is safe to call from rank code for
    diagnostics. If GPU memory cannot be queried, ``free_bytes`` may be ``None``.

    Returns a dict with keys: ``batch_size``, ``bytes_per_sample``,
    ``free_bytes``, ``est_batch_bytes``, ``num_batches``.
    """
    bps = _bytes_per_sample_estimate(nf, nt, nmu)
    free_b = None
    try:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        free_b, _ = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
    except Exception:
        free_b = None
    bs = _adaptive_batch_size(na, nf, nt, nmu)
    return {
        "batch_size": int(bs),
        "bytes_per_sample": int(bps),
        "free_bytes": int(free_b) if free_b is not None else None,
        "est_batch_bytes": int(bps * bs),
        "num_batches": int(math.ceil(max(1, na) / max(1, bs))),
    }


def safe_svd(
    A: np.ndarray, full_matrices: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a numerically safe SVD on GPU and return NumPy arrays.

    Input is sanitized (NaN/Inf to finite, clipping extremes) before sending
    to CuPy. The factorization is performed with ``cp.linalg.svd`` and the
    results are transferred back to host as NumPy arrays.

    Args:
      A: Real-valued matrix to factorize. Will be converted to float64, C-order.
      full_matrices: Whether to compute full-sized U and Vh (as in NumPy/CuPy).

    Returns:
      A tuple (U, s, Vh) where:
        - U: Left singular vectors, shape (m, m) if full_matrices else (m, k).
        - s: Singular values (non-negative), shape (k,), k = min(m, n).
        - Vh: Right singular vectors (Hermitian transpose), shape (n, n) if
          full_matrices else (k, n).

    Raises:
      RuntimeError: If the CuPy-backed SVD fails for any reason.
    """
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
      A: Real-valued matrix to invert. Converted to float64.
      rcond: Relative cutoff below which singular values are set to zero.

    Returns:
      The Moore-Penrose pseudo-inverse of ``A`` as a NumPy float64 array.
    """
    A = np.asarray(A, dtype=np.float64, order="C")
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
    A = np.clip(A, -1e12, 1e12, out=A)
    U, s, Vh = safe_svd(A, full_matrices=False)
    tol = np.max(s) * rcond if s.size else rcond
    s_inv = np.array([1.0 / x if x > tol else 0.0 for x in s], dtype=np.float64)
    return (Vh.T * s_inv) @ U.T


def dem_inv_gsvd(A: np.ndarray, B: np.ndarray):
    """Compute GSVD-equivalent factors using SVD of ``A @ pinv(B)``.

    This mirrors the vendor's GSVD usage but uses an SVD of the product
    AB1 = A @ pinv(B) padded to a square matrix for stability. The outputs
    are arranged to be drop-in compatible with the vendor function.

    Args:
      A: Data matrix of shape (m, n).
      B: Regularization operator of shape (p, n). Only its pseudo-inverse and
        right-multiplication behavior are used in this construction.

    Returns:
      A 5-tuple ``(alpha, beta, U_T, V_T, W)`` where:
        - alpha: Array of shape (k,) with alpha = s / sqrt(1+s^2).
        - beta: Array of shape (k,) with beta = 1 / sqrt(1+s^2).
        - U_T: Transposed U (i.e., U.T) sliced to (k, m) to match vendor.
        - V_T: Transposed V (i.e., V.T), shape (n, n).
        - W: Transformation matrix satisfying vendor parity.
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


def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    """Select the regularization parameter via the discrepancy principle.

    Follows the vendor implementation by evaluating a grid of ``mu`` values
    derived from the generalized singular values and choosing the one that
    best satisfies Morozov's discrepancy principle.

    Note: ``W`` is unused but preserved for interface parity.

    Args:
      sigmaa: 1D array of alpha-like singular values (length >= nf).
      sigmab: 1D array of beta-like singular values (length >= nf).
      U: Left singular vectors so rows can be addressed by filter index.
      W: Unused (kept for vendor parity).
      data: 1D array of data values, shape (nf,).
      err: 1D array of errors per data point, shape (nf,).
      reg_tweak: Scalar multiplier applied in the discrepancy.
      nmu: Number of points in the ``mu`` geometric grid (>= 2).

    Returns:
      The selected regularization parameter ``mu`` as a float.

    Raises:
      ValueError: If U does not have at least ``nf`` rows after shape
        normalization.
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


def dem_pix(*_a, **_k):  # pragma: no cover
    """Unsupported in the multiGPU module.

    Raises:
      RuntimeError: Always, to signal that pixel-wise DEM is not implemented
        in this module. Use ``demmap_pos`` instead.
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
    nmu=42,
    warn=False,
    l_emd=False,
    rscl=False,
):
    """Reconstruct Differential Emission Measure (DEM) for many samples.

    This function mirrors the vendor baseline (GSVD parity via an
    SVD-of-AB1 construction) while accelerating batched cases on GPU.
    It includes a self-normalized L pass when ``dem_norm0`` is trivial,
    optional GLOCI-based weighting, Morozov discrepancy selection for the
    regularization parameter, a positivity loop, and vendor-aligned
    computations for ``edem`` and ``elogt``.

        GPU requirements and error behavior:
        - If no CUDA device is available (``cp.cuda.runtime.getDeviceCount() == 0``),
            this function raises ``RuntimeError``. This module is GPU-only by design.
        - If GPU memory is insufficient for the current batch, the batch size is
            automatically reduced and retried. If retries are exhausted, a
            ``RuntimeError`` is raised with guidance to reduce the batch size.

    Args:
      dd: Data array of shape (na, nf) with measured intensities.
      ed: Error array of shape (na, nf) with per-channel uncertainties.
      rmatrix: Response matrix of shape (nt, nf).
      logt: Log-temperature grid of shape (nt,).
      dlogt: Log-temperature bin widths of shape (nt,).
      glc: GLOCI selector array (shape (nf,)) with positive entries selecting
        channels for the GLOCI prior. If none positive, self-normalized L is
        used when ``dem_norm0`` is trivial.
      reg_tweak: Discrepancy multiplier (>= 0). Default 1.0.
      max_iter: Max positivity loop iterations. Default 10.
      rgt_fact: Factor to increase ``reg_tweak`` when negativity is detected.
      dem_norm0: Optional prior weights. If None or trivial, a self-normalized
        pass is performed to construct L. Shape (na, nt) or (nt,).
      nmu: Number of mu grid points in the discrepancy search (>= 2).
      warn: If True, prints a warning when positivity loop hits ``max_iter``.
      l_emd: If True, use absolute weights for L (vendor option for EMD).
      rscl: If True, rescales DEM and EDEM by the mean data/prediction ratio
        (vendor ``rscl`` behavior).

    Returns:
      A 5-tuple (dem, edem, elogt, chisq, dn_reg) where:
        - dem: DEM map, shape (na, nt).
        - edem: DEM uncertainties, shape (na, nt).
        - elogt: Temperature width proxy, shape (na, nt).
        - chisq: Reduced chi-square per sample, shape (na,).
        - dn_reg: Predicted data, shape (na, nf).
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
                    pin_limit_env = os.environ.get("MULTIGPU_PINNED_POOL_LIMIT_BYTES", None)
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
                            frac_env = float(os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7"))
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
            """Apply vendor-equivalent smoothing and clamping to weights.

            Performs a centered 5-point moving average on the interior of
            wraw (dropping the first/last element) and normalizes by the
            per-sample maximum, then clamps to a minimum of 1e-8.

            Args:
              wraw_b: Raw weights, shape (batch, nt).

            Returns:
              Smoothed and clamped weights of shape (batch, nt-2-2) which
              matches the vendor's centered-window handling.
            """
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

                            rmatrixin_b = (
                                rmatrix_d[None, :, :] * (1.0 / ed_b)[:, None, :]
                            )
                            A_b = cp.transpose(rmatrixin_b, (0, 2, 1))
                            dprime_b = dn_b / ed_b

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
                                err_sq = cp.sum(ed_b**2, axis=1)
                                discr0 = arg_sum0 - (err_sq * float(reg_tweak))[:, None]
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
                            err_sq = cp.sum(ed_b**2, axis=1)

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
                                resid = (dn_b - dn_pred) / ed_b
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
                                resid = (dn_b - dn_pred) / ed_b
                                chisq_b = cp.sum(resid**2, axis=1) / nf
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
                            frac_env = float(os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7"))
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
