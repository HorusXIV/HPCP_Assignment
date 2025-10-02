from __future__ import annotations

"""
Single-GPU helpers using CuPy (preferred) and a clean fallback to CPU/vendor.

This file:
- accepts inputs (H,W,6) or (F,H,W,6)
- uses CuPy batched GEMM for throughput when available
- chunk-processes flattened pixels to avoid OOM and memory accumulation
- falls back to vendor demmap_pos or CPU solver if needed
"""

from typing import Optional, Tuple
import importlib
import os
import math
import numpy as np

from src.common.gpu import available as cupy_available
from src.common.solver import get_logt_bins_once


def _numba_cuda_available() -> bool:
    # API compatibility; not used by this implementation
    try:
        from numba import cuda  # type: ignore
        return bool(cuda.is_available())
    except Exception:
        return False


def gpu_ready() -> bool:
    return bool(cupy_available())


# ---------- responses + small helpers ----------
def _synth_responses(nt: int, nf: int = 6):
    """Return (T_RESP (nt,nf), logT_centers (nt,), temps(nt+1,))"""
    logT = np.linspace(5.5, 7.5, 200, dtype=np.float32)
    temps = np.logspace(5.5, 7.5, nt + 1, dtype=np.float32)
    centers = np.linspace(5.7, 7.3, nf, dtype=np.float32)
    T_RESP = np.exp(-0.5 * ((logT[:, None] - centers[None, :]) / 0.20) ** 2) + 1e-30
    return T_RESP.astype(np.float32), logT.astype(np.float32), temps.astype(np.float32)


def _err_sqrt_cpu(counts6: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.sqrt(a * np.clip(counts6, 0, None) + b).astype(np.float32, copy=False)


# ---------- GPU helpers ----------
def _get_gpu_free_bytes(cp) -> Optional[int]:
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return None


def _estimate_pixels_per_chunk(free_bytes: Optional[int], nf: int, nt: int, safety_factor: float = 0.40) -> int:
    """
    Estimate number of pixels we can fit in one chunk based on free GPU memory.
    Conservative default safety_factor to avoid OOM.
    """
    # bytes per pixel needed: dn(nf) + err(nf) + dem(nt) + edem(nt)
    bytes_per_pixel = (nf + nf + nt + nt) * 4
    if free_bytes is None or free_bytes <= 0:
        # fallback default
        return max(1, 100_000)
    usable = int(free_bytes * safety_factor)
    p = max(1, usable // bytes_per_pixel)
    return int(p)


def _gpu_batch_dem_matmul(counts_flat: np.ndarray, T_RESP: np.ndarray, nt: int, cupy_streams: int = 1):
    """
    counts_flat: (pixels, nf) numpy float32
    T_RESP:       (nt, nf) numpy float32
    Returns:
        dem_out (pixels, nt) numpy float32
        edem_out (pixels, nt) numpy float32  -- simplistic error model
    Implementation:
      - chunk counts_flat so each chunk fits memory
      - upload chunk to device, compute chunk @ T_RESP.T -> dem_chunk
      - compute edem_chunk as small fraction of dem (placeholder)
      - copy back to host and assemble
    """
    import cupy as cp

    pixels, nf = counts_flat.shape
    # use device to hold T_RESP.T = (nf, nt)
    rmatrix_T = cp.asarray(T_RESP.T, dtype=cp.float32)  # (nf, nt)
    NT = int(nt)

    free_bytes = _get_gpu_free_bytes(cp)
    chunk_pixels = _estimate_pixels_per_chunk(free_bytes, nf, NT, safety_factor=0.40)
    # clamp chunk size between 1 and pixels
    chunk_pixels = min(max(1, chunk_pixels), pixels)

    dem_out = np.empty((pixels, NT), dtype=np.float32)
    edem_out = np.empty((pixels, NT), dtype=np.float32)

    pos = 0
    # Simple single-threaded loop; could be enhanced with streams + pinned memory.
    while pos < pixels:
        end = min(pixels, pos + chunk_pixels)
        sub = counts_flat[pos:end].astype(np.float32, copy=False)  # (chunk, nf)

        # device compute
        d_sub = cp.asarray(sub)  # copy to device
        # matrix multiply: (chunk, nf) @ (nf, nt) -> (chunk, nt)
        d_dem = cp.matmul(d_sub, rmatrix_T)
        # placeholder error model: relative fraction of dem; replace with proper model if known
        d_edem = 0.1 * cp.abs(d_dem)

        # copy back
        dem_out[pos:end, :] = cp.asnumpy(d_dem)
        edem_out[pos:end, :] = cp.asnumpy(d_edem)

        # free device temp memory ASAP
        del d_sub, d_dem, d_edem
        cp.get_default_memory_pool().free_all_blocks()

        pos = end

    # cleanup
    del rmatrix_T
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    return dem_out.astype(np.float32, copy=False), edem_out.astype(np.float32, copy=False)


def _gpu_batch_err_sqrt(counts_flat: np.ndarray, a: float, b: float) -> np.ndarray:
    """Compute sqrt error on GPU for flattened (pixels,nf) input and return numpy array."""
    import cupy as cp
    d = cp.asarray(counts_flat, dtype=cp.float32)
    d_out = cp.sqrt(a * cp.clip(d, 0.0, None) + b)
    out = cp.asnumpy(d_out)
    del d, d_out
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    return out.astype(np.float32, copy=False)


# ---------- main API ----------
def solve_tile_all_single_gpu(
    counts6: np.ndarray,
    *,
    nmu: Optional[int] = 42,
    nt: Optional[int] = None,
    err_a: float = 1.0,
    err_b: float = 1e-6,
    streams: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch-capable single-GPU solver.

    Accepts:
      - counts6 with shape (H,W,6)  -> single frame
      - counts6 with shape (F,H,W,6) -> batch of frames

    Returns:
      dem, edem, chisq, logT_centers

    dem shapes:
      - single-frame input: dem (H,W,NT)
      - batch input: dem (F,H,W,NT)
    """

    # Accept 3D (H,W,6) or 4D (F,H,W,6)
    if counts6.ndim == 3:
        single_frame = True
        H, W, nf = counts6.shape
        F = 1
        counts = counts6.reshape(1, H, W, nf)
    elif counts6.ndim == 4:
        single_frame = False
        F, H, W, nf = counts6.shape
        counts = counts6
    else:
        raise ValueError("counts6 must have ndim 3 (H,W,6) or 4 (F,H,W,6)")

    if nf != 6:
        raise ValueError(f"expected last axis size 6 (filters); got {nf}")

    # temperature bins
    NT, logT_centers = get_logt_bins_once(nmu=nmu, nt=nt)
    NT = int(NT)

    # Flatten pixels (pixels_total, nf)
    pixels_total = int(F * H * W)
    # sanitize and flatten
    counts_flat = np.clip(np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None).astype(np.float32, copy=False)
    counts_flat = counts_flat.reshape(pixels_total, nf)

    # Compute errors (edn) either on GPU or CPU
    edn_flat: np.ndarray
    if cupy_available():
        try:
            edn_flat = _gpu_batch_err_sqrt(counts_flat, err_a, err_b)
        except Exception:
            # fallback to CPU
            edn_flat = _err_sqrt_cpu(counts_flat, err_a, err_b)
    else:
        edn_flat = _err_sqrt_cpu(counts_flat, err_a, err_b)

    # Build synthetic response matrix (nt x nf) on host
    T_RESP, T_RESP_LOGT, TEMPS = _synth_responses(nt=NT, nf=nf)  # T_RESP (nt, nf)

    # Try vendor demmap_pos if available (it may be optimized)
    try:
        mod = importlib.import_module("src.baseline.vendor.demmap_pos")
        demmap_pos = getattr(mod, "demmap_pos")
    except Exception:
        demmap_pos = None

    if demmap_pos is not None:
        # Attempt to call vendor in one shot if it accepts large arrays.
        # Some vendor implementations expect (pixels, nf) inputs.
        try:
            dem1d, edem1d, _elogt1d, chisq1d, _ = demmap_pos(
                counts_flat,
                edn_flat,
                T_RESP,
                T_RESP_LOGT,
                np.full(nf, 1.0, dtype=np.float32),  # glc
                np.full_like(T_RESP_LOGT, np.median(np.diff(T_RESP_LOGT)), dtype=np.float32),
                reg_tweak=1.0,
                max_iter=10,
                rgt_fact=1.5,
                dem_norm0=None,
                nmu=int(nmu or 42),
                warn=False,
                l_emd=False,
                rscl=False,
            )
        except Exception:
            dem1d = None
            # If vendor fails on large arrays, fall back to chunked CPU vendor or CuPy path below

    else:
        dem1d = None

    # If vendor produced valid dem1d, use it (resample if needed)
    if dem1d is not None:
        # ensure numpy arrays
        dem1d = np.asarray(dem1d, dtype=np.float32, copy=False)
        edem1d = np.asarray(edem1d, dtype=np.float32, copy=False)
        chisq1d = np.asarray(chisq1d, dtype=np.float32, copy=False)
    else:
        # Use CuPy batched matmul path if available; else fallback to CPU solver per-tile
        if cupy_available():
            try:
                dem1d, edem1d = _gpu_batch_dem_matmul(counts_flat, T_RESP, NT, cupy_streams=max(1, int(streams)))
                # naive chisq fallback: zeros (vendor typically supplies chisq)
                chisq1d = np.zeros((pixels_total,), dtype=np.float32)
            except Exception:
                # last resort: CPU solver per-frame/per-tile
                dem1d = None
        else:
            dem1d = None

        if dem1d is None:
            # CPU fallback (use existing solver) — process per-frame to match expected tile solver
            from src.common.solver import solve_tile_all as _solve_cpu
            # _solve_cpu expects (H,W,6) and returns (H,W,NT) etc.
            dem_out = np.empty((F, H, W, NT), dtype=np.float32)
            edem_out = np.empty((F, H, W, NT), dtype=np.float32)
            chisq_out = np.empty((F, H, W), dtype=np.float32)
            for fi in range(F):
                dem_f, edem_f, chisq_f, _ = _solve_cpu(counts[fi], nmu=nmu, nt=NT)
                dem_out[fi] = dem_f
                edem_out[fi] = edem_f
                chisq_out[fi] = chisq_f
            if single_frame:
                return dem_out[0], edem_out[0], chisq_out[0], logT_centers.astype(np.float32, copy=False)
            return dem_out, edem_out, chisq_out, logT_centers.astype(np.float32, copy=False)

    # At this point dem1d/edem1d/chisq1d are available (shape: (pixels, k))
    dem1d = np.asarray(dem1d, dtype=np.float32, copy=False)
    edem1d = np.asarray(edem1d, dtype=np.float32, copy=False)
    chisq1d = np.asarray(chisq1d, dtype=np.float32, copy=False)

    # If vendor returned a different NT, resample/interp as before
    if dem1d.ndim == 2 and dem1d.shape[1] != NT:
        src_len = int(dem1d.shape[1])
        if src_len == int(T_RESP_LOGT.shape[0]):
            DEM_nt = np.empty((dem1d.shape[0], NT), dtype=np.float32)
            EDEM_nt = np.empty_like(DEM_nt)
            for i in range(dem1d.shape[0]):
                DEM_nt[i, :] = np.interp(logT_centers, T_RESP_LOGT, dem1d[i, :]).astype(np.float32, copy=False)
                EDEM_nt[i, :] = np.interp(logT_centers, T_RESP_LOGT, edem1d[i, :]).astype(np.float32, copy=False)
            dem1d = DEM_nt
            edem1d = EDEM_nt
        else:
            m = min(src_len, NT)
            DEM_nt = np.zeros((dem1d.shape[0], NT), dtype=np.float32)
            EDEM_nt = np.zeros_like(DEM_nt)
            DEM_nt[:, :m] = dem1d[:, :m]
            EDEM_nt[:, :m] = edem1d[:, :m]
            dem1d = DEM_nt
            edem1d = EDEM_nt

    # reshape back to (F,H,W,NT) and (F,H,W) for chisq
    dem = dem1d.reshape((F, H, W, NT)).astype(np.float32, copy=False)
    edem = edem1d.reshape((F, H, W, NT)).astype(np.float32, copy=False)

    # If chisq1d has appropriate length -> reshape else create zero array
    if chisq1d is not None and chisq1d.size == pixels_total:
        chisq = chisq1d.reshape((F, H, W)).astype(np.float32, copy=False)
    else:
        chisq = np.zeros((F, H, W), dtype=np.float32)

    # Return single-frame shapes to preserve old API
    if single_frame:
        return dem[0], edem[0], chisq[0], logT_centers.astype(np.float32, copy=False)

    return dem, edem, chisq, logT_centers.astype(np.float32, copy=False)


__all__ = [
    "gpu_ready",
    "solve_tile_all_single_gpu",
]
