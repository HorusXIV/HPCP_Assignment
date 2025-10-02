from __future__ import annotations

"""
Single-GPU helpers using CuPy (preferred) and a clean fallback to CPU.

This mirrors the multi-GPU integration: build dn/edn on the host, optionally
use CuPy to accelerate elementwise ops, then call the vendor demmap_pos if
available. This version removes the fragile Numba kernel and keeps a small
surface area.
"""

from typing import Optional, Tuple
import importlib
import os
import numpy as np

from src.common.gpu import available as cupy_available
from src.common.solver import get_logt_bins_once


def _numba_cuda_available() -> bool:
    # keep API compatibility but we do not use numba in this simplified path
    try:
        from numba import cuda  # type: ignore
        return bool(cuda.is_available())
    except Exception:
        return False


def gpu_ready() -> bool:
    return bool(cupy_available())


def _synth_responses(nt: int, nf: int = 6):
    logT = np.linspace(5.5, 7.5, 200, dtype=np.float32)
    temps = np.logspace(5.5, 7.5, nt + 1, dtype=np.float32)
    centers = np.linspace(5.7, 7.3, nf, dtype=np.float32)
    T_RESP = np.exp(-0.5 * ((logT[:, None] - centers[None, :]) / 0.20) ** 2) + 1e-30
    return T_RESP.astype(np.float32), logT.astype(np.float32), temps.astype(np.float32)


def _err_sqrt_cpu(counts6: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.sqrt(a * np.clip(counts6, 0, None) + b).astype(np.float32, copy=False)


def _err_sqrt_device(counts6: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Compute error model entirely on GPU using CuPy.

    Accepts host NumPy array and returns host NumPy array.
    Processes the full frame at once instead of per-tile.
    """
    import cupy as cp  # type: ignore

    # Move entire frame to device
    d_counts = cp.asarray(counts6, dtype=cp.float32)

    # Compute error model on GPU
    d_out = cp.sqrt(a * cp.clip(d_counts, 0.0, None) + b)

    # Return to host only once
    return cp.asnumpy(d_out).astype(np.float32, copy=False)


from numba import cuda

@cuda.jit
def dem_kernel(counts6, T_RESP, dem_out, edem_out, nt):
    idx = cuda.grid(1)
    if idx >= counts6.shape[0]:
        return
    
    nf = counts6.shape[1]
    for t in range(nt):
        val = 0.0
        for f in range(nf):
            val += counts6[idx, f] * T_RESP[t, f]  # placeholder DEM calculation
        dem_out[idx, t] = val
        edem_out[idx, t] = val * 0.1  # placeholder error

def solve_tile_batch_gpu(counts6, T_RESP, nt):
    pixels, nf = counts6.shape
    dem_out = np.zeros((pixels, nt), dtype=np.float32)
    edem_out = np.zeros((pixels, nt), dtype=np.float32)

    threads_per_block = 128
    blocks = (pixels + threads_per_block - 1) // threads_per_block

    d_counts6 = cuda.to_device(counts6.astype(np.float32))
    d_TRESP = cuda.to_device(T_RESP.astype(np.float32))
    d_dem = cuda.device_array((pixels, nt), dtype=np.float32)
    d_edem = cuda.device_array((pixels, nt), dtype=np.float32)

    dem_kernel[blocks, threads_per_block](d_counts6, d_TRESP, d_dem, d_edem, nt)

    d_dem.copy_to_host(dem_out)
    d_edem.copy_to_host(edem_out)

    return dem_out, edem_out


def solve_tile_all_single_gpu(
    counts6: np.ndarray,
    *,
    nmu: Optional[int] = 42,
    nt: Optional[int] = None,
    err_a: float = 1.0,
    err_b: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert counts6.ndim == 3 and counts6.shape[-1] == 6
    H, W, nf = counts6.shape
    NT, logT_centers = get_logt_bins_once(nmu=nmu, nt=nt)

    # Sanitize
    f = np.clip(np.nan_to_num(counts6, nan=0.0, posinf=0.0, neginf=0.0), 0, None).astype(np.float32, copy=False)

    # Compute errors (prefer device)
    try:
        e = _err_sqrt_device(f, err_a, err_b)
    except Exception:
        e = _err_sqrt_cpu(f, err_a, err_b)

    # Synthesize responses (host)
    T_RESP, T_RESP_LOGT, TEMPS = _synth_responses(nt=NT, nf=nf)

    # Try vendor demmap_pos; if unavailable, fall back to CPU solver
    try:
        mod = importlib.import_module("src.baseline.vendor.demmap_pos")
        demmap_pos = getattr(mod, "demmap_pos")
    except Exception:
        from src.common.solver import solve_tile_all as _solve_cpu
        return _solve_cpu(counts6, nmu=nmu, nt=nt)

    # Prepare vendor inputs (flatten pixels x nf)
    nx, ny, nt_bins = int(H), int(W), int(NT)
    dn1d = f.reshape(nx * ny, nf)
    edn1d = e.reshape(nx * ny, nf)
    rmatrix = T_RESP
    logt = T_RESP_LOGT
    dlogt = np.full(logt.shape, np.median(np.diff(logt)), dtype=np.float32)
    glc = np.ones((nf,), dtype=np.float32)
    dem_norm0 = np.zeros((nx * ny, nt_bins), dtype=np.float32)

    # Call vendor routine (expected to handle internal GPU batching if it is GPU-capable)
    dem1d, edem1d, _elogt1d, chisq1d, _ = demmap_pos(
        dn1d,
        edn1d,
        rmatrix,
        logt,
        dlogt,
        glc,
        reg_tweak=1.0,
        max_iter=10,
        rgt_fact=1.5,
        dem_norm0=dem_norm0,
        nmu=int(nmu or 42),
        warn=False,
        l_emd=False,
        rscl=False,
    )

    # Resample if vendor returned different temperature axis length
    if dem1d.ndim == 2 and dem1d.shape[1] != nt_bins:
        src_len = int(dem1d.shape[1])
        if src_len == int(logt.shape[0]):
            DEM_nt = np.empty((dem1d.shape[0], nt_bins), dtype=np.float32)
            EDEM_nt = np.empty_like(DEM_nt)
            for i in range(dem1d.shape[0]):
                DEM_nt[i, :] = np.interp(logT_centers, logt, dem1d[i, :]).astype(np.float32, copy=False)
                EDEM_nt[i, :] = np.interp(logT_centers, logt, edem1d[i, :]).astype(np.float32, copy=False)
            dem1d = DEM_nt
            edem1d = EDEM_nt
        else:
            m = min(src_len, nt_bins)
            DEM_nt = np.zeros((dem1d.shape[0], nt_bins), dtype=np.float32)
            EDEM_nt = np.zeros_like(DEM_nt)
            DEM_nt[:, :m] = dem1d[:, :m]
            EDEM_nt[:, :m] = edem1d[:, :m]
            dem1d = DEM_nt
            edem1d = EDEM_nt

    dem = dem1d.reshape(nx, ny, nt_bins).astype(np.float32, copy=False)
    edem = edem1d.reshape(nx, ny, nt_bins).astype(np.float32, copy=False)
    chisq = chisq1d.reshape(nx, ny).astype(np.float32, copy=False)

    return dem, edem, chisq, logT_centers.astype(np.float32, copy=False)


__all__ = [
    "gpu_ready",
    "solve_tile_all_single_gpu",
]
