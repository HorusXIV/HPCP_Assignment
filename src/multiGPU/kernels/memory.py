"""
Memory planning utilities for multiGPU kernels.

- _bytes_per_sample_estimate: heuristic for peak memory per sample
- _adaptive_batch_size: choose batch size given free GPU memory
- estimate_batch_plan: public diagnostic helper
"""

from __future__ import annotations

import math
import os
from typing import Dict, Any


def _bytes_per_sample_estimate(nf: int, nt: int, nmu: int) -> int:
    """Estimate peak bytes/sample (float64) for the batched DEM solver.

    This accounts for:
    - Two SVD passes (initial L0 and main pass, worst-case when gloci=0)
    - Lambda selection buffers (mu grid, vals, coef, discr)
    - Downstream matrices for error/elogt (kVT, kdag, kdagk)
    - Primary per-sample tensors (normalized A_b/AB1 and intermediates)

    The estimate intentionally errs on the conservative side to avoid OOM
    retries that drastically reduce batch size.
    """
    k = min(int(nf), int(nt))
    nmu_eff = max(int(nmu), 2)

    # Core per-sample floats (worst-case two SVD passes)
    # Base matrices (normalized + transposed variants)
    base = 3 * (nf * nt)  # rmatrixin_b, A_b, AB1

    # Two SVDs worth of outputs (U, s, Vh) and one V
    svd_2passes = 2 * (nf * k + k + k * nt) + (nf * k)

    # Lambda selection buffers (two passes): vals (k*nmu), discr (nmu), coef (k)
    lam_sel = 2 * (k * nmu_eff + nmu_eff + k)

    # Weights and outputs (bvec, dem, edem, elogt, dn_pred)
    out_terms = (3 * nt) + nt + nt + nf

    # EDEM/ELOGT internals: kVT (nt*k), kdag (nt*nf), kdagk (nt*nt)
    err_terms = (nt * k) + (nt * nf) + (nt * nt)

    floats_per_sample = base + svd_2passes + lam_sel + out_terms + err_terms
    # Safety for cuSOLVER workspaces, allocator overhead, ping-pong staging, etc.
    safety = 1.5
    bytes_f64 = 8.0
    est = bytes_f64 * safety * (floats_per_sample + 64)
    # Optional runtime tuner: multiply estimate by a scale factor
    # e.g., MULTIGPU_BPS_SCALE=0.9 to attempt slightly larger batches.
    try:
        scale = float(os.environ.get("MULTIGPU_BPS_SCALE", "1.0"))
        scale = float(min(max(scale, 0.25), 4.0))
    except Exception:
        scale = 1.0
    return int(est * scale)


def _adaptive_batch_size(na: int, nf: int, nt: int, nmu: int) -> int:
    """Pick a batch size based on free GPU memory and problem size.

    Uses CuPy's memGetInfo and targets a fraction of free memory controlled
    by the environment variable MULTIGPU_BATCH_MEM_FRAC.

    Args:
        na (int): Number of samples.
        nf (int): Number of filters.
        nt (int): Number of temperature bins.
        nmu (int): Number of candidate regularization values.

    Returns:
        int: Chosen batch size (at least 1, at most na).
    """
    import cupy as cp

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
            # Default target fraction of free memory; configurable via env
            # Keep consistent with demmap_pos OOM paths (default 0.7)
            frac_env = float(os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7"))
            mem_frac = float(min(max(frac_env, 0.1), 0.9))
        except Exception:
            mem_frac = 0.7
        est = int((free_b * mem_frac) // bytes_per)
        return max(1, min(est, na))
    except Exception:
        return default


def estimate_batch_plan(na: int, nf: int, nt: int, nmu: int) -> Dict[str, Any]:
    """Provide a batch sizing and memory usage estimate.

    Args:
        na (int): Number of samples.
        nf (int): Number of filters.
        nt (int): Number of temperature bins.
        nmu (int): Number of candidate regularization values.

    Returns:
        Dict[str, Any]: Dictionary with keys: batch_size, bytes_per_sample,
        free_bytes, est_batch_bytes, num_batches.
    """
    import cupy as cp

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


__all__ = ["estimate_batch_plan", "_bytes_per_sample_estimate", "_adaptive_batch_size"]
