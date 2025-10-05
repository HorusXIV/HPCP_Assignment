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
    """Estimate peak bytes per sample used by the batched solver (Float64).

    Args:
        nf (int): Number of filters.
        nt (int): Number of temperature bins.
        nmu (int): Number of candidate regularization values.

    Returns:
        int: Estimated peak bytes per sample.
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
            frac_env = float(os.environ.get("MULTIGPU_BATCH_MEM_FRAC", "0.7"))
            mem_frac = float(min(max(frac_env, 0.1), 0.9))
        except Exception:
            mem_frac = 0.55
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
