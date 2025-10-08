"""
Modular GPU kernel package for multiGPU DEM inversion.

This package mirrors the organization of the vendor baseline by splitting
kernels and helpers into focused modules:

- demmap_pos: Batched CuPy-accelerated DEM reconstruction
- dem_inv_gsvd: GSVD-equivalent factorization via SVD of A @ pinv(B)
- dem_reg_map: Discrepancy-principle regularization parameter selection
- linalg: Numerically safe SVD and pseudo-inverse helpers
- memory: Batch-size and memory footprint estimation utilities
- utils: NVTX ranges, verbosity toggles, pinned host memory helpers

Public API: import directly from this package, e.g.:

    from src.multiGPU.kernels import demmap_pos, estimate_batch_plan
"""

from .demmap_pos import demmap_pos  # noqa: F401
from .dem_inv_gsvd import dem_inv_gsvd  # noqa: F401
from .dem_reg_map import dem_reg_map  # noqa: F401
from .linalg import safe_svd, safe_pinv  # noqa: F401
from .memory import estimate_batch_plan  # noqa: F401
from .utils import nvtx_range, verbose_enabled  # noqa: F401

__all__ = [
    "demmap_pos",
    "dem_inv_gsvd",
    "dem_reg_map",
    "safe_svd",
    "safe_pinv",
    "estimate_batch_plan",
    "nvtx_range",
    "verbose_enabled",
]
