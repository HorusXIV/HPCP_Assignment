"""I/O helpers for multiGPU package.

We provide a minimal loader to read `.npz` files used by the assignment data
and convert them to structures that the compute kernels expect.
"""
import numpy as np
from typing import Any, Dict


def load_npz(path: str) -> Dict[str, Any]:
    """Load an npz file and return a dict of arrays."""
    data = np.load(path, allow_pickle=True)
    return dict(data)


def ensure_2d_dn(dn: np.ndarray) -> np.ndarray:
    """Ensure `dn` has shape (n_samples, n_filters) as expected by demmap_pos wrapper.

    Accepts 1D/2D/3D and flattens spatial dims to the first axis.
    """
    dn = np.asarray(dn)
    if dn.ndim == 1:
        return dn.reshape(1, -1)
    if dn.ndim >= 2:
        # collapse leading dims except last (filters)
        return dn.reshape(-1, dn.shape[-1])
    return dn
