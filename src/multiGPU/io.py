"""I/O helpers for the multi-GPU DEM solver.

Minimal utilities to load ``.npz`` inputs and ensure array shapes match the
expectations of the GPU kernels.
"""

import numpy as np
from typing import Any, Dict


def load_npz(path: str) -> Dict[str, Any]:
    """Load ``.npz`` content into a regular dictionary.

    Args:
        path: Filesystem path to a ``.npz`` file.

    Returns:
        Dict mapping array names to NumPy arrays or objects stored within.
    """
    data = np.load(path, allow_pickle=True)
    return dict(data)


def ensure_2d_dn(dn: np.ndarray) -> np.ndarray:
    """Return ``dn`` with shape ``(n_samples, n_filters)``.

    Accepts 1D/2D/3D inputs and flattens leading spatial dimensions to the
    sample axis. This is a convenience for simple dataset variants.

    Args:
        dn: Input array of band measurements.

    Returns:
        2D NumPy array with samples along axis 0 and filters along axis 1.
    """
    dn = np.asarray(dn)
    if dn.ndim == 1:
        return dn.reshape(1, -1)
    if dn.ndim >= 2:
        # collapse leading dims except last (filters)
        return dn.reshape(-1, dn.shape[-1])
    return dn
