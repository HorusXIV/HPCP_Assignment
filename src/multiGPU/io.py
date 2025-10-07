"""I/O helpers for the multi-GPU DEM solver.

This module provides small utilities to load ``.npz`` inputs and coerce array
shapes to match the GPU kernels' expectations, keeping orchestration code in
``main.py`` concise.
"""

import numpy as np
from typing import Any, Dict


def load_npz(path: str) -> Dict[str, Any]:
    """Load ``.npz`` content into a regular dictionary.

    Args:
        path: Filesystem path to a ``.npz`` file.

    Returns:
        Mapping from array names to NumPy arrays or stored objects.
    """
    data = np.load(path, allow_pickle=True)
    return dict(data)


def ensure_2d_dn(dn: np.ndarray) -> np.ndarray:
    """Coerce a measurement array to shape ``(n_samples, n_filters)``.

    Accepts 1D, 2D, or 3D inputs and flattens leading spatial dimensions to
    the sample axis.

    Args:
        dn: Input array of band measurements.

    Returns:
        Two-dimensional array with samples along axis 0 and filters along
        axis 1.
    """
    dn = np.asarray(dn)
    if dn.ndim == 1:
        return dn.reshape(1, -1)
    if dn.ndim >= 2:
        return dn.reshape(-1, dn.shape[-1])
    return dn
