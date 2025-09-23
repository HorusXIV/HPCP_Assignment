# src/baseline/vendor/__init__.py
"""
Vendorized reference solver (provided code). Do not modify solver logic.

This __init__ adds this directory to sys.path so the vendor modules can
import each other using their original absolute names (e.g., 'demmap_pos').
"""

from pathlib import Path
import sys as _sys

_vendor_dir = Path(__file__).parent
if str(_vendor_dir) not in _sys.path:
    _sys.path.insert(0, str(_vendor_dir))

# Optional: re-export for convenience
from .dn2dem_pos import dn2dem_pos  # noqa: E402
from .demmap_pos import demmap_pos, dem_pix  # noqa: E402
from .dem_inv_gsvd import dem_inv_gsvd  # noqa: E402
from .dem_reg_map import dem_reg_map  # noqa: E402

__all__ = ("dn2dem_pos", "demmap_pos", "dem_pix", "dem_inv_gsvd", "dem_reg_map")
