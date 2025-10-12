# src/baseline/vendor/__init__.py
"""
Vendorized reference solver (provided code). Do not modify solver logic.

This __init__ adds this directory to sys.path so the vendor modules can
import each other using the original upstream names.
"""

import sys
from pathlib import Path

# Ensure this package directory is importable first
_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

# Re-export the main entry points for convenience
try:
    from .demmap_pos import demmap_pos
    from .dn2dem_pos import dn2dem_pos
    from .dem_inv_gsvd import dem_inv_gsvd
    from .dem_reg_map import dem_reg_map
except Exception as _e:  # pragma: no cover
    # Keep import errors soft during tooling time; real runs will fail loudly.
    pass
