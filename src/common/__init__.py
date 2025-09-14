"""
Common utilities shared across CPU/GPU/Dask variants.

Typical imports:
    from src.common import load_np_stack, prepare_synthetic_responses
    from src.common import dn2dem, dem_to_temp_maps
    from src.common import run_baseline_suite
"""

from .io import load_np_stack, default_files, frame_for_solver
from .responses import prepare_synthetic_responses
from .dem_api import dn2dem
from .post import dem_to_temp_maps
from .profiling import run_baseline_suite

__all__ = (
    "load_np_stack",
    "default_files",
    "frame_for_solver",
    "prepare_synthetic_responses",
    "dn2dem",
    "dem_to_temp_maps",
    "run_baseline_suite",
)
