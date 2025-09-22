# src/common/dataio/__init__.py
"""
Data I/O public API (stable entry point).

Usage:
    from src.common.dataio import default_files, build_lazy_npz_stack
    from src.common.dataio import load_np_stack, frame_for_solver
    from src.common.dataio import write_manifest_and_hash
"""

from .files import default_files
from .datasets import build_lazy_npz_stack
from .stacks import load_np_stack, frame_for_solver
from .manifest import write_manifest_and_hash

__all__ = [
    "default_files",
    "build_lazy_npz_stack",
    "load_np_stack",
    "frame_for_solver",
    "write_manifest_and_hash",
]
